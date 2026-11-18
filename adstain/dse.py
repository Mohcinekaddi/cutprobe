import hashlib
import json
import math
import os
import time

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize

from . import analytical as am
from .config import resources as load_resources
from .onnx_graph import forbidden_cuts

PENALTY = 1e6


def decode(x, n_free):
    x = np.asarray(x, dtype=int)
    return np.concatenate(([0], x[:n_free])), np.argsort(x[n_free:], kind="stable")


def perm_to_keys(perm):
    perm = np.asarray(perm, dtype=int)
    inv = np.empty(perm.size, dtype=int)
    inv[perm] = np.arange(perm.size)
    return inv


def encode(cuts, perm):
    return np.concatenate([np.asarray(cuts, dtype=int)[1:], perm_to_keys(perm)])


def equiv_groups(resources):
    groups = {}
    for i, r in enumerate(resources):
        groups.setdefault(r.equiv_key, []).append(i)
    return groups


def canonical_perm(perm, groups):
    perm = np.asarray(perm, dtype=int).copy()
    for members in groups.values():
        if len(members) < 2:
            continue
        member_set = set(members)
        positions = [s for s, r in enumerate(perm) if r in member_set]
        for pos, member in zip(positions, sorted(members)):
            perm[pos] = member
    return perm


def nearest_valid(pos, forbidden, lo, hi):
    pos = int(np.clip(pos, lo, hi))
    if pos not in forbidden:
        return pos
    for d in range(1, hi - lo + 1):
        if pos + d <= hi and pos + d not in forbidden:
            return pos + d
        if pos - d >= lo and pos - d not in forbidden:
            return pos - d
    return pos


class CutRepair(Repair):
    def __init__(self, model_len, n_ranks, min_segment, groups, canonicalize, forbidden):
        super().__init__()
        self.model_len = model_len
        self.n_free = n_ranks - 1
        self.m = min_segment
        self.groups = groups
        self.canonicalize = canonicalize
        self.forbidden = set(forbidden)
        self.lower = np.array([(j + 1) * self.m for j in range(self.n_free)], dtype=int)
        self.upper = np.array([model_len - (self.n_free - j) * self.m for j in range(self.n_free)],
                              dtype=int)

    def _keys(self, keys):
        perm = np.argsort(np.asarray(keys, dtype=int), kind="stable")
        if self.canonicalize:
            perm = canonical_perm(perm, self.groups)
        return perm_to_keys(perm)

    def _avoid_forbidden(self, cuts):
        if not self.forbidden:
            return cuts
        out = cuts.copy()
        prev = 0
        for j in range(len(out)):
            lo = max(int(self.lower[j]), prev + self.m)
            hi = int(self.upper[j])
            if hi < lo:
                out[j] = int(np.clip(out[j], 0, self.model_len - 1))
            else:
                out[j] = nearest_valid(int(np.clip(out[j], lo, hi)), self.forbidden, lo, hi)
            prev = out[j]
        return out

    def _do(self, problem, Z, **kwargs):
        Z = np.atleast_2d(np.asarray(Z, dtype=int)).copy()
        nc, m = self.n_free, self.m
        for i in range(len(Z)):
            row = Z[i].copy()
            if nc:
                free = np.sort(np.clip(row[:nc], 0, self.model_len - 1))
                free = np.clip(free, self.lower, self.upper)
                fwd, bwd = free.copy(), free.copy()
                for j in range(1, nc):
                    fwd[j] = max(fwd[j], fwd[j - 1] + m)
                for j in range(nc - 2, -1, -1):
                    bwd[j] = min(bwd[j], bwd[j + 1] - m)
                comb = np.round((fwd + bwd) / 2.0).astype(int)
                for j in range(1, nc):
                    comb[j] = max(comb[j], comb[j - 1] + m)
                comb = np.clip(comb, self.lower, self.upper)
                for j in range(nc - 2, -1, -1):
                    comb[j] = min(comb[j], comb[j + 1] - m)
                row[:nc] = self._avoid_forbidden(comb)
            row[nc:] = self._keys(row[nc:])
            Z[i] = row
        return Z


class CutSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        nc = problem.n_free
        X = np.empty((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            X[i, :nc] = np.random.randint(problem.xl[:nc], problem.xu[:nc] + 1)
            X[i, nc:] = np.random.permutation(problem.n_ranks)
        return problem.repair_op._do(problem, X)


class CutPlacementMutation(PM):
    def __init__(self, n_free, n_ranks, prob_swap=0.5, **kwargs):
        super().__init__(**kwargs)
        self.n_free = n_free
        self.n_ranks = n_ranks
        self.prob_swap = prob_swap

    def _do(self, problem, X, **kwargs):
        Xp = np.array(super()._do(problem, X, **kwargs), dtype=float)
        nc = self.n_free
        for i in range(Xp.shape[0]):
            if self.n_ranks < 2 or np.random.random() > self.prob_swap:
                continue
            perm = np.argsort(np.round(Xp[i, nc:]).astype(int), kind="stable")
            a, b = np.random.choice(self.n_ranks, size=2, replace=False)
            perm[[a, b]] = perm[[b, a]]
            Xp[i, nc:] = perm_to_keys(perm)
        return Xp


class AnalyticEvaluator:
    name = "analytic"
    verbose = False

    def __init__(self, model):
        self.model = model

    def __call__(self, cuts, placement):
        stage_ms, mem = self.model.predict(cuts, placement)
        return {"stage_ms": stage_ms, "memory_mb": mem, "period_ms": max(stage_ms)}


class DryEvaluator:
    name = "dry"
    verbose = True

    def __init__(self, model_len, res_by_name):
        self.model_len = model_len
        self.res = res_by_name
        self.runs = 0

    def __call__(self, cuts, placement):
        self.runs += 1
        sizes = np.diff(np.append(np.asarray(cuts), self.model_len)).astype(float)
        speed = np.array([4.0 if self.res[p].kind == "gpu" else self.res[p].threads * 0.5
                          for p in placement])
        stage = list(sizes / speed * 10.0)
        mem = list(sizes * 12.0 + 100.0)
        return {"stage_ms": stage, "compute_ms": stage, "wall_ms": [max(stage)] * len(stage),
                "memory_mb": mem, "period_ms": max(stage)}

    def remeasure(self, cuts, placement, n):
        return [self(cuts, placement) for _ in range(n)]


class MeasureEvaluator:
    name = "measure"
    verbose = True

    def __init__(self, deployer, log_path, signature):
        self.deployer = deployer
        self.log_path = log_path
        self.signature = signature
        self.cache = {}
        if os.path.exists(log_path):
            with open(log_path) as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec.get("signature") == signature and "error" not in rec:
                        self.cache[(tuple(rec["cuts"]), tuple(rec["placement"]))] = rec["metrics"]
            if self.cache:
                print(f"[measure] {len(self.cache)} earlier measurement(s) reused from {log_path}")

    @property
    def runs(self):
        return self.deployer.runs

    def __call__(self, cuts, placement):
        key = (tuple(int(c) for c in cuts), tuple(placement))
        if key in self.cache:
            m = self.cache[key]
        else:
            try:
                m = self.deployer.measure(list(key[0]), list(key[1]))
            except Exception as e:
                self._log({"cuts": list(key[0]), "placement": list(key[1]), "error": str(e)})
                raise
            self.cache[key] = m
            self._log({"cuts": list(key[0]), "placement": list(key[1]), "metrics": m})
        return {"stage_ms": list(m["compute_ms"]), "compute_ms": list(m["compute_ms"]),
                "wall_ms": list(m["wall_ms"]), "memory_mb": list(m["memory_mb"]),
                "period_ms": max(m["wall_ms"])}

    def remeasure(self, cuts, placement, n):
        out = []
        for _ in range(n):
            try:
                m = self.deployer.measure(list(cuts), list(placement))
            except Exception as e:
                print(f"  [repeat] cuts={list(cuts)}: {str(e).splitlines()[0]}")
                continue
            self._log({"cuts": list(cuts), "placement": list(placement), "metrics": m, "repeat": True})
            out.append(m)
        return out

    def records(self):
        return [{"cuts": list(c), "placement": list(p), **m} for (c, p), m in self.cache.items()]

    def _log(self, rec):
        rec = dict(rec, signature=self.signature, time=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(self.log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")


class PartitionProblem(Problem):
    def __init__(self, model_len, resources, evaluator, min_segment, canonicalize, forbidden):
        self.model_len = model_len
        self.resources = list(resources)
        self.names = [r.name for r in self.resources]
        self.n_ranks = len(self.resources)
        self.n_free = self.n_ranks - 1
        self.min_segment = min_segment
        self.forbidden = set(forbidden)
        self.evaluator = evaluator
        self.repair_op = CutRepair(model_len, self.n_ranks, min_segment,
                                   equiv_groups(self.resources), canonicalize, self.forbidden)
        self.history = []
        self.cache = {}
        if min_segment * self.n_ranks > model_len:
            raise ValueError(f"{self.n_ranks} segments of at least {min_segment} layers do not "
                             f"fit in {model_len} layers")
        n_var = self.n_free + self.n_ranks
        xl = np.zeros(n_var, dtype=int)
        xu = np.full(n_var, self.n_ranks - 1, dtype=int)
        xl[:self.n_free] = self.repair_op.lower
        xu[:self.n_free] = self.repair_op.upper
        super().__init__(n_var=n_var, n_obj=2, n_ieq_constr=1 if self.forbidden else 0,
                         xl=xl, xu=xu, vtype=int)

    def describe(self, x):
        cuts, perm = decode(x, self.n_free)
        return [int(c) for c in cuts], [self.names[int(p)] for p in perm]

    def _evaluate(self, X, out, *args, **kwargs):
        n = len(X)
        F = np.full((n, 2), PENALTY)
        G = np.zeros((n, 1))
        rng = np.random.default_rng()
        verbose = self.evaluator.verbose
        for i in range(n):
            x = np.asarray(X[i], dtype=int)
            key = tuple(x.tolist())
            if key in self.cache:
                F[i], G[i] = self.cache[key]
                continue
            cuts, placement = self.describe(x)
            bad = [c for c in cuts if c in self.forbidden]
            G[i, 0] = float(len(bad))
            if bad:
                F[i] = PENALTY + rng.uniform(0, 1e-3)
                self.history.append({"cuts": cuts, "placement": placement,
                                     "error": f"forbidden cut(s) {bad}"})
            else:
                try:
                    r = self.evaluator(cuts, placement)
                    rec = self._record(cuts, placement, r)
                    F[i, 0] = -rec["throughput"]
                    F[i, 1] = rec["max_rank_memory"]
                    self.history.append(rec)
                    if verbose:
                        print(f"  [{self.evaluator.name}] cuts={cuts} {' | '.join(placement)}  "
                              f"T={rec['throughput']:.3f} img/s  M={rec['max_rank_memory']:.1f} MB  "
                              f"bottleneck={rec['bottleneck_resource']}")
                except Exception as e:
                    F[i] = PENALTY + rng.uniform(0, 1e-3)
                    self.history.append({"cuts": cuts, "placement": placement, "error": str(e)})
                    if verbose:
                        print(f"  [invalid] cuts={cuts} {' | '.join(placement)}: "
                              f"{str(e).splitlines()[0]}")
            self.cache[key] = (F[i].copy(), G[i].copy())
        out["F"] = F
        if self.n_ieq_constr:
            out["G"] = G

    def _record(self, cuts, placement, r):
        stage = r["stage_ms"]
        b = int(np.argmax(stage))
        rec = {"cuts": cuts, "placement": placement,
               "segment_sizes": np.diff(np.append(cuts, self.model_len)).tolist(),
               "bottleneck_rank": b, "bottleneck_resource": placement[b],
               "throughput": 1000.0 / r["period_ms"], "seconds_per_image": r["period_ms"] / 1000.0,
               "max_rank_memory": max(r["memory_mb"]), "stage_ms": stage,
               "memory_mb": r["memory_mb"], "source": self.evaluator.name}
        for k in ("compute_ms", "wall_ms"):
            if k in r:
                rec[k] = r[k]
        return rec


def promote(X, prev, nxt):
    X = np.atleast_2d(np.asarray(X, dtype=int))
    if nxt.n_ranks <= prev.n_ranks or len(X) == 0:
        return X
    index_of = {n: i for i, n in enumerate(nxt.names)}
    new_slots = [i for i, n in enumerate(nxt.names) if n not in prev.names]
    m = nxt.min_segment
    out = []
    for x in X:
        cuts, perm = decode(x, prev.n_free)
        bounds = list(cuts) + [prev.model_len]
        new_cuts, new_perm, spare = [], [], list(new_slots)
        for s in range(prev.n_ranks):
            a, b = bounds[s], bounds[s + 1]
            old = index_of[prev.names[int(perm[s])]]
            mid = a + max((b - a) // 2, m)
            if mid > b - m or not spare:
                new_cuts.append(a)
                new_perm.append(old)
                continue
            new_cuts += [a, mid]
            new_perm += [old, spare.pop(0)]
        while len(new_cuts) < nxt.n_ranks and spare:
            new_cuts.append(min(new_cuts[-1] + m, nxt.model_len - m))
            new_perm.append(spare.pop(0))
        if len(new_cuts) == nxt.n_ranks:
            out.append(encode(np.array(new_cuts), np.array(new_perm)))
    if not out:
        return np.empty((0, nxt.n_var), dtype=int)
    return nxt.repair_op._do(nxt, np.array(out, dtype=int))


def seed_population(problem, X0, pop_size):
    if X0 is not None and len(X0):
        X0 = problem.repair_op._do(problem, np.atleast_2d(np.asarray(X0, dtype=int)))
        _, idx = np.unique(X0, axis=0, return_index=True)
        X0 = X0[np.sort(idx)][:pop_size]
    else:
        X0 = np.empty((0, problem.n_var), dtype=int)
    missing = pop_size - len(X0)
    if missing > 0:
        fresh = CutSampling()._do(problem, missing)
        X0 = np.vstack([X0, fresh]) if len(X0) else fresh
    return np.asarray(X0, dtype=int)


def run_level(problem, pop, gen, X0, sc, label):
    t0 = time.time()
    print(f"\n  >>> {label}: pop={pop}, gen={gen}, {problem.n_ranks} resource(s)")
    algorithm = NSGA2(
        pop_size=pop,
        sampling=seed_population(problem, X0, pop),
        crossover=SBX(prob=sc["crossover"], eta=15, vtype=float, repair=RoundingRepair()),
        mutation=CutPlacementMutation(problem.n_free, problem.n_ranks, prob_swap=sc["swap"],
                                      prob=sc["mutation"], eta=20, vtype=float,
                                      repair=RoundingRepair()),
        repair=problem.repair_op,
        eliminate_duplicates=True,
    )
    res = minimize(problem, algorithm, ("n_gen", gen), seed=sc["seed"], verbose=False)
    if res.X is None:
        print("      no feasible solution")
        return np.empty((0, problem.n_var), dtype=int), np.empty((0, 2))
    X = np.atleast_2d(res.X).astype(int)
    F = np.atleast_2d(res.F)
    print(f"      {len(X)} non-dominated, {len(problem.cache)} evaluations, {time.time() - t0:.1f} s")
    for x, f in zip(X, F):
        cuts, placement = problem.describe(x)
        print(f"      cuts={str(cuts):<26} T={-f[0]:9.4f} img/s  M={f[1]:8.1f} MB  "
              + " | ".join(placement))
    return X, F


def _signature(cfg, model_path, resources, task=None):
    ad = cfg["autodice"]
    pp = {k: v for k, v in cfg["preprocess"].items()
          if not (k == "letterbox" and not v) and not (k == "pad_value" and not cfg["preprocess"].get("letterbox"))}
    sig = {"model": model_path, "res": [(r.name, r.slots, r.kind, r.device) for r in resources],
           "warmup": ad["warmup"], "loops": ad["loops"],
           "det": ad["deterministic"], "image": ad["bench_image"],
           "pp": pp, "mpi": ad["mpirun_args"], "timing": "median"}
    if task is not None and task.detection:
        sig["head"] = task.head["format"]
    return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]


def prepare(cfg):
    from .model import prepare as prepare_model
    return prepare_model(cfg)


def make_measure_evaluator(cfg, model_path, model, resources, task=None):
    from .deploy import Deployer
    deployer = Deployer(cfg, model, resources, task=task)
    return MeasureEvaluator(deployer, os.path.join(cfg["workdir"], "measurements.jsonl"),
                            _signature(cfg, model_path, resources, task))


def spread_subset(X, F, n):
    if len(X) <= n:
        return X
    order = np.argsort(F[:, 0])
    pick = np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))
    return X[order[pick]]


def _all_records(measurer, history):
    recs = am.records_from_history(history)
    seen = {(tuple(r["cuts"]), tuple(r["placement"])) for r in recs}
    if hasattr(measurer, "records"):
        for r in am.records_from_history(measurer.records()):
            if (tuple(r["cuts"]), tuple(r["placement"])) not in seen:
                recs.append(r)
    return recs


def refit(costs, resources, old, records, lam, path, when):
    if len(records) < 4:
        return old
    new = am.calibrate(costs, resources, records, lam, verbose=False)
    for name, th in old.theta_time.items():
        new.theta_time.setdefault(name, th)
    for name, th in old.theta_mem.items():
        new.theta_mem.setdefault(name, th)
    new.save(path)
    e = am.objective_errors(new, records)
    print(f"[calib] {when}: model refitted on {len(records)} measured mapping(s) "
          f"(fit error throughput {e[0]:.1f} %, memory {e[1]:.1f} %), saved {path}")
    return new


def calibrate_from_measurements(cfg, costs, resources, evaluator, forbidden, out_path):
    sc = cfg["search"]
    plans = am.plan_calibration(costs, resources, sc["min_segment"], sc["calib_runs"],
                                forbidden=forbidden)
    print(f"\n[calib] measuring {len(plans)} mapping(s) chosen for identifiability")
    problem = PartitionProblem(costs.n, resources, evaluator, sc["min_segment"],
                               sc["canonicalize"], forbidden)
    index = {r.name: i for i, r in enumerate(resources)}
    X = [encode(np.array(c), np.array([index[p] for p in pl])) for c, pl in plans]
    X = problem.repair_op._do(problem, np.array(X, dtype=int))
    problem._evaluate(X, {})
    records = am.records_from_history(problem.history)
    if len(records) < 2:
        raise SystemExit(f"only {len(records)} mapping(s) measured successfully: cannot "
                         f"calibrate (see {cfg['workdir']}/deploy/*.log)")
    model = am.calibrate(costs, resources, records, sc["ridge_lambda"])
    am.cross_validate(costs, resources, records, sc["ridge_lambda"])
    model.save(out_path)
    print(f"[calib] written {out_path}")
    return model, problem.history


def search(cfg, levels=None, dry=False):
    sc = cfg["search"]
    levels = levels or sc["levels"]
    if levels not in ("analytic+measure", "analytic", "measure"):
        raise ValueError(f"unknown levels {levels!r}")
    all_res = load_resources(cfg)
    by_name = {r.name: r for r in all_res}
    order = [by_name[n] for n in (sc["resource_order"] or [r.name for r in all_res])]
    model_path, model, task = prepare(cfg)
    n_layers = len(model.graph.node)
    forbidden = forbidden_cuts(model, cfg["autodice"]["dummy_marker"], sc["dummy_offsets"])
    n_pe = len(order)
    K = max(int(math.ceil(math.log2(n_pe))), 1) if sc["multistage"] and n_pe > 1 else 1
    results_path, calib_path = sc["results"], sc["calibration"]
    if dry:
        results_path = os.path.splitext(results_path)[0] + "_dry.json"
        calib_path = os.path.splitext(calib_path)[0] + "_dry.json"

    print("=" * 74)
    print(f"model {os.path.basename(cfg['model'])}: {n_layers} layers")
    print(f"resources: {[r.name for r in order]}")
    print(f"levels={levels}, stages K={K}, dry_run={dry}")
    if forbidden:
        print(f"forbidden cut positions ({len(forbidden)}): {sorted(forbidden)}")
    print("=" * 74)

    need_measure = levels != "analytic" or (sc["auto_calibrate"] and not os.path.exists(calib_path))
    if dry:
        measurer = DryEvaluator(n_layers, by_name)
    elif need_measure:
        measurer = make_measure_evaluator(cfg, model_path, model, order, task)
    else:
        measurer = None

    measured = []
    analytic = None
    if levels != "measure":
        costs = am.LayerCosts(model_path)
        print(f"\nstatic costs: {costs.macs.sum() / 1e9:.2f} GMAC, "
              f"{costs.m_coeffs.sum() * 4 / 1e6:.0f} MB of weights")
        if os.path.exists(calib_path):
            try:
                analytic = am.AnalyticalModel.load(calib_path, costs, all_res)
                print(f"calibration loaded: {calib_path}")
            except RuntimeError as e:
                print(f"[calib] {e}\n[calib] tip: 'calibrate --history <workdir>/measurements.jsonl' "
                      f"refits it from earlier measurements without new deployments")
                if not (sc["auto_calibrate"] and measurer is not None):
                    raise SystemExit(1)
        if analytic is None and sc["auto_calibrate"] and measurer is not None:
            analytic, hist = calibrate_from_measurements(cfg, costs, order, measurer, forbidden,
                                                         calib_path)
            measured += hist
        elif analytic is None:
            print("no calibration file: level 1 uses uncalibrated priors")
            analytic = am.AnalyticalModel(costs, all_res)

    stages, prev_problem, prev_X = [], None, None
    X = F = problem = None
    for k in range(1, K + 1):
        subset = order[:2 ** k] if K > 1 else order
        print("\n" + "-" * 74)
        print(f"STAGE {k}/{K}: {len(subset)} PE: {', '.join(r.name for r in subset)}")
        print("-" * 74)
        stage = {"stage": k, "resources": [r.name for r in subset]}

        if levels != "measure":
            p1 = PartitionProblem(n_layers, subset, AnalyticEvaluator(analytic),
                                  sc["min_segment"], sc["canonicalize"], forbidden)
            X0 = promote(prev_X, prev_problem, p1) if prev_X is not None else None
            X1, F1 = run_level(p1, sc["l1_pop"], sc["l1_gen"], X0, sc, "LEVEL 1 (analytical model)")
            stage["level1"] = _front(p1, X1, F1)
            X, F, problem = X1, F1, p1

        if levels != "analytic":
            p2 = PartitionProblem(n_layers, subset, measurer, sc["min_segment"],
                                  sc["canonicalize"], forbidden)
            if X is not None and len(X) and problem is not None and problem.n_ranks == p2.n_ranks:
                X0 = spread_subset(X, F, sc["l2_pop"])
            else:
                X0 = promote(prev_X, prev_problem, p2) if prev_X is not None else None
            X2, F2 = run_level(p2, sc["l2_pop"], sc["l2_gen"], X0, sc,
                               "LEVEL 2 (dry run)" if dry else "LEVEL 2 (AutoDiCE measurement)")
            stage["level2"] = _front(p2, X2, F2)
            measured += p2.history
            X, F, problem = X2, F2, p2
            if sc["recalibrate"] and analytic is not None and levels == "analytic+measure" and k < K:
                analytic = refit(costs, all_res, analytic, _all_records(measurer, measured),
                                 sc["ridge_lambda"], calib_path, f"after stage {k}")

        stages.append(stage)
        prev_X, prev_problem = X, problem

    best_cuts = [problem.describe(x)[0] for x in X]
    best_place = [problem.describe(x)[1] for x in X]
    keep = [i for i, c in enumerate(best_cuts) if not set(c) & forbidden]
    if len(keep) < len(best_cuts):
        print(f"[search] {len(best_cuts) - len(keep)} solution(s) dropped: forbidden cut")
    X, F = X[keep], F[keep]
    best_cuts = [best_cuts[i] for i in keep]
    best_place = [best_place[i] for i in keep]

    print("\n===== Pareto front (throughput / memory) =====")
    for cuts, placement, f in zip(best_cuts, best_place, F):
        print(f"cuts {cuts}  " + " | ".join(f"r{r}={n}" for r, n in enumerate(placement)))
        print(f"   throughput {-f[0]:.4f} img/s ({1.0 / -f[0]:.4f} s/img), memory {f[1]:.1f} MB")

    repeats = None
    n_rep = int(sc["final_repeats"])
    if n_rep > 0 and best_cuts and hasattr(problem.evaluator, "remeasure"):
        repeats = []
        print(f"\n[repeat] measuring the {len(best_cuts)} front solution(s) {n_rep} more time(s)")
        spread_t, spread_m = [], []
        for cuts, placement, f in zip(best_cuts, best_place, F):
            runs = problem.evaluator.remeasure(cuts, placement, n_rep)
            thr = [float(-f[0])] + [1000.0 / max(m["wall_ms"]) for m in runs]
            mem = [float(f[1])] + [max(m["memory_mb"]) for m in runs]
            rep = {"cuts": cuts, "placement": placement, "n": len(thr),
                   "throughput_mean": float(np.mean(thr)), "throughput_std": float(np.std(thr)),
                   "memory_mean": float(np.mean(mem)), "memory_std": float(np.std(mem)),
                   "throughput_runs": thr, "memory_runs": mem}
            repeats.append(rep)
            spread_t.append(rep["throughput_std"] / max(rep["throughput_mean"], 1e-9))
            spread_m.append(rep["memory_std"] / max(rep["memory_mean"], 1e-9))
            print(f"  cuts {str(cuts):<24} {rep['throughput_mean']:8.3f} +/- {rep['throughput_std']:.3f} img/s   "
                  f"{rep['memory_mean']:7.1f} +/- {rep['memory_std']:.1f} MB")
        print(f"[repeat] run-to-run spread: throughput {100 * np.mean(spread_t):.1f} %, "
              f"memory {100 * np.mean(spread_m):.1f} % (no model can be more precise than this)")

    validation = None
    if analytic is not None and problem.evaluator.name != "analytic" and best_cuts:
        validation = []
        et, em = [], []
        for i, (cuts, placement, f) in enumerate(zip(best_cuts, best_place, F)):
            t_pred, m_pred = analytic.predict(cuts, placement)
            tm = repeats[i]["throughput_mean"] if repeats else -f[0]
            mm = repeats[i]["memory_mean"] if repeats else f[1]
            tp, mp = 1000.0 / max(t_pred), max(m_pred)
            validation.append({"cuts": cuts, "placement": placement,
                               "throughput_measured": float(tm), "throughput_predicted": float(tp),
                               "memory_measured": float(mm), "memory_predicted": float(mp)})
            et.append(abs(tp - tm) / max(tm, 1e-9))
            em.append(abs(mp - mm) / max(mm, 1e-9))
        print(f"\n[validation] analytical vs measured on the final front: throughput "
              f"{100 * np.mean(et):.1f} % mean error, memory {100 * np.mean(em):.1f} %")
        if sc["recalibrate"] and levels == "analytic+measure":
            refit(costs, all_res, analytic, _all_records(measurer, measured), sc["ridge_lambda"],
                  calib_path, "end of search (for the next run)")

    seen, raw = set(), []
    for h in measured:
        if "error" in h:
            continue
        key = (tuple(h["cuts"]), tuple(h["placement"]))
        if key not in seen:
            seen.add(key)
            raw.append(h)

    results = {
        "model": cfg["model"],
        "formatted_model": model_path,
        "task": task.kind,
        "methodology": f"hierarchical NSGA-II (Guo et al., IEEE IoT-J 2023): K={K}, levels={levels}",
        "objectives": ["throughput_img_per_s (max)", "max_rank_rss_mb (min)"],
        "resources": {str(i + 1): r.name for i, r in enumerate(order)},
        "forbidden_cuts": sorted(forbidden),
        "budgets": {"l1": [sc["l1_pop"], sc["l1_gen"]], "l2": [sc["l2_pop"], sc["l2_gen"]]},
        "autodice_runs": getattr(measurer, "runs", 0),
        "calibration": calib_path if analytic is not None and analytic.calibrated else None,
        "dry_run": dry,
        "stages": stages,
        "validation": validation,
        "repeats": repeats,
        "best_x": X.tolist(),
        "best_cuts": best_cuts,
        "best_placements": best_place,
        "best_genes": best_cuts,
        "best_objectives": [[float(-f[0]), float(f[1])] for f in F],
        "seconds_per_image": [float(1.0 / -f[0]) for f in F],
        "raw_history": raw,
    }
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nAutoDiCE runs: {results['autodice_runs']}\nwritten {results_path}")
    try:
        from .plots import plot_pareto
        png = os.path.splitext(results_path)[0] + "_pareto.png"
        if plot_pareto(results, png):
            print(f"written {png}")
    except Exception as e:
        print(f"[plot] pareto figure skipped: {e}")
    return results


def _front(problem, X, F):
    return {"cuts": [problem.describe(x)[0] for x in X],
            "placements": [problem.describe(x)[1] for x in X],
            "objectives": [[float(-f[0]), float(f[1])] for f in F],
            "n_evaluations": len(problem.cache)}
