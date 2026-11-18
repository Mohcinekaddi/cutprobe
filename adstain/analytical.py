import json
import os
import sys

import numpy as np

from .onnx_graph import initializer_names, load_model, data_inputs

try:
    from scipy.optimize import nnls
except ImportError:
    nnls = None

ELEM_BYTES = 4.0
GMAC = 1e9
MB = 1e6

ELEMENTWISE_COST = {
    "Relu": 1.0, "LeakyRelu": 2.0, "PRelu": 2.0, "Clip": 1.0, "Sigmoid": 4.0, "Tanh": 4.0,
    "HardSigmoid": 2.0, "HardSwish": 3.0, "Add": 1.0, "Sum": 1.0, "Sub": 1.0, "Mul": 1.0,
    "Div": 2.0, "Concat": 1.0, "Dropout": 0.0, "Identity": 0.0, "Pad": 0.5,
    "Flatten": 0.0, "Reshape": 0.0, "Squeeze": 0.0, "Unsqueeze": 0.0, "Transpose": 1.0,
    "Softmax": 3.0, "BatchNormalization": 2.0, "LRN": 10.0, "GlobalAveragePool": 1.0,
    "GlobalMaxPool": 1.0, "Constant": 0.0, "Shape": 0.0, "Gather": 0.0,
}

TIME_TERMS = ("macs", "n_layers", "one")
MEM_TERMS = ("coeffs_mb", "outbuffs_mb", "inbuffs_mb", "one")
DEFAULT_BANDWIDTH = {"shm": 6.0, "pcie": 6.0, "net": 0.118}
DEFAULT_LATENCY_MS = {"shm": 0.01, "pcie": 0.05, "net": 0.20}
PRIOR_TIME = {"gpu": {"macs": 2.0, "n_layers": 0.20, "one": 1.0},
              "cpu": {"macs": 45.0, "n_layers": 0.50, "one": 2.0}}
PRIOR_MEM = {"gpu": {"coeffs_mb": 2.0, "outbuffs_mb": 0.3, "inbuffs_mb": 1.0, "one": 350.0},
             "cpu": {"coeffs_mb": 1.0, "outbuffs_mb": 0.3, "inbuffs_mb": 1.0, "one": 120.0}}
RIDGE_PRIOR = {"cpu": {"coeffs_mb": 1.0, "outbuffs_mb": 0.5, "inbuffs_mb": 1.0},
               "gpu": {"coeffs_mb": 2.0, "outbuffs_mb": 0.5, "inbuffs_mb": 1.0}}


def _prod(dims):
    out = 1.0
    for d in dims:
        out *= max(int(d), 1)
    return out


def _shape(vi):
    return [d.dim_value if d.HasField("dim_value") else 1
            for d in vi.type.tensor_type.shape.dim]


class LayerCosts:
    def __init__(self, model_path):
        model = load_model(model_path)
        g = model.graph
        self.model_path = os.path.abspath(model_path)
        self.names = [n.name for n in g.node]
        self.op_types = [n.op_type for n in g.node]
        self.n = len(g.node)

        inits = initializer_names(g)
        init_dims = {t.name: list(t.dims) for t in g.initializer}
        shapes = {vi.name: _shape(vi) for vi in list(g.input) + list(g.output) + list(g.value_info)}
        real_inputs = data_inputs(g)
        self.input_numel = _prod(shapes[real_inputs[0]]) if real_inputs and real_inputs[0] in shapes else 0.0

        self.macs = np.zeros(self.n)
        self.m_coeffs = np.zeros(self.n)
        self.m_outbuffs = np.zeros(self.n)
        self.tensor_numel = {}
        self.producer = {}
        self.consumers = {}
        unknown = set()

        for i, node in enumerate(g.node):
            for o in node.output:
                self.producer[o] = i
                self.tensor_numel[o] = _prod(shapes[o]) if o in shapes else 0.0
            for x in node.input:
                if x and x not in inits:
                    self.consumers.setdefault(x, []).append(i)
            self.m_coeffs[i] = sum(_prod(init_dims[x]) for x in node.input if x in init_dims)
            out_numel = self.tensor_numel.get(node.output[0], 0.0)
            self.m_outbuffs[i] = out_numel
            self.macs[i] = self._node_macs(node, out_numel, init_dims, shapes)
            if self.macs[i] == 0.0 and node.op_type not in ELEMENTWISE_COST:
                unknown.add(node.op_type)
        if unknown:
            print("[costs] ops without a MAC model (treated as elementwise): "
                  + ", ".join(sorted(unknown)), file=sys.stderr)

        tensors = list(self.producer)
        self._t_size = np.array([self.tensor_numel.get(t, 0.0) for t in tensors])
        e_t, e_src, e_dst = [], [], []
        for k, t in enumerate(tensors):
            for d in self.consumers.get(t, []):
                e_t.append(k)
                e_src.append(self.producer[t])
                e_dst.append(d)
        self._e_t = np.asarray(e_t, dtype=np.int64)
        self._e_src = np.asarray(e_src, dtype=np.int64)
        self._e_dst = np.asarray(e_dst, dtype=np.int64)

    @staticmethod
    def _ints(node, name, default):
        for a in node.attribute:
            if a.name == name and a.ints:
                return list(a.ints)
        return default

    def _node_macs(self, node, out_numel, init_dims, shapes):
        op = node.op_type
        if op in ("Conv", "ConvTranspose"):
            for x in node.input[1:2]:
                if x in init_dims and len(init_dims[x]) >= 2:
                    return out_numel * _prod(init_dims[x][1:])
            return out_numel * _prod(self._ints(node, "kernel_shape", [3, 3])) * 3.0
        if op in ("Gemm", "MatMul"):
            for x in node.input[1:2]:
                if x in init_dims:
                    return _prod(init_dims[x])
            a = shapes.get(node.input[0])
            return out_numel * (a[-1] if a else 1.0)
        if op in ("MaxPool", "AveragePool"):
            return out_numel * _prod(self._ints(node, "kernel_shape", [2, 2]))
        return out_numel * ELEMENTWISE_COST.get(op, 1.0)

    def cut_traffic(self, cuts):
        k = len(cuts)
        cuts = np.asarray(cuts, dtype=np.int64)
        s_src = np.searchsorted(cuts, self._e_src, side="right") - 1
        s_dst = np.searchsorted(cuts, self._e_dst, side="right") - 1
        cross = s_src != s_dst
        pairs = np.unique(self._e_t[cross] * k + s_dst[cross])
        t_id, d_seg = pairs // k, pairs % k
        size = self._t_size[t_id]
        src_seg = np.searchsorted(cuts, self._e_src[np.searchsorted(self._e_t, t_id)], side="right") - 1
        recv, send = np.zeros(k), np.zeros(k)
        np.add.at(recv, d_seg, size)
        np.add.at(send, src_seg, size)
        return recv * ELEM_BYTES, send * ELEM_BYTES

    def segment_terms(self, cuts):
        cuts = [int(c) for c in cuts]
        bounds = cuts + [self.n]
        recv_b, send_b = self.cut_traffic(cuts)
        terms = []
        for s in range(len(cuts)):
            a, b = bounds[s], bounds[s + 1]
            terms.append({
                "macs": float(self.macs[a:b].sum()) / GMAC,
                "n_layers": float(b - a),
                "recv_mb": float(recv_b[s]) / MB,
                "send_mb": float(send_b[s]) / MB,
                "coeffs_mb": float(self.m_coeffs[a:b].sum()) * ELEM_BYTES / MB,
                "outbuffs_mb": float(self.m_outbuffs[a:b].sum()) * ELEM_BYTES / MB,
                "inbuffs_mb": float(recv_b[s] / ELEM_BYTES if a > 0 else self.input_numel)
                * ELEM_BYTES / MB,
                "one": 1.0,
            })
        return terms


def term_value(terms, name, kind):
    if name == "paper_mb":
        w = 2.0 if kind == "gpu" else 1.0
        return w * terms["coeffs_mb"] + terms["outbuffs_mb"] + terms["inbuffs_mb"]
    return terms[name]


def link_kind(a, b):
    if a.host != b.host:
        return "net"
    return "pcie" if a.kind != b.kind else "shm"


class AnalyticalModel:
    def __init__(self, costs, resources, theta_time=None, theta_mem=None, bandwidth=None,
                 latency_ms=None, calibrated=False, metadata=None):
        self.costs = costs
        self.res = {r.name: r for r in resources}
        self.theta_time = theta_time or {}
        self.theta_mem = theta_mem or {}
        self.bandwidth = dict(DEFAULT_BANDWIDTH, **(bandwidth or {}))
        self.latency_ms = dict(DEFAULT_LATENCY_MS, **(latency_ms or {}))
        self.calibrated = calibrated
        self.metadata = metadata or {}
        self._warned = set()

    def _coefs(self, table, name, prior, terms):
        if name in table:
            return np.asarray(table[name], dtype=float)
        if name not in self._warned:
            print(f"[model] {name}: not calibrated, using priors", file=sys.stderr)
            self._warned.add(name)
        p = prior[self.res[name].kind]
        return np.array([p.get(t, 0.0) for t in terms], dtype=float)

    def comm_ms(self, s, placement, recv_mb, send_mb):
        t = 0.0
        if s > 0 and recv_mb > 0:
            lk = link_kind(self.res[placement[s - 1]], self.res[placement[s]])
            t += recv_mb / self.bandwidth[lk] + self.latency_ms[lk]
        if s < len(placement) - 1 and send_mb > 0:
            lk = link_kind(self.res[placement[s]], self.res[placement[s + 1]])
            t += send_mb / self.bandwidth[lk] + self.latency_ms[lk]
        return t

    def compute_ms(self, terms, name):
        kind = self.res[name].kind
        th = self._coefs(self.theta_time, name, PRIOR_TIME, TIME_TERMS)
        return float(np.array([term_value(terms, k, kind) for k in TIME_TERMS]) @ th)

    def memory_mb(self, terms, name):
        kind = self.res[name].kind
        ph = self._coefs(self.theta_mem, name, PRIOR_MEM, MEM_TERMS)
        return max(float(np.array([term_value(terms, k, kind) for k in MEM_TERMS]) @ ph), 0.0)

    def predict(self, cuts, placement):
        terms = self.costs.segment_terms(cuts)
        stage_ms, mem_mb = [], []
        for s, name in enumerate(placement):
            t = terms[s]
            stage_ms.append(max(self.compute_ms(t, name)
                                + self.comm_ms(s, placement, t["recv_mb"], t["send_mb"]), 1e-3))
            mem_mb.append(self.memory_mb(t, name))
        return stage_ms, mem_mb

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({
                "model_path": self.costs.model_path,
                "model_len": self.costs.n,
                "time_terms": list(TIME_TERMS),
                "mem_terms": list(MEM_TERMS),
                "theta_time": {k: list(map(float, v)) for k, v in self.theta_time.items()},
                "theta_mem": {k: list(map(float, v)) for k, v in self.theta_mem.items()},
                "bandwidth": self.bandwidth,
                "latency_ms": self.latency_ms,
                "calibrated": self.calibrated,
                "metadata": self.metadata,
            }, fh, indent=2)

    @classmethod
    def load(cls, path, costs, resources):
        with open(path) as fh:
            d = json.load(fh)
        if costs.n != d["model_len"]:
            raise RuntimeError(f"{path}: calibrated on {d['model_len']} layers, "
                               f"model has {costs.n}")
        if tuple(d.get("time_terms", ())) != TIME_TERMS or tuple(d.get("mem_terms", ())) != MEM_TERMS:
            raise RuntimeError(f"{path}: term set differs from this version, recalibrate")
        return cls(costs, resources, d["theta_time"], d["theta_mem"], d.get("bandwidth"),
                   d.get("latency_ms"), d.get("calibrated", False), d.get("metadata"))


def _solve(A, y, prior=None, lam=0.0):
    if prior is not None and lam > 0:
        w = np.linalg.norm(A, axis=0) * np.sqrt(lam)
        w = np.where(np.isnan(prior), 0.0, w)
        A = np.vstack([A, np.diag(w)])
        y = np.concatenate([y, w * np.nan_to_num(prior)])
    if nnls is not None:
        return nnls(A, y)[0]
    return np.clip(np.linalg.lstsq(A, y, rcond=None)[0], 0.0, None)


def _condition(A):
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    return float(np.linalg.cond(A / norms))


def _fit(rows, targets, names, label, kind, lam, verbose):
    A = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    names = list(names)
    while len(names) > 1 and A.shape[0] < len(names) + 1:
        drop = names[-2]
        keep = [i for i, n in enumerate(names) if n != drop]
        A, names = A[:, keep], [names[i] for i in keep]
        if verbose:
            print(f"[calib] {label}: {A.shape[0]} samples, dropping term '{drop}'", file=sys.stderr)
    prior = None
    if lam > 0:
        table = RIDGE_PRIOR.get(kind, {})
        prior = np.array([table.get(n, np.nan) for n in names], dtype=float)
    cond = _condition(A)
    if cond > 30 and verbose:
        print(f"[calib] {label}: condition number {cond:.0f}, the calibration plan is not "
              f"varied enough to separate the terms", file=sys.stderr)
    coef = _solve(A, y, prior, lam)
    denom = np.where(np.abs(y) > 1e-9, np.abs(y), 1.0)
    mape = float(np.mean(np.abs(A @ coef - y) / denom) * 100.0)
    return coef, names, mape, A.shape[0], cond


def _expand(coef, fitted, full):
    lut = dict(zip(fitted, coef))
    return [float(lut.get(n, 0.0)) for n in full]


def records_from_history(history):
    out = []
    for h in history:
        if "error" in h or not h.get("cuts") or not h.get("placement"):
            continue
        comp = h.get("compute_ms")
        if comp is None or h.get("memory_mb") is None or len(comp) != len(h["placement"]):
            continue
        out.append({k: h[k] for k in ("cuts", "placement", "compute_ms", "memory_mb", "wall_ms")
                    if k in h})
    return out


def calibrate(costs, resources, records, lam=0.0, verbose=True):
    res = {r.name: r for r in resources}
    rows_t, y_t, rows_m, y_m = {}, {}, {}, {}
    for rec in records:
        terms = costs.segment_terms(rec["cuts"])
        for s, name in enumerate(rec["placement"]):
            kind = res[name].kind
            t = terms[s]
            rows_t.setdefault(name, []).append([term_value(t, k, kind) for k in TIME_TERMS])
            y_t.setdefault(name, []).append(rec["compute_ms"][s])
            rows_m.setdefault(name, []).append([term_value(t, k, kind) for k in MEM_TERMS])
            y_m.setdefault(name, []).append(rec["memory_mb"][s])

    theta_time, theta_mem, report = {}, {}, {}
    for name in sorted(rows_t):
        kind = res[name].kind
        ct, nt, et, n, kt = _fit(rows_t[name], y_t[name], TIME_TERMS, f"{name}/time", kind, 0.0,
                                 verbose)
        cm, nm, em, _, km = _fit(rows_m[name], y_m[name], MEM_TERMS, f"{name}/memory", kind, lam,
                                 verbose)
        theta_time[name] = _expand(ct, nt, TIME_TERMS)
        theta_mem[name] = _expand(cm, nm, MEM_TERMS)
        report[name] = {"n_stages": n, "mape_time_pct": et, "mape_memory_pct": em,
                        "cond_time": kt, "cond_memory": km}
        if verbose:
            a = dict(zip(TIME_TERMS, theta_time[name]))["macs"]
            print(f"\n--- {name} ({n} measured stages) ---")
            print("  t = " + " + ".join(f"{c:.4g}*{k}" for c, k in zip(theta_time[name], TIME_TERMS))
                  + f" [ms]   MAPE {et:.1f} %" + (f"   R = {1000.0 / a:.1f} GMAC/s" if a > 1e-6 else ""))
            print("  M = " + " + ".join(f"{c:.4g}*{k}" for c, k in zip(theta_mem[name], MEM_TERMS))
                  + f" [MB]   MAPE {em:.1f} %")

    model = AnalyticalModel(costs, resources, theta_time, theta_mem, calibrated=True,
                            metadata={"report": report, "n_records": len(records),
                                      "ridge_lambda": lam})
    if verbose:
        err = objective_errors(model, records)
        if err:
            print(f"\n[calib] objective error on {len(records)} mappings: "
                  f"throughput {err[0]:.1f} % (mean), memory {err[1]:.1f} % (mean)")
    return model


def _measured_period(rec):
    wall = rec.get("wall_ms")
    return max(wall) if wall else max(rec["compute_ms"])


def objective_errors(model, records):
    et, em = [], []
    for rec in records:
        p_ms, p_mb = model.predict(rec["cuts"], rec["placement"])
        thr = 1000.0 / _measured_period(rec)
        et.append(abs(1000.0 / max(p_ms) - thr) / thr)
        mem = max(rec["memory_mb"])
        em.append(abs(max(p_mb) - mem) / max(mem, 1e-9))
    if not et:
        return None
    return 100 * float(np.mean(et)), 100 * float(np.mean(em))


def cross_validate(costs, resources, records, lam=0.0):
    if len(records) < 3:
        return None
    et, em = [], []
    for i in range(len(records)):
        train = [r for j, r in enumerate(records) if j != i]
        test = records[i]
        m = calibrate(costs, resources, train, lam, verbose=False)
        if any(name not in m.theta_time for name in test["placement"]):
            continue
        e = objective_errors(m, [test])
        et.append(e[0])
        em.append(e[1])
    if not et:
        return None
    print(f"[calib] leave-one-out ({len(et)} folds): throughput {np.mean(et):.1f} % mean / "
          f"{np.max(et):.1f} % max, memory {np.mean(em):.1f} % mean / {np.max(em):.1f} % max")
    return {"throughput_pct": et, "memory_pct": em}


def repair_cuts(cuts, model_len, m):
    k = len(cuts)
    if k * m > model_len:
        raise ValueError(f"{model_len} layers cannot hold {k} segments of at least {m}")
    cuts = sorted(int(c) for c in cuts)
    cuts[0] = 0
    for j in range(1, k):
        cuts[j] = max(cuts[j], cuts[j - 1] + m)
    cuts[k - 1] = min(cuts[k - 1], model_len - m)
    for j in range(k - 2, 0, -1):
        cuts[j] = min(cuts[j], cuts[j + 1] - m)
    return cuts


def _plan_score(costs, kinds, plans):
    rows = {r: [] for r in kinds}
    for cuts, placement in plans:
        terms = costs.segment_terms(cuts)
        for s, r in enumerate(placement):
            rows[r].append(terms[s])
    need = max(len(TIME_TERMS), len(MEM_TERMS)) + 1
    penalty, worst = 0.0, 0.0
    for r, kind in kinds.items():
        if len(rows[r]) < need:
            penalty += need - len(rows[r])
            continue
        for names in (TIME_TERMS, MEM_TERMS):
            A = np.array([[term_value(t, k, kind) for k in names] for t in rows[r]])
            worst = max(worst, _condition(A))
    return penalty * 1e4 + worst


def plan_calibration(costs, resources, min_segment=2, n_runs=12, seed=0, n_candidates=400,
                     forbidden=()):
    rng = np.random.default_rng(seed)
    names = [r.name for r in resources]
    kinds = {r.name: r.kind for r in resources}
    k = len(names)
    forbidden = set(forbidden)

    def make(frac):
        f = np.asarray(frac, dtype=float)
        f = f / f.sum()
        cuts = [0] + list(np.cumsum(np.round(f * costs.n).astype(int))[:k - 1])
        cuts = repair_cuts(cuts, costs.n, min_segment)
        return None if set(cuts[1:]) & forbidden else cuts

    candidates = []
    patterns = [[1.0] * k]
    if k > 1:
        patterns += [[0.6] + [0.4 / (k - 1)] * (k - 1), [0.4 / (k - 1)] * (k - 1) + [0.6]]
    if k >= 3:
        patterns.append([0.1, 0.6] + [0.3 / (k - 2)] * (k - 2))
    for frac in patterns:
        cuts = make(frac)
        if cuts is None:
            continue
        for shift in range(k):
            candidates.append((cuts, [names[(i + shift) % k] for i in range(k)]))
    tries = 0
    while len(candidates) < n_candidates and tries < 20 * n_candidates:
        tries += 1
        cuts = make(rng.dirichlet(np.full(k, 0.6)))
        if cuts is not None:
            candidates.append((cuts, [names[i] for i in rng.permutation(k)]))

    selected = []
    for _ in range(min(max(n_runs, 2 * k), len(candidates))):
        best, best_score = None, float("inf")
        for cand in candidates:
            score = _plan_score(costs, kinds, selected + [cand])
            if score < best_score:
                best, best_score = cand, score
        selected.append(best)
        candidates = [c for c in candidates if c is not best]
    return selected
