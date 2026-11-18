import json
import os

import numpy as np
import pandas as pd

FAMILIES = ("mfgsm", "mgna", "mjsma")
NAMES = {"mfgsm": "mFGSM", "mgna": "mGNA", "mjsma": "mJSMA"}
ACC = {False: ("accuracy_attack_pct", "accuracy_baseline_pct"), True: ("map50_attack", "map50_baseline")}
ANY = "any"


def _num(x):
    return f"{float(x):g}"


def config_label(row):
    fam = row["attack"]
    if fam == "mfgsm" and pd.notna(row.get("eps_rel")):
        return f"mFGSM ε {_num(row['eps_rel'])}"
    if fam == "mgna" and pd.notna(row.get("sigma_rel")):
        return f"mGNA σ {_num(row['sigma_rel'])} γ {_num(row['gamma'])}"
    if fam == "mjsma" and pd.notna(row.get("theta")):
        return f"mJSMA θ {_num(row['theta'])}"
    return str(row.get("config_label", row["config_tag"]))


def _order(row):
    fam = row["attack"]
    k = FAMILIES.index(fam) if fam in FAMILIES else len(FAMILIES)
    strength = {"mfgsm": row.get("eps_rel"), "mgna": row.get("sigma_rel"), "mjsma": row.get("theta")}.get(fam)
    gamma = row.get("gamma") if fam == "mgna" else 0.0
    return (k, float(gamma) if pd.notna(gamma) else 0.0, float(strength) if pd.notna(strength) else 0.0,
            row["config_tag"])


def load_results(dirs):
    frames = [pd.read_csv(os.path.join(d, "attack_summary_gpu.csv")) for d in dirs
              if os.path.exists(os.path.join(d, "attack_summary_gpu.csv"))]
    if not frames:
        raise FileNotFoundError(f"no attack_summary_gpu.csv in {list(dirs)}")
    a = pd.concat(frames, ignore_index=True).drop_duplicates(["layer_name", "config_tag"], keep="last")
    det = "map50_attack" in a.columns and a["map50_attack"].notna().any()
    acc, base = ACC[det]
    a = a[a[acc].notna()].copy()
    baselines = a[base].dropna().round(4).unique()
    if len(baselines) > 1:
        print(f"[rank] WARNING: the merged campaigns have different clean accuracies {sorted(baselines)} "
              f"(different test sets?): the first one is used")
    clean = float(a[base].dropna().iloc[0])
    a["acc"] = a[acc].astype(float)
    a["delta"] = 100.0 * np.maximum((a["std_attacked"] - a["std_clean"]).abs(),
                                    (a["mean_attacked"] - a["mean_clean"]).abs()) / a["std_clean"]
    a["label"] = a.apply(config_label, axis=1)
    return a, clean, det


def partitionings(cfg, model=None):
    from ..taps import attack_units, load_genes
    path = cfg["attack"]["genes"]
    with open(path) as fh:
        s = json.load(fh)
    genes, placements = load_genes(path)
    if model is None:
        from ..model import formatted_path, load_model, prepare
        dst = formatted_path(cfg)
        model = load_model(dst) if os.path.exists(dst) else prepare(cfg, verbose=False)[1]
    units, per_gene = attack_units(model, genes, placements, marker=cfg["autodice"]["dummy_marker"],
                                   semantics=cfg["attack"].get("semantics", "device"))
    tensor = {u["name"]: u["tap"] for u in units}
    key = lambda c, p: (tuple(int(x) for x in c), tuple(p or ()))
    reps = {key(r["cuts"], r["placement"]): r for r in s.get("repeats", [])}
    places = s.get("best_placements") or [None] * len(s["best_cuts"])
    rows = []
    for i, (cuts, place, obj) in enumerate(zip(s["best_cuts"], places, s["best_objectives"]), 1):
        r = reps.get(key(cuts, place), {})
        rows.append(dict(gene=i, cuts=[int(c) for c in cuts], placement=list(place or []),
                         throughput=float(r.get("throughput_mean", obj[0])),
                         memory=float(r.get("memory_mean", obj[1])),
                         units=list(per_gene.get(i, {}).get("taps", []))))
    return pd.DataFrame(rows), tensor


def rank_budget(res, P, clean, tags, band=None):
    r = res[res["config_tag"].isin(tags)]
    if band is not None:
        r = r[r["delta"] <= band]
    best = {}
    for u, g in r.groupby("layer_name"):
        j = int(np.argmin(g["acc"].to_numpy()))
        best[u] = (float(g["acc"].iloc[j]), g["config_tag"].iloc[j])
    rows = []
    for p in P.itertuples():
        cuts = []
        for u in p.units:
            acc, tag = best.get(u, (clean, None))
            if acc >= clean:
                acc, tag = clean, None
            cuts.append((acc, u, tag))
        accs = np.array([c[0] for c in cuts])
        k = int(np.argmin(accs))
        rows.append(dict(gene=p.gene, throughput=p.throughput, memory=p.memory, n_cuts=len(cuts),
                         mean_pct=100.0 * accs.mean() / clean, worst_pct=100.0 * accs.min() / clean,
                         weakest_cut=cuts[k][1], weakest_config=cuts[k][2] or "none"))
    D = pd.DataFrame(rows)
    D["rank"] = D["mean_pct"].round(6).rank(method="min", ascending=False).astype(int)
    return D.sort_values(["rank", "gene"]).reset_index(drop=True)


def rank_all(res, P, clean, tags=None, band=None):
    front = sorted({u for us in P["units"] for u in us})
    done = res[res["layer_name"].isin(front)]
    lost = [u for u in front if u not in set(done["layer_name"])]
    if lost:
        raise SystemExit(f"[rank] no campaign result for {len(lost)} cut(s) of the front: {lost[:5]}"
                         f"{' ...' if len(lost) > 5 else ''} (attack.semantics or the attack folders do not match "
                         f"the search results)")
    cov = done.groupby("config_tag")["layer_name"].nunique()
    complete = set(cov[cov == len(front)].index)
    meta = done.drop_duplicates("config_tag").set_index("config_tag")
    wanted = list(tags) if tags else list(meta.index)
    unknown = [t for t in wanted if t not in meta.index]
    if unknown:
        raise SystemExit(f"[rank] configuration(s) {unknown} absent from the campaign results; known: "
                         f"{sorted(meta.index)}")
    left_out = [t for t in wanted if t not in complete]
    budget = sorted((t for t in wanted if t in complete),
                    key=lambda t: _order(dict(meta.loc[t].to_dict(), config_tag=t)))
    if not budget:
        raise SystemExit("[rank] no configuration of the budget was evaluated on every cut of the front")
    budgets = [("budget", "all", budget)]
    for fam in FAMILIES:
        ts = [t for t in budget if meta.loc[t, "attack"] == fam]
        if ts:
            budgets.append(("attack", NAMES[fam], ts))
    budgets += [("strength", meta.loc[t, "label"], [t]) for t in budget]
    vis = [(ANY, None)] + ([(f"stealthy (Δ ≤ {band:g} %)", band)] if band is not None else [])
    fastest = P.loc[P["throughput"].idxmax(), "gene"]
    lightest = P.loc[P["memory"].idxmin(), "gene"]
    out = []
    for kind, name, ts in budgets:
        for vname, b in vis:
            D = rank_budget(res, P, clean, ts, b)
            D.insert(0, "visibility", vname)
            D.insert(0, "budget", name)
            D.insert(0, "budget_kind", kind)
            D["weakest_config"] = D["weakest_config"].map(lambda t: meta.loc[t, "label"] if t in meta.index else t)
            D["fastest"] = D["gene"] == fastest
            D["lightest"] = D["gene"] == lightest
            out.append(D)
    T = pd.concat(out, ignore_index=True)
    cuts = P.set_index("gene")
    T["cuts"] = T["gene"].map(lambda g: " ".join(map(str, cuts.loc[g, "cuts"])))
    T["placement"] = T["gene"].map(lambda g: " ".join(cuts.loc[g, "placement"]))
    return T, [meta.loc[t, "label"] for t in budget], left_out


def _summary(T, model_name, task, clean, n_cuts, budget_labels, left_out, band, semantics):
    unit = "mAP@.5" if task else "Top-1"
    lines = [f"Robustness ranking of the Pareto-optimal partitionings - {model_name}",
             f"clean {unit} {clean:.2f} %; {T['gene'].nunique()} partitionings, {n_cuts} distinct cuts "
             f"({semantics} semantics)",
             f"budget: {len(budget_labels)} configuration(s): {', '.join(budget_labels)}"]
    if left_out:
        lines.append(f"left out (not evaluated on every cut of the front): {', '.join(left_out)}")
    lines += ["Mean, Worst: accuracy under attack in % of the clean accuracy, averaged over the devices / on the "
              "weakest device; A_s = most damaging configuration of the budget at the cut of D_s (null attack "
              "included). Ranked by Mean.",
              f"stealthy: only the (cut, configuration) pairs whose shift Δ stays within ±{band:g} % of the clean "
              f"std (mean and std of the feature map)" if band is not None else "no stealth band",
              "† fastest, ‡ lightest partitioning", ""]
    head = f"{'rank':>4} {'π':>4}  {'T':>7} {'M':>7} {'Mean':>6} {'Worst':>6}  weakest cut (configuration)"
    for vis, D in T[T["budget_kind"] == "budget"].groupby("visibility", sort=False):
        lines += [f"== whole budget, {vis}", head]
        for r in D.itertuples():
            mark = ("†" if r.fastest else "") + ("‡" if r.lightest else "")
            lines.append(f"{r.rank:>4} {str(r.gene) + mark:>4}  {r.throughput:7.1f} {r.memory:7.1f} "
                         f"{r.mean_pct:6.1f} {r.worst_pct:6.1f}  {r.weakest_cut} ({r.weakest_config})")
        lines.append("")
    for kind, title in (("attack", "per attack"), ("strength", "per strength")):
        S = T[T["budget_kind"] == kind]
        if S.empty:
            continue
        lines += [f"== {title}: most robust partitioning π* (rank 1), ranges over the front, partitionings that "
                  f"lose more than half of their accuracy on their weakest device",
                  f"{'budget':<24} {'visibility':<22} {'π*':>8} {'Mean':>6} {'Worst':>6}  "
                  f"{'Mean range':>12} {'Worst range':>12} {'Worst<50%':>9}"]
        for (name, vis), D in S.groupby(["budget", "visibility"], sort=False):
            top = D[D["rank"] == 1]
            pis = ("all" if len(top) == len(D) else ",".join(map(str, top["gene"][:3]))
                   + (",…" if len(top) > 3 else ""))
            lines.append(f"{name:<24} {vis:<22} {pis:>8} {top['mean_pct'].iloc[0]:6.1f} "
                         f"{top['worst_pct'].max():6.1f}  "
                         f"{D['mean_pct'].min():5.1f}-{D['mean_pct'].max():5.1f} "
                         f"{D['worst_pct'].min():5.1f}-{D['worst_pct'].max():5.1f} "
                         f"{int((D['worst_pct'] < 50).sum()):>5}/{len(D)}")
        lines.append("")
    return "\n".join(lines)


def default_dirs(cfg):
    dirs = [d if os.path.isabs(d) else os.path.join(cfg["workdir"], d)
            for d in cfg.get("report", {}).get("attack_dirs") or []]
    dirs.append(cfg["attack"]["out"])
    return list(dict.fromkeys(os.path.normpath(d) for d in dirs))


def write(cfg, attack_dirs=None, tags=None, band="default", out=None, model=None):
    dirs = [os.path.normpath(d if os.path.isabs(d) else os.path.join(cfg["workdir"], d)) for d in attack_dirs] \
        if attack_dirs else default_dirs(cfg)
    if band == "default":
        band = cfg.get("report", {}).get("stealth_band")
    band = float(band) if band not in (None, 0, 0.0) else None
    res, clean, det = load_results(dirs)
    P, _ = partitionings(cfg, model)
    T, labels, left_out = rank_all(res, P, clean, tags, band)
    out = out or os.path.join(cfg["attack"]["out"], "ranking")
    os.makedirs(out, exist_ok=True)
    T.to_csv(os.path.join(out, "ranking.csv"), index=False, float_format="%.4f")
    n_cuts = len({u for us in P["units"] for u in us})
    text = _summary(T, cfg.get("name", "model"), det, clean, n_cuts, labels, left_out, band,
                    cfg["attack"].get("semantics", "device"))
    with open(os.path.join(out, "summary.txt"), "w") as fh:
        fh.write(text + "\n")
    print(f"[rank] {len(labels)} configuration(s), {T['gene'].nunique()} partitionings, folders: "
          f"{', '.join(os.path.basename(d) for d in dirs)} -> {out}/ranking.csv, summary.txt")
    return T
