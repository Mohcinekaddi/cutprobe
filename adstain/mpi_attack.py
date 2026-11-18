import json
import os
import struct

import numpy as np
import pandas as pd
import torch

from . import attacks as A
from . import detection as D
from .taps import side_inputs


def _model(cfg):
    from .model import prepare
    return prepare(cfg, verbose=False)


def make_image_lists(cfg, out_dir, n_images, n_history, task=None):
    at = cfg["attack"]
    if task is not None and task.detection:
        return _det_image_lists(cfg, out_dir, n_images, n_history, task)
    if not at["csv"]:
        return None
    df = pd.read_csv(at["csv"])
    if "classname" not in df.columns:
        df["classname"] = df["label"].astype(str)

    def absp(p):
        return p if os.path.isabs(p) or not at["dataset_dir"] else os.path.join(at["dataset_dir"], p)

    n = min(int(n_images), len(df) - int(n_history))
    test = df.head(n)
    pool = df.drop(index=test.index)
    hist = pool.sample(n=min(int(n_history), len(pool)), random_state=42)
    paths = [absp(p) for p in test["filepath"]]
    with open(os.path.join(out_dir, "test_images.txt"), "w") as fh:
        fh.write("\n".join(paths) + "\n")
    pd.DataFrame({"index": np.arange(n), "filepath": paths, "label": test["label"].astype(int),
                  "classname": test["classname"]}).to_csv(os.path.join(out_dir, "test_images.csv"),
                                                          index=False)
    with open(os.path.join(out_dir, "history_images.txt"), "w") as fh:
        fh.write("\n".join(absp(p) for p in hist["filepath"]) + "\n")
    return n, len(hist)


def _det_image_lists(cfg, out_dir, n_images, n_history, task):
    at = cfg["attack"]
    if not (at.get("annotations") or at.get("csv") or at.get("dataset_dir")):
        return None
    ds = D.load_dataset(at, task, cfg["detection"])
    test, hist, _ = D.split_dataset(ds, n_images, n_history)
    with open(os.path.join(out_dir, "test_images.txt"), "w") as fh:
        fh.write("\n".join(test.paths) + "\n")
    sizes = test.sizes if test.sizes is not None else np.array([D._image_size(p) for p in test.paths])
    pd.DataFrame({"index": np.arange(len(test)), "filepath": test.paths, "width": sizes[:, 0],
                  "height": sizes[:, 1]}).to_csv(os.path.join(out_dir, "test_images.csv"), index=False)
    gt_path = os.path.join(out_dir, "test_gt.csv")
    if test.has_gt:
        test.gt_frame().to_csv(gt_path, index=False)
    elif os.path.exists(gt_path):
        os.remove(gt_path)
    with open(os.path.join(out_dir, "history_images.txt"), "w") as fh:
        fh.write("\n".join(hist) + "\n")
    return len(test), len(hist)


def open_records(path):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        _, c, h, w = struct.unpack("4i", fh.read(16))
    rec = 16 + 4 * c * h * w
    if size % rec:
        raise ValueError(f"{path}: truncated ({size} bytes, record size {rec})")
    dtype = np.dtype([("idx", "<i4"), ("c", "<i4"), ("h", "<i4"), ("w", "<i4"),
                      ("data", "<f4", (c * h * w,))])
    return np.memmap(path, dtype=dtype, mode="r", shape=(size // rec,)), (c, h, w)


def _tensor_shape(entry, chw):
    shape = entry.get("shape") or []
    dims = [int(d) for d in shape[1:]]
    if dims and all(d > 0 for d in dims) and int(np.prod(dims)) == chw[0] * chw[1] * chw[2]:
        return dims
    return list(chw)


def _pick_config(cfg, tag):
    configs = A.build_configs(cfg["attack"]["configs"])
    if tag is None:
        return configs[0]
    for c in configs:
        if c["tag"] == tag:
            return c
    raise SystemExit(f"unknown attack tag {tag!r}; available: {[c['tag'] for c in configs]}")


def build_payload(cfg, build_dir, layer, tag=None, fm_dir="fm", hist_dir="fm_hist", side=True,
                  device=None, chunk=64):
    with open(os.path.join(build_dir, "buffer_manifest.json")) as fh:
        manifest = json.load(fh)
    entries = [e for e in manifest if int(e["attack_target_layer"]) == int(layer)]
    if not entries:
        layers = sorted({e["attack_target_layer"] for e in manifest})
        raise SystemExit(f"no tensor is received at layer {layer}; attackable layers: {layers}")
    c = _pick_config(cfg, tag)
    fm_dir = fm_dir if os.path.isabs(fm_dir) else os.path.join(build_dir, fm_dir)
    hist_dir = hist_dir if os.path.isabs(hist_dir) else os.path.join(build_dir, hist_dir)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = _model(cfg)[1]
    pos = {o: k for k, n in enumerate(model.graph.node) for o in n.output}
    tap = max((e["buffer_name"] for e in entries), key=lambda b: pos.get(b, -1))
    side_order = side_inputs(model, [tap])[tap]

    print(f"[payload] layer {layer} (cut {entries[0]['cut']}, received by rank {entries[0]['rank']} "
          f"{entries[0]['resource']}), attack {c['tag']} ({c['label']}), device {dev}")
    report = {"layer": int(layer), "tag": c["tag"], "label": c["label"], "buffers": {}}
    with torch.inference_mode():
        for e in entries:
            name = e["buffer_name"]
            out_path = os.path.join(fm_dir, e["attack_filename"])
            if os.path.exists(out_path):
                os.remove(out_path)
            if name != tap and not side:
                print(f"  {name}: side branch left clean (--no-side)")
                continue
            p = c["params"] if name == tap else A.side_params(
                c["params"], side_order.index(name) if name in side_order else 0)
            clean_path = os.path.join(fm_dir, e["filename"])
            if not os.path.exists(clean_path):
                raise SystemExit(f"{clean_path} missing: run the binary with DUMP_BUFFER=1 "
                                 f"FM_DIR={os.path.relpath(fm_dir, build_dir)} first")
            recs, chw = open_records(clean_path)
            shape = _tensor_shape(e, chw)
            ctx = None
            if c["cfg"].needs_history:
                hist_path = os.path.join(hist_dir, e["filename"])
                if not os.path.exists(hist_path):
                    raise SystemExit(f"{hist_path} missing: mFGSM needs a history dump "
                                     f"(DUMP_BUFFER=1 FM_DIR={os.path.relpath(hist_dir, build_dir)} "
                                     f"on history_images.txt)")
                hrecs, _ = open_records(hist_path)
                hist = torch.from_numpy(np.ascontiguousarray(hrecs["data"])).to(dev)
                ctx = A.History(hist.reshape([len(hrecs)] + shape), p["dom_quantile"])
            stats = A.FMStats()
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as fh:
                for a in range(0, len(recs), chunk):
                    blk = recs[a:a + chunk]
                    idx = np.asarray(blk["idx"], dtype=np.int64)
                    x = torch.from_numpy(np.ascontiguousarray(blk["data"])).to(dev)
                    x = x.reshape([len(blk)] + shape)
                    mfm, diag = A.apply_attack(x, ctx, c["attack"], p, idx)
                    stats.update(x, mfm, diag, torch.ones(len(blk), dtype=torch.bool, device=dev))
                    out = np.empty(len(blk), dtype=recs.dtype)
                    for k in ("idx", "c", "h", "w"):
                        out[k] = blk[k]
                    out["data"] = mfm.reshape(len(blk), -1).float().cpu().numpy()
                    out.tofile(fh)
            os.replace(tmp, out_path)
            st = stats.finalize()
            report["buffers"][name] = {"role": "tap" if name == tap else "side",
                                       "records": int(len(recs)), "file": out_path,
                                       **{k: st.get(k) for k in ("amp_rel", "l2_rel_pct", "snr_db",
                                                                 "d_mean_pct", "d_var_pct",
                                                                 "neg_frac_attacked_pct",
                                                                 "match_fallback_pct")}}
            print(f"  {name} ({'tap' if name == tap else 'side'}): {len(recs)} records, "
                  f"rms(d)/std {st.get('amp_rel', float('nan')):.4f}, "
                  f"||d||/||FM|| {st.get('l2_rel_pct', float('nan')):.2f} %, "
                  f"dVar {st.get('d_var_pct', float('nan')):+.3f} % -> {out_path}")
    with open(os.path.join(build_dir, f"payload_L{layer}_{c['tag']}.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report


def build_task(build_dir):
    p = os.path.join(build_dir, "task.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def quality_change(row):
    if "map50_attack" in row:
        return f"mAP@.5 {row.get('map50_clean')} -> {row.get('map50_attack')} %"
    if "vanished_pct" in row:
        return f"vanished {row.get('vanished_pct')} %, ghost {row.get('ghost_pct')} %"
    return f"accuracy {row.get('accuracy_clean_pct')} -> {row.get('accuracy_attack_pct')} %"


def score(build_dir, clean_csv="clean.csv", attacked_csv="attacked.csv", layer=None, tag=None,
          label=None, plot=True):
    def path(p):
        return p if os.path.isabs(p) else os.path.join(build_dir, p)

    info = build_task(build_dir)
    if info and info.get("task") == "detection":
        return _score_detection(build_dir, info, path(clean_csv), path(attacked_csv), layer, tag,
                                label, plot, os.path.basename(attacked_csv))
    clean = pd.read_csv(path(clean_csv)).set_index("index")
    atk = pd.read_csv(path(attacked_csv)).set_index("index")
    idx = clean.index.intersection(atk.index)
    row = {"layer": layer, "tag": tag, "attacked": os.path.basename(attacked_csv),
           "n_images": int(len(idx)),
           "top1_changed_pct": round(100.0 * float((clean.loc[idx, "top1"] != atk.loc[idx, "top1"]).mean()), 2)
           if len(idx) else np.nan}
    images = path("test_images.csv")
    plots = []
    if os.path.exists(images):
        info = pd.read_csv(images).set_index("index")
        idx = idx.intersection(info.index)
        g = info.loc[idx, "label"].to_numpy()
        base = clean.loc[idx, "top1"].to_numpy()
        row["accuracy_clean_pct"] = round(100.0 * float((base == g).mean()), 2) if len(idx) else np.nan
        pm = A.paired_metrics(base, atk.loc[idx, "top1"].to_numpy(), g, np.ones(len(idx), dtype=bool))
        row.update(pm)
        if plot and len(idx):
            plots.append(_plot_pair(build_dir, info, clean, atk, idx, pm, layer, tag, label))

    return _save_score(build_dir, row, plots, plot)


def _save_score(build_dir, row, plots, plot):
    out = os.path.join(build_dir, "scores.csv")
    df = pd.DataFrame([row])
    if os.path.exists(out):
        old = pd.read_csv(out)
        if "attacked" not in old.columns:
            old["attacked"] = "attacked.csv"
        same = ((old["layer"].astype(str) == str(row["layer"])) & (old["tag"].astype(str) == str(row["tag"]))
                & (old["attacked"].astype(str) == row["attacked"]))
        df = pd.concat([old[~same], df], ignore_index=True)
    df.to_csv(out, index=False)
    if plot:
        plots.append(_plot_summary(build_dir, df))
    row["plots"] = [p for p in plots if p]
    return row


def _det_scoring_inputs(build_dir, info):
    images = pd.read_csv(os.path.join(build_dir, "test_images.csv")).set_index("index")
    n = int(images.index.max()) + 1 if len(images) else 0
    pp, hw = info["preprocess"], info["input_hw"]
    size = np.zeros((n, 2), np.int64)
    size[images.index.to_numpy()] = images[["width", "height"]].to_numpy()
    geos = [D.geometry(w, h, pp, hw) for w, h in size]
    gts = None
    gt_path = os.path.join(build_dir, "test_gt.csv")
    if os.path.exists(gt_path):
        g = pd.read_csv(gt_path)
        empty = (np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, bool))
        gts = [empty] * n
        for i, gg in g.groupby("index"):
            if int(i) < n:
                gts[int(i)] = (gg[["x1", "y1", "x2", "y2"]].to_numpy(np.float32),
                               gg["class"].to_numpy(np.int64), gg["iscrowd"].to_numpy().astype(bool))
    return images, n, geos, size, gts


def _score_detection(build_dir, info, clean_path, atk_path, layer, tag, label, plot, attacked_name):
    from .task import Task
    task = Task.from_dict(info)
    det = dict(D.DET_DEFAULTS, **(info.get("detection") or {}))
    conf, miou = float(det["conf"]), float(det["match_iou"])
    images, n, geos, size, gts = _det_scoring_inputs(build_dir, info)
    geo_of = lambda i: (geos[i], size[i])
    clean, done_c = D.store_from_csv(clean_path, n, geo_of)
    atk, done_a = D.store_from_csv(atk_path, n, geo_of)
    valid = done_c & done_a
    pm, img = D.paired_metrics(gts, clean, atk, valid, conf, miou)
    row = {"layer": layer, "tag": tag, "attacked": attacked_name,
           "n_images": int(valid.sum())}
    qc = qa = {}
    if gts is not None:
        qc = D.summary_metrics(gts, clean, valid, conf, miou)
        qa = D.summary_metrics(gts, atk, valid, conf, miou)
        row.update({"map50_95_clean": qc["map50_95"], "map50_clean": qc["map50"],
                    "map50_95_attack": qa["map50_95"], "map50_attack": qa["map50"],
                    "delta_map50_pp": round(qc["map50"] - qa["map50"], 2)})
    row.update(pm)
    plots = []
    if plot and valid.any():
        from .plots import plot_det_config, plot_det_examples
        os.makedirs(os.path.join(build_dir, "plots"), exist_ok=True)
        cut = _cut_names(build_dir).get(int(layer), "?") if layer is not None else "?"
        out = os.path.join(build_dir, "plots", f"L{layer}_{tag}.png")
        lost, base = D.lost_per_class(gts, clean, atk, valid, conf, miou) if gts is not None else ({}, {})
        try:
            if plot_det_config(out, f"L{layer} {cut}", label or tag or "attack", pm, {}, qc, qa, lost,
                               base, task.class_name, img, engine="ncnn MPI"):
                plots.append(out)
            ex = os.path.join(build_dir, "plots", f"L{layer}_{tag}_examples.jpg")
            if plot_det_examples(ex, f"L{layer} {cut}", label or tag or "attack", img,
                                 images["filepath"].reindex(range(n)).fillna("").tolist(), clean, atk,
                                 gts, conf, task.class_name, 4):
                plots.append(ex)
        except Exception as e:
            print(f"[score] figure skipped: {e}")
    return _save_score(build_dir, row, plots, plot)


def _cut_names(build_dir):
    try:
        with open(os.path.join(build_dir, "buffer_manifest.json")) as fh:
            manifest = json.load(fh)
    except OSError:
        return {}
    names = {}
    for e in manifest:
        names.setdefault(int(e["attack_target_layer"]), []).append(e["buffer_name"])
    return {k: " + ".join(v) for k, v in names.items()}


def _plot_pair(build_dir, info, clean, atk, idx, pm, layer, tag, label):
    from .plots import plot_layer_config
    n = int(info.index.max()) + 1
    gt = np.full(n, -1, dtype=np.int64)
    top_c = np.full(n, -2, dtype=np.int64)
    top_a = np.full(n, -2, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    gt[info.index.to_numpy()] = info["label"].to_numpy()
    top_c[idx.to_numpy()] = clean.loc[idx, "top1"].to_numpy()
    top_a[idx.to_numpy()] = atk.loc[idx, "top1"].to_numpy()
    valid[idx.to_numpy()] = True
    names = dict(zip(info["label"].astype(int), info.get("classname", info["label"]).astype(str)))
    cut = _cut_names(build_dir).get(int(layer), "?") if layer is not None else "?"
    os.makedirs(os.path.join(build_dir, "plots"), exist_ok=True)
    out = os.path.join(build_dir, "plots", f"L{layer}_{tag}.png")
    ok = plot_layer_config(out, f"L{layer} {cut}", label or tag or "attack", gt, top_c, top_a,
                           valid, names, pm, float("nan"), engine="ncnn MPI",
                           clean_label="clean run")
    return out if ok else None


def _plot_summary(build_dir, df):
    from .plots import plot_scores
    df = df.dropna(subset=["layer"])
    if df.empty:
        return None
    os.makedirs(os.path.join(build_dir, "plots"), exist_ok=True)
    out = os.path.join(build_dir, "plots", "summary.png")
    return out if plot_scores(df, out, _cut_names(build_dir)) else None


RUN_KINDS = {
    "clean": ("test_images.txt", {"DUMP_BUFFER": "1", "FM_DIR": "fm"}, "clean.csv"),
    "history": ("history_images.txt", {"DUMP_BUFFER": "1", "FM_DIR": "fm_hist"}, "history.csv"),
    "attacked": ("test_images.txt", {"ATTACK_ENABLED": "1", "FM_DIR": "fm"}, "attacked.csv"),
    "plain": ("test_images.txt", {}, "results.csv"),
}


def _manifest(build_dir):
    with open(os.path.join(build_dir, "buffer_manifest.json")) as fh:
        return json.load(fh)


def _record_bytes(entry):
    dims = [int(d) for d in (entry.get("shape") or [])[1:]]
    return 16 + 4 * int(np.prod(dims)) if dims and all(d > 0 for d in dims) else 0


def dump_bytes_per_image(manifest):
    total = sum(_record_bytes(e) for e in manifest)
    per_layer = {}
    for e in manifest:
        per_layer[int(e["attack_target_layer"])] = per_layer.get(int(e["attack_target_layer"]), 0) + _record_bytes(e)
    return total, per_layer


def _count_lines(path):
    with open(path) as fh:
        return sum(1 for line in fh if line.strip())


def _gb(n):
    return n / 1e9


def run_binary(cfg, build_dir, kind, layer=None, results_file=None, image_list=None, fm_dir=None,
               offset=0):
    import shutil
    import subprocess
    import time
    if kind not in RUN_KINDS:
        raise SystemExit(f"unknown kind {kind!r} ({list(RUN_KINDS)})")
    if not os.path.exists(os.path.join(build_dir, "multinode")):
        raise SystemExit(f"no multinode binary in {build_dir}: run 'build' first")
    default_list, env_vars, default_results = RUN_KINDS[kind]
    env_vars = dict(env_vars, RESULTS_FILE=results_file or default_results)
    if fm_dir and "FM_DIR" in env_vars:
        env_vars["FM_DIR"] = fm_dir
    if kind == "attacked":
        if layer is None:
            raise SystemExit("--layer is required for an attacked run")
        env_vars["ATTACK_TARGET_LAYER"] = str(int(layer))
    image_list = image_list or default_list
    if not os.path.exists(os.path.join(build_dir, image_list)):
        raise SystemExit(f"{image_list} not found in {build_dir}")
    env_vars["IMG_OFFSET"] = str(int(offset))
    if kind in ("clean", "history"):
        per_image, _ = dump_bytes_per_image(_manifest(build_dir))
        need = per_image * _count_lines(os.path.join(build_dir, image_list))
        free = shutil.disk_usage(build_dir).free
        if need > 0.95 * free:
            raise SystemExit(f"the {kind} dump needs {_gb(need):.1f} GB ({per_image / 1e6:.2f} MB per image) "
                             f"but only {_gb(free):.1f} GB are free on this disk: use fewer images, or "
                             f"'mpi-pareto', which works in chunks")
    with open(os.path.join(build_dir, "rankfile")) as fh:
        n = sum(1 for line in fh if line.strip())
    ad = cfg["autodice"]
    exports = []
    for k in env_vars:
        exports += ["-x", k]
    cmd = ([ad["mpirun"]] + [str(a) for a in ad["mpirun_args"]] + exports
           + ["-np", str(n), "-rf", "rankfile", "./multinode", image_list])
    print("$ cd " + build_dir, flush=True)
    shown = [ad["mpirun"]] + [str(a) for a in ad["mpirun_args"]] + cmd[len(exports) + 1 + len(ad["mpirun_args"]):]
    print("$ " + " ".join(f"{k}={v}" for k, v in env_vars.items()) + " " + " ".join(shown), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=build_dir, env=dict(os.environ, **env_vars))
    if p.returncode:
        raise SystemExit(f"mpirun exited with {p.returncode}")
    print(f"[run] {kind} done in {time.time() - t0:.1f} s -> {env_vars['RESULTS_FILE']}", flush=True)
    return env_vars["RESULTS_FILE"]


def sweep(cfg, build_dir, layers=None, tags=None, device=None, side=True):
    with open(os.path.join(build_dir, "buffer_manifest.json")) as fh:
        manifest = json.load(fh)
    layers = layers or sorted({int(e["attack_target_layer"]) for e in manifest})
    configs = A.build_configs(cfg["attack"]["configs"])
    labels = {c["tag"]: c["label"] for c in configs}
    tags = tags or [c["tag"] for c in configs]
    unknown = [t for t in tags if t not in labels]
    if unknown:
        raise SystemExit(f"unknown tag(s) {unknown}; available: {list(labels)}")
    if not os.path.exists(os.path.join(build_dir, "clean.csv")):
        raise SystemExit("clean.csv missing: run the clean run first")
    needs_hist = any(c["cfg"].needs_history for c in configs if c["tag"] in tags)
    if needs_hist and not os.path.isdir(os.path.join(build_dir, "fm_hist")):
        raise SystemExit("mFGSM needs the history run first (fm_hist/ missing)")
    total, k = len(layers) * len(tags), 0
    rows = []
    for layer in layers:
        for tag in tags:
            k += 1
            print(f"\n===== [{k}/{total}] layer {layer}, {tag} =====", flush=True)
            build_payload(cfg, build_dir, layer, tag, side=side, device=device)
            name = f"attacked_L{layer}_{tag}.csv"
            run_binary(cfg, build_dir, "attacked", layer=layer, results_file=name)
            _remove_payloads(build_dir, layer)
            row = score(build_dir, "clean.csv", name, layer, tag, labels[tag])
            rows.append(row)
            print(f"[score] layer {layer} {tag}: ASR {row.get('attack_success_rate_pct')} %, "
                  f"{quality_change(row)}", flush=True)
    return rows


def pareto_campaign(cfg, genes=None, tags=None, n_images=500, n_history=None, device=None,
                    side=True, keep=False, results_path=None, chunk=None):
    import shutil
    from .config import resources as load_resources
    from .deploy import Deployer
    path = results_path or cfg["search"]["results"]
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run 'search' first or pass --results")
    with open(path) as fh:
        R = json.load(fh)
    n_sol = len(R["best_cuts"])
    genes = genes or list(range(1, n_sol + 1))
    bad = [g for g in genes if not 1 <= g <= n_sol]
    if bad:
        raise SystemExit(f"gene(s) {bad} out of range 1..{n_sol}")
    configs = A.build_configs(cfg["attack"]["configs"])
    labels = {c["tag"]: c["label"] for c in configs}
    tags = tags or list(labels)
    unknown = [t for t in tags if t not in labels]
    if unknown:
        raise SystemExit(f"unknown tag(s) {unknown}; available: {list(labels)}")
    needs_hist = any(c["cfg"].needs_history for c in configs if c["tag"] in tags)
    n_history = int(cfg["attack"]["n_buffer"] if n_history is None else n_history)

    root = os.path.join(cfg["workdir"], "mpi_pareto")
    os.makedirs(root, exist_ok=True)
    _, model, task = _model(cfg)
    res = load_resources(cfg)
    summary_path = os.path.join(root, "summary.csv")
    state = {"results": os.path.abspath(path), "images": int(n_images), "history": n_history,
             "tags": tags, "side": bool(side)}
    print(f"[pareto] {len(genes)} Pareto solution(s) of {path}, attacks {tags}, "
          f"{n_images} test images, results in {root}", flush=True)

    for k, g in enumerate(genes, 1):
        cuts, placement = R["best_cuts"][g - 1], R["best_placements"][g - 1]
        thr, mem = R["best_objectives"][g - 1][:2]
        gdir = os.path.join(root, f"gene_{g:03d}")
        done = os.path.join(gdir, "done.json")
        here = dict(state, cuts=cuts, placement=placement)
        if os.path.exists(done):
            with open(done) as fh:
                if json.load(fh) == here:
                    print(f"\n[pareto] gene {g}: already done, skipped", flush=True)
                    continue
        print(f"\n{'#' * 74}\n[pareto] gene {g} ({k}/{len(genes)}): {thr:.2f} img/s, {mem:.0f} MB, "
              f"cuts {cuts}, {' | '.join(placement)}\n{'#' * 74}", flush=True)
        os.makedirs(gdir, exist_ok=True)
        build_state = {"cuts": cuts, "placement": placement, "images": int(n_images),
                       "history": n_history}
        build_json = os.path.join(gdir, "build.json")
        same_build = False
        if os.path.exists(build_json) and os.path.exists(os.path.join(gdir, "multinode")):
            with open(build_json) as fh:
                same_build = json.load(fh) == build_state
        if same_build:
            print(f"[pareto] gene {g}: reusing the existing build (resuming)", flush=True)
        else:
            Deployer(cfg, model, res, gdir, task=task).build(cuts, placement, mode="infer")
            make_image_lists(cfg, gdir, n_images, n_history, task)
            for name in ("chunks.json", "done.json"):
                if os.path.exists(os.path.join(gdir, name)):
                    os.remove(os.path.join(gdir, name))
            with open(build_json, "w") as fh:
                json.dump(build_state, fh)
        rows = attack_build(cfg, gdir, tags, device, side, chunk)

        with open(os.path.join(gdir, "buffer_manifest.json")) as fh:
            manifest = json.load(fh)
        cut_of = {int(e["attack_target_layer"]): int(e["cut"]) for e in manifest}
        names = _cut_names(gdir)
        new = pd.DataFrame([{"gene": g, "throughput": thr, "memory_mb": mem,
                             "cuts": ",".join(map(str, cuts)), "placement": ",".join(placement),
                             "cut": cut_of.get(int(r["layer"])), "tensors": names.get(int(r["layer"]), ""),
                             "label": labels.get(r["tag"], r["tag"]),
                             **{key: v for key, v in r.items() if key != "plots"}} for r in rows])
        if os.path.exists(summary_path):
            old = pd.read_csv(summary_path)
            new = pd.concat([old[old["gene"] != g], new], ignore_index=True)
        new.sort_values(["gene", "layer", "tag"]).to_csv(summary_path, index=False)

        if not keep:
            shutil.rmtree(os.path.join(gdir, "fm_hist"), ignore_errors=True)
            for name in os.listdir(gdir):
                if name.endswith((".onnx", ".bin")):
                    os.remove(os.path.join(gdir, name))
        with open(done, "w") as fh:
            json.dump(here, fh, indent=1)
        _plot_pareto_summary(root, summary_path)

    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        df = df[df["gene"].isin(genes)]
        print(f"\n[pareto] worst-case ASR (%) over each solution's cuts:")
        table = df.pivot_table(index=["gene", "throughput", "memory_mb"], columns="tag",
                               values="attack_success_rate_pct", aggfunc="max")
        print(table.round(2).to_string())
        print(f"\n[pareto] {summary_path}")
        _plot_pareto_summary(root, summary_path)


def _plot_pareto_summary(root, summary_path):
    from .plots import plot_pareto_attacks
    try:
        out = os.path.join(root, "pareto_attacks.png")
        if plot_pareto_attacks(pd.read_csv(summary_path), out):
            print(f"[pareto] figure {out}", flush=True)
    except Exception as e:
        print(f"[pareto] figure skipped: {e}", flush=True)


def _remove_payloads(build_dir, layer, fm_dir="fm"):
    for e in _manifest(build_dir):
        if int(e["attack_target_layer"]) == int(layer):
            for name in (e["attack_filename"], e["attack_filename"] + ".tmp"):
                p = os.path.join(build_dir, fm_dir, name)
                if os.path.exists(p):
                    os.remove(p)


def _append_results(build_dir, part, total):
    new = pd.read_csv(os.path.join(build_dir, part))
    path = os.path.join(build_dir, total)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[~old["index"].isin(new["index"])], new], ignore_index=True)
    new = new.sort_values("index", kind="stable")
    new.to_csv(path, index=False)
    os.remove(os.path.join(build_dir, part))


def _layer_files(manifest, layer, attacked=False):
    key = "attack_filename" if attacked else "filename"
    return [e[key] for e in manifest if int(e["attack_target_layer"]) == int(layer)]


def _dumps_ok(build_dir, manifest, layers, n_records):
    for layer in layers:
        for e in manifest:
            if int(e["attack_target_layer"]) != int(layer):
                continue
            p = os.path.join(build_dir, "fm", e["filename"])
            rb = _record_bytes(e)
            if not os.path.exists(p) or (rb and os.path.getsize(p) != rb * n_records):
                return False
    return True


def _check_space(build_dir, need, what):
    import shutil
    free = shutil.disk_usage(build_dir).free
    if need > 0.97 * free:
        raise SystemExit(f"{what} needs {_gb(need):.1f} GB but only {_gb(free):.1f} GB are free on "
                         f"{build_dir}: free some space (or use a smaller --chunk) and run the same "
                         f"command again, it resumes here")


def attack_build(cfg, build_dir, tags=None, device=None, side=True, chunk=None, layers=None):
    import shutil
    manifest = _manifest(build_dir)
    layers = layers or sorted({int(e["attack_target_layer"]) for e in manifest})
    configs = A.build_configs(cfg["attack"]["configs"])
    labels = {c["tag"]: c["label"] for c in configs}
    tags = tags or list(labels)
    needs_hist = any(c["cfg"].needs_history for c in configs if c["tag"] in tags)
    with open(os.path.join(build_dir, "test_images.txt")) as fh:
        images = [line.rstrip("\n") for line in fh if line.strip()]
    n = len(images)
    per_image, per_layer = dump_bytes_per_image(manifest)
    peak_per_image = per_image + max(per_layer[l] for l in layers)
    n_hist = _count_lines(os.path.join(build_dir, "history_images.txt")) if needs_hist else 0
    hist_bytes = n_hist * per_image

    state = {"images": n, "layers": layers, "tags": tags, "side": bool(side)}
    prog_path = os.path.join(build_dir, "chunks.json")
    prog = {}
    if os.path.exists(prog_path):
        with open(prog_path) as fh:
            prog = json.load(fh)
    if prog.get("state") != state:
        prog = {"state": state}
        for name in os.listdir(build_dir):
            if name.startswith(("clean", "attacked_")) and name.endswith(".csv"):
                os.remove(os.path.join(build_dir, name))
        shutil.rmtree(os.path.join(build_dir, "fm"), ignore_errors=True)
    for k in ("done", "clean", "combos"):
        prog.setdefault(k, [])

    def save():
        with open(prog_path + ".tmp", "w") as fh:
            json.dump(prog, fh)
        os.replace(prog_path + ".tmp", prog_path)

    hist_ready = needs_hist and prog.get("history") and os.path.isdir(os.path.join(build_dir, "fm_hist"))
    if prog.get("chunk") is None:
        free = shutil.disk_usage(build_dir).free
        budget = 0.85 * free - (0 if hist_ready else hist_bytes)
        prog["chunk"] = int(chunk or min(n, max(budget, 0) // peak_per_image, 20000))
        if prog["chunk"] < 1:
            raise SystemExit(f"not enough disk: one image needs {peak_per_image / 1e6:.2f} MB of dumps "
                             f"(+{_gb(hist_bytes):.1f} GB of history), {_gb(free):.1f} GB free on {build_dir}")
        save()
    size = int(prog["chunk"])
    chunks = [(a, min(a + size, n)) for a in range(0, n, size)]
    print(f"[attack] {n} images in {len(chunks)} chunk(s) of {size}: {per_image / 1e6:.2f} MB of dumps "
          f"per image, peak disk use about {_gb(size * peak_per_image + hist_bytes):.1f} GB; "
          f"{len(layers)} cut(s) x {len(tags)} attack(s); progress in {prog_path}", flush=True)

    if needs_hist and not hist_ready:
        _check_space(build_dir, hist_bytes, "the history dump")
        run_binary(cfg, build_dir, "history")
        prog["history"] = True
        save()

    for ci, (a, b) in enumerate(chunks):
        if ci in prog["done"]:
            print(f"[attack] chunk {ci + 1}/{len(chunks)} already done", flush=True)
            continue
        todo = [l for l in layers if any(f"{ci}:{l}:{t}" not in prog["combos"] for t in tags)]
        with open(os.path.join(build_dir, "chunk.txt"), "w") as fh:
            fh.write("\n".join(images[a:b]) + "\n")
        if ci in prog["clean"] and _dumps_ok(build_dir, manifest, todo, b - a):
            print(f"\n[attack] chunk {ci + 1}/{len(chunks)}: images {a}..{b - 1}, clean run already "
                  f"done, resuming at cut(s) {todo}", flush=True)
        else:
            print(f"\n[attack] chunk {ci + 1}/{len(chunks)}: images {a}..{b - 1}", flush=True)
            shutil.rmtree(os.path.join(build_dir, "fm"), ignore_errors=True)
            _check_space(build_dir, per_image * (b - a), f"the clean dump of chunk {ci + 1}")
            run_binary(cfg, build_dir, "clean", results_file="clean_part.csv", image_list="chunk.txt",
                       offset=a)
            _append_results(build_dir, "clean_part.csv", "clean.csv")
            prog["clean"] = sorted(set(prog["clean"]) | {ci})
            save()
        for layer in layers:
            for tag in tags:
                key = f"{ci}:{layer}:{tag}"
                if key in prog["combos"]:
                    continue
                print(f"\n===== chunk {ci + 1}/{len(chunks)}, layer {layer}, {tag} =====", flush=True)
                _remove_payloads(build_dir, layer)
                _check_space(build_dir, per_layer[layer] * (b - a), f"the payload of layer {layer}")
                build_payload(cfg, build_dir, layer, tag, side=side, device=device)
                part = f"attacked_L{layer}_{tag}_part.csv"
                run_binary(cfg, build_dir, "attacked", layer=layer, results_file=part,
                           image_list="chunk.txt", offset=a)
                _remove_payloads(build_dir, layer)
                _append_results(build_dir, part, f"attacked_L{layer}_{tag}.csv")
                prog["combos"].append(key)
                save()
            for name in _layer_files(manifest, layer):
                p = os.path.join(build_dir, "fm", name)
                if os.path.exists(p):
                    os.remove(p)
                    print(f"[attack] deleted {name} (cut fully attacked for this chunk)", flush=True)
        shutil.rmtree(os.path.join(build_dir, "fm"), ignore_errors=True)
        prog["done"].append(ci)
        save()

    rows = []
    for layer in layers:
        for tag in tags:
            row = score(build_dir, "clean.csv", f"attacked_L{layer}_{tag}.csv", layer, tag, labels[tag])
            rows.append(row)
            print(f"[score] layer {layer} {tag}: ASR {row.get('attack_success_rate_pct')} %, "
                  f"{quality_change(row)}", flush=True)
    return rows
