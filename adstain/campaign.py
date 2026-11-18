import json
import os
import time

import numpy as np
import pandas as pd
import torch

from . import attacks as A
from . import detection as D
from .gpu import GpuSession, check_gpu, is_oom, register_shared_arena
from .images import GpuImageLoader
from .model import prepare
from .onnx_graph import model_input_hw
from .plots import (MPL_ERROR, MPL_OK, gene_figures, plot_det_config, plot_det_examples,
                    plot_layer_config)
from .taps import (CLEAN, SuffixBuilder, attack_units, build_full_with_taps, cut_units, derive_taps, ensure_dir,
                   load_genes, non_float, safe_name, select_units, side_inputs)

STAT_COLS = ("amp_rel", "l2_rel_pct", "snr_db", "d_mean_pct", "d_std_pct", "d_var_pct",
             "d_mean_sigma", "mean_clean", "std_clean", "var_clean", "mean_attacked",
             "std_attacked", "var_attacked", "neg_frac_clean_pct", "neg_frac_attacked_pct",
             "clip_frac_pct", "eps_rel_p05", "eps_rel_p50", "eps_rel_p95", "eps_saturated_pct",
             "sat_frac_pct", "power_p_p05", "power_p_p50", "power_p_p95", "match_fallback_pct")
NOISE_COLS = ("eps_rel_p05", "eps_rel_p50", "eps_rel_p95", "eps_rel_mean", "eps_rel_std",
              "eps_saturated_pct", "power_p_p05", "power_p_p50", "power_p_p95",
              "match_fallback_pct", "sat_frac_pct", "sat_frac_max_pct", "n_images_stats",
              "n_elems_per_image", "mean_clean", "std_clean", "var_clean", "mean_attacked",
              "std_attacked", "var_attacked", "d_mean_pct", "d_std_pct", "d_var_pct",
              "d_mean_sigma", "mean_delta", "std_delta", "absmean_delta", "l2_clean", "l2_delta",
              "l2_mean_per_image", "l2_rel_pct", "amp_rel", "snr_db", "linf_max", "clip_frac_pct",
              "neg_frac_clean_pct", "neg_frac_attacked_pct")
SIDE_COLS = ("amp_rel", "l2_rel_pct", "snr_db", "linf_max", "d_mean_pct", "d_std_pct",
             "d_var_pct", "neg_frac_clean_pct", "neg_frac_attacked_pct", "eps_rel_p50",
             "eps_saturated_pct", "sat_frac_pct", "match_fallback_pct")
PAIRED_COLS = ["n_paired_images", "base_correct_atk_correct", "base_correct_atk_wrong",
               "base_wrong_atk_correct", "base_wrong_atk_wrong", "attack_success_rate_pct",
               "accidental_correction_rate_pct", "net_effect_pct"]
DET_PAIRED_COLS = ["n_paired_images", "n_gt_objects", "base_hit_atk_hit", "base_hit_atk_miss",
                   "base_miss_atk_hit", "base_miss_atk_miss", "attack_success_rate_pct",
                   "accidental_correction_rate_pct", "net_effect_pct",
                   "image_attack_success_rate_pct", "ref_detections", "vanished_pct", "ghost_pct",
                   "relabeled_pct", "image_changed_pct", "asr_reference"]


def _dataset(at):
    if not at["csv"]:
        raise SystemExit("attack.csv is required (columns: filepath,label[,classname])")
    df = pd.read_csv(at["csv"])
    miss = {"filepath", "label"} - set(df.columns)
    if miss:
        raise SystemExit(f"{at['csv']}: missing column(s) {sorted(miss)}")
    if "classname" not in df.columns:
        df["classname"] = df["label"].astype(str)
    return df


def _split(n_total, n_buf, limit, what):
    n = min(limit, n_total - n_buf)
    if n <= 0:
        raise SystemExit(f"{what}: not enough images ({n_total}) for {n_buf} history images")
    if n < min(limit, n_total):
        print(f"[data] test set reduced to {n} images to keep {n_buf} disjoint history images")
    return n


def run(cfg, taps=None):
    at = cfg["attack"]
    pp = cfg["preprocess"]
    out_dir = ensure_dir(at["out"])
    work = ensure_dir(os.path.join(out_dir, "onnx_gpu"))
    strict = bool(at["strict_gpu"])
    attack_side = bool(at["attack_side_inputs"])

    configs = A.build_configs(at["configs"])
    need_hist = any(c["cfg"].needs_history for c in configs)
    print(f"[attacks] {len(configs)} configuration(s): "
          + " | ".join(f"{c['tag']} ({c['label']})" for c in configs))

    model_path, model, task = prepare(cfg)
    detect = task.detection
    n_buf = int(at["n_buffer"]) if need_hist else 0

    def absp(p):
        return p if os.path.isabs(p) or not at["dataset_dir"] else os.path.join(at["dataset_dir"], p)

    if detect:
        ds = D.load_dataset(at, task, cfg["detection"])
        test, buf_paths, reduced = D.split_dataset(ds, at["max_images"] or len(ds), n_buf)
        n = len(test)
        if reduced:
            print(f"[data] test set reduced to {n} images to keep {n_buf} disjoint history images")
        test_paths = test.paths
        gt = None
        names = {i: task.class_name(i) for i in range(task.num_classes or 0)}
        print(f"[data] {ds.source}: {n} test images"
              + (f", {test.n_objects()} ground-truth objects ({ds.class_map_note})" if test.has_gt
                 else " (no ground truth: attacked detections are compared with the clean ones)"))
    else:
        df = _dataset(at)
        limit = int(at["max_images"]) if at["max_images"] else len(df)
        n = _split(len(df), n_buf, limit, at["csv"])
        df_test = df.head(n)
        pool = df.drop(index=df_test.index)
        df_buf = pool.sample(n=min(n_buf, len(pool)), random_state=42) if n_buf else pool.head(0)
        test_paths = [absp(p) for p in df_test["filepath"]]
        gt = df_test["label"].astype(int).to_numpy()
        buf_paths = [absp(p) for p in df_buf["filepath"]]
        names = dict(zip(df["label"].astype(int), df["classname"]))

    dev = check_gpu(int(at["gpu"]), bool(at["tf32"]))
    t0 = time.time()
    finals = [o.name for o in model.graph.output] if detect else [model.graph.output[0].name]
    final = finals[0]
    marker = cfg["autodice"]["dummy_marker"]
    explicit = taps or at["taps"]
    if isinstance(explicit, str):
        explicit = [t.strip() for t in explicit.split(",") if t.strip()]
    semantics = at.get("semantics", "device")
    if explicit:
        units = cut_units(model, derive_taps(model, explicit=explicit, marker=marker)[0])
        per_gene = {}
    else:
        if not os.path.exists(at["genes"]):
            raise SystemExit(f"no genes file {at['genes']}: run 'search' first or set attack.taps")
        genes, placements = load_genes(at["genes"])
        units, per_gene = attack_units(model, genes, placements, marker=marker, semantics=semantics)
    if at.get("units"):
        try:
            units = select_units(units, at["units"])
        except ValueError as e:
            raise SystemExit(str(e))
        kept = {u["name"] for u in units}
        per_gene = {gid: info for gid, info in per_gene.items() if kept & set(info["taps"])}
        print(f"[model] attack.units: {len(units)} cut unit(s) kept, in {len(per_gene)} partitioning(s)")
    for u in list(units):
        bad = non_float(model, [u["tap"]] + u["side"] + u["bypass"] + u["dup"])
        if bad:
            print(f"[model] cut {u['name']} skipped: non-float tensor(s) cross it {bad}")
            units.remove(u)
    if not units:
        raise SystemExit("no attackable cut tensor")
    unit = {u["name"]: u for u in units}
    tap_list = [u["name"] for u in units]
    side = {u["name"]: u["side"] for u in units}
    seed_k = {u["name"]: {s: k for k, s in enumerate(side_inputs(model, [u["tap"]])[u["tap"]])}
              for u in units}
    for u in units:
        for s in u["side"]:
            seed_k[u["name"]].setdefault(s, len(seed_k[u["name"]]))
    print(f"[model] {os.path.basename(cfg['model'])}: output{'s' if len(finals) > 1 else ''} "
          f"{', '.join(repr(f) for f in finals)}, {len(tap_list)} cut tensor(s): {tap_list}")
    partial = [u for u in units if not u["exact"]]
    if partial and semantics == "device":
        print(f"[model] {len(partial)} cut(s) crossed by tensors that the compromised device does not produce: "
              f"they come straight from an upstream device and stay clean")
    elif partial:
        print(f"[model] {len(partial)} cut(s) where some crossing tensors skip the receiving rank: as in the "
              f"MPI deployment, only what that rank receives is attacked, the rest stays clean")
        for u in partial:
            print(f"           {u['name']}: clean {u['bypass'] + [d + ' (later ranks)' for d in u['dup']]}")
    input_hw = model_input_hw(model, pp["input_hw"])
    if input_hw is None:
        raise SystemExit("the model input size is dynamic: set preprocess.input_hw")
    print(f"[model] input {input_hw[0]}x{input_hw[1]}, {pp['channel_order']} x*{pp['pixel_scale']:g}, "
          f"mean={pp['mean']}, std={pp['std']}, "
          + (f"letterbox (pad {pp.get('pad_value', 114):g})" if pp.get("letterbox") else
             f"short side {pp['resize_short']} + center crop" if pp["resize_short"] else "direct resize"))
    n_res = sum(1 for t in tap_list if side[t])
    if n_res:
        print(f"[model] {n_res} cut(s) also carry a side branch, "
              + ("attacked with the same configuration" if attack_side else "passed clean"))
        for t in tap_list:
            if side[t]:
                print(f"           {t} + {side[t]}")
    exposed = list(dict.fromkeys(t for u in units for t in [u["tap"]] + u["side"] + u["bypass"] + u["dup"]))
    attacked = list(dict.fromkeys(t for u in units for t in [u["tap"]] + (u["side"] if attack_side else [])))

    shared = bool(at["shared_arena"]) and register_shared_arena(int(at["gpu"]))
    sess_kw = dict(strict=strict, tf32=bool(at["tf32"]), shared_arena=shared,
                   algo_search=at["cudnn_algo_search"], max_workspace=bool(at["cudnn_max_workspace"]))
    full = GpuSession(build_full_with_taps(model_path, exposed, os.path.join(work, "full_with_taps.onnx")),
                      int(at["gpu"]), **sess_kw)
    builder = SuffixBuilder(model)
    suffixes = {}
    for t in tap_list:
        path = builder.build_unit(unit[t], finals if detect else final,
                                  os.path.join(work, f"suffix_{safe_name(t)}.onnx"))
        suffixes[t] = GpuSession(path, int(at["gpu"]), in_name=unit[t]["tap"], **sess_kw)
    sessions = [full] + list(suffixes.values())
    free_b, total_b = torch.cuda.mem_get_info(dev)
    print(f"[model] {len(sessions)} CUDA session(s) ready in {time.time() - t0:.1f} s, "
          f"strict={strict}; GPU memory free {free_b / 2**30:.2f} / {total_b / 2**30:.2f} GiB")

    def release_arenas():
        for s in sessions:
            try:
                s.release()
            except Exception as e:
                print(f"  [memory] arena release failed ({os.path.basename(s.path)}): "
                      f"{str(e).splitlines()[0][:200]}")
                return
            if shared and s._tails is not None:
                return

    loader = GpuImageLoader(dev, pp, input_hw, decode=at.get("jpeg_decode", "cpu"),
                            workers=int(at.get("decode_workers", 8)))
    bs = int(at["batch"])
    decoder = D.Decoder(task.head, input_hw, cfg["detection"]) if detect else None
    cur = {}

    if detect:
        Store = D.DetStore

        def put(store, idx, outs):
            store.put(idx, decoder({f: outs[f] for f in finals}), cur["geos"], cur["sizes"])
    else:
        Store = A.PredStore

        def put(store, idx, outs):
            store.put(idx, outs[final])

    with torch.inference_mode():
        hist_raw = {}
        if need_hist:
            ts = time.time()
            hist_names = attacked
            chunks = {t: [] for t in hist_names}
            for a in range(0, len(buf_paths), bs):
                x, valid = loader.load(buf_paths[a:a + bs])
                outs = full.run(x)
                vm = torch.tensor(valid, device=dev)
                for t in hist_names:
                    chunks[t].append(outs[t][vm])
                del outs, x
            hist_raw = {t: torch.cat(v, 0) for t, v in chunks.items()}
            mem = sum(v.numel() * 4 for v in hist_raw.values()) / 2 ** 20
            print(f"[history] {len(buf_paths)} images, {len(hist_raw)} tensor(s) kept on GPU "
                  f"({mem:.1f} MiB) in {time.time() - ts:.1f} s")

        ctx_cache = {}

        def get_ctx(name, attack, p):
            if attack != "mfgsm":
                return None
            key = (name, p["dom_quantile"])
            if key not in ctx_cache:
                ctx_cache[key] = A.History(hist_raw[name], p["dom_quantile"])
            return ctx_cache[key]

        base_full = Store(n)
        clean_suf = {t: Store(n) for t in tap_list}
        atk_pred = {(t, c["tag"]): Store(n) for t in tap_list for c in configs}
        stats = {(t, c["tag"]): A.FMStats() for t in tap_list for c in configs}
        stats.update({(t, c["tag"], "side"): A.FMStats()
                      for t in tap_list if attack_side and side[t] for c in configs})
        t_atk = {(t, c["tag"]): 0.0 for t in tap_list for c in configs}
        valid_all = np.zeros(n, dtype=bool)
        sizes_all = np.zeros((n, 2), dtype=np.int64)
        timers = {"load": 0.0, "full": 0.0, "clean": 0.0}

        def run_chunk(a, b):
            idx = np.arange(a, b)
            ts = time.time()
            x, valid, sizes = loader.load(test_paths[a:b], with_sizes=True)
            valid_all[a:b] = valid
            sizes_all[a:b] = sizes
            if detect:
                cur["sizes"] = sizes
                cur["geos"] = [D.geometry(w, h, pp, input_hw) for w, h in sizes]
            vm = torch.tensor(valid, device=dev)
            torch.cuda.synchronize()
            timers["load"] += time.time() - ts
            ts = time.time()
            outs = full.run(x)
            put(base_full, idx, outs)
            timers["full"] += time.time() - ts
            del x
            for t in tap_list:
                u = unit[t]
                fm = outs[u["tap"]]
                extra = {s: outs[s] for s in side[t]}
                keep = {b: outs[b] for b in u["bypass"]}
                keep.update({d + CLEAN: outs[d] for d in u["dup"]})
                ts = time.time()
                put(clean_suf[t], idx, suffixes[t].run(fm, {**extra, **keep}))
                timers["clean"] += time.time() - ts
                for c in configs:
                    key = (t, c["tag"])
                    p = c["params"]
                    ts = time.time()
                    mfm, diag = A.apply_attack(fm, get_ctx(u["tap"], c["attack"], p), c["attack"], p, idx)
                    mextra = extra
                    if attack_side and extra:
                        mextra, sdiag = {}, {}
                        for s, xs in extra.items():
                            mextra[s], sdiag = A.apply_attack(xs, get_ctx(s, c["attack"], p), c["attack"],
                                                              A.side_params(p, seed_k[t][s]), idx)
                        stats[key + ("side",)].update(A.flat_cat(list(extra.values())),
                                                      A.flat_cat(list(mextra.values())),
                                                      sdiag if len(extra) == 1 else {}, vm)
                        del sdiag
                    put(atk_pred[key], idx, suffixes[t].run(mfm, {**mextra, **keep}))
                    stats[key].update(fm, mfm, diag, vm)
                    del mfm, diag, mextra
                    torch.cuda.synchronize()
                    t_atk[key] += time.time() - ts
            del outs

        n_oom = [0]

        def process(a, b, depth=0):
            marks = {k: st.mark() for k, st in stats.items()}
            oom = False
            try:
                run_chunk(a, b)
            except Exception as e:
                if not is_oom(e) or (b - a) <= 1:
                    raise
                oom = True
            if not oom:
                return
            for k, st in stats.items():
                st.rollback(marks[k])
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            release_arenas()
            n_oom[0] += 1
            mid = (a + b) // 2
            if depth == 0:
                print(f"  [memory] out of memory on images [{a}, {b}): split {b - a} -> "
                      f"{mid - a} + {b - mid}; attack.batch={max(1, (b - a) // 2)} avoids this")
            process(a, mid, depth + 1)
            process(mid, b, depth + 1)

        torch.cuda.reset_peak_memory_stats(dev)
        t_start = time.time()
        for bi, a in enumerate(range(0, n, bs)):
            b = min(a + bs, n)
            process(a, b)
            if bi == 0 or (bi + 1) % 10 == 0 or b == n:
                el = time.time() - t_start
                print(f"  [{b}/{n}] {el:.0f} s, ETA {el / b * (n - b):.0f} s, torch peak "
                      f"{torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB"
                      + (f", cv2 fallbacks {loader.n_cpu}" if loader.n_cpu else ""))
        t_total = time.time() - t_start

    run_info = dict(cfg=cfg, at=at, pp=pp, out_dir=out_dir, configs=configs, tap_list=tap_list,
                    per_gene=per_gene, side=side, attack_side=attack_side, n_res=n_res, n=n,
                    test_paths=test_paths, buf_paths=buf_paths, valid=valid_all, base_full=base_full,
                    clean_suf=clean_suf, atk_pred=atk_pred, stats=stats, t_atk=t_atk, names=names,
                    dev=dev, strict=strict, bs=bs, loader=loader, t_total=t_total, timers=timers,
                    n_oom=n_oom[0], task=task, units=unit)
    if detect:
        return _finish_detection(run_info, test, sizes_all, input_hw)
    return _finish_classification(run_info, gt)


def _attack_cols(c, side, t, attack_side, sst, units=None):
    p = c["params"]
    mf = c["attack"] == "mfgsm"
    side_cols = {"side_buffers": ";".join(side[t]),
                 "side_attacked": bool(attack_side and side[t]),
                 **{f"side_{k}": sst.get(k, np.nan) for k in SIDE_COLS}}
    u = (units or {}).get(t)
    if units and any(not x["exact"] for x in units.values()):
        side_cols.update({"cut": u["cut"], "next_cut": u["next_cut"],
                          "clean_buffers": ";".join(u["bypass"] + [d + "@later" for d in u["dup"]])})
    match_mode = p["match_mode"] if mf and p["eps_mode"] == "match_variance" else None
    eps_rel = p["eps_rel"] if mf and p["eps_mode"] == "fixed" else np.nan
    head = {"layer_name": t, "buffer_name": u["tap"] if u else t, "attack": c["attack"],
            "config_tag": c["tag"], "config_label": c["label"],
            "eps_mode": p.get("eps_mode", "fixed"), "dc_correct": p.get("dc_correct", False),
            "clip": p.get("clip", "none"), "match_mode": match_mode, "eps_rel": eps_rel,
            "gamma": p.get("gamma", np.nan), "sigma_rel": p.get("sigma_rel", np.nan),
            "theta": p.get("theta", np.nan)}
    return head, side_cols, match_mode, eps_rel


def _finish_classification(r, gt):
    cfg, at, out_dir, configs = r["cfg"], r["at"], r["out_dir"], r["configs"]
    tap_list, side, attack_side, n = r["tap_list"], r["side"], r["attack_side"], r["n"]
    base_full, clean_suf, atk_pred, stats, t_atk = (r["base_full"], r["clean_suf"], r["atk_pred"],
                                                     r["stats"], r["t_atk"])
    test_paths, names, v = r["test_paths"], r["names"], r["valid"]
    acc_full = float((base_full.top1 == gt)[v].mean()) if v.any() else float("nan")
    print(f"\n[baseline] full model accuracy {acc_full * 100:.2f} % on {int(v.sum())} images")
    if np.isfinite(acc_full) and acc_full < 0.5:
        print("[baseline] low accuracy: check preprocess (channel_order, pixel_scale, mean, std, "
              "resize) and the label indices of the CSV")

    pd.DataFrame({"index": np.arange(n), "img_path": test_paths, "gt_class": gt,
                  "top1_class": base_full.top1, "correct": base_full.top1 == gt, "valid": v,
                  "top3": [",".join(map(str, r)) for r in base_full.top3]}
                 ).to_csv(os.path.join(out_dir, "baseline_gpu.csv"), index=False)

    ncnn_agree = None
    ref_path = at["reference_baseline"]
    if ref_path and os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        ref = ref[(ref["index"] >= 0) & (ref["index"] < n)]
        ri = ref["index"].to_numpy()
        ri = ri[v[ri]]
        if ri.size:
            col = "top1_class" if "top1_class" in ref.columns else "top1"
            ref_top1 = ref.set_index("index").loc[ri, col].to_numpy()
            ncnn_agree = float((ref_top1 == base_full.top1[ri]).mean() * 100)
            print(f"[baseline] top-1 agreement GPU vs reference {ncnn_agree:.2f} % on {ri.size} images")

    rows, pred_rows, noise_rows = [], [], []
    n_fig = 0
    for t in tap_list:
        ctop1 = clean_suf[t].top1
        acc_clean = float((ctop1 == gt)[v].mean() * 100)
        agree = float((ctop1 == base_full.top1)[v].mean() * 100)
        print(f"\n=== {t} === clean suffix accuracy {acc_clean:.2f} %, agreement with full "
              f"model {agree:.2f} %")
        for c in configs:
            key = (t, c["tag"])
            pm = A.paired_metrics(ctop1, atk_pred[key].top1, gt, v)
            st = stats[key].finalize()
            sst = stats[key + ("side",)].finalize() if key + ("side",) in stats else {}
            head, side_cols, match_mode, eps_rel = _attack_cols(c, side, t, attack_side, sst, r.get("units"))
            row = {
                **head,
                "accuracy_baseline_pct": round(acc_full * 100, 2),
                "accuracy_clean_suffix_pct": round(acc_clean, 2),
                **pm,
                "delta_acc_pp": round(acc_full * 100 - pm["accuracy_attack_pct"], 2),
                "images_processed": pm["n_paired_images"],
                "attack_run_s": round(t_atk[key], 2),
                "eval_backend": "gpu_ort_cuda",
                "pairing_baseline": "gpu_clean_same_suffix",
                "linf_delta": st.get("linf_max", np.nan),
                **{k: st.get(k, np.nan) for k in STAT_COLS},
                **side_cols,
            }
            rows.append(row)
            noise_rows.append({"layer_name": t, "buffer_name": head["buffer_name"], "attack": c["attack"],
                               "config_tag": c["tag"], "config_label": c["label"],
                               "eps_rel": eps_rel, "eps_mode": row["eps_mode"],
                               "match_mode": match_mode,
                               "attack_success_rate_pct": pm["attack_success_rate_pct"],
                               **{k: st.get(k, np.nan) for k in NOISE_COLS}, **side_cols})
            print(f"  {c['label']:30s} ASR {row['attack_success_rate_pct']:6.2f} %  "
                  f"acc {row['accuracy_attack_pct']:6.2f} %  rms(d)/std {row['amp_rel']:.4f}  "
                  f"dVar {row['d_var_pct']:+.3f} %  neg {row['neg_frac_attacked_pct']:.2f} %"
                  + (f"  fallback {row['match_fallback_pct']:.1f} %" if match_mode == "power" else "")
                  + (f"  | side rms(d)/std {row['side_amp_rel']:.4f}" if row["side_attacked"] else "")
                  + f"  ({t_atk[key]:.1f} s)")

            atop1 = atk_pred[key].top1
            for i in np.nonzero(v)[0]:
                pred_rows.append((int(i), t, c["tag"], int(gt[i]), int(ctop1[i]), int(atop1[i])))

            cdir = ensure_dir(os.path.join(out_dir, "layers", safe_name(t), c["tag"]))
            bad = v & (atop1 != gt)
            bi = np.nonzero(bad)[0]
            pd.DataFrame({"index": bi, "img_path": [test_paths[i] for i in bi],
                          "gt_class": gt[bad],
                          "gt_classname": [names.get(int(x), "") for x in gt[bad]],
                          "top1_clean": ctop1[bad], "top1_attack": atop1[bad],
                          "top1_attack_classname": [names.get(int(x), "") for x in atop1[bad]]}
                         ).to_csv(os.path.join(cdir, "misclassified.csv"), index=False)
            if at["figures"]:
                try:
                    if plot_layer_config(os.path.join(cdir, "results.png"), t, c["label"], gt, ctop1,
                                         atop1, v, names, pm, acc_full * 100):
                        n_fig += 1
                except Exception as e:
                    print(f"    [figure] {t} / {c['tag']}: {e}")

    pred = pd.DataFrame(pred_rows, columns=["index", "layer_name", "config_tag", "gt_class", "top1_clean",
                                            "top1_attack"])
    synth_metrics = [f"full model accuracy: {acc_full * 100:.2f} %",
                     (f"top-1 agreement with reference: {ncnn_agree:.2f} %" if ncnn_agree is not None
                      else "top-1 agreement with reference: n/a")]
    synth_rows = [f"{x['layer_name']} | {x['config_label']}: ASR {x['attack_success_rate_pct']:.2f} %, "
                  f"ACR {x['accidental_correction_rate_pct']:.2f} %, acc {x['accuracy_attack_pct']:.2f} %, "
                  f"amp_rel {x['amp_rel']:.4f}"
                  + (f", side_amp_rel {x['side_amp_rel']:.4f}" if x["side_attacked"] else "")
                  for x in rows]
    return _write_outputs(r, rows, noise_rows, pred, PAIRED_COLS, n_fig, synth_metrics, synth_rows)


def _finish_detection(r, test, sizes, input_hw):
    cfg, at, out_dir, configs = r["cfg"], r["at"], r["out_dir"], r["configs"]
    tap_list, side, attack_side, n = r["tap_list"], r["side"], r["attack_side"], r["n"]
    base_full, clean_suf, atk_pred, stats, t_atk = (r["base_full"], r["clean_suf"], r["atk_pred"],
                                                     r["stats"], r["t_atk"])
    test_paths, v, task = r["test_paths"], r["valid"], r["task"]
    det = cfg["detection"]
    conf, miou = float(det["conf"]), float(det["match_iou"])
    gts = test.gts
    cname = task.class_name
    te = time.time()

    base = D.summary_metrics(gts, base_full, v, conf, miou)
    base.pop("per_class", None)
    if gts is not None:
        print(f"\n[baseline] full model on {int(v.sum())} images, {test.n_objects()} objects: "
              f"mAP@[.5:.95] {base['map50_95']:.2f} %, mAP@.5 {base['map50']:.2f} %, "
              f"recall {base['recall_pct']:.2f} % / precision {base['precision_pct']:.2f} % at "
              f"score >= {conf:g}")
        if base["map50"] < 10:
            print("[baseline] very low mAP: check preprocess (letterbox, channel_order, pixel_scale, "
                  "mean, std), the label mapping (detection.label_map) and the head decoding")
    else:
        n_det = sum(len(base_full.at(i, conf)[1]) for i in np.nonzero(v)[0])
        print(f"\n[baseline] full model: {n_det} detections with score >= {conf:g} on "
              f"{int(v.sum())} images (no ground truth)")
    bf = base_full.to_frame(0.0, v, task.names)
    bf.insert(1, "img_path", [test_paths[i] for i in bf["index"]])
    bf.to_csv(os.path.join(out_dir, "baseline_gpu_detections.csv.gz"), index=False)
    pd.DataFrame({"index": np.arange(n), "img_path": test_paths, "width": sizes[:, 0],
                  "height": sizes[:, 1], "valid": v,
                  "n_det": [len(base_full.at(i, conf)[1]) for i in range(n)],
                  "n_gt": [int((~g[2]).sum()) for g in gts] if gts is not None else 0}
                 ).to_csv(os.path.join(out_dir, "baseline_gpu.csv"), index=False)
    if gts is not None:
        test.gt_frame().to_csv(os.path.join(out_dir, "ground_truth.csv"), index=False)

    rows, noise_rows, per_image = [], [], []
    n_fig = 0
    for t in tap_list:
        clean = D.summary_metrics(gts, clean_suf[t], v, conf, miou)
        clean.pop("per_class", None)
        fid, _ = D.paired_metrics(None, base_full, clean_suf[t], v, conf, miou)
        print(f"\n=== {t} === clean suffix"
              + (f" mAP@.5 {clean['map50']:.2f} %, mAP@[.5:.95] {clean['map50_95']:.2f} %," if gts is not None else "")
              + f" reproduces {100 - fid['vanished_pct']:.2f} % of the full model's detections")
        for c in configs:
            key = (t, c["tag"])
            pm, img = D.paired_metrics(gts, clean_suf[t], atk_pred[key], v, conf, miou)
            atk = D.summary_metrics(gts, atk_pred[key], v, conf, miou)
            atk.pop("per_class", None)
            st = stats[key].finalize()
            sst = stats[key + ("side",)].finalize() if key + ("side",) in stats else {}
            head, side_cols, match_mode, eps_rel = _attack_cols(c, side, t, attack_side, sst, r.get("units"))
            q = {}
            if gts is not None:
                q = {"map50_95_baseline": base["map50_95"], "map50_baseline": base["map50"],
                     "map50_95_clean_suffix": clean["map50_95"], "map50_clean_suffix": clean["map50"],
                     "map50_95_attack": atk["map50_95"], "map50_attack": atk["map50"],
                     "delta_map50_95_pp": round(clean["map50_95"] - atk["map50_95"], 2),
                     "delta_map50_pp": round(clean["map50"] - atk["map50"], 2)}
            row = {
                **head, "task": "detection", **q, **pm,
                "conf": conf, "match_iou": miou,
                "images_processed": pm["n_paired_images"],
                "attack_run_s": round(t_atk[key], 2),
                "eval_backend": "gpu_ort_cuda",
                "pairing_baseline": "gpu_clean_same_suffix",
                "linf_delta": st.get("linf_max", np.nan),
                **{k: st.get(k, np.nan) for k in STAT_COLS},
                **side_cols,
            }
            rows.append(row)
            noise_rows.append({"layer_name": t, "buffer_name": head["buffer_name"], "attack": c["attack"],
                               "config_tag": c["tag"], "config_label": c["label"],
                               "eps_rel": eps_rel, "eps_mode": row["eps_mode"],
                               "match_mode": match_mode,
                               "attack_success_rate_pct": pm["attack_success_rate_pct"],
                               **{k: st.get(k, np.nan) for k in NOISE_COLS}, **side_cols})
            print(f"  {c['label']:30s} ASR {pm['attack_success_rate_pct']:6.2f} %"
                  + (f"  mAP@.5 {clean['map50']:.2f} -> {atk['map50']:.2f}" if gts is not None else "")
                  + f"  vanished {pm['vanished_pct']:.2f} %  ghost {pm['ghost_pct']:.2f} %"
                  + f"  rms(d)/std {row['amp_rel']:.4f}  dVar {row['d_var_pct']:+.3f} %"
                  + (f"  | side rms(d)/std {row['side_amp_rel']:.4f}" if row["side_attacked"] else "")
                  + f"  ({t_atk[key]:.1f} s)")
            img.insert(1, "layer_name", t)
            img.insert(2, "config_tag", c["tag"])
            per_image.append(img)

            cdir = ensure_dir(os.path.join(out_dir, "layers", safe_name(t), c["tag"]))
            dets = atk_pred[key].to_frame(conf, v, task.names)
            dets.insert(1, "img_path", [test_paths[i] for i in dets["index"]])
            dets.to_csv(os.path.join(cdir, "detections.csv"), index=False)
            if gts is not None:
                _lost_objects(gts, clean_suf[t], atk_pred[key], v, conf, miou, test_paths, cname
                              ).to_csv(os.path.join(cdir, "lost_objects.csv"), index=False)
            if at["figures"]:
                try:
                    lost, base_cnt = (D.lost_per_class(gts, clean_suf[t], atk_pred[key], v, conf, miou)
                                      if gts is not None else ({}, {}))
                    if plot_det_config(os.path.join(cdir, "results.png"), t, c["label"], pm, base,
                                       clean, atk, lost, base_cnt, cname, img):
                        n_fig += 1
                    if int(at.get("examples", 4)) > 0 and plot_det_examples(
                            os.path.join(cdir, "examples.jpg"), t, c["label"], img, test_paths,
                            clean_suf[t], atk_pred[key], gts, conf, cname, int(at.get("examples", 4))):
                        n_fig += 1
                except Exception as e:
                    print(f"    [figure] {t} / {c['tag']}: {e}")

    pred = pd.concat(per_image, ignore_index=True) if per_image else pd.DataFrame()
    print(f"\n[detection] metrics computed in {time.time() - te:.1f} s")
    synth_metrics = [f"detection head: {task.head['format']}, {task.num_classes} classes; "
                     f"score >= {conf:g} and IoU >= {miou:g} for the paired object metrics",
                     f"ground truth: {test.source}" + (f", {test.n_objects()} objects" if gts is not None else "")]
    if gts is not None:
        synth_metrics.append(f"full model mAP@[.5:.95] {base['map50_95']:.2f} %, mAP@.5 {base['map50']:.2f} %")
    synth_rows = [f"{x['layer_name']} | {x['config_label']}: ASR {x['attack_success_rate_pct']:.2f} %"
                  + (f", mAP@.5 {x['map50_clean_suffix']:.2f} -> {x['map50_attack']:.2f}" if gts is not None else "")
                  + f", vanished {x['vanished_pct']:.2f} %, ghost {x['ghost_pct']:.2f} %, amp_rel {x['amp_rel']:.4f}"
                  + (f", side_amp_rel {x['side_amp_rel']:.4f}" if x["side_attacked"] else "")
                  for x in rows]
    return _write_outputs(r, rows, noise_rows, pred, [c for c in DET_PAIRED_COLS if not rows or c in rows[0]],
                          n_fig, synth_metrics, synth_rows)


def _lost_objects(gts, clean, atk, valid, conf, miou, paths, cname):
    rows = []
    for i in np.nonzero(valid)[0]:
        gb, gl, gc = gts[i]
        h_c, _ = D.hits(clean, gts, i, conf, miou)
        h_a, _ = D.hits(atk, gts, i, conf, miou)
        keep = np.nonzero(~gc)[0]
        for k in np.nonzero(h_c & ~h_a)[0]:
            g = keep[k]
            rows.append((int(i), paths[i], int(gl[g]), cname(gl[g]), *[round(float(x), 1) for x in gb[g]]))
    return pd.DataFrame(rows, columns=["index", "img_path", "class", "classname", "x1", "y1", "x2", "y2"])


def _write_outputs(r, rows, noise_rows, pred, paired_cols, n_fig, synth_metrics, synth_rows):
    cfg, at, out_dir, tap_list, per_gene = r["cfg"], r["at"], r["out_dir"], r["tap_list"], r["per_gene"]
    side, attack_side, task = r["side"], r["attack_side"], r["task"]
    summary = pd.DataFrame(rows)
    noise = pd.DataFrame(noise_rows)
    summary.to_csv(os.path.join(out_dir, "attack_summary_gpu.csv"), index=False)
    noise.to_csv(os.path.join(out_dir, "fm_noise_stats.csv"), index=False)
    pred.to_csv(os.path.join(out_dir, "predictions_gpu.csv"), index=False)
    summary[["layer_name", "attack", "config_tag"] + paired_cols].to_csv(
        os.path.join(out_dir, "attack_paired_breakdown.csv"), index=False)
    try:
        pivot = summary.pivot_table(index="layer_name", columns="config_label",
                                    values="attack_success_rate_pct").reindex(tap_list)
        print("\n=== ASR (%) per cut and configuration ===")
        print(pivot.round(2).to_string())
    except Exception as e:
        print(f"[pivot] {e}")

    n_fig_gene = 0
    for gid, info in per_gene.items():
        gdir = ensure_dir(os.path.join(out_dir, f"gene_{gid:03d}"))
        g = summary[summary["layer_name"].isin(info["taps"])].copy()
        g.insert(0, "gene_id", gid)
        g.insert(1, "split_layer", g["layer_name"].map(
            info.get("attack_target_layer") or {tn: k + 2 for k, tn in enumerate(info["taps"])}))
        g.to_csv(os.path.join(gdir, "attack_summary_table.csv"), index=False)
        gn = noise[noise["layer_name"].isin(info["taps"])].copy()
        gn.insert(0, "gene_id", gid)
        gn.to_csv(os.path.join(gdir, "fm_noise_stats.csv"), index=False)
        g[["split_layer", "layer_name", "attack", "config_tag"] + paired_cols].to_csv(
            os.path.join(gdir, "attack_paired_breakdown.csv"), index=False)
        units = r.get("units") or {}
        clean = {t: units[t]["bypass"] + [d + "@later" for d in units[t]["dup"]]
                 for t in info["taps"] if t in units and not units[t]["exact"]}
        with open(os.path.join(gdir, "gene_info.json"), "w") as fh:
            json.dump({"gene_id": gid, **info, "engine": "gpu_ort_cuda",
                       "side_inputs": {t: side[t] for t in info["taps"] if side.get(t)},
                       "side_inputs_attacked": attack_side,
                       **({"clean_buffers": clean} if clean else {})}, fh, indent=4)
        if at["figures"]:
            n_fig_gene += len(gene_figures(g, gn, gdir, gid, task.kind))

    if not MPL_OK:
        print(f"[figures] matplotlib unavailable ({MPL_ERROR}): CSV only")
    dev, loader, timers, t_atk = r["dev"], r["loader"], r["timers"], r["t_atk"]
    n, pp, configs = r["n"], r["pp"], r["configs"]
    synth = [
        "StAIn GPU campaign",
        "=" * 60,
        f"model: {cfg['model']} ({task.kind})",
        f"GPU: {torch.cuda.get_device_name(dev)} (cuda:{at['gpu']}), strict={r['strict']}, tf32={at['tf32']}",
        f"test images: {n} ({int(r['valid'].sum())} usable), history: {len(r['buf_paths'])}, batch {r['bs']}",
        f"decoding: {loader.n_gpu} JPEG on GPU, {loader.n_cpu_jpeg} on CPU ({loader.workers} threads), "
        f"{loader.n_png} PNG, {loader.n_cpu} cv2 fallback, "
        f"{loader.n_failed} unreadable",
        f"cut tensors: {tap_list}",
        f"side branches: { {t: s for t, s in side.items() if s} or 'none'}"
        + ((" (attacked)" if attack_side else " (clean)") if r["n_res"] else ""),
        f"preprocess: {pp}",
        "configurations: " + " | ".join(f"{c['tag']} ({c['label']})" for c in configs),
        *synth_metrics,
        f"time {r['t_total']:.1f} s: load {timers['load']:.1f}, full {timers['full']:.1f}, "
        f"clean suffixes {timers['clean']:.1f}, attacks {sum(t_atk.values()):.1f}",
        f"torch peak memory {torch.cuda.max_memory_allocated(dev) / 2**30:.2f} GiB"
        + (f", {r['n_oom']} out-of-memory split(s)" if r["n_oom"] else ""),
        f"figures: {n_fig} per cut/config, {n_fig_gene} per gene",
        "",
    ]
    head_len = len(synth) - 1
    synth += synth_rows
    with open(os.path.join(out_dir, "synthesis.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(synth) + "\n")
    print("\n" + "\n".join(synth[:head_len]))
    print(f"results: {out_dir}")
    return summary
