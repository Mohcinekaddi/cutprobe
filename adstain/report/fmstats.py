#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
import time

import numpy as np


def signature(cfg):
    at = cfg["attack"]
    from ..attacks import build_configs
    return dict(model=cfg["model"], csv=at["csv"], annotations=at["annotations"], dataset_dir=at["dataset_dir"],
                max_images=at["max_images"], n_buffer=at["n_buffer"], genes=at["genes"],
                tags=[c["tag"] for c in build_configs(at["configs"])])


def up_to_date(cfg, out):
    path = os.path.join(out, f"{cfg['name']}_meta.json")
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        old = json.load(fh).get("signature")
    new = signature(cfg)
    return bool(old) and all(old.get(k) == v for k, v in new.items() if k != "tags") \
        and set(new["tags"]) <= set(old.get("tags", []))


def moments(t):
    f = t.reshape(t.shape[0], -1).double()
    return f.mean(dim=1), f.var(dim=1, unbiased=False).sqrt()


def rms(t):
    f = t.reshape(t.shape[0], -1).double()
    return f.pow(2).mean(dim=1).sqrt()


def attacked_tensors(cfg, model):
    from ..taps import attack_units, load_genes, non_float, select_units
    at = cfg["attack"]
    genes, placements = load_genes(at["genes"])
    units, _ = attack_units(model, genes, placements, marker=cfg["autodice"]["dummy_marker"],
                            semantics=at.get("semantics", "device"))
    units = select_units(units, at.get("units"))
    units = [u for u in units if not non_float(model, [u["tap"]] + u["side"] + u["bypass"] + u["dup"])]
    return units, list(dict.fromkeys(u["tap"] for u in units))


def session(cfg, model_path, model, taps, work):
    from ..gpu import GpuSession, check_gpu, register_shared_arena
    from ..images import GpuImageLoader
    from ..onnx_graph import model_input_hw
    from ..taps import build_full_with_taps
    at, pp = cfg["attack"], cfg["preprocess"]
    dev = check_gpu(int(at["gpu"]), bool(at["tf32"]))
    os.makedirs(work, exist_ok=True)
    shared = bool(at["shared_arena"]) and register_shared_arena(int(at["gpu"]))
    full = GpuSession(build_full_with_taps(model_path, taps, os.path.join(work, "full_with_taps.onnx")),
                      int(at["gpu"]), strict=bool(at["strict_gpu"]), tf32=bool(at["tf32"]), shared_arena=shared,
                      algo_search=at["cudnn_algo_search"], max_workspace=bool(at["cudnn_max_workspace"]))
    loader = GpuImageLoader(dev, pp, model_input_hw(model, pp["input_hw"]), decode=at.get("jpeg_decode", "cpu"),
                            workers=int(at.get("decode_workers", 8)))
    return dev, full, loader


def _cleanup(work):
    shutil.rmtree(work, ignore_errors=True)
    try:
        os.rmdir(os.path.dirname(work))
    except OSError:
        pass


IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def image_list(spec, cfg):
    import pandas as pd
    if os.path.isdir(spec):
        return sorted(os.path.join(r, f) for r, _, fs in os.walk(spec) for f in fs if f.lower().endswith(IMAGE_EXT))
    if spec.endswith(".csv"):
        df = pd.read_csv(spec)
        if "filepath" not in df.columns:
            raise SystemExit(f"{spec}: needs a filepath column")
        paths = [str(p) for p in df["filepath"]]
    else:
        with open(spec) as fh:
            paths = [ln.strip() for ln in fh if ln.strip()]
    base = cfg["attack"]["dataset_dir"] or os.path.dirname(os.path.abspath(spec))
    return [p if os.path.isabs(p) else os.path.join(base, p) for p in paths]


def validation_signature(cfg):
    th = cfg["report"]["threshold"]
    return dict(model=cfg["model"], genes=cfg["attack"]["genes"], validation=th["validation"],
                max_validation=th["max_validation"])


def validation_up_to_date(cfg, out):
    path = os.path.join(out, "validation", f"{cfg['name']}_meta.json")
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        return json.load(fh).get("signature") == validation_signature(cfg)


def collect_validation(cfg, out, batch=None, work=None):
    import torch

    from ..model import prepare

    th, at = cfg["report"]["threshold"], cfg["attack"]
    paths = image_list(th["validation"], cfg)
    if th["max_validation"]:
        paths = paths[:int(th["max_validation"])]
    if not paths:
        raise SystemExit(f"{th['validation']}: no image")
    model_path, model, _ = prepare(cfg)
    units, taps = attacked_tensors(cfg, model)
    n, T = len(paths), len(taps)
    print(f"[study] {cfg['name']}: {n} validation images (clean), {T} attacked tensor(s)")
    out = os.path.join(out, "validation")
    work = os.path.join(work or os.path.join(out, "work"), cfg["name"])
    dev, full, loader = session(cfg, model_path, model, taps, work)
    bs = int(batch or at["batch"])
    clean_mean, clean_std = np.full((n, T), np.nan), np.full((n, T), np.nan)
    valid = np.zeros(n, dtype=bool)
    t0 = time.time()
    with torch.inference_mode():
        for bi, a in enumerate(range(0, n, bs)):
            b = min(a + bs, n)
            x, ok = loader.load(paths[a:b])
            outs = full.run(x)
            del x
            valid[a:b] = ok
            for j, t in enumerate(taps):
                mu, sd = moments(outs[t])
                clean_mean[a:b, j], clean_std[a:b, j] = mu.cpu().numpy(), sd.cpu().numpy()
            del outs
            if bi == 0 or (bi + 1) % 50 == 0 or b == n:
                el = time.time() - t0
                print(f"  [{b}/{n}] {el:.0f} s, ETA {el / b * (n - b):.0f} s")
    clean_mean[~valid], clean_std[~valid] = np.nan, np.nan
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, f"{cfg['name']}_fmstats.npz")
    np.savez_compressed(dst, taps=np.array(taps), clean_mean=clean_mean, clean_std=clean_std, valid=valid)
    with open(os.path.join(out, f"{cfg['name']}_meta.json"), "w") as fh:
        json.dump(dict(model=cfg["name"], n_images=n, n_valid=int(valid.sum()), tensors=taps,
                       signature=validation_signature(cfg), written=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=1)
    _cleanup(work)
    print(f"[study] wrote {dst}")
    return dst


def collect(cfg, out, batch=None, work=None):
    import torch

    from .. import attacks as A
    from .. import detection as D
    from ..campaign import _dataset, _split
    from ..model import prepare

    work = work or os.path.join(out, "work")
    at = cfg["attack"]
    configs = A.build_configs(at["configs"])
    need_hist = any(c["cfg"].needs_history for c in configs)
    model_path, model, task = prepare(cfg)
    detect = task.detection
    n_buf = int(at["n_buffer"]) if need_hist else 0

    def absp(p):
        return p if os.path.isabs(p) or not at["dataset_dir"] else os.path.join(at["dataset_dir"], p)

    if detect:
        ds = D.load_dataset(at, task, cfg["detection"])
        test, buf_paths, _ = D.split_dataset(ds, at["max_images"] or len(ds), n_buf)
        test_paths, labels = test.paths, None
    else:
        df = _dataset(at)
        limit = int(at["max_images"]) if at["max_images"] else len(df)
        n = _split(len(df), n_buf, limit, at["csv"])
        df_test = df.head(n)
        pool = df.drop(index=df_test.index)
        df_buf = pool.sample(n=min(n_buf, len(pool)), random_state=42) if n_buf else pool.head(0)
        test_paths = [absp(p) for p in df_test["filepath"]]
        labels = df_test["label"].astype(int).to_numpy()
        buf_paths = [absp(p) for p in df_buf["filepath"]]
    n = len(test_paths)

    units, taps = attacked_tensors(cfg, model)
    layer_to_tap = {u["name"]: u["tap"] for u in units}
    T, K = len(taps), len(configs)
    print(f"[study] {cfg['name']}: {n} test images, {len(units)} cut(s) -> {T} distinct attacked tensor(s), "
          f"{K} attack configuration(s)")

    work = os.path.join(work, cfg["name"])
    dev, full, loader = session(cfg, model_path, model, taps, work)
    bs = int(batch or at["batch"])

    clean_mean = np.full((n, T), np.nan)
    clean_std = np.full((n, T), np.nan)
    valid = np.zeros(n, dtype=bool)
    atk_mean = np.full((n, T, K), np.nan, dtype=np.float32)
    atk_std = np.full((n, T, K), np.nan, dtype=np.float32)
    atk_amp = np.full((n, T, K), np.nan, dtype=np.float32)

    with torch.inference_mode():
        hist = {}
        if need_hist:
            chunks = {t: [] for t in taps}
            for a in range(0, len(buf_paths), bs):
                x, ok = loader.load(buf_paths[a:a + bs])
                outs = full.run(x)
                vm = torch.tensor(ok, device=dev)
                for t in taps:
                    chunks[t].append(outs[t][vm])
                del outs, x
            hist = {t: torch.cat(v, 0) for t, v in chunks.items()}
            print(f"[history] {len(buf_paths)} images")
        ctx_cache = {}

        def ctx(t, attack, p):
            if attack != "mfgsm":
                return None
            key = (t, p["dom_quantile"])
            if key not in ctx_cache:
                ctx_cache[key] = A.History(hist[t], p["dom_quantile"])
            return ctx_cache[key]

        t0 = time.time()
        for bi, a in enumerate(range(0, n, bs)):
            b = min(a + bs, n)
            idx = np.arange(a, b)
            x, ok = loader.load(test_paths[a:b])
            outs = full.run(x)
            del x
            valid[a:b] = ok
            cm = torch.empty((b - a, T), dtype=torch.float64, device=dev)
            cs = torch.empty_like(cm)
            am = torch.empty((b - a, T, K), dtype=torch.float64, device=dev)
            ast = torch.empty_like(am)
            aamp = torch.empty_like(am)
            for j, t in enumerate(taps):
                fm = outs[t]
                cm[:, j], cs[:, j] = moments(fm)
                for k, c in enumerate(configs):
                    mfm, _ = A.apply_attack(fm, ctx(t, c["attack"], c["params"]), c["attack"], c["params"], idx)
                    am[:, j, k], ast[:, j, k] = moments(mfm)
                    aamp[:, j, k] = rms(mfm - fm) / cs[:, j].clamp_min(1e-12)
                    del mfm
            del outs
            clean_mean[a:b], clean_std[a:b] = cm.cpu().numpy(), cs.cpu().numpy()
            atk_mean[a:b], atk_std[a:b], atk_amp[a:b] = am.cpu().numpy(), ast.cpu().numpy(), aamp.cpu().numpy()
            if bi == 0 or (bi + 1) % 20 == 0 or b == n:
                el = time.time() - t0
                print(f"  [{b}/{n}] {el:.0f} s, ETA {el / b * (n - b):.0f} s")
        clean_mean[~valid], clean_std[~valid] = np.nan, np.nan
        atk_mean[~valid], atk_std[~valid], atk_amp[~valid] = np.nan, np.nan, np.nan

    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, f"{cfg['name']}_fmstats.npz")
    np.savez_compressed(
        dst, taps=np.array(taps), layer_names=np.array(list(layer_to_tap)),
        layer_taps=np.array(list(layer_to_tap.values())),
        cfg_tags=np.array([c["tag"] for c in configs]), cfg_labels=np.array([c["label"] for c in configs]),
        cfg_attacks=np.array([c["attack"] for c in configs]),
        clean_mean=clean_mean, clean_std=clean_std, valid=valid,
        labels=labels if labels is not None else np.full(n, -1), atk_mean=atk_mean, atk_std=atk_std,
        atk_amp=atk_amp)
    with open(os.path.join(out, f"{cfg['name']}_meta.json"), "w") as fh:
        json.dump(dict(model=cfg["name"], task="detection" if detect else "classification", n_test=n,
                       n_valid=int(valid.sum()), n_history=len(buf_paths), tensors=taps,
                       configs=[c["label"] for c in configs], config=cfg["config_path"], signature=signature(cfg),
                       written=time.strftime("%Y-%m-%d %H:%M:%S")), fh, indent=1)
    _cleanup(work)
    print(f"[study] wrote {dst}")
    return dst


def main():
    from .. import config as C
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--batch", type=int, help="images per batch (default attack.batch)")
    ap.add_argument("--out", required=True, help="output folder (the stealth study reads it as its data folder)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    collect(C.load(args.config, args.set), args.out, args.batch)


if __name__ == "__main__":
    main()
