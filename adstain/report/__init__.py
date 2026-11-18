import os
import traceback
from pathlib import Path


def runs_dir(cfg):
    return Path(cfg["report"]["runs"] or os.path.dirname(cfg["workdir"]))


def attack_dirs(cfg):
    dirs = list(cfg["report"]["attack_dirs"] or ["attack"])
    own = os.path.basename(cfg["attack"]["out"])
    return dirs if own in dirs else dirs + [own]


def _step(name, fn, *a, **kw):
    try:
        return True, fn(*a, **kw)
    except Exception:
        print(f"[report] {name} FAILED:\n{traceback.format_exc()}")
        return False, None


def generate(cfg):
    from . import campaign_figures, fmstats, stealth_study
    rp = cfg["report"]
    band = float(rp["stealth_band"])
    runs = runs_dir(cfg)
    out = Path(rp["out"] or runs / "report")
    dirs = attack_dirs(cfg)
    success = float(rp["success_asr"])
    print(f"[report] runs {runs} (attack folders {', '.join(dirs)}), tolerance ±{band:g} %, successful attack: "
          f"ASR ≥ {success:g} %, output {out}")

    ok, models = _step("campaign tables", campaign_figures.generate, runs, out / "campaign",
                       bands=[band] + list(rp["extra_bands"] or []), attack_dirs=dirs)
    if not rp["stealth_study"]:
        return ok
    study = out / "stealth"
    data = study / "data"
    if rp["collect"] == "always" or (rp["collect"] == "auto" and not fmstats.up_to_date(cfg, data)):
        print(f"[report] feature-map statistics of {cfg['name']} (one GPU pass over the test set)")
        ok &= _step("feature-map statistics", fmstats.collect, cfg, str(data), rp["batch"])[0]
    th = rp["threshold"]
    if th["mode"] != "fixed" and th["source"] == "validation" and (
            rp["collect"] == "always" or (rp["collect"] == "auto" and not fmstats.validation_up_to_date(cfg, data))):
        print(f"[report] clean statistics of the validation images of {cfg['name']} (calibrated threshold)")
        ok &= _step("validation statistics", fmstats.collect_validation, cfg, str(data), rp["batch"])[0]
    names = [m["name"] for m in models] if models else None
    good, txt = _step("stealth study", stealth_study.generate, runs, data, study, names, band, dirs,
                      cfg["autodice"]["dummy_marker"], threshold=th, success=success,
                      semantics=cfg["attack"].get("semantics", "device"))
    ok &= good
    if good:
        print(txt)
    print(f"[report] {'done' if ok else 'done with errors'}: {out}")
    return ok
