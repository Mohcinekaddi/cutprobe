import copy
import os
import re
import sys

import yaml

TOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_ATTACKS = [
    {"attack": "mfgsm", "eps_mode": "match_variance", "dc_correct": True,
     "match_mode": "power", "eps_max": 3.0, "metric": "cosine", "topk": 1},
    {"attack": "mgna", "gamma": [1.0], "sigma_rel": [0.3]},
    {"attack": "mjsma", "theta": [0.08], "scope": "image"},
]

DEFAULTS = {
    "model": None,
    "workdir": None,
    "task": "auto",
    "preprocess": {
        "channel_order": "RGB",
        "pixel_scale": 1.0 / 255.0,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "resize_short": 256,
        "letterbox": False,
        "pad_value": 114,
        "input_hw": None,
    },
    "detection": {
        "head": "auto",
        "conf": 0.25,
        "eval_conf": 0.001,
        "iou": 0.6,
        "max_det": 300,
        "max_nms": 30000,
        "match_iou": 0.5,
        "agnostic_nms": False,
        "label_map": "auto",
        "names": None,
        "num_classes": None,
        "anchors": None,
        "layout": None,
        "objectness": None,
        "box_format": None,
        "normalized": None,
        "background": None,
    },
    "platform": {
        "resources": [],
        "gpu_slot": 0,
    },
    "autodice": {
        "root": os.environ.get("AUTODICE_ROOT", os.path.join(TOOL_DIR, "third_party", "AutoDiCE")),
        "onnx2ncnn": "{root}/build/tools/onnx/onnx2ncnn",
        "mpicxx": "{prefix}/bin/mpic++",
        "mpirun": "{prefix}/bin/mpirun",
        "mpirun_args": [],
        "include_dirs": ["{root}/src", "{root}/build/src", "{prefix}/include/opencv4"],
        "lib_dirs": ["{root}/build/src",
                     "{root}/build/glslang/glslang",
                     "{root}/build/glslang/glslang/OSDependent/Unix",
                     "{root}/build/glslang/OGLCompilersDLL",
                     "{root}/build/glslang/SPIRV"],
        "libs": ["ncnn", "opencv_core", "opencv_imgproc", "opencv_imgcodecs", "vulkan",
                 "glslang", "MachineIndependent", "GenericCodeGen", "OSDependent",
                 "OGLCompiler", "SPIRV", "pthread"],
        "cxxflags": ["-fopenmp", "-std=c++17", "-Wl,-rpath,{prefix}/lib"],
        "bench_image": None,
        "labels": None,
        "warmup": 4,
        "loops": 50,
        "deterministic": True,
        "dump_feature_maps": False,
        "dummy_marker": "DUMMY",
        "timeout_s": 900,
    },
    "search": {
        "levels": "analytic+measure",
        "multistage": True,
        "resource_order": None,
        "min_segment": 2,
        "l1_pop": 200,
        "l1_gen": 800,
        "l2_pop": 36,
        "l2_gen": 30,
        "crossover": 0.8,
        "mutation": 0.2,
        "swap": 0.5,
        "seed": 1,
        "canonicalize": True,
        "dummy_offsets": [0, 1],
        "auto_calibrate": True,
        "calib_runs": 12,
        "ridge_lambda": 0.0,
        "recalibrate": True,
        "final_repeats": 5,
        "results": "search_results.json",
        "calibration": "analytical_model.json",
    },
    "attack": {
        "csv": None,
        "dataset_dir": None,
        "genes": None,
        "taps": None,
        "semantics": "device",
        "units": None,
        "out": "attack",
        "max_images": None,
        "n_buffer": 200,
        "batch": 8,
        "gpu": 0,
        "strict_gpu": False,
        "tf32": False,
        "attack_side_inputs": True,
        "shared_arena": True,
        "cudnn_algo_search": "HEURISTIC",
        "cudnn_max_workspace": False,
        "reference_baseline": None,
        "figures": True,
        "jpeg_decode": "cpu",
        "decode_workers": 8,
        "annotations": None,
        "examples": 4,
        "configs": DEFAULT_ATTACKS,
    },
    "report": {
        "enabled": True,
        "stealth_band": 10.0,
        "extra_bands": [],
        "success_asr": 50.0,
        "stealth_study": True,
        "collect": "auto",
        "batch": None,
        "attack_dirs": ["attack"],
        "runs": None,
        "out": None,
        "threshold": {
            "mode": "fixed",
            "source": "test",
            "calibration": 0.3,
            "validation": None,
            "max_validation": None,
            "false_alarm": 0.01,
            "window": 1000,
            "drift": False,
            "margin": 1.0,
            "repeats": 5,
        },
    },
}


class Resource:
    def __init__(self, name, host, kind, cores, device=0):
        if "/" in name:
            raise ValueError(f"resource name may not contain '/': {name!r}")
        self.name = name
        self.host = host
        self.kind = kind
        self.cores = list(cores)
        self.device = int(device)

    @property
    def threads(self):
        return len(self.cores) if self.kind == "cpu" else 1

    @property
    def slots(self):
        return ",".join(str(c) for c in self.cores)

    @property
    def ident(self):
        return re.sub(r"\W", "_", self.name)

    @property
    def equiv_key(self):
        return (self.host, self.kind, self.threads)

    def __repr__(self):
        return f"Resource({self.name}, {self.kind}, cores={self.slots})"


_RES_RE = re.compile(r"^(?P<host>.+)_(?P<kind>cpu|arm|gpu)(?P<ids>[0-9.\-]*)$")


def _parse_cores(ids):
    if not ids:
        return []
    if "." in ids or "," in ids:
        return [int(x) for x in re.split(r"[.,]", ids) if x]
    if "-" in ids:
        a, b = ids.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(ch) for ch in ids]


def parse_resource(spec, gpu_slot=0):
    fields = dict(spec) if isinstance(spec, dict) else {"name": str(spec)}
    name = fields["name"]
    m = _RES_RE.match(name)
    host = fields.get("host", m.group("host") if m else None)
    kind = fields.get("kind", ("cpu" if m.group("kind") == "arm" else m.group("kind")) if m else None)
    if host is None or kind not in ("cpu", "gpu"):
        raise ValueError(f"cannot parse resource {name!r}: expected <host>_cpu<cores> or "
                         f"<host>_gpu<index>, or a dict with host/kind/cores")
    if kind == "gpu":
        device = fields.get("device", int(m.group("ids") or 0) if m else 0)
        cores = fields.get("cores", [gpu_slot])
    else:
        device = 0
        cores = fields.get("cores", _parse_cores(m.group("ids")) if m else [])
    if isinstance(cores, (int, str)):
        cores = _parse_cores(str(cores)) if isinstance(cores, str) else [cores]
    if not cores:
        raise ValueError(f"resource {name!r} has no cores")
    return Resource(name, host, kind, cores, device)


def resources(cfg):
    plat = cfg["platform"]
    out = [parse_resource(r, plat["gpu_slot"]) for r in plat["resources"]]
    idents = [r.ident for r in out]
    if len(set(idents)) != len(idents):
        raise ValueError(f"duplicate resource names: {[r.name for r in out]}")
    if not out:
        raise ValueError("platform.resources is empty")
    return out


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def _set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def _abs(path, base):
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def load(path, overrides=()):
    path = os.path.abspath(path)
    with open(path) as fh:
        user = yaml.safe_load(fh) or {}
    cfg = _merge(copy.deepcopy(DEFAULTS), user)
    user_pp = set((user.get("preprocess") or {}).keys())
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {item!r}")
        _set_dotted(cfg, key.strip(), yaml.safe_load(raw))
        if key.strip().startswith("preprocess."):
            user_pp.add(key.strip().split(".", 1)[1])
    cfg["_user_preprocess"] = sorted(user_pp)

    base = os.path.dirname(path)
    cfg["config_path"] = path
    cfg["name"] = os.path.splitext(os.path.basename(path))[0]
    if not cfg["model"]:
        raise ValueError(f"{path}: 'model' is required")
    cfg["model"] = _abs(cfg["model"], base)
    cfg["workdir"] = _abs(cfg["workdir"] or os.path.join(TOOL_DIR, "runs", cfg["name"]), base)

    ad = cfg["autodice"]
    subst = {"root": _abs(ad["root"], base), "prefix": sys.prefix}
    ad["root"] = subst["root"]

    def tpl(p):
        return _abs(str(p).format(**subst), base)

    ad["onnx2ncnn"] = tpl(ad["onnx2ncnn"])
    for k in ("mpicxx", "mpirun"):
        v = str(ad[k]).format(**subst)
        ad[k] = v if "/" not in v or os.path.exists(v) else os.path.basename(v)
    ad["cxxflags"] = [str(f).format(**subst) for f in ad["cxxflags"]]
    ad["include_dirs"] = [tpl(p) for p in ad["include_dirs"]]
    ad["lib_dirs"] = [tpl(p) for p in ad["lib_dirs"]]
    for k in ("bench_image", "labels"):
        ad[k] = _abs(ad[k], base)

    work = cfg["workdir"]
    se = cfg["search"]
    se["results"] = _abs(se["results"], work)
    se["calibration"] = _abs(se["calibration"], work)

    at = cfg["attack"]
    for k in ("csv", "dataset_dir", "reference_baseline", "annotations"):
        at[k] = _abs(at[k], base)
    at["out"] = _abs(at["out"], work)
    at["genes"] = _abs(at["genes"], work) if at["genes"] else se["results"]

    rp = cfg["report"]
    for k in ("runs", "out"):
        rp[k] = _abs(rp[k], base)
    th = rp["threshold"]
    th["validation"] = _abs(th["validation"], base)
    if th["mode"] not in ("fixed", "calibrated", "fixed_window") or th["source"] not in ("test", "validation"):
        raise ValueError("report.threshold.mode must be fixed, calibrated or fixed_window, "
                         "report.threshold.source test or validation")
    if not 0 < float(th["calibration"]) < 1 or not 0 < float(th["false_alarm"]) < 0.5 or int(th["window"]) < 1 \
            or not float(th["margin"]) > 0:
        raise ValueError("report.threshold: calibration in (0, 1), false_alarm in (0, 0.5), window >= 1, margin > 0")
    if th["mode"] != "fixed" and th["source"] == "validation" and not th["validation"]:
        raise ValueError("report.threshold.source validation needs report.threshold.validation (clean images)")
    if not float(rp["stealth_band"]) > 0 or not all(float(b) > 0 for b in rp["extra_bands"] or []):
        raise ValueError("report.stealth_band and report.extra_bands must be > 0 (percent of the clean std)")
    if not 0 < float(rp["success_asr"]) <= 100:
        raise ValueError("report.success_asr must be in (0, 100] (attack success rate, %)")
    if rp["collect"] not in ("auto", "always", "never"):
        raise ValueError("report.collect must be auto, always or never")

    pp = cfg["preprocess"]
    if pp["channel_order"] not in ("RGB", "BGR"):
        raise ValueError("preprocess.channel_order must be RGB or BGR")
    if len(pp["mean"]) != 3 or len(pp["std"]) != 3:
        raise ValueError("preprocess.mean and preprocess.std need 3 values")
    if cfg["task"] not in ("auto", "classification", "detection"):
        raise ValueError("task must be auto, classification or detection")
    det = cfg["detection"]
    for k in ("conf", "eval_conf", "iou", "match_iou"):
        if not 0.0 <= float(det[k]) <= 1.0:
            raise ValueError(f"detection.{k} must be in [0, 1]")
    if det["label_map"] not in ("auto", "names", "contiguous", "identity"):
        raise ValueError("detection.label_map must be auto, names, contiguous or identity")
    return cfg


def public(cfg):
    return {k: v for k, v in cfg.items() if k not in ("config_path",) and not k.startswith("_")}


def dump(cfg, path):
    with open(path, "w") as fh:
        yaml.safe_dump(public(cfg), fh, sort_keys=False)
