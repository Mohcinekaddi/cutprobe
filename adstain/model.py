import json
import os

import onnx
from onnx import shape_inference

from .onnx_graph import load_model, model_input_hw, normalize, rename_tensors
from .task import Task, analyse, resolve_preprocess

FORMAT_VERSION = 3
HEAD_KEYS = ("head", "names", "num_classes", "anchors", "layout", "objectness", "box_format",
             "normalized", "background")


def formatted_path(cfg):
    return os.path.join(cfg["workdir"], "format_" + os.path.basename(cfg["model"]))


def _code_hash():
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha1()
    for f in ("task.py", "detection.py", "onnx_graph.py"):
        with open(os.path.join(here, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:12]


def _key(cfg):
    src = cfg["model"]
    det = cfg.get("detection") or {}
    return {"version": FORMAT_VERSION, "code": _code_hash(), "source": src, "mtime": os.path.getmtime(src),
            "size": os.path.getsize(src), "input_hw": cfg["preprocess"].get("input_hw"),
            "task": cfg.get("task", "auto"), "head": {k: det.get(k) for k in HEAD_KEYS}}


def _build(cfg, dst):
    pp = cfg["preprocess"]
    model = onnx.load(cfg["model"])
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    notes = normalize(model, pp.get("input_hw"), os.path.dirname(dst) or ".")
    model = rename_tensors(shape_inference.infer_shapes(model))
    hw = model_input_hw(model, pp.get("input_hw"))
    n0 = len(model.graph.node)
    task = analyse(model, cfg.get("task", "auto"), cfg.get("detection"), hw)
    if len(model.graph.node) != n0:
        notes.append(f"decoding head cut off: {n0 - len(model.graph.node)} node(s) removed, the network "
                     f"ends at {len(model.graph.output)} raw map(s)")
        model = shape_inference.infer_shapes(model)
    task.notes = notes + task.notes
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    onnx.save(model, dst)
    return task


def prepare(cfg, verbose=True):
    dst = formatted_path(cfg)
    side = os.path.splitext(dst)[0] + ".task.json"
    key = _key(cfg)
    info = None
    if os.path.exists(dst) and os.path.exists(side):
        try:
            with open(side) as fh:
                info = json.load(fh)
        except (OSError, ValueError):
            info = None
        if info and (info.get("key") != key or os.path.getmtime(dst) < key["mtime"]):
            info = None
    fresh = info is None
    if fresh:
        task = _build(cfg, dst)
        info = {"key": key, "task": task.as_dict()}
        with open(side, "w") as fh:
            json.dump(info, fh, indent=1)
    task = Task.from_dict(info["task"])
    model = load_model(dst)
    hw = model_input_hw(model, cfg["preprocess"].get("input_hw"))
    applied = resolve_preprocess(cfg, task, hw)
    if verbose:
        report(cfg, task, applied, fresh)
    return dst, model, task


def report(cfg, task, applied, fresh=False):
    name = os.path.basename(cfg["model"])
    lines = task.describe()
    print(f"[model] {name}: " + lines[0])
    for line in lines[1:]:
        print(f"[model]   {line}")
    if fresh:
        for n in task.notes:
            print(f"[model]   {n}")
    if applied:
        pp = cfg["preprocess"]
        print(f"[preprocess] {task.preset or 'detection'} defaults for "
              + ", ".join(f"{k}={pp[k]}" for k in applied) + " (set them in the config to override)")


def task_of(cfg):
    return prepare(cfg, verbose=False)[2]
