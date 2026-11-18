import ast
import re

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .onnx_graph import (data_inputs, initializer_names, metadata, reachable_nodes, tensor_info)

TASKS = ("classification", "detection")
HEADS = ("yolov5", "yolov8", "yolo_decoded", "boxes_scores", "end2end")
DET_OPS = ("NonMaxSuppression", "RoiAlign", "MaxRoiPool")
OTHER_TASKS = {"segment": "instance segmentation", "pose": "pose estimation",
               "obb": "oriented-box detection"}
V5_ANCHORS = [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]],
              [[116, 90], [156, 198], [373, 326]]]
YOLO_PRESET = {"channel_order": "RGB", "pixel_scale": 1.0 / 255.0, "mean": [0.0, 0.0, 0.0],
               "std": [1.0, 1.0, 1.0]}


class Task:
    def __init__(self, kind, reason, head=None, names=None, preset=None, notes=()):
        self.kind = kind
        self.reason = reason
        self.head = head
        self.names = list(names) if names else None
        self.preset = preset
        self.notes = list(notes)

    @property
    def detection(self):
        return self.kind == "detection"

    @property
    def num_classes(self):
        if self.head and self.head.get("num_classes"):
            return int(self.head["num_classes"])
        return len(self.names) if self.names else None

    def class_name(self, c):
        c = int(c)
        return self.names[c] if self.names and 0 <= c < len(self.names) else str(c)

    def as_dict(self):
        return {"task": self.kind, "reason": self.reason, "head": self.head, "names": self.names,
                "preset": self.preset, "notes": self.notes}

    @classmethod
    def from_dict(cls, d):
        return cls(d["task"], d.get("reason", ""), d.get("head"), d.get("names"), d.get("preset"),
                   d.get("notes", ()))

    def describe(self):
        lines = [f"task {self.kind} ({self.reason})"]
        h = self.head
        if h:
            fmt = h["format"]
            extra = []
            if h.get("strides"):
                extra.append("strides " + "/".join(str(s) for s in h["strides"]))
            if fmt == "yolov8":
                extra.append(f"DFL reg_max {h['reg_max']}" if h["reg_max"] > 1 else "no DFL")
            if h.get("end2end"):
                extra.append(f"NMS-free top-{h.get('topk')} selection")
            if fmt == "yolov5":
                extra.append("anchors " + " | ".join(
                    ",".join(f"{w:g}x{hh:g}" for w, hh in lvl) for lvl in h["anchors"]))
            if fmt in ("yolo_decoded", "boxes_scores"):
                extra.append(f"boxes {h.get('box_format')}" + (" normalised" if h.get("normalized") else ""))
            if fmt == "yolo_decoded":
                extra.append(("objectness, " if h.get("objectness") else "") + str(h.get("layout")))
            nc = h.get("num_classes")
            lines.append(f"head {fmt}: " + ", ".join([f"{nc} classes" if nc else "class ids from the model"] + extra))
            lines.append("decoder input(s): " + ", ".join(h["outputs"]))
            if h.get("check"):
                lines.append(h["check"])
        return lines


def parse_names(model):
    raw = metadata(model).get("names")
    if not raw:
        return None
    try:
        v = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if isinstance(v, dict):
        try:
            return [str(v[k]) for k in sorted(v, key=int)]
        except (ValueError, TypeError):
            return None
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return None


def _out_dims(model):
    info = tensor_info(model.graph)
    return [(o.name, info.get(o.name, (0, []))[1]) for o in model.graph.output]


def guess_task(model):
    ops = {n.op_type for n in model.graph.node}
    hit = [op for op in DET_OPS if op in ops]
    if hit:
        return "detection", f"the graph contains {hit[0]}"
    outs = _out_dims(model)
    if len(outs) == 1:
        name, d = outs[0]
        if len(d) == 2:
            return "classification", f"single output {name} [N, {d[1]}]"
        if len(d) == 4 and d[2] == 1 and d[3] == 1:
            return "classification", f"single output {name} [N, {d[1]}, 1, 1]"
        if len(d) == 3 and min(d[1], d[2]) >= 5:
            return "detection", f"single output {name} {d}: box candidates"
    else:
        boxes = [n for n, d in outs if len(d) >= 3 and d[-1] == 4]
        if boxes:
            return "detection", f"output {boxes[0]} holds boxes (last dimension 4)"
        named = [n for n, _ in outs if re.search(r"box|bbox|loc|anchor|detect|nms|score|label|num_det", n, re.I)]
        if named:
            return "detection", f"detection-like output names ({', '.join(named)})"
        name, d = outs[0]
        if len(d) == 2:
            return "classification", f"first output {name} [N, {d[1]}] (other outputs ignored)"
    raise SystemExit("cannot tell whether this model is a classifier or a detector (outputs "
                     + ", ".join(f"{n} {d}" for n, d in outs) + "): set 'task: classification' or "
                     "'task: detection' in the config")


def decide_kind(model, setting="auto"):
    meta = metadata(model)
    mt = (meta.get("task") or "").lower()
    if setting in TASKS:
        return setting, "set in the config"
    if setting not in (None, "auto"):
        raise SystemExit(f"task must be auto, classification or detection (got {setting!r})")
    if mt in ("classify", "classification"):
        return "classification", f"model metadata task={mt}"
    if mt in ("detect", "detection"):
        return "detection", f"model metadata task={mt}"
    if mt in OTHER_TASKS:
        raise SystemExit(f"this is an {OTHER_TASKS[mt]} model (metadata task={mt}): only "
                         f"classification and detection models are supported")
    return guess_task(model)


def _is_ultralytics(model):
    return metadata(model).get("author", "").lower().startswith("ultralytics")


def analyse(model, setting="auto", det=None, input_hw=None, log=print):
    det = det or {}
    kind, reason = decide_kind(model, setting)
    names = det.get("names") or parse_names(model)
    if kind == "classification":
        return Task(kind, reason, names=names, preset="yolo" if _is_ultralytics(model) else None)
    want = det.get("head") or "auto"
    if want not in ("auto",) + HEADS:
        raise SystemExit(f"detection.head must be auto or one of {HEADS} (got {want!r})")
    notes = []
    head = None
    if want in ("auto", "yolov5", "yolov8"):
        head, why = _raw_yolo_head(model, input_hw, det, names, want)
        if head is None:
            notes.append(f"no raw YOLO head: {why}")
            if want != "auto":
                raise SystemExit(f"detection.head={want}: {why}")
    if head is None:
        head = _decoded_head(model, det, names, None if want == "auto" else want)
    nc = head.get("num_classes")
    if names and nc and len(names) != nc:
        notes.append(f"{len(names)} class names for {nc} classes: names ignored")
        names = None
    preset = "yolo" if (_is_ultralytics(model) or head["format"] in ("yolov5", "yolov8")) else None
    return Task(kind, reason, head=head, names=names, preset=preset, notes=notes)


def _producers(g):
    return {o: n for n in g.node for o in n.output}


def _consumers(g):
    out = {}
    for n in g.node:
        for x in n.input:
            out.setdefault(x, []).append(n)
    return out


def pyramid_maps(model, input_hw):
    if not input_hw:
        return None
    g = model.graph
    info = tensor_info(g)
    inits = initializer_names(g)
    producer = _producers(g)
    inputs = set(data_inputs(g))
    H, W = input_hw

    def stride(t):
        dims = info.get(t, (0, []))[1]
        if len(dims) != 4 or min(dims[1:]) <= 0:
            return None
        for s in (4, 8, 16, 32, 64, 128):
            if dims[2] * s == H and dims[3] * s == W:
                return s
        return None

    maps, seen, stack = {}, set(), [o.name for o in g.output]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        s = stride(t)
        if s is not None:
            maps[t] = s
            continue
        if t in inputs:
            return None
        n = producer.get(t)
        if n is not None:
            stack.extend(x for x in n.input if x and x not in inits)
    return maps or None


def _const(g, name):
    for t in g.initializer:
        if t.name == name:
            return numpy_helper.to_array(t)
    return None


def _dfl_reg_max(model, nodes):
    g = model.graph
    by_name = {n.name: n for n in g.node}
    has_softmax = any(by_name[n].op_type == "Softmax" for n in nodes)
    for n in nodes:
        node = by_name[n]
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        w = _const(g, node.input[1])
        if w is None or w.ndim != 4 or w.shape[0] != 1 or w.shape[2:] != (1, 1):
            continue
        r = w.shape[1]
        if r > 1 and np.allclose(w.reshape(-1), np.arange(r)) and has_softmax:
            return int(r)
    return None


def _dfl_inputs(model, nodes):
    g = model.graph
    by_name = {n.name: n for n in g.node}
    producer = _producers(g)
    keep = set(nodes)
    stack = [x for n in nodes if by_name[n].op_type == "Softmax" for x in by_name[n].input]
    seen = set()
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        n = producer.get(t)
        if n is not None and n.name in keep:
            stack.extend(x for x in n.input if x)
    return seen


def _v5_layout(model, maps):
    g = model.graph
    cons = _consumers(g)
    shapes = set()
    for t in maps:
        for n in cons.get(t, []):
            if n.op_type == "Reshape" and len(n.input) > 1:
                s = _const(g, n.input[1])
                if s is not None and s.size == 5:
                    shapes.add((int(s[1]), int(s[2])))
    return shapes.pop() if len(shapes) == 1 else None


def _v5_anchors(model, maps_by_level, na):
    g = model.graph
    cons = _consumers(g)
    out = []
    for t in maps_by_level:
        found, seen, stack = None, set(), [t]
        while stack and found is None:
            x = stack.pop()
            for n in cons.get(x, []):
                if n.name in seen:
                    continue
                seen.add(n.name)
                for i in n.input:
                    c = _const(g, i) if n.op_type == "Mul" else None
                    if c is not None and c.ndim == 5 and c.shape[1] == na and c.shape[-1] == 2:
                        flat = c.reshape(na, -1, 2)
                        if np.allclose(flat, flat[:, :1, :]):
                            found = flat[:, 0, :]
                            break
                stack.extend(n.output)
        if found is None:
            return None
        out.append([[float(v) for v in a] for a in found])
    return out


def _raw_yolo_head(model, input_hw, det, names, want):
    maps = pyramid_maps(model, input_hw)
    if not maps:
        return None, "no 4-D pyramid map (H, W = input / 8, 16, 32...) separates the outputs from the image"
    g = model.graph
    info = tensor_info(g)
    pos = {o: k for k, n in enumerate(g.node) for o in n.output}
    order = sorted(maps, key=lambda t: (maps[t], pos.get(t, 0)))
    if len(g.output) != 1:
        return None, f"{len(g.output)} outputs (a raw YOLO head ends in one candidate tensor)"
    final = g.output[0].name
    nodes = reachable_nodes(model, order, [final])
    levels = sorted(set(maps.values()))
    per = {s: [t for t in order if maps[t] == s] for s in levels}
    reg_max = _dfl_reg_max(model, nodes)
    pairs_ok = all(len(v) == 2 for v in per.values())
    if (not reg_max and pairs_ok and want in ("auto", "yolov8")
            and all(sum(info[t][1][1] == 4 for t in per[s]) == 1 for s in levels)):
        reg_max = 1
    if reg_max and want in ("auto", "yolov8"):
        if all(len(v) == 1 for v in per.values()) and reg_max > 1:
            channels = {info[t][1][1] for t in order}
            if len(channels) != 1:
                return None, f"pyramid maps with different channel counts {sorted(channels)}"
            nc = channels.pop() - 4 * reg_max
            split = False
        elif pairs_ok:
            dfl = _dfl_inputs(model, nodes) if reg_max > 1 else None
            pairs = []
            for s in levels:
                box = [t for t in per[s] if (t in dfl if dfl is not None else info[t][1][1] == 4)]
                if len(box) != 1 or info[box[0]][1][1] != 4 * reg_max:
                    return None, f"stride {s}: no box map with 4x{reg_max} channels"
                pairs += [box[0], next(t for t in per[s] if t != box[0])]
            order = pairs
            ncs = {info[t][1][1] for t in order[1::2]}
            if len(ncs) != 1:
                return None, f"class maps with different channel counts {sorted(ncs)}"
            nc = ncs.pop()
            split = True
        else:
            return None, "pyramid maps neither one per level nor a box + class pair per level"
        if nc < 1:
            return None, f"the maps cannot hold 4x{reg_max} DFL bins and classes"
        head = {"format": "yolov8", "num_classes": nc, "reg_max": reg_max, "strides": levels,
                "split": split}
    elif want in ("auto", "yolov5"):
        if any(len(v) != 1 for v in per.values()):
            return None, "several pyramid maps per level without DFL decoding"
        channels = {info[t][1][1] for t in order}
        if len(channels) != 1:
            return None, f"pyramid maps with different channel counts {sorted(channels)}"
        C = channels.pop()
        strides = [maps[t] for t in order]
        lay = _v5_layout(model, order)
        if lay is None:
            na = len(det.get("anchors") or V5_ANCHORS[0])
            no = C // na
            if no * na != C or no < 6:
                return None, "no DFL decoding and no [1, na, 5+nc, H, W] reshape"
        else:
            na, no = lay
        anchors = det.get("anchors") or _v5_anchors(model, order, na)
        if anchors is None and len(order) == 3 and na == 3:
            anchors = V5_ANCHORS
        if anchors is None or len(anchors) != len(order):
            return None, "anchor sizes not found (set detection.anchors)"
        head = {"format": "yolov5", "num_classes": no - 5, "strides": strides,
                "anchors": [[[float(v) for v in a] for a in lvl] for lvl in anchors]}
    else:
        return None, f"no {want} decoding found"
    head["outputs"] = order
    head["box_format"] = "cxcywh"
    by_name = {n.name: n for n in g.node}
    topk = [by_name[n] for n in nodes if by_name[n].op_type == "TopK"]
    cand = final
    if topk:
        k = _const(g, topk[-1].input[1]) if len(topk[-1].input) > 1 else None
        head["end2end"] = True
        head["topk"] = int(np.asarray(k).reshape(-1)[0]) if k is not None else 300
        cand = _candidate_tensor(model, nodes, head, info, maps)
    ok, msg = _check_raw(model, head, final, cand)
    if not ok:
        return None, f"{head['format']} decoding does not reproduce the model output ({msg})"
    head["check"] = f"decoder check vs the model's own decoding: {msg}"
    removed = truncate(model, order)
    head["removed_nodes"] = removed
    return head, ""


def sample_outputs(model, extra=()):
    import onnxruntime as ort
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for t in extra:
        if t not in have:
            m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    x = np.random.RandomState(0).rand(*shape).astype(np.float32)
    return dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {inp.name: x})))


def _candidate_tensor(model, nodes, head, info, maps):
    by_name = {n.name: n for n in model.graph.node}
    A = sum(info[t][1][2] * info[t][1][3] for t in maps if t in head["outputs"][::2 if head.get("split") else 1])
    for n in nodes:
        o = by_name[n].output[0]
        if info.get(o, (0, []))[1] == [1, 4 + head["num_classes"], A]:
            return o
    return None


def _check_raw(model, head, final, cand=None):
    import torch
    from .detection import e2e_rows, raw_to_decoded
    extra = list(head["outputs"]) + ([cand] if cand else [])
    outs = sample_outputs(model, extra)
    maps = [torch.from_numpy(outs[t]) for t in head["outputs"]]
    msg = "no candidate tensor to compare"
    if cand:
        ref = outs[cand]
        ok = False
        for box_out in (("cxcywh", "xyxy") if head["format"] == "yolov8" else ("cxcywh",)):
            head["box_out"] = box_out
            with torch.no_grad():
                mine = raw_to_decoded(maps, head).numpy()
            if mine.shape != ref.shape:
                msg = f"shape {list(mine.shape)} vs {list(ref.shape)}"
                continue
            err = np.abs(mine - ref)
            worst = float(err.max())
            if (err <= 1e-3 + 1e-4 * np.abs(ref)).all():
                ok, msg = True, f"max |diff| {worst:.2g} over {ref.size} values"
                break
            msg = f"max |diff| {worst:.3g}"
        if not ok:
            return False, msg
        if head.get("box_out") == "cxcywh":
            head.pop("box_out")
    if head.get("end2end"):
        ref = outs[final].reshape(-1, outs[final].shape[-1])
        with torch.no_grad():
            mine = e2e_rows(maps, head).numpy().reshape(ref.shape)
        if mine.shape != ref.shape:
            return False, f"selection shape {list(mine.shape)} vs {list(ref.shape)}"
        hit = 0
        for row in ref:
            same = (mine[:, 5] == row[5]) & (np.abs(mine[:, 4] - row[4]) <= 1e-4) \
                & (np.abs(mine[:, :4] - row[:4]).max(1) <= 1e-2)
            hit += int(same.any())
        if hit < 0.98 * len(ref):
            return False, f"top-{head['topk']} selection: {hit}/{len(ref)} rows reproduced"
        msg += f"; top-{head['topk']} selection {hit}/{len(ref)} rows reproduced"
    return True, msg


def truncate(model, outputs):
    g = model.graph
    info = tensor_info(g)
    inputs = data_inputs(g)
    keep = set(reachable_nodes(model, inputs, outputs))
    nodes = [n for n in g.node if n.name in keep]
    removed = len(g.node) - len(nodes)
    used = {x for n in nodes for x in n.input}
    produced = {o for n in nodes for o in n.output} | set(inputs)
    inits = [t for t in g.initializer if t.name in used]
    graph_inputs = [vi for vi in g.input if vi.name in inputs or vi.name in used]
    value_info = [vi for vi in g.value_info if vi.name in produced and vi.name not in outputs]
    del g.node[:]
    g.node.extend(nodes)
    del g.initializer[:]
    g.initializer.extend(inits)
    del g.input[:]
    g.input.extend(graph_inputs)
    del g.value_info[:]
    g.value_info.extend(value_info)
    del g.output[:]
    for t in outputs:
        g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, info[t][1]))
    return removed


def _decoded_head(model, det, names, want):
    outs = _out_dims(model)
    nc_names = len(names) if names else None
    nc_cfg = det.get("num_classes")

    def finish(head):
        for k in ("box_format", "normalized", "objectness", "background", "layout"):
            if det.get(k) is not None:
                head[k] = det[k]
        if nc_cfg:
            head["num_classes"] = int(nc_cfg)
        if head.get("normalized") is None:
            b = sample_outputs(model)[head["outputs"][0]]
            head["normalized"] = bool(np.abs(b[..., :4]).max() <= 2.0)
        return head

    if len(outs) == 1:
        name, d = outs[0]
        if len(d) != 3:
            raise SystemExit(f"detection output {name} {d}: expected [N, A, K] or [N, K, A]; set "
                             f"detection.head")
        a, b = d[1], d[2]
        if want in (None, "end2end") and b in (6, 7) and a <= 1000:
            if want == "end2end" or not (nc_names and nc_names + 5 == b):
                return finish({"format": "end2end", "outputs": [name], "layout": "anchors_first",
                               "box_format": "xyxy", "normalized": None,
                               "num_classes": nc_names or nc_cfg})
        if want not in (None, "yolo_decoded"):
            raise SystemExit(f"detection.head={want} does not fit the output {name} {d}")
        first = det.get("layout") or ("anchors_first" if a > b else "anchors_last")
        k = b if first == "anchors_first" else a
        obj = det.get("objectness")
        if obj is None:
            if nc_cfg or nc_names:
                obj = k - int(nc_cfg or nc_names) == 5
            else:
                obj = first == "anchors_first"
        nc = k - (5 if obj else 4)
        return finish({"format": "yolo_decoded", "outputs": [name], "layout": first,
                       "objectness": bool(obj), "num_classes": nc, "box_format": "cxcywh",
                       "normalized": False})
    boxes = [(n, d) for n, d in outs if len(d) >= 3 and d[-1] == 4]
    if not boxes:
        raise SystemExit("detection outputs " + ", ".join(f"{n} {d}" for n, d in outs)
                         + ": no box tensor [N, A, 4]; set detection.head")
    bname, bd = boxes[0]
    A = bd[1]
    rest = [(n, d) for n, d in outs if n != bname]
    scores = [(n, d) for n, d in rest if len(d) == 3 and A in (d[1], d[2])]
    if scores and want in (None, "boxes_scores"):
        sname, sd = scores[0]
        layout = "anchors_first" if sd[1] == A else "anchors_last"
        C = sd[2] if layout == "anchors_first" else sd[1]
        return finish({"format": "boxes_scores", "outputs": [bname, sname], "layout": layout,
                       "num_classes": C, "box_format": "xyxy", "normalized": None,
                       "background": None})
    flat = [(n, d) for n, d in rest if len(d) == 2 and d[1] == A]
    if len(flat) >= 2 and want in (None, "end2end"):
        sname = next((n for n, _ in flat if re.search(r"score|conf|prob", n, re.I)), flat[0][0])
        lname = next((n for n, _ in flat if n != sname), None)
        return finish({"format": "end2end", "outputs": [bname, sname, lname], "layout": "split",
                       "box_format": "xyxy", "normalized": None,
                       "num_classes": nc_names or nc_cfg})
    raise SystemExit("cannot interpret the detection outputs " + ", ".join(f"{n} {d}" for n, d in outs)
                     + ": set detection.head")


def resolve_preprocess(cfg, task, input_hw):
    pp = cfg["preprocess"]
    user = set(cfg.get("_user_preprocess") or ())
    applied = []
    if task.preset == "yolo":
        preset = dict(YOLO_PRESET)
        if task.detection:
            preset.update(letterbox=True, resize_short=None)
        else:
            preset.update(letterbox=False, resize_short=int(min(input_hw)) if input_hw else None)
        for k, v in preset.items():
            if k not in user and pp.get(k) != v:
                pp[k] = v
                applied.append(k)
    elif task.detection and "resize_short" not in user and "letterbox" not in user and pp.get("resize_short"):
        pp["resize_short"] = None
        applied.append("resize_short")
    return applied
