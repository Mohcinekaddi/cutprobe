import os
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

SHAPE_OPS = ("Shape", "Size", "ConstantOfShape", "Range", "NonZero")

SEMANTIC_SINCE = {
    "Split": 18, "Pad": 19, "Resize": 19, "LpPool": 18, "ScatterElements": 18, "ScatterND": 18,
    "ReduceL1": 18, "ReduceL2": 18, "ReduceLogSum": 18, "ReduceLogSumExp": 18, "ReduceMax": 18,
    "ReduceMean": 18, "ReduceMin": 18, "ReduceProd": 18, "ReduceSumSquare": 18,
    "Col2Im": 18, "CenterCropPad": 18, "Mish": 18, "BitwiseAnd": 18, "BitwiseOr": 18,
    "BitwiseXor": 18, "BitwiseNot": 18, "AveragePool": 19, "DeformConv": 19,
    "AffineGrid": 20, "Gelu": 20, "GridSample": 20, "DFT": 20, "ImageDecoder": 20,
    "GroupNormalization": 21, "Attention": 23, "RMSNormalization": 23, "RotaryEmbedding": 23,
}
KNOWN_OPSET = 23


def _rename(s):
    return s.replace("/", "_")


def rename_tensors(model):
    g = model.graph
    for t in g.initializer:
        t.name = _rename(t.name)
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        vi.name = _rename(vi.name)
    for n in g.node:
        for i in range(len(n.input)):
            n.input[i] = _rename(n.input[i])
        for i in range(len(n.output)):
            n.output[i] = _rename(n.output[i])
        n.name = n.output[0]
    return model


_ORT_OPSET = []


def ort_max_opset():
    if not _ORT_OPSET:
        import onnxruntime as ort
        found = None
        for v in range(30, 6, -1):
            g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "probe",
                                  [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", v)])
            m.ir_version = 7
            try:
                ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
                found = v
                break
            except Exception:
                continue
        _ORT_OPSET.append(found)
    return _ORT_OPSET[0]


def _fp16_to_fp32(model):
    g = model.graph
    n = 0
    for t in g.initializer:
        if t.data_type == TensorProto.FLOAT16:
            t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float32), t.name))
            n += 1
    for node in g.node:
        for a in node.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT16:
                a.t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(a.t).astype(np.float32),
                                                     a.t.name))
                n += 1
            elif a.name == "to" and node.op_type == "Cast" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
                n += 1
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
            n += 1
    return n


def _hoist_constants(model):
    g = model.graph
    keep, n = [], 0
    for node in g.node:
        if (node.op_type == "Constant" and len(node.attribute) == 1
                and node.attribute[0].name == "value"):
            t = onnx.TensorProto()
            t.CopyFrom(node.attribute[0].t)
            t.name = node.output[0]
            g.initializer.append(t)
            n += 1
        else:
            keep.append(node)
    if n:
        del g.node[:]
        g.node.extend(keep)
    return n


def metadata(model):
    return {p.key: p.value for p in model.metadata_props}


def meta_imgsz(model):
    import ast
    v = metadata(model).get("imgsz")
    try:
        v = ast.literal_eval(v) if v else None
    except (ValueError, SyntaxError):
        return None
    if isinstance(v, int):
        return (v, v)
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    return None


def _fix_input_dims(model, input_hw):
    inits = initializer_names(model.graph)
    for vi in model.graph.input:
        if vi.name in inits:
            continue
        dims = vi.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") and d.dim_value > 0 for d in dims[1:]):
            return False
        input_hw = input_hw or meta_imgsz(model)
        if len(dims) != 4 or not input_hw:
            raise ValueError(f"input {vi.name!r} has dynamic dimensions "
                             f"{[d.dim_value or d.dim_param for d in dims]}: set preprocess.input_hw")
        channels = dims[1].dim_value if dims[1].HasField("dim_value") and dims[1].dim_value > 0 else 3
        for d, v in zip(dims, (1, channels, int(input_hw[0]), int(input_hw[1]))):
            d.ClearField("dim_param")
            d.dim_value = v
        return True
    return False


def _fold_with_ort(model, tmp_dir=None):
    import onnxruntime as ort
    for vi in list(model.graph.output):
        for d in vi.type.tensor_type.shape.dim:
            d.ClearField("dim_param")
            d.ClearField("dim_value")
    del model.graph.value_info[:]
    tmp = tempfile.mkdtemp(prefix="adstain_fold_", dir=tmp_dir)
    src, dst = os.path.join(tmp, "in.onnx"), os.path.join(tmp, "out.onnx")
    try:
        onnx.save(model, src)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = dst
        so.log_severity_level = 3
        ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
        out = onnx.load(dst)
    finally:
        for p in (src, dst):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(tmp)
    used = {n.domain or "" for n in out.graph.node}
    keep = [o for o in out.opset_import if (o.domain or "") in used | {""}]
    del out.opset_import[:]
    out.opset_import.extend(keep)
    out.ClearField("metadata_props")
    out.metadata_props.extend(model.metadata_props)
    return out


def normalize(model, input_hw=None, tmp_dir=None):
    notes = []
    n = _fp16_to_fp32(model)
    if n:
        notes.append(f"float16 -> float32 ({n} tensors)")
    top = ort_max_opset()
    for o in model.opset_import:
        if (o.domain or "") in ("", "ai.onnx") and top and o.version > top:
            ops = {n.op_type for n in model.graph.node if (n.domain or "") in ("", "ai.onnx")}
            bad = sorted(op for op in ops if top < SEMANTIC_SINCE.get(op, 0) <= o.version)
            ort_v = __import__("onnxruntime").__version__
            if bad:
                raise SystemExit(f"the model is opset {o.version} and onnxruntime {ort_v} loads at most {top}: "
                                 f"{', '.join(bad)} changed behaviour after opset {top}, so it cannot be relabelled. "
                                 f"Re-export the model with opset <= {top} (e.g. yolo export ... opset={top}).")
            notes.append(f"opset {o.version} -> {top} (highest onnxruntime {ort_v} loads; no operator of "
                         f"this model changed behaviour in between)"
                         + (f" [opset > {KNOWN_OPSET}: unverified]" if o.version > KNOWN_OPSET else ""))
            o.version = top
    n = _hoist_constants(model)
    if n:
        notes.append(f"{n} Constant node(s) -> initializers")
    fixed = _fix_input_dims(model, input_hw)
    if fixed or any(n.op_type in SHAPE_OPS for n in model.graph.node):
        before = len(model.graph.node)
        if not fixed and not _static_batch(model):
            _fix_batch(model)
        folded = _fold_with_ort(model, tmp_dir)
        model.CopyFrom(folded)
        left = sorted({n.op_type for n in model.graph.node if n.op_type in SHAPE_OPS})
        hw = model_input_hw(model, None)
        notes.append((f"input fixed to {hw[0]}x{hw[1]}, " if fixed and hw else "")
                     + f"shape computations folded ({before} -> {len(model.graph.node)} nodes)"
                     + (f"; still dynamic: {left}" if left else ""))
    return notes


def _static_batch(model):
    inits = initializer_names(model.graph)
    for vi in model.graph.input:
        if vi.name not in inits:
            d = vi.type.tensor_type.shape.dim
            return bool(len(d)) and d[0].HasField("dim_value") and d[0].dim_value > 0
    return True


def _fix_batch(model):
    inits = initializer_names(model.graph)
    for vi in model.graph.input:
        if vi.name not in inits and len(vi.type.tensor_type.shape.dim):
            d = vi.type.tensor_type.shape.dim[0]
            d.ClearField("dim_param")
            d.dim_value = 1


def load_model(path, infer=True):
    model = onnx.load(path)
    if infer:
        model = shape_inference.infer_shapes(model)
    for n in model.graph.node:
        if not n.name:
            n.name = n.output[0]
    return model


def initializer_names(graph):
    return {t.name for t in graph.initializer}


def data_inputs(graph):
    inits = initializer_names(graph)
    return [vi.name for vi in graph.input if vi.name not in inits]


def _dims(vi):
    return [d.dim_value if d.HasField("dim_value") else 0
            for d in vi.type.tensor_type.shape.dim]


def tensor_info(graph):
    info = {}
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        info[vi.name] = (vi.type.tensor_type.elem_type, _dims(vi))
    for t in graph.initializer:
        info.setdefault(t.name, (t.data_type, list(t.dims)))
    return info


def model_input_hw(model, default):
    inits = initializer_names(model.graph)
    for vi in model.graph.input:
        if vi.name in inits:
            continue
        dims = _dims(vi)
        if len(dims) == 4 and dims[2] > 0 and dims[3] > 0:
            return (dims[2], dims[3])
        break
    return tuple(default) if default else None


def dummy_nodes(model, marker):
    marker = marker.upper()
    out = set()
    for i, n in enumerate(model.graph.node):
        tokens = list(n.output) + [n.name or "", n.op_type or ""]
        if any(marker in str(t).upper() for t in tokens):
            out.add(i)
    return out


NCNN_COMPAT_VERSION = 2

AXES_AS_INPUT = ("Squeeze", "Unsqueeze", "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
                 "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
                 "ReduceSumSquare")


def _int64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def _unroll_batched_matmul(model):
    g = model.graph
    info = tensor_info(g)
    inits = initializer_names(g)
    nodes, extra, changed = [], [], 0
    for n in g.node:
        a = info.get(n.input[0], (0, []))[1] if n.op_type == "MatMul" else []
        b = info.get(n.input[1], (0, []))[1] if n.op_type == "MatMul" else []
        if not (n.op_type == "MatMul" and n.input[0] not in inits and n.input[1] not in inits
                and len(a) == 4 and len(b) == 4 and a[0] == 1 and b[0] == 1 and a[1] == b[1] > 1
                and min(a + b) > 0):
            nodes.append(n)
            continue
        h, M, K, N = a[1], a[2], a[3], b[3]
        y = n.output[0]
        sa = [f"{y}__a{i}" for i in range(h)]
        sb = [f"{y}__b{i}" for i in range(h)]
        nodes.append(helper.make_node("Split", [n.input[0]], sa, name=y + "__split_a", axis=1))
        nodes.append(helper.make_node("Split", [n.input[1]], sb, name=y + "__split_b", axis=1))
        extra += [_int64(y + "__shape_a", [1, M, K]), _int64(y + "__shape_b", [1, K, N]),
                  _int64(y + "__shape_c", [1, 1, M, N])]
        parts = []
        for i in range(h):
            nodes += [helper.make_node("Reshape", [sa[i], y + "__shape_a"], [f"{y}__a{i}_2d"], name=f"{y}__a{i}_2d"),
                      helper.make_node("Reshape", [sb[i], y + "__shape_b"], [f"{y}__b{i}_2d"], name=f"{y}__b{i}_2d"),
                      helper.make_node("MatMul", [f"{y}__a{i}_2d", f"{y}__b{i}_2d"], [f"{y}__c{i}_2d"],
                                       name=f"{y}__c{i}_2d"),
                      helper.make_node("Reshape", [f"{y}__c{i}_2d", y + "__shape_c"], [f"{y}__c{i}"],
                                       name=f"{y}__c{i}")]
            parts.append(f"{y}__c{i}")
        nodes.append(helper.make_node("Concat", parts, [y], name=y, axis=1))
        changed += 1
    if changed:
        del g.node[:]
        g.node.extend(nodes)
        g.initializer.extend(extra)
    return changed


def _positive_axes(model):
    info = tensor_info(model.graph)
    changed = 0
    for n in model.graph.node:
        rank = len(info.get(n.input[0], (0, []))[1]) if n.input else 0
        for a in n.attribute:
            if a.name == "axis" and a.type == onnx.AttributeProto.INT and a.i < 0 and rank:
                a.i += rank
                changed += 1
    return changed


def ncnn_compat(model):
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    changed = _positive_axes(model) + _unroll_batched_matmul(model)
    for n in g.node:
        name = None
        if n.op_type == "Split" and len(n.input) > 1 and n.input[1] in inits:
            name, attr = n.input[1], "split"
        elif n.op_type in AXES_AS_INPUT and len(n.input) > 1 and n.input[1] in inits:
            name, attr = n.input[1], "axes"
        if name is None:
            continue
        vals = [int(v) for v in numpy_helper.to_array(inits[name]).reshape(-1)]
        keep = [a for a in n.attribute if a.name != attr]
        del n.attribute[:]
        n.attribute.extend(keep)
        if not (attr == "split" and len(set(vals)) == 1):
            n.attribute.append(helper.make_attribute(attr, vals))
        del n.input[1:]
        changed += 1
    if changed:
        used = {x for n in g.node for x in n.input}
        keep = [t for t in g.initializer if t.name in used]
        del g.initializer[:]
        g.initializer.extend(keep)
        ins = [vi for vi in g.input if vi.name in used or vi.name not in inits]
        del g.input[:]
        g.input.extend(ins)
    return changed


def forbidden_cuts(model, marker, offsets):
    n = len(model.graph.node)
    out = {i + off for i in dummy_nodes(model, marker) for off in offsets if 0 <= i + off < n}
    return out | unpartitionable_cuts(model)


def carriable(entry):
    if entry is None:
        return False
    elem, dims = entry
    return elem in (0, TensorProto.FLOAT) and 2 <= len(dims) <= 4 and all(d > 0 for d in dims[1:])


def unpartitionable_cuts(model):
    g = model.graph
    n = len(g.node)
    info = tensor_info(g)
    inits = initializer_names(g)
    prod = {x: -1 for x in data_inputs(g)}
    last = {}
    for k, node in enumerate(g.node):
        for x in node.input:
            if x and x not in inits:
                last[x] = k
        for o in node.output:
            prod[o] = k
    first_out = min((prod[o.name] for o in g.output if o.name in prod), default=n)
    bad = set(range(first_out + 1, n))
    diff = np.zeros(n + 2, dtype=np.int64)
    for t, p in prod.items():
        if t in last and last[t] > p and not carriable(info.get(t)):
            diff[p + 1] += 1
            diff[last[t] + 1] -= 1
    cum = np.cumsum(diff)
    bad.update(c for c in range(1, n) if cum[c] > 0)
    return bad


def crossing_tensors(model, cut):
    g = model.graph
    inits = initializer_names(g)
    produced = set(data_inputs(g))
    for n in g.node[:cut]:
        produced.update(n.output)
    out = []
    for n in g.node[cut:]:
        for x in n.input:
            if x in produced and x not in inits:
                out.append(x)
    finals = [o.name for o in g.output if o.name in produced]
    return list(dict.fromkeys(out + finals))


def extract_submodel(model, node_names, out_path=None):
    g = model.graph
    keep = set(node_names)
    nodes = [n for n in g.node if n.name in keep]
    produced = {o for n in nodes for o in n.output}
    used = list(dict.fromkeys(x for n in nodes for x in n.input if x))
    inits = {t.name: t for t in g.initializer}
    orig_inputs = {vi.name: vi for vi in g.input}
    vinfo = {vi.name: vi for vi in list(g.value_info) + list(g.output) + list(g.input)}

    inputs = []
    for name in used:
        if name in produced:
            continue
        if name in orig_inputs:
            inputs.append(orig_inputs[name])
        elif name not in inits:
            vi = vinfo.get(name)
            if vi is None:
                raise ValueError(f"no shape information for tensor {name!r}")
            inputs.append(helper.make_tensor_value_info(
                name, vi.type.tensor_type.elem_type or TensorProto.FLOAT, _dims(vi)))
    outputs = [vi for vi in g.output if vi.name in produced]
    value_info = [vi for vi in g.value_info if vi.name in produced or vi.name in used]
    graph = helper.make_graph(nodes, g.name or "submodel", inputs, outputs,
                              initializer=[inits[x] for x in used if x in inits],
                              value_info=value_info)
    sub = helper.make_model(graph, opset_imports=list(model.opset_import),
                            producer_name="adstain")
    sub.ir_version = model.ir_version
    if out_path:
        onnx.save(sub, out_path)
    return sub


def reachable_nodes(model, inputs, outputs):
    producer = {o: n for n in model.graph.node for o in n.output}
    stop = set(inputs)
    seen, keep = set(), set()
    stack = list(outputs)
    while stack:
        t = stack.pop()
        if t in seen or t in stop:
            continue
        seen.add(t)
        n = producer.get(t)
        if n is None or n.name in keep:
            continue
        keep.add(n.name)
        stack.extend(x for x in n.input if x)
    return [n.name for n in model.graph.node if n.name in keep]


def _patch_reshape_batch(model):
    patched = []
    inits = {t.name: t for t in model.graph.initializer}
    consts = {}
    for n in model.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = a
    for n in model.graph.node:
        if n.op_type != "Reshape" or len(n.input) < 2:
            continue
        if any(a.name == "allowzero" and a.i == 1 for a in n.attribute):
            continue
        name = n.input[1]
        if name in inits:
            arr = numpy_helper.to_array(inits[name])
            if arr.ndim == 1 and arr.size and int(arr[0]) == 1:
                arr = arr.astype(np.int64).copy()
                arr[0] = 0
                inits[name].CopyFrom(numpy_helper.from_array(arr, name))
                patched.append(n.name)
        elif name in consts:
            attr = consts[name]
            arr = numpy_helper.to_array(attr.t)
            if arr.ndim == 1 and arr.size and int(arr[0]) == 1:
                arr = arr.astype(np.int64).copy()
                arr[0] = 0
                attr.t.CopyFrom(numpy_helper.from_array(arr, attr.t.name))
                patched.append(n.name)
    return patched


def make_batch_dynamic(model, symbol="N"):
    _patch_reshape_batch(model)
    inits = initializer_names(model.graph)
    if any(vi.name in inits for vi in model.graph.input):
        keep = [vi for vi in model.graph.input if vi.name not in inits]
        del model.graph.input[:]
        model.graph.input.extend(keep)
        model.ir_version = max(model.ir_version, 4)
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = vi.type.tensor_type.shape.dim
        if len(dims):
            dims[0].ClearField("dim_value")
            dims[0].dim_param = symbol
    del model.graph.value_info[:]
    return model


def resolve_dangling_inputs(sub, full):
    full_inits = {t.name: t for t in full.graph.initializer}
    have = initializer_names(sub.graph)
    keep = []
    for vi in sub.graph.input:
        if vi.name in have:
            continue
        if vi.name in full_inits:
            sub.graph.initializer.append(full_inits[vi.name])
            have.add(vi.name)
            continue
        keep.append(vi)
    if len(keep) != len(sub.graph.input):
        del sub.graph.input[:]
        sub.graph.input.extend(keep)
    return sub
