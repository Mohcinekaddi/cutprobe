import hashlib
import json
import os
import subprocess

import numpy as np
import onnx
from onnx import TensorProto, helper

from .deploy import DeployError, compile_cmd, to_ncnn
from .onnx_graph import NCNN_COMPAT_VERSION, data_inputs, model_input_hw, tensor_info

TOL = 1e-2
HARNESS = r"""#include "net.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

int main(int argc, char** argv)
{
    if (argc != 10) return 1;
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_winograd_convolution = false;
    net.opt.num_threads = 4;
    if (net.load_param(argv[1]) != 0 || net.load_model(argv[2]) != 0) return 2;
    const int c = atoi(argv[5]), h = atoi(argv[6]), w = atoi(argv[7]);
    ncnn::Mat in(w, h, c);
    FILE* f = fopen(argv[4], "rb");
    if (!f) return 3;
    for (int q = 0; q < c; q++)
        if (fread(in.channel(q), sizeof(float), (size_t)w * h, f) != (size_t)w * h) return 3;
    fclose(f);
    ncnn::Extractor ex = net.create_extractor();
    ex.input(argv[3], in);
    std::ifstream names(argv[8]);
    std::string name;
    int k = 0;
    while (std::getline(names, name))
    {
        char path[4096];
        snprintf(path, sizeof(path), "%s/%d.raw", argv[9], k++);
        ncnn::Mat out;
        if (ex.extract(name.c_str(), out) != 0) continue;
        FILE* g = fopen(path, "wb");
        if (!g) return 4;
        int hdr[4] = {out.dims, out.w, out.h, out.c};
        fwrite(hdr, sizeof(int), 4, g);
        for (int q = 0; q < out.c; q++)
            fwrite((const float*)out.channel(q), sizeof(float), (size_t)out.w * out.h, g);
        fclose(g);
    }
    return 0;
}
"""


def _run(cmd, log, cwd=None):
    with open(log, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd)
    if p.returncode != 0:
        with open(log, errors="replace") as fh:
            tail = "".join(fh.readlines()[-15:])
        raise DeployError(f"{os.path.basename(cmd[0])} exited with {p.returncode} (see {log})\n{tail}")


def input_tensor(cfg, image, input_hw):
    import torch
    from .images import GpuImageLoader
    loader = GpuImageLoader(torch.device("cpu"), cfg["preprocess"], input_hw, decode="cpu", workers=1)
    x, valid = loader.load([image])
    if not valid[0]:
        raise SystemExit(f"cannot read the check image {image}")
    return x.numpy().astype(np.float32)


def _key(cfg, model_path, image):
    st = os.stat(model_path)
    return {"model": model_path, "mtime": st.st_mtime, "size": st.st_size, "image": image,
            "pp": cfg["preprocess"], "onnx2ncnn": cfg["autodice"]["onnx2ncnn"],
            "harness": hashlib.sha1(HARNESS.encode()).hexdigest()[:12], "tol": TOL,
            "compat": NCNN_COMPAT_VERSION}


def check(cfg, model_path, model, task, image=None, force=False, verbose=True):
    ad = cfg["autodice"]
    image = image or ad["bench_image"]
    if not image or not os.path.exists(image):
        if verbose:
            print("[ncnn-check] skipped: no autodice.bench_image to run")
        return None
    d = os.path.join(cfg["workdir"], "ncnn_check")
    os.makedirs(d, exist_ok=True)
    cache = os.path.join(d, "check.json")
    key = _key(cfg, model_path, image)
    if not force and os.path.exists(cache):
        try:
            with open(cache) as fh:
                res = json.load(fh)
            if res.get("key") == key:
                if verbose:
                    report(res)
                return res
        except (OSError, ValueError):
            pass

    to_ncnn(ad, model_path, os.path.join(d, "full"), lambda cmd, log: _run(cmd, log))
    src, exe = os.path.join(d, "harness.cpp"), os.path.join(d, "harness")
    old = open(src).read() if os.path.exists(src) else None
    if old != HARNESS or not os.path.exists(exe):
        with open(src, "w") as fh:
            fh.write(HARNESS)
        _run(compile_cmd(ad, src, exe), os.path.join(d, "compile.log"))

    hw = model_input_hw(model, cfg["preprocess"]["input_hw"])
    x = input_tensor(cfg, image, hw)
    g = model.graph
    info = tensor_info(g)
    op_of = {o: (k, n.op_type) for k, n in enumerate(g.node) for o in n.output}
    finals = [o.name for o in g.output]
    names = [o for n in g.node for o in n.output
             if o and info.get(o, (0, None))[0] in (0, TensorProto.FLOAT)]
    m = onnx.load(model_path)
    have = {o.name for o in m.graph.output}
    for t in names:
        if t not in have:
            m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    refs = dict(zip([o.name for o in sess.get_outputs()],
                    sess.run(None, {data_inputs(g)[0]: x})))

    out_dir = os.path.join(d, "out")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))
    x[0].tofile(os.path.join(d, "input.raw"))
    with open(os.path.join(d, "names.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    _run([exe, os.path.join(d, "full.param"), os.path.join(d, "full.bin"), data_inputs(g)[0],
          os.path.join(d, "input.raw"), str(x.shape[1]), str(x.shape[2]), str(x.shape[3]),
          os.path.join(d, "names.txt"), out_dir], os.path.join(d, "run.log"))

    first, worst, compared, outs, got_finals = None, 0.0, 0, {}, {}
    for k, t in enumerate(names):
        p = os.path.join(out_dir, f"{k}.raw")
        if not os.path.exists(p):
            continue
        got = np.fromfile(p, dtype=np.float32, offset=16)
        ref = refs[t]
        compared += 1
        if got.size != ref.size:
            msg, rel = f"ncnn gives {got.size} values for {list(ref.shape)}", float("inf")
        else:
            rel = float(np.abs(got - ref.reshape(-1)).max() / (np.abs(ref).max() + 1e-6))
            msg = f"relative difference {rel:.2e}"
        if t in finals:
            outs[t] = msg
            if got.size == ref.size:
                got_finals[t] = got.reshape(ref.shape)
        if rel > TOL and first is None:
            first = {"index": op_of[t][0], "op": op_of[t][1], "tensor": t, "detail": msg}
        if np.isfinite(rel):
            worst = max(worst, rel)
    res = {"key": key, "ok": first is None and len(got_finals) == len(finals), "compared": compared,
           "image": image, "worst": worst, "first_bad": first, "outputs": outs}
    if task is not None and len(got_finals) == len(finals):
        res["prediction"] = _prediction_agreement(task, refs, got_finals, finals, cfg, hw)
    with open(cache, "w") as fh:
        json.dump(res, fh, indent=1)
    if verbose:
        report(res)
    return res


def _prediction_agreement(task, refs, got, finals, cfg, hw):
    import torch
    if not task.detection:
        a, b = refs[finals[0]].reshape(-1), got[finals[0]].reshape(-1)
        return f"top-1 onnxruntime {int(a.argmax())} vs ncnn {int(b.argmax())}"
    from . import detection as D
    dec = D.Decoder(task.head, hw, cfg["detection"])
    conf = float(cfg["detection"]["conf"])
    ra = dec({f: torch.from_numpy(refs[f]) for f in finals}, conf)[0]
    rb = dec({f: torch.from_numpy(got[f]) for f in finals}, conf)[0]
    to_np = lambda d: tuple(v.numpy() for v in d)
    van, gho, rel, _ = D._ref_match(to_np(ra), to_np(rb), 0.5)
    return (f"{len(ra[1])} detection(s) with onnxruntime, {len(rb[1])} with ncnn at score >= {conf:g}: "
            f"{van} missing, {gho} extra in ncnn")


def report(res):
    if res is None:
        return
    name = os.path.basename(res["image"])
    if res["ok"]:
        print(f"[ncnn-check] ncnn matches onnxruntime on {name}: {res['compared']} layer outputs, largest "
              f"relative difference {res['worst']:.1e}" + (f"; {res['prediction']}" if res.get("prediction") else ""))
        return
    fb = res.get("first_bad")
    where = (f"from layer {fb['index']} ({fb['op']}, {fb['tensor']}): {fb['detail']}" if fb
             else "on the graph outputs")
    print(f"[ncnn-check] WARNING: this ncnn build does not compute the model like onnxruntime, {where}."
          f"\n[ncnn-check]   The real MPI deployment (deploy/build/mpi-pareto) will not match the GPU "
          f"campaign for this model (the GPU campaign itself is unaffected)."
          + (f"\n[ncnn-check]   {res['prediction']}" if res.get("prediction") else ""))
