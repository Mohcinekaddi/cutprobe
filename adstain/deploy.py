import glob
import json
import os
import shutil
import subprocess

import onnx

from .codegen import (buffer_manifest, build_stages, generate_cpp, mapping_from_cuts,
                      write_hostfile, write_rankfile)
from .onnx_graph import crossing_tensors, model_input_hw, ncnn_compat, unpartitionable_cuts


class DeployError(RuntimeError):
    pass


def _tail(path, n=25):
    try:
        with open(path, errors="replace") as fh:
            return "".join(fh.readlines()[-n:])
    except OSError:
        return ""


def compile_cmd(ad, src, out):
    return ([ad["mpicxx"], src, "-o", out]
            + ["-I" + d for d in ad["include_dirs"]]
            + ["-L" + d for d in ad["lib_dirs"]]
            + ["-l" + lib for lib in ad["libs"]]
            + list(ad["cxxflags"]))


def to_ncnn(ad, onnx_path, stem, run):
    src = onnx_path
    m = onnx.load(onnx_path)
    if ncnn_compat(m):
        src = stem + ".ncnn.onnx"
        onnx.save(m, src)
    run([ad["onnx2ncnn"], src, stem + ".param", stem + ".bin"], stem + ".onnx2ncnn.log")


def read_metric(path):
    try:
        with open(path) as fh:
            line = fh.readline().strip()
        return float(line) if line else None
    except (OSError, ValueError):
        return None


class Deployer:
    def __init__(self, cfg, model, resources, out_dir=None, task=None):
        self.cfg = cfg
        self.ad = cfg["autodice"]
        self.model = model
        self.task = task
        self.resources = resources
        self.dir = out_dir or os.path.join(cfg["workdir"], "deploy")
        self.binary = "bench"
        self.input_hw = model_input_hw(model, cfg["preprocess"]["input_hw"])
        if self.input_hw is None:
            raise ValueError("the model input size is dynamic: set preprocess.input_hw")
        self.runs = 0
        os.makedirs(self.dir, exist_ok=True)

    def _exec(self, cmd, log, timeout=None, env=None):
        with open(log, "w") as fh:
            try:
                p = subprocess.run(cmd, cwd=self.dir, stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=timeout, env=env)
            except subprocess.TimeoutExpired:
                raise DeployError(f"timeout after {timeout} s: {cmd[0]} (see {log})")
            except OSError as e:
                raise DeployError(f"cannot run {cmd[0]}: {e}")
        if p.returncode != 0:
            raise DeployError(f"{os.path.basename(cmd[0])} exited with {p.returncode} "
                              f"(see {log})\n{_tail(log)}")

    def _clean(self):
        for pat in ("*.onnx", "*.param", "*.bin", "*_PERFORMANCE.txt", "*_COMPUTE.txt",
                    "*_MEMORY.txt", "*.csv", "*.log", "buffer_manifest.json", "task.json"):
            for p in glob.glob(os.path.join(self.dir, pat)):
                os.remove(p)
        for d in ("fm", "fm_hist", "plots"):
            shutil.rmtree(os.path.join(self.dir, d), ignore_errors=True)

    def check_cuts(self, cuts):
        bad = sorted(set(int(c) for c in cuts[1:]) & unpartitionable_cuts(self.model))
        if bad:
            raise DeployError(f"cut(s) {bad} cannot be deployed: a crossing tensor cannot be sent between "
                              f"ranks, or a graph output would be computed before the last rank "
                              f"(see the 'cut' column of 'inspect --layers')")
        marker = self.ad["dummy_marker"]
        if not marker:
            return
        for c in cuts[1:]:
            bad = [t for t in crossing_tensors(self.model, int(c)) if marker.upper() in t.upper()]
            if bad:
                raise DeployError(f"cut {c} crosses {marker} tensor(s) {bad}")

    def build(self, cuts, placement, mode="bench"):
        if mode not in ("bench", "infer"):
            raise ValueError(f"unknown mode {mode!r}")
        os.makedirs(self.dir, exist_ok=True)
        self.check_cuts(cuts)
        self._clean()
        mapping = mapping_from_cuts(self.model, cuts, placement)
        with open(os.path.join(self.dir, "mapping.json"), "w") as fh:
            json.dump(mapping, fh)
        stages = build_stages(self.model, mapping, self.resources, self.dir)
        write_rankfile(os.path.join(self.dir, "rankfile"), stages)
        write_hostfile(os.path.join(self.dir, "hostfile"), stages)
        with open(os.path.join(self.dir, "comm.json"), "w") as fh:
            json.dump({st.resource.name: {"rank": st.rank, "recv": st.recv, "send": st.send}
                       for st in stages}, fh, indent=1)
        with open(os.path.join(self.dir, "buffer_manifest.json"), "w") as fh:
            json.dump(buffer_manifest(stages, self.model, cuts), fh, indent=2)
        if self.task is not None:
            with open(os.path.join(self.dir, "task.json"), "w") as fh:
                json.dump(dict(self.task.as_dict(), input_hw=list(self.input_hw),
                               preprocess=self.cfg["preprocess"], detection=self.cfg["detection"]),
                          fh, indent=1)

        for st in stages:
            stem = os.path.join(self.dir, st.resource.ident)
            to_ncnn(self.ad, stem + ".onnx", stem, self._exec)

        opts = {"mode": mode,
                "warmup": int(self.ad["warmup"]),
                "loops": max(int(self.ad["loops"]), 1),
                "deterministic": bool(self.ad["deterministic"]),
                "labels": self.ad["labels"],
                "dump": bool(self.ad["dump_feature_maps"]),
                "marker": self.ad["dummy_marker"],
                "task": self.task,
                "det": self.cfg["detection"]}
        src = os.path.join(self.dir, "multinode.cpp")
        with open(src, "w") as fh:
            fh.write(generate_cpp(stages, self.model, self.cfg["preprocess"], self.input_hw, opts))
        self.binary = "bench" if mode == "bench" else "multinode"
        self._exec(compile_cmd(self.ad, src, os.path.join(self.dir, self.binary)),
                   os.path.join(self.dir, "compile.log"))
        return stages

    def mpirun_cmd(self, n_ranks, arg, env_vars=()):
        exports = []
        for k in env_vars:
            exports += ["-x", k]
        return ([self.ad["mpirun"]] + [str(a) for a in self.ad["mpirun_args"]] + exports
                + ["-np", str(n_ranks), "-wdir", self.dir, "-rf", "rankfile",
                   "./" + self.binary, arg])

    def run(self, n_ranks, arg, env=None, log="run.log", timeout=None):
        full_env = dict(os.environ, **(env or {}))
        self._exec(self.mpirun_cmd(n_ranks, arg, sorted(env or {})), os.path.join(self.dir, log),
                   timeout=timeout or self.ad["timeout_s"], env=full_env)

    def measure(self, cuts, placement, image=None):
        image = image or self.ad["bench_image"]
        if not image or not os.path.exists(image):
            raise DeployError(f"autodice.bench_image not found: {image}")
        stages = self.build(cuts, placement, mode="bench")
        self.runs += 1
        self.run(len(stages), image)
        out = {"compute_ms": [], "wall_ms": [], "memory_mb": []}
        for r in range(len(stages)):
            vals = [read_metric(os.path.join(self.dir, f"{r}_{k}.txt"))
                    for k in ("COMPUTE", "PERFORMANCE", "MEMORY")]
            if any(v is None for v in vals):
                raise DeployError(f"missing metrics for rank {r} (see {self.dir}/run.log)")
            for key, v in zip(("compute_ms", "wall_ms", "memory_mb"), vals):
                out[key].append(v)
        return out
