import os

import numpy as np
import torch
import onnxruntime as ort


def check_gpu(gpu_id, tf32):
    if not torch.cuda.is_available():
        raise SystemExit("[gpu] torch sees no CUDA device")
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" not in avail:
        raise SystemExit(f"[gpu] onnxruntime has no CUDA provider ({avail}); "
                         f"install onnxruntime-gpu")
    dev = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    print(f"[gpu] {torch.cuda.get_device_name(dev)}, torch {torch.__version__}, "
          f"onnxruntime {ort.__version__}, TF32={'on' if tf32 else 'off'}")
    return dev


def register_shared_arena(gpu_id):
    try:
        mem = ort.OrtMemoryInfo("Cuda", ort.OrtAllocatorType.ORT_ARENA_ALLOCATOR, gpu_id,
                                ort.OrtMemType.DEFAULT)
        cfg = ort.OrtArenaCfg(0, 1, -1, -1)
        ort.create_and_register_allocator_v2("CUDAExecutionProvider", mem,
                                             {"device_id": str(gpu_id)}, cfg)
        return True
    except Exception as e:
        print(f"[memory] shared CUDA arena unavailable ({e}), one arena per session")
        return False


def is_oom(exc):
    if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in ("out of memory", "cuda_error_out_of_memory",
                                  "failed to allocate memory", "cudnn_status_alloc_failed",
                                  "cudnn_status_internal_error", "cublas_status_alloc_failed"))


class GpuSession:
    def __init__(self, path, gpu_id, strict=True, tf32=False, in_name=None, shared_arena=False,
                 algo_search="HEURISTIC", max_workspace=False):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        if strict:
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        so.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
        if shared_arena:
            so.add_session_config_entry("session.use_env_allocators", "1")
        base = {"device_id": str(gpu_id), "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": algo_search,
                "cudnn_conv_use_max_workspace": "1" if max_workspace else "0"}
        self.sess, last = None, None
        for opts in (dict(base, use_tf32="1" if tf32 else "0"), base):
            try:
                self.sess = ort.InferenceSession(path, so, providers=[("CUDAExecutionProvider", opts)])
                break
            except Exception as e:
                last = e
        if self.sess is None:
            raise RuntimeError(f"cannot create a CUDA session for {os.path.basename(path)}: {last}"
                               + ("\n(strict_gpu: a node has no CUDA kernel; set "
                                  "attack.strict_gpu=false to diagnose)" if strict else ""))
        if self.sess.get_providers()[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"CUDA provider inactive: {self.sess.get_providers()}")
        self.gpu_id = gpu_id
        self.path = path
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.in_name = in_name or self.in_names[0]
        if self.in_name not in self.in_names:
            raise RuntimeError(f"{os.path.basename(path)}: no input {self.in_name!r} "
                               f"(inputs: {self.in_names})")
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self._tails = None
        self._in_tails = None
        self.batch_ok = None
        self.per_image = False

    def _bind_input(self, b, name, x):
        b.bind_input(name=name, device_type="cuda", device_id=self.gpu_id,
                     element_type=np.float32, shape=tuple(x.shape), buffer_ptr=x.data_ptr())

    def _bind_output(self, b, name, t):
        b.bind_output(name=name, device_type="cuda", device_id=self.gpu_id,
                      element_type=np.float32, shape=tuple(t.shape), buffer_ptr=t.data_ptr())

    def _feeds(self, x, extra):
        feeds = {self.in_name: x}
        for k, v in (extra or {}).items():
            feeds[k] = v.to(torch.float32).contiguous()
        if set(feeds) != set(self.in_names):
            raise RuntimeError(f"{os.path.basename(self.path)}: got inputs {sorted(feeds)}, "
                               f"expected {sorted(self.in_names)}")
        return feeds

    def _discover(self, feeds):
        b = self.sess.io_binding()
        ones = {k: v[:1].contiguous() for k, v in feeds.items()}
        for k, v in ones.items():
            self._bind_input(b, k, v)
        for n in self.out_names:
            b.bind_output(n, "cuda", self.gpu_id)
        torch.cuda.synchronize()
        self.sess.run_with_iobinding(b)
        b.synchronize_outputs()
        self._tails = {n: tuple(ov.shape()[1:]) for n, ov in zip(self.out_names, b.get_outputs())}
        self._in_tails = {k: tuple(v.shape[1:]) for k, v in ones.items()}

    def run(self, x, extra=None):
        x = x.to(torch.float32).contiguous()
        feeds = self._feeds(x, extra)
        if self._tails is None:
            self._discover(feeds)
        B = x.shape[0]
        if self.per_image and B > 1:
            return self._per_item(x, extra)
        outs = self._run(feeds, B, x.device)
        if self.batch_ok is None and B > 1:
            single = self._per_item(x[:2], {k: v[:2] for k, v in (extra or {}).items()})
            self.batch_ok = all(
                float((outs[n][:2] - single[n]).abs().max()) <= 1e-3 * (float(single[n].abs().max()) + 1.0)
                for n in self.out_names)
            if not self.batch_ok:
                self.per_image = True
                print(f"  [gpu] {os.path.basename(self.path)}: batched results differ from single-image "
                      f"ones (batch-dependent ops, e.g. TopK / NMS in the graph): run one image at a time")
                return self._per_item(x, extra)
        return outs

    def _per_item(self, x, extra):
        parts = [self._run(self._feeds(x[i:i + 1].contiguous(),
                                       {k: v[i:i + 1] for k, v in (extra or {}).items()}), 1, x.device)
                 for i in range(x.shape[0])]
        return {n: torch.cat([p[n] for p in parts], 0) for n in self.out_names}

    def _run(self, feeds, B, device):
        outs = {n: torch.empty((B,) + self._tails[n], device=device, dtype=torch.float32)
                for n in self.out_names}
        b = self.sess.io_binding()
        for k, v in feeds.items():
            self._bind_input(b, k, v)
        for n, t in outs.items():
            self._bind_output(b, n, t)
        torch.cuda.synchronize()
        self.sess.run_with_iobinding(b)
        b.synchronize_outputs()
        return outs

    def release(self):
        if self._in_tails is None or self._tails is None:
            return
        dev = torch.device(f"cuda:{self.gpu_id}")
        b = self.sess.io_binding()
        keep = []
        for k, s in self._in_tails.items():
            t = torch.zeros((1,) + s, device=dev)
            keep.append(t)
            self._bind_input(b, k, t)
        for n in self.out_names:
            t = torch.empty((1,) + self._tails[n], device=dev)
            keep.append(t)
            self._bind_output(b, n, t)
        ro = ort.RunOptions()
        ro.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{self.gpu_id}")
        torch.cuda.synchronize()
        self.sess.run_with_iobinding(b, ro)
        b.synchronize_outputs()
