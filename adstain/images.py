import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torchvision.io import ImageReadMode, decode_jpeg, decode_png, read_file
    TV_OK = True
except Exception:
    TV_OK = False


def _read_rgb_cv2(path):
    import cv2
    m = cv2.imread(path, cv2.IMREAD_COLOR)
    return None if m is None else cv2.cvtColor(m, cv2.COLOR_BGR2RGB)


class GpuImageLoader:

    def __init__(self, dev, preprocess, input_hw, decode="cpu", workers=8):
        self.dev = dev
        self.input_hw = tuple(input_hw)
        self.order = preprocess["channel_order"]
        self.scale = float(preprocess["pixel_scale"])
        self.resize_short = preprocess.get("resize_short")
        self.letterbox = bool(preprocess.get("letterbox"))
        self.pad = float(preprocess.get("pad_value", 114))
        self.mean = torch.tensor(preprocess["mean"], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(preprocess["std"], device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        self.n_gpu = self.n_png = self.n_cpu = self.n_failed = 0
        self.n_cpu_jpeg = 0
        self.decode = "gpu" if (decode == "gpu" and TV_OK) else "cpu"
        self.workers = max(1, int(workers))
        self._pool = ThreadPoolExecutor(self.workers) if self.decode == "cpu" else None

    def _decode_jpeg(self, path):
        data = read_file(path)
        try:
            img = decode_jpeg(data, mode=ImageReadMode.RGB, device=self.dev,
                              apply_exif_orientation=True)
        except TypeError:
            img = decode_jpeg(data, mode=ImageReadMode.RGB, device=self.dev)
        return img.expand(3, -1, -1) if img.shape[0] == 1 else img

    def _decode_png(self, path):
        return decode_png(read_file(path), mode=ImageReadMode.RGB).to(self.dev)

    def _decode_cv2(self, path):
        import cv2
        m = cv2.imread(path, cv2.IMREAD_COLOR)
        if m is None:
            return None
        return torch.from_numpy(m[:, :, ::-1].copy()).permute(2, 0, 1).contiguous().to(self.dev)

    def _resize_crop(self, x):
        h, w = int(x.shape[-2]), int(x.shape[-1])
        s = np.float32(self.resize_short)
        if w < h:
            rw, rh = int(self.resize_short), int(np.float32(h) * s / np.float32(w) + np.float32(0.5))
        else:
            rh, rw = int(self.resize_short), int(np.float32(w) * s / np.float32(h) + np.float32(0.5))
        if (rh, rw) != (h, w):
            x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
            x = x.round_().clamp_(0.0, 255.0)
        ch, cw = self.input_hw
        y0, x0 = (rh - ch) // 2, (rw - cw) // 2
        return x[..., y0:y0 + ch, x0:x0 + cw]

    def _letterbox(self, x):
        h, w = int(x.shape[-2]), int(x.shape[-1])
        H, W = self.input_hw
        r = min(H / h, W / w)
        nh, nw = int(math.floor(h * r + 0.5)), int(math.floor(w * r + 0.5))
        if (nh, nw) != (h, w):
            x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
            x = x.round_().clamp_(0.0, 255.0)
        out = torch.full((1, 3, H, W), self.pad, device=x.device, dtype=x.dtype)
        top, left = (H - nh) // 2, (W - nw) // 2
        out[..., top:top + nh, left:left + nw] = x
        return out

    def _prep(self, rgb_u8):
        if self.order == "BGR":
            rgb_u8 = rgb_u8[[2, 1, 0]]
        x = rgb_u8.unsqueeze(0).float()
        if self.letterbox:
            x = self._letterbox(x)
        elif self.resize_short:
            x = self._resize_crop(x)
        elif tuple(x.shape[-2:]) != self.input_hw:
            x = F.interpolate(x, size=self.input_hw, mode="bilinear", align_corners=False)
            x = x.round_().clamp_(0.0, 255.0)
        return (x * self.scale - self.mean) / self.std

    def _load_cpu(self, paths):
        out, valid, sizes = [], [], []
        for p, rgb in zip(paths, self._pool.map(_read_rgb_cv2, paths)):
            if rgb is None:
                self.n_failed += 1
                print(f"  [images] unreadable, excluded: {p}")
                out.append(torch.zeros((1, 3) + self.input_hw, device=self.dev))
                valid.append(False)
                sizes.append((0, 0))
                continue
            self.n_cpu_jpeg += 1
            sizes.append((rgb.shape[1], rgb.shape[0]))
            img = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(self.dev)
            out.append(self._prep(img))
            valid.append(True)
        return torch.cat(out, 0), valid, sizes

    def load(self, paths, with_sizes=False):
        if self.decode == "cpu":
            x, valid, sizes = self._load_cpu(paths)
            return (x, valid, sizes) if with_sizes else (x, valid)
        out, valid, sizes = [], [], []
        for p in paths:
            img = None
            if TV_OK:
                try:
                    if p.lower().endswith(".png"):
                        img = self._decode_png(p)
                        self.n_png += 1
                    else:
                        img = self._decode_jpeg(p)
                        self.n_gpu += 1
                except Exception:
                    img = None
            if img is None:
                img = self._decode_cv2(p)
                if img is not None:
                    self.n_cpu += 1
            if img is None:
                self.n_failed += 1
                print(f"  [images] unreadable, excluded: {p}")
                out.append(torch.zeros((1, 3) + self.input_hw, device=self.dev))
                valid.append(False)
                sizes.append((0, 0))
                continue
            sizes.append((int(img.shape[-1]), int(img.shape[-2])))
            out.append(self._prep(img))
            valid.append(True)
        x = torch.cat(out, 0)
        return (x, valid, sizes) if with_sizes else (x, valid)
