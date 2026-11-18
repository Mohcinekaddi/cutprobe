import json
import math
import os
import re

import numpy as np
import pandas as pd
import torch

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

DET_DEFAULTS = {"conf": 0.25, "eval_conf": 0.001, "iou": 0.6, "max_det": 300, "max_nms": 30000,
                "match_iou": 0.5, "agnostic_nms": False}


def _grid(h, w, device, dtype):
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype), indexing="ij")
    return xs, ys


def raw_to_decoded(maps, head):
    fmt = head["format"]
    if fmt == "yolov5":
        outs = []
        for t, s, anc in zip(maps, head["strides"], head["anchors"]):
            B, C, H, W = t.shape
            na = len(anc)
            no = C // na
            y = t.reshape(B, na, no, H, W).permute(0, 1, 3, 4, 2).sigmoid()
            gx, gy = _grid(H, W, t.device, t.dtype)
            g = torch.stack((gx, gy), -1).view(1, 1, H, W, 2)
            a = torch.tensor(anc, device=t.device, dtype=t.dtype).view(1, na, 1, 1, 2)
            xy = (y[..., 0:2] * 2.0 - 0.5 + g) * float(s)
            wh = (y[..., 2:4] * 2.0) ** 2 * a
            outs.append(torch.cat((xy, wh, y[..., 4:]), -1).reshape(B, na * H * W, no))
        return torch.cat(outs, 1)
    if fmt == "yolov8":
        R = int(head["reg_max"])
        box, cls, anc, strd = [], [], [], []
        pairs = list(zip(maps[0::2], maps[1::2])) if head.get("split") else [(t, t) for t in maps]
        for (tb, tc), s in zip(pairs, head["strides"]):
            B, _, H, W = tb.shape
            fb = tb.reshape(B, -1, H * W)
            fc = tc.reshape(B, -1, H * W)
            box.append(fb[:, :4 * R])
            cls.append(fc if head.get("split") else fc[:, 4 * R:])
            t = tb
            gx, gy = _grid(H, W, t.device, t.dtype)
            anc.append(torch.stack((gx.reshape(-1) + 0.5, gy.reshape(-1) + 0.5), 0))
            strd.append(torch.full((H * W,), float(s), device=t.device, dtype=t.dtype))
        box = torch.cat(box, 2)
        B, _, A = box.shape
        if R > 1:
            bins = torch.arange(R, device=box.device, dtype=box.dtype).view(1, 1, R, 1)
            dist = (box.reshape(B, 4, R, A).softmax(2) * bins).sum(2)
        else:
            dist = box
        anc = torch.cat(anc, 1).unsqueeze(0)
        lt, rb = dist[:, :2], dist[:, 2:]
        x1y1, x2y2 = anc - lt, anc + rb
        if head.get("box_out") == "xyxy":
            dbox = torch.cat((x1y1, x2y2), 1)
        else:
            dbox = torch.cat(((x1y1 + x2y2) / 2.0, x2y2 - x1y1), 1)
        dbox = dbox * torch.cat(strd).view(1, 1, A)
        return torch.cat((dbox, torch.cat(cls, 2).sigmoid()), 1)
    raise ValueError(f"no raw decoding for head {fmt!r}")


def _xyxy(b, fmt):
    if fmt == "cxcywh":
        xy, wh = b[..., :2], b[..., 2:4] / 2.0
        return torch.cat((xy - wh, xy + wh), -1)
    if fmt == "xywh":
        return torch.cat((b[..., :2], b[..., :2] + b[..., 2:4]), -1)
    return b[..., :4]


def _scale(b, normalized, input_hw):
    if not normalized:
        return b
    H, W = input_hw
    return b * torch.tensor([W, H, W, H], device=b.device, dtype=b.dtype)


def _from_yolo(y, layout, objectness, box_format, normalized, input_hw):
    if layout == "anchors_last":
        y = y.transpose(1, 2)
    if objectness:
        conf, lab = y[..., 5:].max(-1)
        score = conf * y[..., 4]
    else:
        score, lab = y[..., 4:].max(-1)
    return _scale(_xyxy(y[..., :4], box_format), normalized, input_hw), score, lab


def e2e_select(boxes, scores, k):
    B, A, nc = scores.shape
    k1 = min(int(k), A)
    idx = scores.amax(-1).topk(k1, dim=-1).indices
    bx = boxes.gather(1, idx.unsqueeze(-1).expand(-1, -1, 4))
    sc = scores.gather(1, idx.unsqueeze(-1).expand(-1, -1, nc))
    s2, i2 = sc.flatten(1).topk(min(int(k), k1 * nc), dim=-1)
    return bx.gather(1, (i2 // nc).unsqueeze(-1).expand(-1, -1, 4)), s2, i2 % nc


def e2e_rows(maps, head):
    y = raw_to_decoded(maps, head).transpose(1, 2)
    b, s, l = e2e_select(_xyxy(y[..., :4], head.get("box_out", "cxcywh")), y[..., 4:], head["topk"])
    return torch.cat((b, s.unsqueeze(-1), l.unsqueeze(-1).to(b.dtype)), -1)


def candidates(outs, head, input_hw):
    fmt = head["format"]
    names = head["outputs"]
    if fmt in ("yolov5", "yolov8") and head.get("end2end"):
        r = e2e_rows([outs[t] for t in names], head)
        return r[..., :4], r[..., 4], r[..., 5].long(), False
    if fmt in ("yolov5", "yolov8"):
        y = raw_to_decoded([outs[t] for t in names], head)
        b, s, l = _from_yolo(y, "anchors_first" if fmt == "yolov5" else "anchors_last",
                             fmt == "yolov5", head.get("box_out", "cxcywh"), False, input_hw)
        return b, s, l, True
    if fmt == "yolo_decoded":
        b, s, l = _from_yolo(outs[names[0]], head["layout"], head["objectness"],
                             head.get("box_format", "cxcywh"), head.get("normalized"), input_hw)
        return b, s, l, True
    if fmt == "boxes_scores":
        boxes = outs[names[0]]
        boxes = boxes.reshape(boxes.shape[0], -1, 4)
        sc = outs[names[1]]
        if head.get("layout") == "anchors_last":
            sc = sc.transpose(1, 2)
        bg = head.get("background")
        if bg is not None:
            sc = sc.clone()
            sc[..., int(bg)] = -1.0
        s, l = sc.max(-1)
        return _scale(_xyxy(boxes, head.get("box_format", "xyxy")), head.get("normalized"), input_hw), s, l, True
    if fmt == "end2end":
        if head.get("layout") == "split":
            boxes = outs[names[0]].reshape(outs[names[0]].shape[0], -1, 4)
            s = outs[names[1]].reshape(boxes.shape[0], -1)
            l = outs[names[2]].reshape(boxes.shape[0], -1).long() if names[2] else torch.zeros_like(s).long()
        else:
            y = outs[names[0]]
            boxes, s, l = y[..., :4], y[..., 4], y[..., 5].long()
        return _scale(_xyxy(boxes, head.get("box_format", "xyxy")), head.get("normalized"), input_hw), s, l, False
    raise ValueError(f"unknown detection head {fmt!r}")


def _nms(boxes, scores, iou):
    try:
        from torchvision.ops import nms
        return nms(boxes, scores, iou)
    except Exception:
        order = scores.argsort(descending=True)
        keep = []
        while order.numel():
            i = order[0]
            keep.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            ov = box_iou_t(boxes[i:i + 1], boxes[rest])[0]
            order = rest[ov <= iou]
        return torch.stack(keep) if keep else order


def box_iou_t(a, b):
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp_min(0).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp_min(1e-12)


def select(boxes, scores, labels, conf, iou, max_det, nms=True, agnostic=False, max_nms=30000):
    out = []
    for b in range(boxes.shape[0]):
        m = scores[b] > conf
        bx, sc, lb = boxes[b][m], scores[b][m], labels[b][m]
        if sc.numel() > max_nms:
            top = sc.topk(max_nms).indices
            bx, sc, lb = bx[top], sc[top], lb[top]
        if nms and sc.numel():
            span = (bx.max() - bx.min()).clamp_min(0.0) + 1.0
            off = 0 if agnostic else lb.to(bx.dtype).unsqueeze(1) * span
            keep = _nms(bx - bx.min() + off, sc, iou)[:max_det]
        else:
            keep = sc.argsort(descending=True)[:max_det]
        out.append((bx[keep], sc[keep], lb[keep]))
    return out


class Decoder:
    def __init__(self, head, input_hw, det):
        self.head = head
        self.input_hw = tuple(input_hw)
        self.p = dict(DET_DEFAULTS, **{k: v for k, v in (det or {}).items() if k in DET_DEFAULTS and v is not None})

    def __call__(self, outs, conf=None):
        b, s, l, nms = candidates(outs, self.head, self.input_hw)
        p = self.p
        return select(b.float(), s.float(), l, p["eval_conf"] if conf is None else conf, p["iou"],
                      int(p["max_det"]), nms=nms, agnostic=bool(p["agnostic_nms"]),
                      max_nms=int(p["max_nms"]))


def geometry(w0, h0, pp, input_hw):
    H, W = int(input_hw[0]), int(input_hw[1])
    w0, h0 = int(w0), int(h0)
    if w0 <= 0 or h0 <= 0:
        return 1.0, 1.0, 0.0, 0.0
    if pp.get("letterbox"):
        r = min(H / h0, W / w0)
        nh, nw = int(math.floor(h0 * r + 0.5)), int(math.floor(w0 * r + 0.5))
        return nw / w0, nh / h0, float((W - nw) // 2), float((H - nh) // 2)
    if pp.get("resize_short"):
        s = np.float32(pp["resize_short"])
        if w0 < h0:
            rw, rh = int(s), int(np.float32(h0) * s / np.float32(w0) + np.float32(0.5))
        else:
            rh, rw = int(s), int(np.float32(w0) * s / np.float32(h0) + np.float32(0.5))
        return rw / w0, rh / h0, -float((rw - W) // 2), -float((rh - H) // 2)
    return W / w0, H / h0, 0.0, 0.0


def to_original(boxes, geo, size):
    sx, sy, ox, oy = geo
    w0, h0 = size
    if torch.is_tensor(boxes):
        o = torch.tensor([ox, oy, ox, oy], device=boxes.device, dtype=boxes.dtype)
        s = torch.tensor([sx, sy, sx, sy], device=boxes.device, dtype=boxes.dtype)
        lim = torch.tensor([w0, h0, w0, h0], device=boxes.device, dtype=boxes.dtype)
        return torch.minimum(((boxes - o) / s).clamp_min(0.0), lim)
    b = (np.asarray(boxes, dtype=np.float64) - [ox, oy, ox, oy]) / [sx, sy, sx, sy]
    return np.minimum(np.maximum(b, 0.0), [w0, h0, w0, h0]).astype(np.float32)


_EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64))


class DetStore:

    def __init__(self, n):
        self.n = n
        self.dets = [None] * n

    def put(self, idx, dets, geos, sizes):
        for i, (b, s, l), geo, size in zip(idx, dets, geos, sizes):
            self.dets[int(i)] = (to_original(b, geo, size).cpu().numpy().astype(np.float32),
                                 s.cpu().numpy().astype(np.float32), l.cpu().numpy().astype(np.int64))

    def set(self, i, boxes, scores, labels):
        self.dets[int(i)] = (np.asarray(boxes, np.float32).reshape(-1, 4),
                             np.asarray(scores, np.float32).reshape(-1),
                             np.asarray(labels, np.int64).reshape(-1))

    def get(self, i):
        d = self.dets[int(i)]
        return _EMPTY if d is None else d

    def at(self, i, conf):
        b, s, l = self.get(i)
        m = s >= conf
        return b[m], s[m], l[m]

    def to_frame(self, conf=0.0, valid=None, names=None):
        rows = []
        for i in range(self.n):
            if valid is not None and not valid[i]:
                continue
            b, s, l = self.at(i, conf)
            for k in range(len(s)):
                rows.append((i, *[round(float(v), 2) for v in b[k]], round(float(s[k]), 5), int(l[k])))
        df = pd.DataFrame(rows, columns=["index", "x1", "y1", "x2", "y2", "score", "class"])
        if names:
            df["classname"] = [names[c] if 0 <= c < len(names) else str(c) for c in df["class"]]
        return df


def store_from_csv(path, n, geo_of=None):
    st = DetStore(n)
    df = pd.read_csv(path)
    seen = set(df["index"].astype(int).tolist())
    df = df[df["class"] >= 0]
    for i, g in df.groupby("index"):
        i = int(i)
        if i >= n:
            continue
        b = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)
        if geo_of is not None:
            geo, size = geo_of(i)
            b = to_original(b, geo, size)
        st.set(i, b, g["score"].to_numpy(np.float32), g["class"].to_numpy(np.int64))
    for i in seen:
        if i < n and st.dets[i] is None:
            st.set(i, _EMPTY[0], _EMPTY[1], _EMPTY[2])
    processed = np.zeros(n, dtype=bool)
    processed[[i for i in seen if i < n]] = True
    return st, processed


class DetDataset:
    def __init__(self, paths, sizes=None, gts=None, ids=None, source="", class_map_note=""):
        self.paths = list(paths)
        self.sizes = sizes
        self.gts = gts
        self.ids = ids
        self.source = source
        self.class_map_note = class_map_note

    def __len__(self):
        return len(self.paths)

    @property
    def has_gt(self):
        return self.gts is not None

    def take(self, index):
        index = list(index)
        return DetDataset([self.paths[i] for i in index],
                          None if self.sizes is None else self.sizes[index],
                          None if self.gts is None else [self.gts[i] for i in index],
                          None if self.ids is None else [self.ids[i] for i in index],
                          self.source, self.class_map_note)

    def n_objects(self):
        return 0 if self.gts is None else int(sum((~g[2]).sum() for g in self.gts))

    def gt_frame(self):
        rows = []
        for i, (b, l, c) in enumerate(self.gts or []):
            for k in range(len(l)):
                rows.append((i, *[round(float(v), 2) for v in b[k]], int(l[k]), int(bool(c[k]))))
        return pd.DataFrame(rows, columns=["index", "x1", "y1", "x2", "y2", "class", "iscrowd"])


def split_dataset(ds, limit, n_buf):
    n = min(int(limit), len(ds) - int(n_buf))
    if n <= 0:
        raise SystemExit(f"{ds.source}: not enough images ({len(ds)}) for {n_buf} history images")
    pool = np.arange(n, len(ds))
    buf = np.random.RandomState(42).choice(pool, size=min(int(n_buf), len(pool)), replace=False) \
        if n_buf else np.zeros(0, np.int64)
    return ds.take(range(n)), [ds.paths[i] for i in buf], n < min(int(limit), len(ds))


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def coco_class_map(cats, names, nc, mode="auto"):
    ids = sorted(int(c["id"]) for c in cats)
    if mode in ("auto", "names") and names:
        by = {_norm(n): i for i, n in enumerate(names)}
        m = {int(c["id"]): by.get(_norm(c["name"])) for c in cats}
        hit = sum(v is not None for v in m.values())
        if mode == "names" or hit >= 0.9 * len(cats):
            return m, f"categories matched to model classes by name ({hit}/{len(cats)})"
    if mode in ("auto", "contiguous") and (mode == "contiguous" or nc == len(ids)):
        return {cid: i for i, cid in enumerate(ids)}, f"category ids -> 0..{len(ids) - 1} in id order"
    if mode in ("auto", "identity") and (mode == "identity" or (nc and nc > max(ids))):
        return {cid: cid for cid in ids}, "category id = model class index"
    raise SystemExit(f"cannot map the {len(ids)} annotation categories to the model's {nc} classes: "
                     f"set detection.label_map (names | contiguous | identity)")


def _image_size(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        import cv2
        m = cv2.imread(path)
        return (m.shape[1], m.shape[0]) if m is not None else (0, 0)


def _list_images(at):
    if at.get("csv"):
        df = pd.read_csv(at["csv"])
        if "filepath" not in df.columns:
            raise SystemExit(f"{at['csv']}: needs a 'filepath' column")
        base = at.get("dataset_dir")
        return [p if os.path.isabs(p) or not base else os.path.join(base, p) for p in df["filepath"]]
    d = at.get("dataset_dir")
    if d and os.path.isdir(d):
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
    raise SystemExit("detection needs images: set attack.annotations (COCO json / YOLO labels), "
                     "attack.csv (filepath column) or attack.dataset_dir")


def load_dataset(at, task, det):
    ann = at.get("annotations")
    names, nc = task.names, task.num_classes
    if ann and os.path.isfile(ann) and ann.lower().endswith(".json"):
        with open(ann) as fh:
            coco = json.load(fh)
        cmap, note = coco_class_map(coco.get("categories", []), names, nc, det.get("label_map") or "auto")
        images = sorted(coco["images"], key=lambda im: im["id"])
        base = at.get("dataset_dir") or os.path.dirname(ann)
        by_img = {}
        dropped = 0
        for a in coco.get("annotations", []):
            c = cmap.get(int(a["category_id"]))
            if c is None:
                dropped += 1
                continue
            by_img.setdefault(a["image_id"], []).append((a["bbox"], c, int(a.get("iscrowd", 0))))
        paths, sizes, gts, ids = [], [], [], []
        for im in images:
            paths.append(os.path.join(base, im["file_name"]))
            sizes.append((int(im.get("width", 0)), int(im.get("height", 0))))
            objs = by_img.get(im["id"], [])
            b = np.array([[x, y, x + w, y + h] for (x, y, w, h), _, _ in objs], np.float32).reshape(-1, 4)
            gts.append((b, np.array([c for _, c, _ in objs], np.int64), np.array([bool(z) for _, _, z in objs], bool)))
            ids.append(int(im["id"]))
        ds = DetDataset(paths, np.array(sizes, np.int64), gts, ids, f"COCO {os.path.basename(ann)}",
                        note + (f"; {dropped} object(s) of unmapped categories dropped" if dropped else ""))
        if at.get("csv"):
            keep = {os.path.basename(p): i for i, p in enumerate(ds.paths)}
            order = [keep[os.path.basename(p)] for p in _list_images(at) if os.path.basename(p) in keep]
            if not order:
                raise SystemExit(f"no image of {at['csv']} is in {ann}")
            ds = ds.take(order)
        return ds
    paths = _list_images(at)
    sizes = np.array([_image_size(p) for p in paths], np.int64)
    if ann and os.path.isdir(ann):
        gts = []
        for p, (w, h) in zip(paths, sizes):
            lab = os.path.join(ann, os.path.splitext(os.path.basename(p))[0] + ".txt")
            rows = np.loadtxt(lab, ndmin=2) if os.path.exists(lab) and os.path.getsize(lab) else np.zeros((0, 5))
            rows = rows[:, :5].reshape(-1, 5)
            cx, cy, bw, bh = rows[:, 1] * w, rows[:, 2] * h, rows[:, 3] * w, rows[:, 4] * h
            b = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32)
            gts.append((b.reshape(-1, 4), rows[:, 0].astype(np.int64), np.zeros(len(rows), bool)))
        return DetDataset(paths, sizes, gts, None, f"YOLO labels {ann}", "label = model class index")
    if ann:
        raise SystemExit(f"attack.annotations {ann}: expected a COCO .json file or a YOLO labels folder")
    return DetDataset(paths, sizes, None, None, "images only (no ground truth)")


def _iou_np(a, b, crowd=None):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    iw = (np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0])).clip(0)
    ih = (np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])).clip(0)
    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    if crowd is not None and crowd.any():
        union = np.where(crowd[None, :], area_a[:, None], union)
    return np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)


def match(det_boxes, gt_boxes, gt_crowd, thrs, iou=None):
    T, D, G = len(thrs), len(det_boxes), len(gt_boxes)
    tp = np.zeros((T, D), bool)
    ign = np.zeros((T, D), bool)
    gi = np.full((T, D), -1, np.int64)
    if D == 0 or G == 0:
        return tp, ign, gi
    order = np.argsort(gt_crowd, kind="mergesort")
    crowd = gt_crowd[order]
    iou = _iou_np(det_boxes, gt_boxes[order], crowd) if iou is None else iou[:, order]
    thr = np.minimum(np.asarray(thrs, np.float64), 1 - 1e-10)
    regular = ~crowd
    matched = np.zeros((T, G), bool)
    rows = np.nonzero(iou.max(1) >= thr.min())[0]
    ar = np.arange(T)
    for d in rows:
        row = iou[d]
        ok = regular[None, :] & ~matched & (row[None, :] >= thr[:, None])
        has = ok.any(1)
        if has.any():
            vals = np.where(ok, row[None, :], -1.0)
            j = G - 1 - vals[:, ::-1].argmax(1)
            matched[ar[has], j[has]] = True
            tp[has, d] = True
            gi[has, d] = order[j[has]]
        if crowd.any() and not has.all():
            cmax = np.where(crowd, row, -1.0)
            jc = G - 1 - cmax[::-1].argmax()
            ig = ~has & (cmax[jc] >= thr)
            ign[ig, d] = True
    return tp, ign, gi


def coco_map(store, gts, valid, max_dets=100):
    T = len(IOU_THRS)
    acc, npig = {}, {}
    for i in np.nonzero(valid)[0]:
        gb, gl, gc = gts[i]
        db, ds, dl = store.get(i)
        iou = _iou_np(db, gb, gc) if len(db) and len(gb) else None
        for c in np.unique(np.concatenate((gl, dl))):
            gm = np.nonzero(gl == c)[0]
            npig[c] = npig.get(c, 0) + int((~gc[gm]).sum())
            dm = np.nonzero(dl == c)[0]
            if not len(dm):
                continue
            dm = dm[np.argsort(-ds[dm], kind="mergesort")[:max_dets]]
            if len(gm):
                tp, ign, _ = match(db[dm], gb[gm], gc[gm], IOU_THRS, iou[np.ix_(dm, gm)])
            else:
                tp = ign = np.zeros((T, len(dm)), bool)
            acc.setdefault(c, []).append((ds[dm], tp, ign))
    ap = {}
    for c, n in npig.items():
        if n == 0:
            continue
        parts = acc.get(c)
        if not parts:
            ap[c] = np.zeros(T)
            continue
        s = np.concatenate([p[0] for p in parts])
        tp = np.concatenate([p[1] for p in parts], 1)
        ig = np.concatenate([p[2] for p in parts], 1)
        o = np.argsort(-s, kind="mergesort")
        tp, ig = tp[:, o], ig[:, o]
        tps = np.cumsum(tp & ~ig, 1).astype(np.float64)
        fps = np.cumsum(~tp & ~ig, 1).astype(np.float64)
        res = np.zeros(T)
        for t in range(T):
            rc = tps[t] / n
            pr = tps[t] / (fps[t] + tps[t] + np.spacing(1))
            pr = np.maximum.accumulate(pr[::-1])[::-1] if pr.size else pr
            k = np.searchsorted(rc, REC_THRS, side="left")
            q = np.zeros(len(REC_THRS))
            ok = k < len(pr)
            q[ok] = pr[k[ok]]
            if not ok.all():
                q[np.argmin(ok):] = 0.0
            res[t] = q.mean()
        ap[c] = res
    if not ap:
        return {"map": float("nan"), "map50": float("nan"), "map75": float("nan"), "per_class": {}}
    A = np.stack(list(ap.values()))
    return {"map": float(A.mean()), "map50": float(A[:, 0].mean()), "map75": float(A[:, 5].mean()),
            "per_class": {int(c): (float(v.mean()), float(v[0])) for c, v in ap.items()}}


def _hits(gt, dets, iou_thr):
    gb, gl, gc = gt
    db, ds, dl = dets
    hit = np.zeros(len(gl), bool)
    if not len(dl):
        return hit[~gc], 0
    fp = 0
    iou = _iou_np(db, gb, gc) if len(gb) else None
    for c in np.unique(dl):
        dm = np.nonzero(dl == c)[0]
        gm = np.nonzero(gl == c)[0]
        if not len(gm):
            fp += len(dm)
            continue
        dm = dm[np.argsort(-ds[dm], kind="mergesort")]
        tp, ign, gi = match(db[dm], gb[gm], gc[gm], [iou_thr], iou[np.ix_(dm, gm)])
        fp += int((~tp[0] & ~ign[0]).sum())
        hit[gm[gi[0][tp[0]]]] = True
    return hit[~gc], fp


def hits(store, gts, i, conf, iou_thr):
    cache = store.__dict__.setdefault("_hit_cache", {})
    key = (int(i), float(conf), float(iou_thr), id(gts))
    if key not in cache:
        cache[key] = _hits(gts[i], store.at(i, conf), iou_thr)
    return cache[key]


def _ref_match(ref, atk, iou_thr):
    rb, rs, rl = ref
    ab, as_, al = atk
    nr, na = len(rs), len(as_)
    if nr == 0 or na == 0:
        return nr, na, 0, []
    ok = np.zeros(nr, bool)
    used = np.zeros(na, bool)
    ious = []
    full = _iou_np(ab, rb)
    for c in np.intersect1d(rl, al):
        ri = np.nonzero(rl == c)[0]
        ai = np.nonzero(al == c)[0]
        ai = ai[np.argsort(-as_[ai], kind="mergesort")]
        sub = full[np.ix_(ai, ri)]
        tp, _, gi = match(ab[ai], rb[ri], np.zeros(len(ri), bool), [iou_thr], sub)
        sel = tp[0]
        ok[ri[gi[0][sel]]] = True
        used[ai[sel]] = True
        if sel.any():
            ious += sub[np.nonzero(sel)[0], gi[0][sel]].tolist()
    left_r, left_a = np.nonzero(~ok)[0], np.nonzero(~used)[0]
    relab = 0
    if len(left_r) and len(left_a):
        relab = int((full[np.ix_(left_a, left_r)].max(0) >= iou_thr).sum())
    return int((~ok).sum()), int((~used).sum()), relab, ious


def paired_metrics(gts, clean, atk, valid, conf, iou_thr=0.5):
    idx = np.nonzero(valid)[0]
    n = len(idx)
    hh = hm = mh = mm = 0
    fp_c = fp_a = 0
    img_any = img_base = 0
    nref = van = gho = rel = changed = 0
    ious = []
    per_image = []
    for i in idx:
        c = clean.at(i, conf)
        a = atk.at(i, conf)
        v, g, r, iu = _ref_match(c, a, iou_thr)
        nref += len(c[1])
        van += v
        gho += g
        rel += r
        ious += iu
        changed += int(v > 0 or g > 0)
        if gts is not None:
            h_c, f_c = hits(clean, gts, i, conf, iou_thr)
            h_a, f_a = hits(atk, gts, i, conf, iou_thr)
            fp_c += f_c
            fp_a += f_a
            hh += int((h_c & h_a).sum())
            hm += int((h_c & ~h_a).sum())
            mh += int((~h_c & h_a).sum())
            mm += int((~h_c & ~h_a).sum())
            if h_c.any():
                img_base += 1
                img_any += int((h_c & ~h_a).any())
            per_image.append((int(i), len(h_c), int(h_c.sum()), int(h_a.sum()), int((h_c & ~h_a).sum()),
                              int((~h_c & h_a).sum()), len(c[1]), len(a[1]), v, g, r))
        else:
            per_image.append((int(i), 0, 0, 0, 0, 0, len(c[1]), len(a[1]), v, g, r))
    pct = lambda a, b: round(100.0 * a / b, 2) if b else float("nan")
    out = {"n_paired_images": n,
           "ref_detections": nref, "vanished_pct": pct(van, nref), "ghost_pct": pct(gho, nref),
           "relabeled_pct": pct(rel, nref), "image_changed_pct": pct(changed, n),
           "mean_iou_matched": round(float(np.mean(ious)), 4) if ious else float("nan")}
    if gts is not None:
        n_obj = hh + hm + mh + mm
        tp_c, tp_a = hh + hm, hh + mh
        out.update({
            "n_gt_objects": n_obj,
            "base_hit_atk_hit": hh, "base_hit_atk_miss": hm, "base_miss_atk_hit": mh,
            "base_miss_atk_miss": mm,
            "attack_success_rate_pct": pct(hm, hh + hm),
            "accidental_correction_rate_pct": pct(mh, mh + mm),
            "net_effect_pct": pct(hm - mh, n_obj),
            "recall_clean_pct": pct(tp_c, n_obj), "recall_attack_pct": pct(tp_a, n_obj),
            "precision_clean_pct": pct(tp_c, tp_c + fp_c), "precision_attack_pct": pct(tp_a, tp_a + fp_a),
            "fp_clean": fp_c, "fp_attack": fp_a,
            "image_attack_success_rate_pct": pct(img_any, img_base),
            "asr_reference": "ground_truth"})
    else:
        out.update({"attack_success_rate_pct": out["vanished_pct"], "asr_reference": "clean_detections"})
    cols = ["index", "n_gt", "hit_clean", "hit_attack", "lost", "gained", "n_det_clean",
            "n_det_attack", "vanished", "ghost", "relabeled"]
    return out, pd.DataFrame(per_image, columns=cols)


def summary_metrics(gts, store, valid, conf, iou_thr=0.5):
    out = {}
    if gts is None:
        return out
    m = coco_map(store, gts, valid)
    out["map50_95"] = round(100.0 * m["map"], 2)
    out["map50"] = round(100.0 * m["map50"], 2)
    tp = n = fp = 0
    for i in np.nonzero(valid)[0]:
        h, f = hits(store, gts, i, conf, iou_thr)
        tp += int(h.sum())
        n += len(h)
        fp += f
    out["recall_pct"] = round(100.0 * tp / n, 2) if n else float("nan")
    out["precision_pct"] = round(100.0 * tp / (tp + fp), 2) if tp + fp else float("nan")
    out["per_class"] = m["per_class"]
    return out


def lost_per_class(gts, clean, atk, valid, conf, iou_thr=0.5):
    lost, base = {}, {}
    for i in np.nonzero(valid)[0]:
        gb, gl, gc = gts[i]
        h_c, _ = hits(clean, gts, i, conf, iou_thr)
        h_a, _ = hits(atk, gts, i, conf, iou_thr)
        lab = gl[~gc]
        for c, a, b in zip(lab, h_c, h_a):
            if a:
                base[int(c)] = base.get(int(c), 0) + 1
                if not b:
                    lost[int(c)] = lost.get(int(c), 0) + 1
    return lost, base
