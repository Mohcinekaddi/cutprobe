import itertools
import math

import numpy as np
import torch

STATS_F64_BUDGET = 8_000_000

SPECIFIC = {
    "mfgsm": {"eps_rel": 1.0, "eps_mode": "fixed", "eps_max": 3.0, "dc_correct": False,
              "match_mode": "affine", "clip": "none", "dom_quantile": None,
              "metric": "cosine", "topk": 1},
    "mgna": {"gamma": 1.0, "sigma_rel": 0.5, "seed": 1234},
    "mjsma": {"theta": 0.01, "scope": "image"},
}


def _num(x):
    return f"{float(x):g}".replace(".", "p").replace("-", "m")


class AttackConfig:
    def __init__(self, attack, **params):
        if attack not in SPECIFIC:
            raise ValueError(f"unknown attack {attack!r} (expected {list(SPECIFIC)})")
        self.attack = attack
        self.params = dict(SPECIFIC[attack])
        for k, v in params.items():
            if k not in self.params:
                raise ValueError(f"unknown parameter {k!r} for {attack} "
                                 f"(allowed: {sorted(self.params)})")
            self.params[k] = v
        self._validate()

    def _validate(self):
        p = self.params
        if self.attack == "mfgsm":
            if p["eps_mode"] not in ("fixed", "match_variance"):
                raise ValueError(f"mfgsm eps_mode {p['eps_mode']!r}")
            if p["match_mode"] not in ("affine", "power"):
                raise ValueError(f"mfgsm match_mode {p['match_mode']!r}")
            if p["match_mode"] == "power" and (p["eps_mode"] != "match_variance"
                                               or not p["dc_correct"]):
                raise ValueError("mfgsm match_mode='power' needs eps_mode='match_variance' "
                                 "and dc_correct=true")
            if p["clip"] not in ("none", "domain", "image"):
                raise ValueError(f"mfgsm clip {p['clip']!r}")
            if p["metric"] not in ("cosine", "l2", "sum"):
                raise ValueError(f"mfgsm metric {p['metric']!r}")
            if int(p["topk"]) < 1:
                raise ValueError("mfgsm topk must be >= 1")
        elif self.attack == "mgna":
            if not 0.0 <= float(p["gamma"]) <= 1.0 or float(p["sigma_rel"]) < 0.0:
                raise ValueError("mgna needs 0 <= gamma <= 1 and sigma_rel >= 0")
        else:
            if not 0.0 <= float(p["theta"]) <= 1.0:
                raise ValueError("mjsma needs 0 <= theta <= 1")
            if p["scope"] not in ("image", "channel"):
                raise ValueError(f"mjsma scope {p['scope']!r}")

    @property
    def needs_history(self):
        return self.attack == "mfgsm"

    def tag(self):
        p = self.params
        if self.attack == "mfgsm":
            amp = "solved" if p["eps_mode"] == "match_variance" else f"eps_{_num(p['eps_rel'])}"
            extra = "" if p["clip"] == "none" else f"_clip{p['clip']}"
            return f"mfgsm_{amp}{extra}{'_pow' if p['match_mode'] == 'power' else ''}"
        if self.attack == "mgna":
            return f"mgna_g{_num(p['gamma'])}_s{_num(p['sigma_rel'])}"
        return f"mjsma_t{_num(p['theta'])}{'_ch' if p['scope'] == 'channel' else ''}"

    def label(self):
        p = self.params
        if self.attack == "mfgsm":
            amp = "eps*" if p["eps_mode"] == "match_variance" else f"eps={float(p['eps_rel']):g}"
            return f"mFGSM {amp}{' [power]' if p['match_mode'] == 'power' else ''}"
        if self.attack == "mgna":
            return f"mGNA g={float(p['gamma']):g} sigma={float(p['sigma_rel']):g}"
        return f"mJSMA theta={float(p['theta']):g}" + (" [channel]" if p["scope"] == "channel" else "")


def build_configs(entries):
    configs, seen = [], set()
    for entry in entries:
        entry = dict(entry)
        attack = entry.pop("attack")
        keys = [k for k, v in entry.items() if isinstance(v, (list, tuple))]
        combos = itertools.product(*(entry[k] for k in keys)) if keys else [()]
        for combo in combos:
            params = dict(entry)
            params.update(zip(keys, combo))
            cfg = AttackConfig(attack, **params)
            tag = cfg.tag()
            if tag in seen:
                raise ValueError(f"duplicate attack configuration {tag!r}")
            seen.add(tag)
            configs.append({"attack": attack, "params": cfg.params, "cfg": cfg, "tag": tag,
                            "label": cfg.label()})
    return configs


def as_bchw(t):
    d = t.dim()
    if d == 4:
        return t, None
    orig = tuple(t.shape)
    if d == 2:
        return t.reshape(orig[0], 1, 1, orig[1]), orig
    if d == 3:
        return t.reshape(orig[0], orig[1], 1, orig[2]), orig
    raise ValueError(f"unsupported rank {d} for a cut tensor {orig}")


class History:
    QUANTILE_SAMPLE = 4_000_000

    def __init__(self, hist, dom_quantile=None):
        self.hist = as_bchw(hist)[0].contiguous()
        n = self.hist.shape[0]
        self.flat = self.hist.reshape(n, -1)
        self.unit = self.flat / self.flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.sums = self.flat.sum(dim=1)
        flat = self.flat.reshape(-1)
        self.dom_lo = float(flat.min())
        if dom_quantile is None:
            self.dom_hi = float(flat.max())
        else:
            stride = max(1, int(math.ceil(flat.numel() / self.QUANTILE_SAMPLE)))
            self.dom_hi = float(torch.quantile(flat[::stride].float(), 1.0 - float(dom_quantile)))


def _zeros(B, device, dtype=torch.float64):
    return torch.zeros(B, dtype=dtype, device=device)


def _apply_clip(fm, mfm, ctx, clip):
    B = mfm.shape[0]
    n = mfm[0].numel()
    if clip == "none":
        return mfm, _zeros(B, mfm.device)
    if clip == "domain":
        if ctx is None:
            raise ValueError("clip='domain' needs the history buffer")
        lo = torch.tensor(ctx.dom_lo, dtype=mfm.dtype, device=mfm.device)
        hi = torch.tensor(ctx.dom_hi, dtype=mfm.dtype, device=mfm.device)
    else:
        f = fm.reshape(B, -1)
        lo = f.min(dim=1).values.view(B, 1, 1, 1)
        hi = f.max(dim=1).values.view(B, 1, 1, 1)
    mask = (mfm < lo) | (mfm > hi)
    frac = mask.reshape(B, -1).sum(dim=1).to(torch.float64) / n
    del mask
    return torch.minimum(torch.maximum(mfm, lo), hi), frac


def _solve_eps_match_variance(f, d, budget=STATS_F64_BUDGET):
    B, n = f.shape
    step = max(1, int(budget // max(n, 1)))
    eps = torch.empty(B, dtype=torch.float64, device=f.device)
    for s in range(0, B, step):
        fs = f[s:s + step].double()
        ds = d[s:s + step].double()
        mf, md = fs.mean(dim=1), ds.mean(dim=1)
        cov = (fs * ds).mean(dim=1) - mf * md
        vd = ds.abs().mean(dim=1) - md * md
        eps[s:s + step] = 2.0 * cov / vd.clamp_min(1e-12)
    return eps.to(f.dtype)


def _moment_match_power(fm, mfm, p_bounds=(1.0 / 64.0, 64.0), tol=1e-12, max_iter=60,
                        budget=STATS_F64_BUDGET):
    B = fm.shape[0]
    f = fm.reshape(B, -1)
    a = mfm.reshape(B, -1)
    step = max(1, int(budget // max(f.shape[1], 1)))
    dev = fm.device
    out = mfm.clone()
    o = out.reshape(B, -1)
    p_all = torch.full((B,), float("nan"), dtype=torch.float64, device=dev)
    ok_all = torch.zeros(B, dtype=torch.bool, device=dev)
    u_lo, u_hi = math.log(p_bounds[0]), math.log(p_bounds[1])
    tiny = 1e-300

    for s0 in range(0, B, step):
        fs = f[s0:s0 + step].double()
        gs = a[s0:s0 + step].double().clamp_min(0.0)
        b = fs.shape[0]
        m1 = fs.mean(dim=1)
        target = torch.log((fs * fs).mean(dim=1).clamp_min(tiny) / (m1 * m1).clamp_min(tiny))
        pos = gs > 0
        ok = (fs.amin(dim=1) >= 0.0) & (m1 > 0.0) & (pos.sum(dim=1) >= 2)
        del fs
        gmax = gs.amax(dim=1, keepdim=True).clamp_min(tiny)
        logg = torch.where(pos, torch.log(gs.clamp_min(tiny) / gmax), torch.zeros_like(gs))
        del gs

        def h(u):
            p = u.exp().unsqueeze(1)
            gp = torch.where(pos, torch.exp(p * logg), torch.zeros_like(logg))
            a1 = gp.mean(dim=1)
            b1 = (gp * logg).mean(dim=1)
            gp.mul_(gp)
            a2 = gp.mean(dim=1)
            b2 = (gp * logg).mean(dim=1)
            a1c, a2c = a1.clamp_min(tiny), a2.clamp_min(tiny)
            return (torch.log(a2c) - 2.0 * torch.log(a1c) - target,
                    p.squeeze(1) * (2.0 * b2 / a2c - 2.0 * b1 / a1c))

        lo = torch.full((b,), u_lo, dtype=torch.float64, device=dev)
        hi = torch.full((b,), u_hi, dtype=torch.float64, device=dev)
        v_lo, _ = h(lo)
        v_hi, _ = h(hi)
        ok &= (v_lo <= 0.0) & (v_hi >= 0.0)

        u = torch.zeros(b, dtype=torch.float64, device=dev)
        val = torch.full((b,), float("inf"), dtype=torch.float64, device=dev)
        for _ in range(max_iter):
            val, der = h(u)
            done = (val.abs() < tol) | ~ok
            if bool(done.all()):
                break
            lo = torch.where(val < 0.0, u, lo)
            hi = torch.where(val > 0.0, u, hi)
            newton = u - val / torch.where(der > 0.0, der, torch.ones_like(der))
            inside = (der > 0.0) & (newton > lo) & (newton < hi)
            u = torch.where(done, u, torch.where(inside, newton, 0.5 * (lo + hi)))
        ok &= val.abs() < 1e-9

        p = u.exp()
        gp = torch.where(pos, torch.exp(p.unsqueeze(1) * logg), torch.zeros_like(logg))
        scale = m1 / gp.mean(dim=1).clamp_min(tiny)
        res = (scale.unsqueeze(1) * gp).to(o.dtype)
        o[s0:s0 + b] = torch.where(ok.unsqueeze(1), res, o[s0:s0 + b])
        p_all[s0:s0 + b] = torch.where(ok, p, torch.full_like(p, float("nan")))
        ok_all[s0:s0 + b] = ok
        del logg, pos, gp, res

    return out, p_all, ok_all


def mfgsm(fm, ctx, p):
    if ctx is None:
        raise ValueError("mFGSM needs the history buffer")
    B = fm.shape[0]
    f = fm.reshape(B, -1)
    if p["metric"] == "cosine":
        fu = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        dists = 1.0 - fu @ ctx.unit.T
    elif p["metric"] == "l2":
        dists = torch.cdist(f, ctx.flat)
    else:
        dists = (f.sum(dim=1).unsqueeze(1) - ctx.sums.unsqueeze(0)).abs()
    k = max(1, min(int(p["topk"]), ctx.hist.shape[0]))
    pl = ctx.hist[dists.topk(k=k, dim=1).indices]
    direction = torch.sign(torch.sign(fm.unsqueeze(1) - pl).sum(dim=1))
    del pl, dists

    std_i = f.std(dim=1, unbiased=True)
    if p["eps_mode"] == "fixed":
        eps = float(p["eps_rel"]) * std_i
        saturated = torch.zeros(B, dtype=torch.bool, device=fm.device)
    else:
        eps = _solve_eps_match_variance(f, direction.reshape(B, -1)).clamp_min(0.0)
        cap = float(p["eps_max"]) * std_i
        saturated = eps > cap
        eps = torch.minimum(eps, cap)

    mfm = fm - eps.view(B, 1, 1, 1) * direction
    mfm, clip_frac = _apply_clip(fm, mfm, ctx, p["clip"])
    diag = {"clip_frac": clip_frac,
            "eps_rel_used": (eps / std_i.clamp_min(1e-12)).double(),
            "eps_saturated": saturated}
    if p["match_mode"] == "power":
        mfm, power_p, ok = _moment_match_power(fm, mfm)
        fb = ~ok
        shift = f.mean(dim=1) - mfm.reshape(B, -1).mean(dim=1)
        mfm = mfm + (shift * fb.to(mfm.dtype)).view(B, 1, 1, 1)
        diag.update(power_p=power_p, match_fallback=fb)
    elif p["dc_correct"]:
        shift = f.mean(dim=1) - mfm.reshape(B, -1).mean(dim=1)
        mfm = mfm + shift.view(B, 1, 1, 1)
    return mfm, diag


def mgna(fm, p, img_idx):
    B, C = fm.shape[0], fm.shape[1]
    spatial = tuple(fm.shape[2:])
    gamma = float(p["gamma"])
    n_sel = int(round(gamma * C))
    if gamma > 0.0:
        n_sel = max(1, n_sel)
    std_i = fm.reshape(B, -1).std(dim=1, unbiased=True)
    noise = torch.zeros_like(fm)
    if n_sel > 0:
        for b in range(B):
            g = torch.Generator(device=fm.device)
            g.manual_seed(int((int(p["seed"]) * 1000003 + int(img_idx[b])) % (2 ** 63 - 1)))
            chans = torch.randperm(C, generator=g, device=fm.device)[:n_sel]
            y = torch.randn((n_sel,) + spatial, generator=g, device=fm.device, dtype=fm.dtype)
            noise[b, chans] = y * (float(p["sigma_rel"]) * std_i[b])
    mfm = fm + noise
    del noise
    return mfm, {"clip_frac": _zeros(B, fm.device),
                 "eps_rel_used": torch.full((B,), float(p["sigma_rel"]), dtype=torch.float64,
                                            device=fm.device),
                 "eps_saturated": torch.zeros(B, dtype=torch.bool, device=fm.device)}


def mjsma(fm, p):
    B = fm.shape[0]
    mfm = fm.clone()
    if p["scope"] == "image":
        rows, view = fm.reshape(B, -1), mfm.reshape(B, -1)
    else:
        rows, view = fm.reshape(B * fm.shape[1], -1), mfm.reshape(B * fm.shape[1], -1)
    n = rows.shape[1]
    k = int(round(float(p["theta"]) * n))
    if k > 0:
        absr = rows.abs()
        thr = absr.topk(k=k, dim=1).values[:, -1:]
        mask = absr >= thr
        del absr
        view[mask] = 0.0
        sat_rows = mask.sum(dim=1).to(torch.float64) / n
        del mask
    else:
        sat_rows = _zeros(rows.shape[0], fm.device)
    sat = sat_rows if p["scope"] == "image" else sat_rows.reshape(B, -1).mean(dim=1)
    return mfm, {"clip_frac": _zeros(B, fm.device), "sat_frac": sat,
                 "eps_saturated": torch.zeros(B, dtype=torch.bool, device=fm.device)}


def apply_attack(fm, ctx, attack, p, img_idx):
    fm4, orig = as_bchw(fm)
    if attack == "mfgsm":
        mfm, diag = mfgsm(fm4, ctx, p)
    elif attack == "mgna":
        mfm, diag = mgna(fm4, p, img_idx)
    elif attack == "mjsma":
        mfm, diag = mjsma(fm4, p)
    else:
        raise ValueError(f"unknown attack {attack!r}")
    if orig is not None:
        mfm = mfm.reshape(orig)
    return mfm, diag


def side_params(p, k):
    return dict(p, seed=int(p["seed"]) + 1 + k) if "seed" in p else p


def flat_cat(ts):
    B = ts[0].shape[0]
    if len(ts) == 1:
        return ts[0].reshape(B, -1)
    return torch.cat([x.reshape(B, -1) for x in ts], 1)


def _rel_pct(new, ref, tiny=1e-12):
    if not (np.isfinite(new) and np.isfinite(ref)) or abs(ref) < tiny:
        return float("nan")
    return 100.0 * (new - ref) / abs(ref)


def _pctiles(prefix, v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {}
    return {f"{prefix}_mean": float(v.mean()), f"{prefix}_std": float(v.std()),
            f"{prefix}_p05": float(np.percentile(v, 5)),
            f"{prefix}_p50": float(np.percentile(v, 50)),
            f"{prefix}_p95": float(np.percentile(v, 95)),
            f"{prefix}_min": float(v.min()), f"{prefix}_max": float(v.max())}


class FMStats:
    DIAG = ("clip_frac", "eps_rel_used", "eps_saturated", "sat_frac", "power_p", "match_fallback")

    def __init__(self):
        self.per = {}
        self.diag = {k: [] for k in self.DIAG}
        self.valid = []
        self.n_elems = 0

    def _push(self, k, v):
        self.per.setdefault(k, []).append(v)

    def mark(self):
        return ({k: len(v) for k, v in self.per.items()},
                {k: len(v) for k, v in self.diag.items()}, len(self.valid))

    def rollback(self, mark):
        per, diag, nv = mark
        for k, v in self.per.items():
            del v[per.get(k, 0):]
        for k, v in self.diag.items():
            del v[diag.get(k, 0):]
        del self.valid[nv:]

    def update(self, clean, atk, diag, valid):
        B = clean.shape[0]
        n = clean[0].numel()
        self.n_elems = n
        step = max(1, int(STATS_F64_BUDGET // max(n, 1)))
        for s in range(0, B, step):
            c = clean[s:s + step].reshape(-1, n).double()
            a = atk[s:s + step].reshape(-1, n).double()
            d = a - c
            vc = c.var(dim=1, unbiased=False)
            va = a.var(dim=1, unbiased=False)
            row = {"mean_c": c.mean(dim=1), "var_c": vc, "std_c": vc.sqrt(),
                   "mean_a": a.mean(dim=1), "var_a": va, "std_a": va.sqrt(),
                   "neg_c": (c < 0).sum(dim=1).double() / n, "neg_a": (a < 0).sum(dim=1).double() / n,
                   "l2_c": c.norm(dim=1)}
            del c, a
            row.update(mean_d=d.mean(dim=1), std_d=d.std(dim=1, unbiased=False), l2_d=d.norm(dim=1))
            d.abs_()
            row.update(absmean_d=d.mean(dim=1), linf_d=d.amax(dim=1))
            del d
            for k, v in zip(row, torch.stack(list(row.values())).cpu()):
                self._push(k, v)
        for k in self.DIAG:
            v = diag.get(k)
            if v is not None:
                self.diag[k].append(torch.as_tensor(v).reshape(-1).double().cpu())
        self.valid.append(valid.reshape(-1))

    def finalize(self):
        nan = float("nan")
        if not self.valid:
            return {"n_images_stats": 0}
        valid = torch.cat(self.valid).cpu().numpy().astype(bool)
        nimg = int(valid.sum())
        if nimg == 0:
            return {"n_images_stats": 0}
        arr = {k: torch.cat(v).cpu().numpy()[valid] for k, v in self.per.items()}
        m = lambda k: float(arr[k].mean())
        std_c = np.maximum(arr["std_c"], 1e-12)
        out = {
            "mean_clean": m("mean_c"), "std_clean": m("std_c"), "var_clean": m("var_c"),
            "mean_attacked": m("mean_a"), "std_attacked": m("std_a"), "var_attacked": m("var_a"),
            "mean_delta": m("mean_d"), "std_delta": m("std_d"), "absmean_delta": m("absmean_d"),
            "l2_clean": m("l2_c"), "l2_delta": m("l2_d"),
            "l2_mean_per_image": float(np.sqrt(np.mean(arr["l2_d"] ** 2))),
            "l2_rel_pct": 100.0 * float(np.mean(arr["l2_d"] / np.maximum(arr["l2_c"], 1e-12))),
            "amp_rel": float(np.mean((arr["l2_d"] / math.sqrt(self.n_elems)) / std_c)),
            "d_mean_sigma": float(np.mean(arr["mean_d"] / std_c)),
            "neg_frac_clean_pct": 100.0 * m("neg_c"),
            "neg_frac_attacked_pct": 100.0 * m("neg_a"),
            "linf_max": float(arr["linf_d"].max()),
            "n_images_stats": nimg, "n_elems_per_image": self.n_elems,
        }
        out["d_mean_pct"] = _rel_pct(out["mean_attacked"], out["mean_clean"])
        out["d_std_pct"] = _rel_pct(out["std_attacked"], out["std_clean"])
        out["d_var_pct"] = _rel_pct(out["var_attacked"], out["var_clean"])
        out["snr_db"] = (20.0 * math.log10(out["l2_clean"] / out["l2_delta"])
                         if out["l2_delta"] > 0 and out["l2_clean"] > 0 else nan)

        dg = {k: torch.cat(v).cpu().numpy()[valid] for k, v in self.diag.items()
              if v and sum(t.numel() for t in v) == valid.size}
        out["clip_frac_pct"] = 100.0 * float(dg["clip_frac"].mean()) if "clip_frac" in dg else nan
        if "eps_rel_used" in dg:
            out.update(_pctiles("eps_rel", dg["eps_rel_used"]))
        if "eps_saturated" in dg:
            out["eps_saturated_pct"] = 100.0 * float(dg["eps_saturated"].mean())
        if "sat_frac" in dg:
            out["sat_frac_pct"] = 100.0 * float(dg["sat_frac"].mean())
            out["sat_frac_max_pct"] = 100.0 * float(dg["sat_frac"].max())
        if "power_p" in dg:
            out.update(_pctiles("power_p", dg["power_p"]))
        if "match_fallback" in dg:
            out["match_fallback_pct"] = 100.0 * float(dg["match_fallback"].mean())
        return out


class PredStore:
    def __init__(self, n):
        self.top3 = np.full((n, 3), -1, dtype=np.int64)
        self.score = np.full((n, 3), np.nan, dtype=np.float32)

    def put(self, idx, out):
        flat = out.reshape(out.shape[0], -1)
        k = min(3, flat.shape[1])
        s, i = flat.topk(k, dim=1)
        self.top3[idx, :k] = i.cpu().numpy()
        self.score[idx, :k] = s.float().cpu().numpy()

    @property
    def top1(self):
        return self.top3[:, 0]


def paired_metrics(base_top1, atk_top1, gt, valid):
    b = (base_top1 == gt)[valid]
    a = (atk_top1 == gt)[valid]
    n = int(valid.sum())
    both_ok = int((b & a).sum())
    b_ok_a_ko = int((b & ~a).sum())
    b_ko_a_ok = int((~b & a).sum())
    both_ko = int((~b & ~a).sum())
    d_asr, d_acr = both_ok + b_ok_a_ko, b_ko_a_ok + both_ko
    return {
        "n_paired_images": n,
        "base_correct_atk_correct": both_ok,
        "base_correct_atk_wrong": b_ok_a_ko,
        "base_wrong_atk_correct": b_ko_a_ok,
        "base_wrong_atk_wrong": both_ko,
        "attack_success_rate_pct": round(100.0 * b_ok_a_ko / d_asr, 2) if d_asr else np.nan,
        "accidental_correction_rate_pct": round(100.0 * b_ko_a_ok / d_acr, 2) if d_acr else np.nan,
        "net_effect_pct": round(100.0 * (b_ok_a_ko - b_ko_a_ok) / n, 2) if n else np.nan,
        "accuracy_attack_pct": round(100.0 * float(a.mean()), 2) if n else np.nan,
    }
