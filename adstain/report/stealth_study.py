#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import campaign_figures as PR
from .campaign_figures import AXIS, INK, INK2, MUTED, SURFACE, plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

DUMMY_MARKER = 'DUMMY'
SEMANTICS = 'device'
BAND = 10.0
SENSITIVITY_BANDS = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]
SUCCESS = 50.0
BANDS = list(SENSITIVITY_BANDS)
N_SPLITS, CHUNK = 1000, 100
THR = dict(mode='fixed', source='test', calibration=0.3, false_alarm=0.01, window=1000, drift=False, margin=1.0,
           repeats=5)
CAL_DRAWS = 2000
THR_LABEL = '±10 %'
SCALE = 'half of the test set'
DETECTED, STEALTHY = 0.95, 0.05
NAT_Q = 0.99
STRONG = 1.0
CONTROLS = ('mgna', 'mjsma')
FAM_COLOR = {f: PR.family_color(f, list(PR.FAMILIES)) for f in PR.FAMILIES}
FAM_SHORT = {'mfgsm': 'mFGSM', 'mgna': 'mGNA', 'mjsma': 'mJSMA'}
SYM = {'mfgsm': 'ε', 'mgna': 'σ', 'mjsma': 'θ'}
STAT_STYLE = {'std': dict(color=INK, marker='o'), 'mean': dict(color=INK2, marker='s')}
STAT_LABEL = {'std': 'Δstd', 'mean': 'Δmean'}
CLASSES = ['stealthy, within natural noise', 'stealthy, above natural noise', 'borderline', 'flagged']
CLASS_COLOR = dict(zip(CLASSES, [PR.BLUES[0], PR.BLUES[4], PR.BLUES[8], PR.BLUES[12]]))
CLASS_INK = dict(zip(CLASSES, [INK, INK, SURFACE, SURFACE]))


def parse_tag(tag):
    num = lambda s: float(s.replace('p', '.'))
    fam = tag.split('_', 1)[0]
    if fam == 'mgna':
        g, s = re.match(r'mgna_g([0-9p]+)_s([0-9p]+)', tag).groups()
        return fam, num(s), num(g)
    s = re.search(r'(?:_eps_|_t)([0-9p]+)', tag)
    return fam, num(s.group(1)) if s else np.nan, np.nan


def load(name, data, runs, attack_dirs=('attack',)):
    z = np.load(data / f'{name}_fmstats.npz')
    meta = json.loads((data / f'{name}_meta.json').read_text())
    v = z['valid']
    m = dict(name=name, pretty=PR.pretty(name), task=meta['task'], taps=[str(t) for t in z['taps']], meta=meta,
             run_dir=runs / name)
    m['cs'], m['cm'] = z['clean_std'][v], z['clean_mean'][v]
    m['as'], m['am'] = z['atk_std'][v].astype(np.float64), z['atk_mean'][v].astype(np.float64)
    m['amp'] = np.nanmean(z['atk_amp'][v].astype(np.float64), axis=0)
    m['n'] = int(v.sum())
    m['tags'] = [str(t) for t in z['cfg_tags']]
    parsed = [parse_tag(t) for t in m['tags']]
    m['cfg_attack'] = [p[0] for p in parsed]
    m['strength'] = [p[1] for p in parsed]
    m['gamma'] = [p[2] for p in parsed]
    m['units'] = pd.DataFrame({'layer_name': [str(x) for x in z['layer_names']],
                               'tensor': [str(x) for x in z['layer_taps']]})
    frames = [pd.read_csv(c) for c in (runs / name / sub / 'attack_summary_gpu.csv' for sub in attack_dirs)
              if c.exists()]
    if frames:
        a = pd.concat(frames, ignore_index=True).drop_duplicates(['layer_name', 'config_tag'], keep='last')
        a['asr'] = a['attack_success_rate_pct']
        a['d_mean_std_pct'] = 100 * a['d_mean_sigma']
        m['campaign'] = a[['layer_name', 'config_tag', 'asr', 'd_std_pct', 'd_mean_std_pct']].rename(
            columns={'d_std_pct': 'campaign_d_std_pct', 'd_mean_std_pct': 'campaign_d_mean_std_pct'})
    else:
        m['campaign'] = pd.DataFrame(columns=['layer_name', 'config_tag', 'asr', 'campaign_d_std_pct',
                                              'campaign_d_mean_std_pct'])
    m['n_cfg_asr'] = len(set(m['tags']) & set(m['campaign']['config_tag']))
    m['val'] = None
    vf = data / 'validation' / f'{name}_fmstats.npz'
    if windowed() and THR['source'] == 'validation' and not vf.exists():
        print(f'  [{name}] no validation statistics ({vf.name}, collected by the report of that model\'s config): '
              f'calibrated on {100 * float(THR["calibration"]):g} % of its test images instead')
    elif windowed() and THR['source'] == 'validation':
        zv = np.load(vf)
        pos = {str(t): i for i, t in enumerate(zv['taps'])}
        miss = [t for t in m['taps'] if t not in pos]
        if miss:
            raise RuntimeError(f'{vf}: tensors {miss[:3]} missing, collected for other cuts')
        idx, vv = [pos[t] for t in m['taps']], zv['valid']
        m['val'] = (zv['clean_mean'][vv][:, idx], zv['clean_std'][vv][:, idx])
    return m


def calibrated():
    return THR['mode'] == 'calibrated'


def windowed():
    return THR['mode'] in ('calibrated', 'fixed_window')


def draw_weights(rng, c, n, k):
    W = np.zeros((c, n))
    for i in range(c):
        W[i] = np.bincount(rng.integers(0, n, k), minlength=n)
    return W / k


def calibration_split(m, rng):
    if m['val'] is not None:
        return m['val'], np.arange(m['n'])
    p = rng.permutation(m['n'])
    k = int(round(THR['calibration'] * m['n']))
    return (m['cm'][np.sort(p[:k])], m['cs'][np.sort(p[:k])]), np.sort(p[k:])


def consecutive(cm, cs, M, S):
    N = min(int(THR['window']), len(cs))
    w = len(cs) // N
    Sw = cs[:w * N].reshape(w, N, -1).mean(1)
    Mw = cm[:w * N].reshape(w, N, -1).mean(1)
    return 100 * (Sw - S) / S, 100 * (Mw - M) / S


def thresholds(cal, rng):
    cm, cs = cal
    n, N = len(cs), int(THR['window'])
    dev = []
    for c0 in range(0, CAL_DRAWS, CHUNK):
        c = min(CHUNK, CAL_DRAWS - c0)
        Wr, Ww = draw_weights(rng, c, n, n), draw_weights(rng, c, n, N)
        Sr, Mr = Wr @ cs, Wr @ cm
        dev.append(100 * np.maximum(np.abs(Ww @ cs - Sr), np.abs(Ww @ cm - Mr)) / Sr)
    tau = np.quantile(np.concatenate(dev), 1 - float(THR['false_alarm']), axis=0)
    ds, dm = consecutive(cm, cs, cm.mean(0), cs.mean(0))
    drift = np.maximum(np.abs(ds), np.abs(dm)).max(0)
    if THR['drift']:
        tau = np.maximum(tau, drift)
    return float(THR['margin']) * tau, drift


def window_means(m, rng, ref, ev, n_windows=N_SPLITS):
    T, K = len(m['taps']), len(m['tags'])
    (Mref, Sref), N, nE = ref, int(THR['window']), len(ev)
    cs, cm = m['cs'][ev], m['cm'][ev]
    as2, am2 = m['as'][ev].reshape(nE, T * K), m['am'][ev].reshape(nE, T * K)
    out = {k: [] for k in ('SBc', 'MBc', 'SBa', 'MBa')}
    for c0 in range(0, n_windows, CHUNK):
        c = min(CHUNK, n_windows - c0)
        W = draw_weights(rng, c, nE, N)
        out['SBc'].append(W @ cs)
        out['MBc'].append(W @ cm)
        out['SBa'].append((W @ as2).reshape(c, T, K))
        out['MBa'].append((W @ am2).reshape(c, T, K))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out['SA'] = np.broadcast_to(Sref, out['SBc'].shape)
    out['MA'] = np.broadcast_to(Mref, out['MBc'].shape)
    return out


def half_means(m, rng):
    n, T, K = m['n'], len(m['taps']), len(m['tags'])
    h = n // 2
    as2, am2 = m['as'].reshape(n, T * K), m['am'].reshape(n, T * K)
    out = {k: [] for k in ('SA', 'MA', 'SBc', 'MBc', 'SBa', 'MBa')}
    for c0 in range(0, N_SPLITS, CHUNK):
        c = min(CHUNK, N_SPLITS - c0)
        WA, WB = np.zeros((c, n)), np.zeros((c, n))
        for i in range(c):
            p = rng.permutation(n)
            WA[i, p[:h]] = 1.0 / h
            WB[i, p[h:2 * h]] = 1.0 / h
        out['SA'].append(WA @ m['cs'])
        out['MA'].append(WA @ m['cm'])
        out['SBc'].append(WB @ m['cs'])
        out['MBc'].append(WB @ m['cm'])
        out['SBa'].append((WB @ as2).reshape(c, T, K))
        out['MBa'].append((WB @ am2).reshape(c, T, K))
    return {k: np.concatenate(v) for k, v in out.items()}


def order_deviation(m):
    h = m['n'] // 2
    halves = [slice(0, h), slice(h, 2 * h)]
    ds, dm = [], []
    for a, b in ((0, 1), (1, 0)):
        SA, MA = m['cs'][halves[a]].mean(0), m['cm'][halves[a]].mean(0)
        SB, MB = m['cs'][halves[b]].mean(0), m['cm'][halves[b]].mean(0)
        ds.append(100 * (SB - SA) / SA)
        dm.append(100 * (MB - MA) / SA)
    return np.array(ds), np.array(dm)


def analyse_model(m, rng):
    T = len(m['taps'])
    if windowed():
        reps = max(1, int(THR['repeats'])) if m['val'] is None else 1
        hms, thr_w, taus, drifts = [], [], [], []
        for r in range(reps):
            cal, ev_r = calibration_split(m, rng)
            ref = (cal[0].mean(0), cal[1].mean(0))
            tau, drift = thresholds(cal, rng)
            if not calibrated():
                tau = np.full(T, BAND)
            n_w = N_SPLITS // reps + (r < N_SPLITS % reps)
            hms.append(window_means(m, rng, ref, ev_r, n_w))
            thr_w.append(np.broadcast_to(tau, (n_w, T)))
            taus.append(tau), drifts.append(drift)
            if r == 0:
                ev, (Mref, Sref) = ev_r, ref
        hm = {k: np.concatenate([h[k] for h in hms]) for k in hms[0]}
        thr_w, thr, drift_cal = np.concatenate(thr_w), np.mean(taus, axis=0), np.mean(drifts, axis=0)
        m['calibration'] = dict(n_cal=len(cal[1]), n_eval=len(ev), repeats=reps,
                                small=len(cal[1]) < 10 * int(THR['window']),
                                source='validation images' if m['val'] is not None else 'held-out test images')
        if m['calibration']['small']:
            print(f"  [{m['name']}] only {len(cal[1])} calibration images for windows of {THR['window']}: the noise of "
                  f"the reference is not negligible; more calibration images (≥ 10 × N) or a smaller window is safer")
    else:
        hm = half_means(m, rng)
        thr, drift_cal = np.full(T, BAND), np.full(T, np.nan)
        thr_w = np.broadcast_to(thr, (N_SPLITS, T))
    SA, MA = hm['SA'], hm['MA']
    dsn, dmn = 100 * (hm['SBc'] - SA) / SA, 100 * (hm['MBc'] - MA) / SA
    dsa = 100 * (hm['SBa'] - SA[:, :, None]) / SA[:, :, None]
    dma = 100 * (hm['MBa'] - MA[:, :, None]) / SA[:, :, None]
    en = np.maximum(np.abs(dsn), np.abs(dmn))
    ea = np.maximum(np.abs(dsa), np.abs(dma))
    os_, om = consecutive(m['cm'][ev], m['cs'][ev], Mref, Sref) if windowed() else order_deviation(m)
    eo = np.maximum(np.abs(os_), np.abs(om))
    b_nat = np.quantile(en, NAT_Q, axis=0)

    nat_rows = []
    for j, t in enumerate(m['taps']):
        row = dict(model=m['name'], tensor=t, n_images=m['n'], n_splits=N_SPLITS,
                   half=int(THR['window']) if windowed() else m['n'] // 2)
        for stat, x in (('std', dsn[:, j]), ('mean', dmn[:, j]), ('either', en[:, j])):
            x = np.abs(x)
            row.update({f'{stat}_q50': np.quantile(x, .5), f'{stat}_q95': np.quantile(x, .95),
                        f'{stat}_q99': np.quantile(x, .99), f'{stat}_max': x.max()})
        row.update(order_std=np.abs(os_[:, j]).max(), order_mean=np.abs(om[:, j]).max(), order_either=eo[:, j].max(),
                   noise_band=b_nat[j], threshold=thr[j], fpr_thr=(en[:, j] > thr_w[:, j]).mean(),
                   order_flag_thr=bool((eo[:, j] > thr[j]).any()), drift_calibration=drift_cal[j])
        if windowed():
            row.update(n_calibration=m['calibration']['n_cal'], n_evaluation=m['calibration']['n_eval'],
                       calibration_source=m['calibration']['source'])
        for b in BANDS:
            row[f'fpr_b{b:g}'] = (en[:, j] > b).mean()
            row[f'order_flag_b{b:g}'] = bool((eo[:, j] > b).any())
        nat_rows.append(row)
    nat = pd.DataFrame(nat_rows)

    det_rows = []
    for j, t in enumerate(m['taps']):
        for k, tag in enumerate(m['tags']):
            row = dict(model=m['name'], tensor=t, config_tag=tag, family=m['cfg_attack'][k], strength=m['strength'][k],
                       gamma=m['gamma'][k], amp_rel=m['amp'][j, k],
                       d_std_median=np.median(dsa[:, j, k]), d_mean_median=np.median(dma[:, j, k]),
                       either_q01=np.quantile(ea[:, j, k], .01), either_median=np.median(ea[:, j, k]),
                       noise_band=b_nat[j], flag_rate_noise_band=(ea[:, j, k] > b_nat[j]).mean(),
                       threshold=thr[j], flag_rate_thr=(ea[:, j, k] > thr_w[:, j]).mean())
            for b in BANDS:
                row[f'flag_rate_b{b:g}'] = (ea[:, j, k] > b).mean()
            det_rows.append(row)
    det = pd.DataFrame(det_rows)
    f10 = det['flag_rate_thr']
    det['class'] = np.select(
        [f10 >= DETECTED, f10 > STEALTHY, det['flag_rate_noise_band'] >= DETECTED],
        ['flagged', 'borderline', 'stealthy, above natural noise'], 'stealthy, within natural noise')
    det = m['units'].merge(det, on='tensor', how='left')
    det = det.merge(m['campaign'], on=['layer_name', 'config_tag'], how='left')
    return nat, det


def strong(d):
    return d['amp_rel'] >= STRONG


def admissible(b):
    ok = (b.fpr_worst <= 0.01) & (b.order_flagged_share == 0)
    for fam in CONTROLS:
        ok &= b[f'{fam}_strong_detected_share'].fillna(1) >= 1
    return ok


def band_table(models, nat, det):
    rows = []
    for m in models:
        g, d = nat[nat.model == m['name']], det[det.model == m['name']]
        for b in BANDS:
            hit = d[f'flag_rate_b{b:g}'] >= DETECTED
            row = dict(model=m['name'], band_pct=b, fpr_worst=g[f'fpr_b{b:g}'].max(),
                       order_flagged_share=g[f'order_flag_b{b:g}'].mean())
            for fam in PR.FAMILIES:
                f = d.family == fam
                row[f'{fam}_detected_share'] = hit[f].mean()
                row[f'{fam}_strong_n'] = int((f & strong(d)).sum())
                row[f'{fam}_strong_detected_share'] = hit[f & strong(d)].mean() if (f & strong(d)).any() else np.nan
            hi = d['asr'] >= SUCCESS
            row['successful_detected_share'] = hit[hi].mean() if hi.any() else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def grid(n, ncols=3, w=4.3, h=3.4):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def band_line(ax, band=None, label=True, horizontal=True):
    band = BAND if band is None else band
    (ax.axhline if horizontal else ax.axvline)(band, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    if label and horizontal:
        ax.annotate(f'±{band:g} % band', (1, band), xycoords=('axes fraction', 'data'), xytext=(-2, 3),
                    textcoords='offset points', ha='right', va='bottom', fontsize=7.5, color=INK)


def footer(fig, handles, ncol, text):
    if handles:
        fig.legend(handles=handles, loc='upper left', ncol=ncol, bbox_to_anchor=(0.01, 0.0))
    rows = int(np.ceil(len(handles) / ncol)) if handles else 0
    fig.text(0.01, -(0.3 + 0.21 * rows) / fig.get_figheight(), text, fontsize=7.5, color=MUTED, va='top',
             linespacing=1.5)


def plain_log(ax, axis='x'):
    from matplotlib.ticker import FuncFormatter
    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def fam_handle(f, label):
    under = f == 'mgna'
    return Line2D([], [], color=FAM_COLOR[f], marker=PR.FAM_MARKER[f], markersize=6 if under else 4,
                  linewidth=4.0 if under else 1.8, label=label)


def fmt_dev(x):
    return f'{x:.2f}' if x < 1 else f'{x:.1f}' if x < 10 else f'{x:.0f}'


def cfg_short(m, k):
    f = m['cfg_attack'][k]
    s = m['strength'][k]
    lab = f'{SYM[f]} {s:g}' if s == s else f"{SYM[f]}*" + ('\npower' if m['tags'][k].endswith('_pow') else '')
    gammas = {g for g, a in zip(m['gamma'], m['cfg_attack']) if a == 'mgna'}
    return lab + (f'\nγ {m["gamma"][k]:g}' if f == 'mgna' and len(gammas) > 1 else '')


def fig_natural(models, nat, out, grid_name):
    fig, ax = plt.subplots(figsize=(max(6.0, 1.75 * len(models) + 1.5), 4.4))
    rng = np.random.default_rng(1)
    floor = 0.01
    for i, m in enumerate(models):
        g = nat[nat.model == m['name']]
        for c, (stat, col, filled) in enumerate((('std', 'std_q99', True), ('std', 'order_std', False),
                                                 ('mean', 'mean_q99', True), ('mean', 'order_mean', False))):
            y = np.maximum(g[col].to_numpy(), floor)
            x = i + (c - 1.5) * 0.2 + (rng.random(len(y)) - 0.5) * 0.13
            st = STAT_STYLE[stat]
            ax.scatter(x, y, s=22, marker=st['marker'], linewidth=1.0, facecolor=st['color'] if filled else SURFACE,
                       edgecolor=SURFACE if filled else st['color'], zorder=3)
    if calibrated():
        for i, m in enumerate(models):
            g = nat[nat.model == m['name']]
            ax.scatter(i + (rng.random(len(g)) - 0.5) * 0.7, g['threshold'], s=30, marker='_', color=INK,
                       linewidth=1.4, zorder=4)
    else:
        band_line(ax)
    ax.set_yscale('log')
    ax.set_ylim(floor * 0.7, None)
    plain_log(ax, 'y')
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"{m['pretty'].replace(' · ', chr(10))}\n" + (
        f"N = {THR['window']}, ref. {m['calibration']['n_cal']}" if windowed() else f"2 × {m['n'] // 2} images")
        for m in models], fontsize=8)
    ax.set_ylabel('|Δ| clean vs clean reference (%)')
    ax.grid(axis='x', visible=False)
    ax.set_title(f'Natural variability of the monitored statistics ({SCALE}): clean vs clean reference',
                 loc='left', fontweight='bold')
    order = ('windows of consecutive images (dataset order, worst)' if windowed() else
             'split in dataset order (first vs second half)')
    handles = [Line2D([], [], linestyle='', marker=STAT_STYLE[s]['marker'], color=STAT_STYLE[s]['color'],
                      markersize=6, label=f'{STAT_LABEL[s]}: 99th percentile over {N_SPLITS} random '
                                          f'{"windows" if windowed() else "splits"}')
               for s in ('std', 'mean')]
    handles += [Line2D([], [], linestyle='', marker=STAT_STYLE[s]['marker'], markerfacecolor=SURFACE,
                       markeredgecolor=STAT_STYLE[s]['color'], markersize=6,
                       label=f'{STAT_LABEL[s]}: {order}') for s in ('std', 'mean')]
    if calibrated():
        handles.append(Line2D([], [], linestyle='', marker='_', color=INK, markersize=9, markeredgewidth=1.4,
                              label=f'calibrated threshold τ of each tensor ({THR_LABEL})'))
    fig.tight_layout()
    footer(fig, handles, 2,
           'One point per attacked tensor. Random splits: the test set shuffled and cut into two disjoint halves, '
           f'{N_SPLITS} times. Dataset order: ImageNet is class-sorted, so its two halves\nhold different classes '
           '(classes 0-479 vs 480-959); CIFAR-10 and COCO are not sorted. Δstd = (S_B − S_A) / S_A, '
           'Δmean = (M_B − M_A) / S_A, S, M = average per-image std\nand mean of the feature map over a half. '
           'Values below 0.01 % are drawn at 0.01 % (YOLO11n attention softmax: per-image mean fixed by construction).')
    PR.save(fig, out / '01_natural_variability_halves.png')


def fig_footprint(models, nat, det, out, grid_name):
    fig, axes = grid(len(models), w=4.4, h=4.0)
    lo, hi = 0.01, 1000
    for ax, m in zip(axes, models):
        g, d = nat[nat.model == m['name']], det[det.model == m['name']]
        qs, qm = g['std_q99'].max(), g['mean_q99'].max()
        sq = g['threshold'].median()
        ax.add_patch(Rectangle((lo, lo), sq - lo, sq - lo, facecolor='none', edgecolor=INK, linewidth=1.0,
                               linestyle=(0, (4, 3)), zorder=2))
        ax.add_patch(Rectangle((lo, lo), qs - lo, qm - lo, facecolor=AXIS, alpha=0.6, linewidth=0, zorder=0))
        for fam in PR.FAMILIES:
            s = d[d.family == fam]
            size = np.where(s['asr'].notna(), 6 + 0.5 * s['asr'].clip(0, 100), 10)
            ax.scatter(np.clip(s['d_std_median'].abs(), lo, hi), np.clip(s['d_mean_median'].abs(), lo, hi),
                       s=size, color=FAM_COLOR[fam], alpha=0.65, edgecolor=SURFACE,
                       linewidth=0.4, marker=PR.FAM_MARKER[fam], zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plain_log(ax, 'x')
        plain_log(ax, 'y')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        n_in = (d['class'].str.startswith('stealthy')).sum()
        ax.set_title(f"{m['pretty']}  ({n_in}/{len(d)} stealthy)", loc='left')
        ax.set_xlabel('|Δstd|, attacked vs clean reference (%)')
        ax.set_ylabel('|Δmean| (%)')
    handles = [Line2D([], [], color=FAM_COLOR[f], marker=PR.FAM_MARKER[f], linestyle='', markersize=6,
                      label=PR.FAMILIES[f][0]) for f in PR.FAMILIES]
    handles += [Line2D([], [], color=INK, linestyle=(0, (4, 3)), linewidth=1.0,
                       label=(f'median τ of the model on both statistics (each tensor has its own τ)' if calibrated()
                              else f'{THR_LABEL} on both statistics (stealthy inside)')),
                Patch(facecolor=AXIS, alpha=0.6, label='natural 99th percentile (worst tensor of the model)')]
    fig.suptitle(f'Footprint of every attacked layer × strength ({SCALE} attacked, {grid_name})',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    footer(fig, handles, 3,
           f'One point per attacked layer × configuration: median over {N_SPLITS} random splits of the deviation of the '
           'attacked half from the clean reference half; marker area ∝ ASR of the campaign\n(small fixed marker when the '
           f'configuration was not in the campaign). Points inside the dashed square are stealthy under {THR_LABEL}; points in '
           'the grey rectangle are also within\nthe natural variability of the clean halves. Values are clipped to '
           '[0.01, 1000] %.')
    PR.save(fig, out / '02_attack_footprint_halves.png')


def fig_layer_maps(models, nat, det, out, grid_name):
    cmap = ListedColormap([CLASS_COLOR[c] for c in CLASSES])
    for m in models:
        d = det[det.model == m['name']]
        g = nat[nat.model == m['name']].set_index('tensor')
        layers = list(m['units']['layer_name'])
        K = len(m['tags'])
        Z = np.zeros((len(layers), K))
        fig, ax = plt.subplots(figsize=(2.8 + 0.6 * K, 1.8 + 0.26 * len(layers)))
        for i, lay in enumerate(layers):
            for k, tag in enumerate(m['tags']):
                r = d[(d.layer_name == lay) & (d.config_tag == tag)].iloc[0]
                Z[i, k] = CLASSES.index(r['class'])
                ax.text(k, i, fmt_dev(r['either_median']), ha='center', va='center', fontsize=6.5,
                        color=CLASS_INK[r['class']])
                if r['asr'] >= SUCCESS:
                    ax.plot(k + 0.36, i - 0.24, marker='o', markersize=2.6, color=CLASS_INK[r['class']],
                            markeredgewidth=0)
        ax.imshow(Z, cmap=cmap, vmin=-0.5, vmax=len(CLASSES) - 0.5, aspect='auto', interpolation='nearest')
        for k in range(1, K):
            ax.axvline(k - 0.5, color=SURFACE, linewidth=1.5)
        for i in range(1, len(layers)):
            ax.axhline(i - 0.5, color=SURFACE, linewidth=1.0)
        fams = m['cfg_attack']
        for f in PR.FAMILIES:
            ks = [k for k in range(K) if fams[k] == f]
            if ks:
                ax.annotate(FAM_SHORT[f], (np.mean(ks), 1), xycoords=('data', 'axes fraction'), xytext=(0, 26),
                            textcoords='offset points', ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                            color=INK)
                if ks[0] > 0:
                    ax.axvline(ks[0] - 0.5, color=SURFACE, linewidth=4)
        ax.set_xticks(range(K))
        ax.set_xticklabels([cfg_short(m, k) for k in range(K)], fontsize=7)
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', pad=1)
        ax.set_yticks(range(len(layers)))
        tens = m['units'].set_index('layer_name')['tensor']
        names = [PR.short_layer(l.split('@')[0], 30) for l in layers]
        dup = {x for x in names if names.count(x) > 1}
        ax.set_yticklabels([f"{x} @{l.split('@')[1]}" if x in dup and '@' in l else x for x, l in zip(names, layers)],
                           fontsize=7)
        col, head = ('threshold', 'τ %') if calibrated() else ('noise_band', 'noise\nq99 %')
        for i, lay in enumerate(layers):
            ax.text(K - 0.35, i, f"{g.loc[tens[lay], col]:.2f}", ha='left', va='center', fontsize=6.5,
                    color=MUTED)
        ax.annotate(head, (K - 0.35, 1), xycoords=('data', 'axes fraction'), xytext=(0, 3),
                    textcoords='offset points', ha='left', va='bottom', fontsize=6.5, color=MUTED)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{m['pretty']}: every attacked layer × tested strength at {THR_LABEL} "
                     f"({SCALE} attacked, {grid_name})", loc='left', fontweight='bold', pad=44)
        handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor=AXIS, linewidth=0.5, label=c) for c in CLASSES]
        handles.append(Line2D([], [], linestyle='', marker='o', markersize=4, color=INK,
                              label=f"campaign ASR ≥ {SUCCESS:g} % ({m['n_cfg_asr']} of the {K} configurations were in the "
                                    'campaign)'))
        fig.tight_layout()
        footer(fig, handles, 5,
               'Cell number = footprint of the attack: median over the splits of max(|Δstd|, |Δmean|) between the attacked '
               f'half and the clean half (%). Colour = class over {N_SPLITS} random splits:\nflagged on ≥ '
               f'{100 * DETECTED:g} % of the splits (footprint > {THR_LABEL}), borderline, or stealthy (flagged on ≤ '
               f'{100 * STEALTHY:g} %); a stealthy pair is "above natural noise" when a band set at the natural '
               f'{100 * NAT_Q:g}th percentile\nof its tensor (right column) still flags it on ≥ {100 * DETECTED:g} % of '
               'the splits. Layers in depth order.')
        PR.save(fig, out / f'03_layers_x_strengths_{m["name"]}.png')


def fig_bands(models, bt, out, grid_name, nat_thr=None):
    nat_thr = nat_thr or {}
    fig, axes = grid(len(models))
    for ax, m in zip(axes, models):
        g = bt[bt.model == m['name']].sort_values('band_pct')
        for b in g.loc[admissible(g), 'band_pct']:
            ax.axvspan(b * 10 ** -0.08, b * 10 ** 0.08, color=AXIS, alpha=0.45, linewidth=0, zorder=0)
        ax.plot(g.band_pct, 100 * g.fpr_worst, color=INK, linewidth=1.8, marker='o', markersize=3.8,
                markeredgecolor=SURFACE, zorder=4)
        ax.plot(g.band_pct, 100 * g.order_flagged_share, color=INK, linewidth=1.2, linestyle=':', marker='o',
                markersize=3.8, markerfacecolor=SURFACE, zorder=4)
        for fam in PR.FAMILIES:
            if g[f'{fam}_strong_n'].iloc[0] == 0:
                continue
            under = fam == 'mgna'
            ax.plot(g.band_pct, 100 * g[f'{fam}_strong_detected_share'], color=FAM_COLOR[fam],
                    linewidth=4.0 if under else 1.8, marker=PR.FAM_MARKER[fam], markersize=6 if under else 4,
                    markeredgecolor=SURFACE, zorder=2 if under else 3)
        if calibrated():
            tau = nat_thr.get(m['name'], [BAND])
            ax.axvspan(min(tau), max(tau), color=INK, alpha=0.08, linewidth=0, zorder=1)
            ax.axvline(float(np.median(tau)), color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        else:
            ax.axvline(BAND, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.set_xscale('log')
        ax.set_xticks(BANDS)
        ax.set_xticklabels([f'{b:g}' for b in BANDS])
        ax.minorticks_off()
        ax.set_xlim(0.8, 62)
        ax.set_ylim(-3, 103)
        n_strong = ', '.join(f"{FAM_SHORT[f]} {g[f'{f}_strong_n'].iloc[0]}" for f in PR.FAMILIES)
        ax.set_title(f"{m['pretty']}", loc='left')
        ax.text(0.99, 0.1, f'strong pairs: {n_strong}', transform=ax.transAxes, ha='right', fontsize=6.5, color=MUTED)
        ax.set_xlabel('band ± b (%)')
        ax.set_ylabel('%')
    handles = [Line2D([], [], color=INK, marker='o', markersize=4, linewidth=1.8,
                      label=f'false alarms, clean half vs clean half (worst tensor, {N_SPLITS} random splits)'),
               Line2D([], [], color=INK, linestyle=':', marker='o', markerfacecolor=SURFACE, markersize=4,
                      label='tensors flagged when the halves follow the dataset order (class-sorted ImageNet)')]
    handles += [fam_handle(f, f'{FAM_SHORT[f]}: strong pairs (amp_rel ≥ {STRONG:g}) flagged on ≥ {100 * DETECTED:g} % '
                              'of the splits') for f in PR.FAMILIES]
    handles += [Line2D([], [], color=INK, linestyle=(0, (4, 3)), linewidth=1.0,
                       label='median calibrated τ (shaded: range over the tensors)' if calibrated() else THR_LABEL),
                Patch(facecolor=AXIS, alpha=0.45, label='admissible band: no false alarm (random or dataset-order '
                                                       'halves), every strong mGNA and mJSMA flagged')]
    fig.suptitle(f'Choosing the band ({SCALE} monitored): false alarms vs detection of strong '
                 f'perturbations ({grid_name})', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    footer(fig, handles, 2,
           'A half is flagged when |Δstd| or |Δmean| exceeds the band. Strong perturbation: amp_rel = RMS of the '
           'perturbation / feature-map std ≥ 1 (perturbation energy ≥ signal energy).\nToo narrow: a clean half with '
           'other classes than the reference raises alarms; too wide: strong blind perturbations (mGNA, mJSMA) pass '
           'unseen. mFGSM is built to stay stealthy\n(it is not a control): its line shows how many of its strong pairs '
           'each band would still catch.')
    PR.save(fig, out / '04_band_sensitivity_halves.png')


def fig_stealthy(models, det, out, grid_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.3 * 3 * len(models) + 1.6), sharey=True)
    labels, rows = [], []
    for m in models:
        for fam in PR.FAMILIES:
            labels.append(f"{m['pretty'].split(' · ')[0]} · {FAM_SHORT[fam]}")
            rows.append((m['name'], fam))
    y = np.arange(len(rows))[::-1]
    n_asr = det['asr'].notna().sum()
    for ax, (title, sel) in zip(axes, (('all tested (layer, strength) pairs', lambda d: d),
                                       (f'pairs with ASR ≥ {SUCCESS:g} % (the attacks worth studying)',
                                        lambda d: d[d.asr >= SUCCESS]))):
        for yi, (name, fam) in zip(y, rows):
            d = sel(det[(det.model == name) & (det.family == fam)])
            n = len(d)
            left = 0.0
            for c in CLASSES:
                share = 100 * (d['class'] == c).mean() if n else 0.0
                if share > 0:
                    ax.barh(yi, share, left=left, height=0.7, color=CLASS_COLOR[c], edgecolor=SURFACE, linewidth=1.5)
                    if share >= 7:
                        ax.text(left + share / 2, yi, f'{(d["class"] == c).sum()}', ha='center', va='center',
                                fontsize=6.5, color=CLASS_INK[c])
                    left += share
            ax.text(101, yi, f'n = {n}', va='center', fontsize=6.5, color=MUTED)
        for k in range(1, len(models)):
            ax.axhline(y[3 * k] + 0.5, color=AXIS, linewidth=0.6)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel('% of the pairs')
        ax.set_title(title, loc='left')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.5)
    fig.suptitle(f'Classification of every attacked layer × strength at {THR_LABEL} ({SCALE} attacked, '
                 f'{grid_name})', x=0.01, ha='left', fontweight='bold')
    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor=AXIS, linewidth=0.5, label=c) for c in CLASSES]
    fig.tight_layout()
    footer(fig, handles, 4,
           f'Numbers in the bars = count of pairs. Classes as in figures 03: over {N_SPLITS} random splits of the test '
           'set into a clean reference half and an attacked half.\nRight panel: only the pairs whose configuration was '
           f'in the campaign have an ASR ({n_asr} of {len(det)} pairs).')
    PR.save(fig, out / '05_stealthy_pairs_summary.png')


def cfg_label(tag):
    fam, s, g = parse_tag(tag)
    if s != s:
        return f'{FAM_SHORT[fam]} {SYM[fam]}*' + (' [power]' if tag.endswith('_pow') else '')
    return f'{FAM_SHORT[fam]} {SYM[fam]} {s:g}' + (f' γ {g:g}' if fam == 'mgna' and g != 1 else '')


def pareto(m):
    import onnx
    from ..taps import attack_units, load_genes
    run = m['run_dir']
    sr = run / 'search_results.json'
    s = json.loads(sr.read_text())
    genes, placements = load_genes(str(sr))
    model = onnx.load(str(next(run.glob('format_*.onnx'))))
    units, per_gene = attack_units(model, genes, placements, marker=DUMMY_MARKER, semantics=SEMANTICS)
    del model
    key = lambda c, p: (tuple(int(x) for x in c), tuple(p))
    reps = {key(r['cuts'], r['placement']): r for r in s.get('repeats', [])}
    known = set(m['units']['layer_name'])
    add = [(u['name'], u['tap']) for u in units if u['name'] not in known and u['tap'] in set(m['taps'])]
    if add:
        m['units'] = pd.concat([m['units'], pd.DataFrame(add, columns=['layer_name', 'tensor'])], ignore_index=True)
        known |= {n for n, _ in add}
    lost = sorted({t for info in per_gene.values() for t in info['taps'] if t not in known})
    if lost:
        print(f"  [{m['name']}] {len(lost)} cut unit(s) without feature-map statistics left out: {lost}")
    rows = []
    for i, (cuts, place, obj) in enumerate(zip(s['best_cuts'], s['best_placements'], s['best_objectives']), 1):
        r = reps.get(key(cuts, place), {})
        rows.append(dict(model=m['name'], gene=i, cuts=list(cuts), placement=list(place),
                         throughput=r.get('throughput_mean', obj[0]), memory=r.get('memory_mean', obj[1]),
                         layers=[t for t in per_gene[i]['taps'] if t in known]))
    return pd.DataFrame(rows)


def fragility(m, det, genes):
    d = det[det.model == m['name']]
    use = {}
    for g, lays in zip(genes['gene'], genes['layers']):
        for l in lays:
            use.setdefault(l, []).append(g)
    rows = []
    for depth, lay in enumerate(m['units']['layer_name']):
        g = d[d.layer_name == lay]
        ga = g[g['asr'].notna()]
        st = g[g['class'].str.startswith('stealthy')]
        sa = st[st['asr'].notna()]
        top = ga.loc[ga['asr'].idxmax()] if len(ga) else None
        stop = sa.loc[sa['asr'].idxmax()] if len(sa) else None
        max_asr = top['asr'] if top is not None else np.nan
        st_asr = stop['asr'] if stop is not None else (0.0 if len(ga) else np.nan)
        noise = g['noise_band'].iloc[0]
        tau = g['threshold'].iloc[0]
        row = dict(
            model=m['name'], layer_name=lay, depth=depth, tensor=g['tensor'].iloc[0],
            pareto_genes=use.get(lay, []), n_pareto_genes=len(use.get(lay, [])),
            natural_noise_q99=noise, threshold=tau,
            max_asr=max_asr, max_asr_config=cfg_label(top['config_tag']) if top is not None else '',
            max_asr_footprint=top['either_median'] if top is not None else np.nan,
            max_asr_amp_rel=top['amp_rel'] if top is not None else np.nan,
            stealthy_asr=st_asr, stealthy_asr_config=cfg_label(stop['config_tag']) if stop is not None else '',
            stealthy_asr_family=stop['family'] if stop is not None else '',
            stealthy_asr_footprint=stop['either_median'] if stop is not None else np.nan,
            stealthy_asr_amp_rel=stop['amp_rel'] if stop is not None else np.nan,
            stealthy_footprint_over_noise=stop['either_median'] / noise if stop is not None else np.nan,
            n_stealthy_configs=len(st), n_configs=len(g), n_configs_with_asr=len(ga),
            n_stealthy_successful=int((sa['asr'] >= SUCCESS).sum()),
            fragility=('fragile: stealthy and successful' if st_asr >= SUCCESS else
                       'vulnerable but visible' if max_asr >= SUCCESS else
                       'robust to the tested attacks' if len(ga) else 'no ASR'))
        for fam in PR.FAMILIES:
            sf = st[st.family == fam]
            sfa = sf[sf['asr'].notna()]
            best = (sfa.loc[sfa['asr'].idxmax()] if len(sfa) else
                    sf.loc[sf['amp_rel'].idxmax()] if len(sf) else None)
            row.update({f'{fam}_n_stealthy': len(sf), f'{fam}_n_stealthy_with_asr': len(sfa),
                        f'{fam}_n_configs_with_asr': int((ga.family == fam).sum()),
                        f'{fam}_stealthy_config': cfg_label(best['config_tag']) if best is not None else '',
                        f'{fam}_stealthy_asr': best['asr'] if best is not None else np.nan,
                        f'{fam}_stealthy_amp_rel': best['amp_rel'] if best is not None else np.nan,
                        f'{fam}_stealthy_footprint': best['either_median'] if best is not None else np.nan})
        rows.append(row)
    lt = pd.DataFrame(rows)
    lti = lt.set_index('layer_name')
    grows = []
    for _, r in genes.iterrows():
        sub = lti.loc[[l for l in r['layers'] if l in lti.index]]
        worst = sub['stealthy_asr'].idxmax() if sub['stealthy_asr'].notna().any() else None
        fams = {}
        for fam in PR.FAMILIES:
            v = sub[f'{fam}_stealthy_asr'].where(sub[f'{fam}_n_stealthy_with_asr'] > 0, 0.0) \
                .where(sub[f'{fam}_n_configs_with_asr'] > 0)
            fams.update({f'{fam}_mean_stealthy_asr': v.mean(), f'{fam}_worst_stealthy_asr': v.max()})
        grows.append(dict(
            model=m['name'], gene=r['gene'], throughput=r['throughput'], memory=r['memory'],
            cuts=' '.join(map(str, r['cuts'])), placement=' '.join(r['placement']),
            layers=' | '.join(r['layers']), n_layers=len(sub),
            n_fragile_layers=int((sub['stealthy_asr'] >= SUCCESS).sum()),
            worst_stealthy_asr=sub['stealthy_asr'].max(), mean_stealthy_asr=sub['stealthy_asr'].mean(),
            worst_layer=worst or '',
            worst_config=sub.loc[worst, 'stealthy_asr_config'] if worst else '',
            max_asr=sub['max_asr'].max(), **fams))
    return lt, pd.DataFrame(grows)


UNK_LO = -24
UNK_X = {'mfgsm': -19, 'mgna': -12.5, 'mjsma': -6}
FRAG_STYLE = {'fragile: stealthy and successful': INK, 'vulnerable but visible': INK2,
              'robust to the tested attacks': MUTED, 'no ASR': MUTED}


def fig_fragile_layers(models, lt, out):
    rows = max(len(m['units']) for m in models)
    ncols = min(3, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, (0.17 * rows + 1.4) * nrows), squeeze=False)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax, m in zip(axes.flat, models):
        g = lt[lt.model == m['name']].sort_values('depth')
        y = np.arange(len(g))
        frag = g['stealthy_asr'] >= SUCCESS
        for yi in y[frag.to_numpy()]:
            ax.axhspan(yi - 0.5, yi + 0.5, color=AXIS, alpha=0.35, linewidth=0, zorder=0)
        ax.axvline(SUCCESS, color=INK, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
        ax.hlines(y, g['stealthy_asr'], g['max_asr'], color=AXIS, linewidth=1.2, zorder=2)
        ax.scatter(g['max_asr'], y, s=26, facecolor=SURFACE, edgecolor=INK2, linewidth=1.1, zorder=3)
        ax.axvspan(UNK_LO, -2, color=AXIS, alpha=0.18, linewidth=0, zorder=0)
        ax.annotate('stealthy,\nASR unknown', ((UNK_LO - 2) / 2, -0.6), ha='center', va='bottom', fontsize=6,
                    color=MUTED, annotation_clip=False)
        for fam in PR.FAMILIES:
            asr = g[f'{fam}_stealthy_asr'].to_numpy()
            has = (g[f'{fam}_n_stealthy'] > 0).to_numpy()
            known = has & np.isfinite(asr)
            ax.scatter(asr[known], y[known], s=30, color=FAM_COLOR[fam], marker=PR.FAM_MARKER[fam],
                       edgecolor=SURFACE, linewidth=0.6, zorder=5 if fam == 'mfgsm' else 4)
            unk = has & ~np.isfinite(asr)
            ax.scatter(np.full(unk.sum(), UNK_X[fam]), y[unk], s=24, facecolor=SURFACE, edgecolor=FAM_COLOR[fam],
                       marker=PR.FAM_MARKER[fam], linewidth=1.1, zorder=4)
        none = (g['stealthy_asr_family'] == '').to_numpy() & g['stealthy_asr'].notna().to_numpy()
        ax.scatter(g['stealthy_asr'][none], y[none], s=14, color=MUTED, marker='|', zorder=4)
        names = [PR.short_layer(l.split('@')[0], 26) for l in g['layer_name']]
        dup = {x for x in names if names.count(x) > 1}
        names = [f"{x} @{l.split('@')[1]}" if x in dup and '@' in l else x for x, l in zip(names, g['layer_name'])]
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.5)
        for lab, f in zip(ax.get_yticklabels(), frag):
            lab.set_color(INK if f else INK2)
            lab.set_fontweight('bold' if f else 'normal')
        for yi, n, nz in zip(y, g['n_pareto_genes'], g['threshold' if calibrated() else 'natural_noise_q99']):
            ax.text(103, yi, f'×{n}', va='center', fontsize=6, color=MUTED)
            ax.text(111, yi, f'{nz:.2f}', va='center', fontsize=6, color=MUTED)
        for yi, x, a in zip(y, g['stealthy_asr'], g['stealthy_asr_amp_rel']):
            if np.isfinite(a) and x >= SUCCESS:
                ax.annotate(f'{a:.2g}', (x, yi), xytext=(-6, 0), textcoords='offset points', ha='right',
                            va='center', fontsize=5.5, color=INK2)
        for x, lab in ((103, '×n'), (111, 'τ %' if calibrated() else 'noise\nq99 %')):
            ax.annotate(lab, (x, -0.6), ha='left', va='bottom', fontsize=6, color=MUTED, annotation_clip=False)
        ax.set_ylim(len(g) - 0.5, -0.5)
        ax.set_xlim(UNK_LO - 1, 122)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.grid(axis='y', visible=False)
        nf = int(frag.sum())
        ax.set_title(f"{m['pretty']}: {nf}/{len(g)} fragile layers", loc='left', pad=22)
        ax.set_xlabel('attack success rate of the campaign (%)')
    handles = [Line2D([], [], linestyle='', marker='o', markerfacecolor=SURFACE, markeredgecolor=INK2, markersize=6,
                      label='strongest attack on the layer (any configuration)')]
    handles += [Line2D([], [], linestyle='', marker=PR.FAM_MARKER[f], color=FAM_COLOR[f], markersize=6,
                       label=f'{FAM_SHORT[f]}: strongest attack stealthy at {THR_LABEL} (at its ASR)')
                for f in PR.FAMILIES]
    handles += [Line2D([], [], linestyle='', marker=PR.FAM_MARKER[f], markerfacecolor=SURFACE,
                       markeredgecolor=FAM_COLOR[f], markersize=6,
                       label=f'{FAM_SHORT[f]}: stealthy, ASR unknown (configuration not in the campaign)')
                for f in PR.FAMILIES]
    handles += [Patch(facecolor=AXIS, alpha=0.35, label=f'fragile layer: a stealthy attack reaches ASR ≥ {SUCCESS:g} % (bold)')]
    fig.suptitle(f'Fragile layers of the Pareto deployments: attack success vs stealth ({SCALE} '
                 f'attacked, {THR_LABEL})', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    n_asr = ', '.join(f"{m['pretty'].split(' · ')[0]} {m['n_cfg_asr']}/{len(m['tags'])}" for m in models)
    footer(fig, handles, 4,
           'One row per attacked layer (every tensor that a Pareto deployment of the search cuts), in depth order; ×n = '
           'number of Pareto deployments that cut there; ' + ('τ = calibrated threshold of the layer (%)' if calibrated()
           else 'noise q99 = natural deviation\nbetween two clean halves of the layer (%)') + '. Number left of a fragile marker = injected noise of that stealthy attack (amp_rel = perturbation '
           'RMS / feature-map std).\nA gap between the two markers = the strongest attacks leave a visible footprint. '
           'ASR comes from the GPU campaign and only exists for the configurations it '
           f'ran (configurations with ASR: {n_asr});\nstealth is measured here for every configuration. Layers without '
           'any stealthy configuration with an ASR are drawn at 0.')
    PR.save(fig, out / '06_fragile_layers.png')


def fig_pareto_fragility(models, gt, out):
    fig, axes = grid(len(models), w=4.4, h=3.6)
    axes = list(axes)
    for ax, m in zip(axes, models):
        g = gt[gt.model == m['name']].sort_values('throughput')
        ax.step(g['throughput'], g['memory'], where='post', color=AXIS, linewidth=1.0, zorder=1)
        known = g['worst_stealthy_asr'].notna()
        sc = ax.scatter(g.loc[known, 'throughput'], g.loc[known, 'memory'], c=g.loc[known, 'worst_stealthy_asr'],
                        cmap=PR.CMAP_MARK, vmin=0, vmax=100, s=70, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.scatter(g.loc[~known, 'throughput'], g.loc[~known, 'memory'], s=60, facecolor=SURFACE, edgecolor=MUTED,
                   zorder=3)
        frag = g['n_fragile_layers'] > 0
        ax.scatter(g.loc[frag, 'throughput'], g.loc[frag, 'memory'], s=150, facecolor='none', edgecolor=INK,
                   linewidth=1.0, zorder=2)
        PR.label_points(ax, g['throughput'].to_numpy(), g['memory'].to_numpy(), [f'g{x}' for x in g['gene']])
        n = int(frag.sum())
        ax.set_title(f"{m['pretty']}\n{n}/{len(g)} deployments with a fragile cut", loc='left')
        ax.set_xlabel('throughput (img/s)')
        ax.set_ylabel('max memory per rank (MB)')
    fig.suptitle('Pareto deployments (throughput vs memory) coloured by their exposure to a stealthy successful attack',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cb = fig.colorbar(sc, cax=fig.add_axes((0.935, 0.3, 0.012, 0.45)))
    cb.set_label('worst stealthy ASR over the cuts (%)')
    cb.outline.set_visible(False)
    handles = [Line2D([], [], linestyle='', marker='o', markerfacecolor='none', markeredgecolor=INK, markersize=11,
                      label=f'at least one cut is a fragile layer (stealthy attack with ASR ≥ {SUCCESS:g} %)')]
    footer(fig, handles, 1,
           'One point per Pareto solution of the search (g = gene number; measured throughput and memory, mean of the '
           f'repeats). Colour = the highest ASR reached by an attack that stays stealthy at {THR_LABEL}\non any of the layers '
           'the deployment cuts (0 = every successful attack is visible); details in tables/6_pareto_deployments.csv.')
    PR.save(fig, out / '07_pareto_fragility.png')


ROBUSTNESS = {
    'mean_stealthy_asr': ('08_pareto_3d_robustness_mean', 'mean over its cuts', None),
    'worst_stealthy_asr': ('09_pareto_3d_robustness_worst_cut', 'worst cut', None),
    **{f'{f}_{k}_stealthy_asr': (f'{n}_pareto_3d_robustness_{f}_{k}' + ('_cut' if k == 'worst' else ''), how, f)
       for k, n, how in (('mean', 11, 'mean over its cuts'), ('worst', 12, 'worst cut')) for f in PR.FAMILIES},
}


def fig_pareto_3d(models, gt, out, col):
    fname, how, fam = ROBUSTNESS[col]
    if not gt[col].notna().any():
        return
    who = f'stealthy {FAM_SHORT[fam]} attacks' if fam else 'stealthy attacks'
    n = len(models)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(5.0 * ncols, 4.6 * nrows))
    sc = None
    for i, m in enumerate(models):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        g = gt[gt.model == m['name']].sort_values('throughput')
        g = g[g[col].notna()]
        x, y, z = g['throughput'].to_numpy(), g['memory'].to_numpy(), 100 - g[col].to_numpy()
        for xi, yi, zi in zip(x, y, z):
            ax.plot([xi, xi], [yi, yi], [0, zi], color=AXIS, linewidth=0.8, zorder=1)
        ax.scatter(x, y, np.zeros_like(z), s=10, color=MUTED, depthshade=False)
        sc = ax.scatter(x, y, z, c=z, cmap=PR.CMAP_MARK, vmin=0, vmax=100, s=55, edgecolor=SURFACE,
                        linewidth=0.8, depthshade=False)
        for xi, yi, zi, gene in zip(x, y, z, g['gene']):
            ax.text(xi, yi, zi + 4, f'g{gene}', fontsize=7, color=INK2, ha='center')
        best = g.assign(rob=z).sort_values(['rob', 'throughput'], ascending=False).iloc[0] if len(g) else None
        ax.set_zlim(0, 105)
        ax.set_xlabel('throughput (img/s)', fontsize=8, labelpad=4)
        ax.set_ylabel('memory per rank (MB)', fontsize=8, labelpad=4)
        ax.set_zlabel('robustness (%)', fontsize=8, labelpad=2)
        ax.tick_params(labelsize=7, pad=1)
        ax.view_init(elev=22, azim=-58)
        sub = (f"most robust: g{best.gene}, {best.rob:.0f} % at {best.throughput:.0f} img/s"
               if best is not None else 'no ASR')
        ax.set_title(f"{m['pretty']}\n{sub}", loc='left', fontsize=9.5)
    fig.suptitle(f'Pareto deployments: throughput × memory × robustness to {who} ({how}, {SCALE})',
                 x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(rect=(0, 0, 0.93, 0.97))
    cb = fig.colorbar(sc, cax=fig.add_axes((0.945, 0.3, 0.011, 0.4)))
    cb.set_label(f'robustness = 100 − stealthy {FAM_SHORT[fam] + " " if fam else ""}ASR, {how} (%)')
    cb.outline.set_visible(False)
    att = f'an {FAM_SHORT[fam]} configuration' if fam else 'an attack'
    none = 'such configuration' if fam else 'attack'
    agg = (f'mean over the layers it cuts of the highest ASR reached there by {att} stealthy at {THR_LABEL}, whatever that '
           f'ASR (0 when no {none} stays stealthy on the layer)' if how.startswith('mean') else
           f'highest ASR reached by {att} stealthy at {THR_LABEL} on any layer it cuts, whatever that ASR (its weakest cut '
           'decides)')
    footer(fig, [], 1,
           'One point per Pareto solution of the search (g = gene number; measured throughput and memory, mean of the '
           f'repeats; grey dot = its projection on the floor).\nRobustness = 100 − stealthy ASR of the deployment, '
           f'with stealthy ASR = {agg}.\nOnly stealthy {FAM_SHORT[fam] + " " if fam else ""}attacks count: a visible '
           'attack is caught by the detector. '
           'Details in tables/6_pareto_deployments.csv.')
    PR.save(fig, out / f'{fname}.png')


def summarize_fragility(models, lt, gt):
    L = [f'Fragile layers of the Pareto deployments (stealthy at {THR_LABEL} AND campaign ASR ≥ {SUCCESS:g} %)', '=' * 72,
         'injected noise: amp_rel = RMS of the perturbation / feature-map std of the stealthy attack; footprint: '
         'max(|Δstd|, |Δmean|) of the attacked half; natural noise: 99th percentile of the same deviation between two '
         'clean halves of the layer.', '']
    for m in models:
        g = lt[lt.model == m['name']].sort_values('stealthy_asr', ascending=False)
        f = g[g['stealthy_asr'] >= SUCCESS]
        gg = gt[gt.model == m['name']]
        L.append(f"{m['pretty']}: {len(f)}/{len(g)} fragile layers; {int((gg.n_fragile_layers > 0).sum())}/{len(gg)} "
                 f"Pareto deployments cut at a fragile layer (ASR known for {m['n_cfg_asr']}/{len(m['tags'])} "
                 'configurations)')
        for fam in PR.FAMILIES:
            n, na = int(g[f'{fam}_n_stealthy'].sum()), int(g[f'{fam}_n_stealthy_with_asr'].sum())
            L.append(f"  {FAM_SHORT[fam]}: {n} stealthy (layer, configuration) pairs on "
                     f"{int((g[f'{fam}_n_stealthy'] > 0).sum())}/{len(g)} layers, {na} with a campaign ASR")
        for _, r in f.iterrows():
            L.append(f"  {PR.short_layer(r.layer_name, 44):44s} stealthy ASR {r.stealthy_asr:5.1f} % "
                     f"({r.stealthy_asr_config}, injected noise amp_rel {r.stealthy_asr_amp_rel:.2f}); footprint "
                     f"{r.stealthy_asr_footprint:.2f} % vs natural noise {r.natural_noise_q99:.2f} % "
                     f"(×{r.stealthy_footprint_over_noise:.1f}); max ASR {r.max_asr:5.1f} %; in "
                     f"{r.n_pareto_genes} deployment(s)")
            for fam in PR.FAMILIES:
                if r[f'{fam}_n_stealthy'] and fam != r.stealthy_asr_family:
                    a = r[f'{fam}_stealthy_asr']
                    L.append(f"  {'':44s}   also stealthy: {r[f'{fam}_stealthy_config']} (injected noise amp_rel "
                             f"{r[f'{fam}_stealthy_amp_rel']:.2f}, footprint {r[f'{fam}_stealthy_footprint']:.2f} %, "
                             + (f"ASR {a:.1f} %)" if a == a else 'ASR unknown)'))
        v = g[(g['fragility'] == 'vulnerable but visible')]
        if len(v):
            L.append('  vulnerable but visible: ' + ', '.join(PR.short_layer(x, 30) for x in v.layer_name))
        safe = gg[gg.n_fragile_layers == 0].sort_values('throughput', ascending=False)
        if len(safe):
            L.append('  deployments without a fragile cut: ' + ', '.join(
                f"g{r.gene} ({r.throughput:.1f} img/s, {r.memory:.0f} MB)" for _, r in safe.iterrows()))
        L.append('')
    return '\n'.join(L) + '\n'


def fig_thresholds(models, nat, out):
    fig, axes = grid(len(models), w=4.6, h=3.5)
    for ax, m in zip(axes, models):
        g = nat[nat.model == m['name']].reset_index(drop=True)
        x = np.arange(len(g))
        ax.plot(x, g['threshold'], color=INK, linewidth=1.6, marker='o', markersize=3.5, markeredgecolor=SURFACE,
                zorder=4)
        ax.plot(x, np.maximum(g['either_q99'], 1e-3), color=INK2, linewidth=1.0, linestyle=':', zorder=3)
        ax.scatter(x, np.maximum(g['order_either'], 1e-3), s=12, color=MUTED, marker='x', linewidth=0.8, zorder=2)
        ax.axhline(BAND, color=INK, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        ax.set_yscale('log')
        plain_log(ax, 'y')
        ax.set_xlim(-0.5, len(g) - 0.5)
        ax.set_xlabel('attacked tensor, shallow → deep')
        ax.set_ylabel('%')
        ax.set_title(f"{m['pretty']}\nτ {g.threshold.min():.2g}-{g.threshold.max():.2g} %, false alarms: median "
                     f"{100 * g.fpr_thr.median():.1f} %, worst {100 * g.fpr_thr.max():.1f} %", loc='left', fontsize=8.5)
    handles = [Line2D([], [], color=INK, linewidth=1.6, marker='o', markersize=4, label='calibrated threshold τ'),
               Line2D([], [], color=INK2, linewidth=1.0, linestyle=':',
                      label='natural noise of the clean evaluation windows (99th percentile)'),
               Line2D([], [], color=MUTED, linestyle='', marker='x', markersize=5,
                      label='worst window of consecutive images (dataset order: class drift)'),
               Line2D([], [], color=INK, linewidth=0.9, linestyle=(0, (4, 3)),
                      label=f'fixed band report.stealth_band (±{BAND:g} %), for comparison')]
    fig.suptitle(f"Calibrated detection threshold τ of every cut (target {100 * float(THR['false_alarm']):g} % false "
                 f"alarms, {SCALE})", x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    src = ' / '.join(sorted({m['calibration']['source'] for m in models}))
    footer(fig, handles, 2,
           f'τ is set on clean {src} only (bootstrap of windows of {THR["window"]} inferences against a re-drawn '
           f'reference); the attacks and the dotted line use the other test images. The title gives the measured\n'
           'false-alarm rate of the worst tensor on clean evaluation windows (the check of the calibration). A τ below '
           'the crosses: the monitor would alarm when the input classes drift.')
    PR.save(fig, out / '10_thresholds_per_cut.png')


def summarize(models, nat, det, bt, grid_name):
    L = []
    P = L.append
    P(f'Stealth-threshold study ({SCALE}, {grid_name})')
    P('=' * 72)
    if calibrated():
        src = ('a separate validation image set' if THR['source'] == 'validation' else
               f"{100 * float(THR['calibration']):g} % of the test images, held out (never attacked)")
        P(f"Threshold calibrated per model and cut on clean images only ({src}): τ = the "
          f"{100 * (1 - float(THR['false_alarm'])):g}th percentile of max(|Δstd|, |Δmean|) of clean windows of "
          f"{THR['window']} inferences vs the reference" + (', at least the largest deviation of windows of '
          'consecutive images (drift)' if THR['drift'] else '') + (f", × {THR['margin']:g}" if float(THR['margin'])
          != 1 else '') + f'. The attacks are evaluated on the other test images: {N_SPLITS} windows of '
          f"{THR['window']} attacked inferences per (layer, configuration). Strong perturbation: amp_rel ≥ {STRONG:g}.")
    elif windowed():
        P(f"Fixed band {THR_LABEL} on every cut, monitored over windows of {THR['window']} inferences against the "
          'reference of clean calibration images (' + ('a separate validation image set' if THR['source'] == 'validation'
          else f"{100 * float(THR['calibration']):g} % of the test images, held out") + f'); the attacks are evaluated '
          f"on the other test images: {N_SPLITS} windows of {THR['window']} attacked inferences per (layer, "
          f'configuration), the same windows as the calibrated mode. Flag: |Δstd| or |Δmean| > band. Strong '
          f'perturbation: amp_rel ≥ {STRONG:g}.')
    else:
        P(f'{N_SPLITS} random splits of the test set into a clean reference half A and a monitored half B (clean or '
          f'attacked). Flag: |Δstd| or |Δmean| > band. Strong perturbation: amp_rel ≥ {STRONG:g}.')
    P('')
    for m in models:
        n = m['name']
        g, d, b = nat[nat.model == n], det[det.model == n], bt[bt.model == n]
        if windowed():
            c = m['calibration']
            P(f"{m['pretty']}: {len(m['taps'])} tensors, {len(d)} (layer, configuration) pairs; reference "
              f"{'and τ ' if calibrated() else ''}from "
              f"{c['n_cal']} clean {c['source']}, attacks on {c['n_eval']} evaluation images"
              + (f" ({c['repeats']} random partitions" + (', τ = their mean)' if calibrated() else ')')
                 if c['repeats'] > 1 else '')
              + f"; campaign ASR for {m['n_cfg_asr']}/{len(m['tags'])} configurations")
            if c['small']:
                P(f"  WARNING: {c['n_cal']} calibration images < 10 × the window ({THR['window']}): the noise of the "
                  'reference itself is not negligible against that of a window, so the false-alarm rate of one deployed '
                  'reference can be well above the target; use more calibration images or a smaller window')
            if calibrated() and g.fpr_thr.median() > 5 * float(THR['false_alarm']):
                P(f"  WARNING: clean evaluation windows raise {100 * g.fpr_thr.median():.0f} % false alarms (median "
                  f"tensor) for a target of {100 * float(THR['false_alarm']):g} %: the calibration images do not "
                  'represent the test images (other classes or scenes); calibrate on a representative set')
            P((f"  threshold τ per tensor: {g.threshold.min():.2f} - {g.threshold.max():.2f} % (median "
               f"{g.threshold.median():.2f} %)" if calibrated() else f'  band {THR_LABEL} on every tensor') +
              f"; measured false alarms on clean evaluation windows: {100 * g.fpr_thr.median():.2f} % (median tensor), "
              f"{100 * g.fpr_thr.max():.2f} % (worst tensor)" + (f", target {100 * float(THR['false_alarm']):g} %"
                                                                  if calibrated() else ''))
            P(f"  natural, random windows: q99 |Δstd| ≤ {g.std_q99.max():.2f} %, |Δmean| ≤ {g.mean_q99.max():.2f} % "
              f"(worst tensor; median tensor {g.std_q99.median():.2f} / {g.mean_q99.median():.2f} %)")
            P(f"  natural, consecutive images (dataset order): worst window {g.order_either.max():.2f} % (median tensor "
              f"{g.order_either.median():.2f} %); tensors beyond their τ: {int(g.order_flag_thr.sum())}/{len(g)}"
              + ('' if THR['drift'] or not calibrated() else ' (τ not calibrated for drift: set report.threshold.drift)'))
        else:
            P(f"{m['pretty']}: {len(m['taps'])} tensors, {len(d)} (layer, configuration) pairs, halves of {m['n'] // 2} "
              f"images; campaign ASR for {m['n_cfg_asr']}/{len(m['tags'])} configurations")
            P(f"  natural, random halves: q99 |Δstd| ≤ {g.std_q99.max():.2f} %, |Δmean| ≤ {g.mean_q99.max():.2f} % "
              f"(worst tensor; median tensor {g.std_q99.median():.2f} / {g.mean_q99.median():.2f} %); max over the "
              f"splits {g.either_max.max():.2f} %")
            P(f"  natural, dataset-order halves: |Δstd| ≤ {g.order_std.max():.2f} %, |Δmean| ≤ {g.order_mean.max():.2f} % "
              f"(median tensor {g.order_either.median():.2f} %); tensors beyond {THR_LABEL}: "
              f"{int(g.order_flag_thr.sum())}/{len(g)}")
            P(f"  false alarms at {THR_LABEL}, random halves: {100 * g.fpr_thr.max():.2f} % (worst tensor)")
        cnt = d['class'].value_counts()
        P(f'  classes at {THR_LABEL}: ' + ', '.join(f'{c} {int(cnt.get(c, 0))}' for c in CLASSES))
        for fam in PR.FAMILIES:
            df = d[d.family == fam]
            st = df[df['class'].str.startswith('stealthy')]
            hi = st[st.asr >= SUCCESS]
            sg = df[strong(df)]
            P(f"    {FAM_SHORT[fam]}: flagged {100 * (df['class'] == 'flagged').mean():.0f} %, stealthy {len(st)}/{len(df)} "
              f"({(st['class'] == 'stealthy, above natural noise').sum()} above noise, "
              f"{(st['class'] == 'stealthy, within natural noise').sum()} within noise); stealthy with ASR ≥ {SUCCESS:g} %: "
              f"{len(hi)}" + (f" (best {hi.asr.max():.1f} %)" if len(hi) else '')
              + f"; strong pairs flagged {(sg['class'] == 'flagged').sum()}/{len(sg)}")
        ok = b.loc[b.fpr_worst <= 0.01, 'band_pct'].min()
        okd = b.loc[b.order_flagged_share == 0, 'band_pct']
        ctl = b.loc[(b.mgna_strong_detected_share.fillna(1) >= 1) & (b.mjsma_strong_detected_share.fillna(1) >= 1),
                    'band_pct']
        P(f"  fixed bands at this scale: ≤ 1 % false alarms (random {'windows' if windowed() else 'halves'}) from ±{ok:g} %; no tensor flagged by the dataset-order halves "
          f"from {'±' + format(okd.min(), 'g') + ' %' if len(okd) else 'no tested band'}; every strong mGNA and mJSMA "
          f"flagged up to {'±' + format(ctl.max(), 'g') + ' %' if len(ctl) else 'none'}; admissible: "
          + (', '.join(f'±{x:g} %' for x in b.loc[admissible(b), 'band_pct']) or 'none'))
        P('')
    return '\n'.join(L) + '\n'


def generate(runs, data, out, models=None, band=10.0, attack_dirs=('attack',), dummy_marker='DUMMY',
             grid_name='campaign attack grid', threshold=None, success=50.0, semantics='device'):
    global BAND, BANDS, DUMMY_MARKER, THR, THR_LABEL, SCALE, SUCCESS, SEMANTICS
    BAND, DUMMY_MARKER, SUCCESS, SEMANTICS = float(band), dummy_marker, float(success), semantics
    BANDS = sorted(set(SENSITIVITY_BANDS) | {BAND})
    THR = dict(THR, **(threshold or {}))
    if calibrated():
        THR_LABEL = f"τ per cut ({100 * float(THR['false_alarm']):g} % false alarms)"
        SCALE = f"windows of {THR['window']} inferences"
    elif windowed():
        THR_LABEL, SCALE = f'±{BAND:g} %', f"windows of {THR['window']} inferences"
    else:
        THR_LABEL, SCALE = f'±{BAND:g} %', 'half of the test set'
    runs, data, out = Path(runs), Path(data), Path(out)
    if models is None:
        models = sorted(p.name[:-len('_meta.json')] for p in data.glob('*_meta.json'))
    names = [n for n in models if (data / f'{n}_meta.json').exists() and (runs / n / 'search_results.json').exists()]
    if not names:
        raise RuntimeError(f'no feature-map statistics in {data} for a model of {runs}')
    models, nats, dets = [], [], []
    for n in names:
        m = load(n, data, runs, attack_dirs)
        nat, det = analyse_model(m, np.random.default_rng(0))
        for k in ('cs', 'cm', 'as', 'am'):
            m.pop(k)
        models.append(m), nats.append(nat), dets.append(det)
        print(f'  analysed {n}')
    nat, det = pd.concat(nats, ignore_index=True), pd.concat(dets, ignore_index=True)
    bt = band_table(models, nat, det)

    tables, figs = out / 'tables', out / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    nat.to_csv(tables / '1_natural_variability_halves.csv', index=False)
    det.to_csv(tables / '2_attacks_halves.csv', index=False)
    bt.to_csv(tables / '3_band_sensitivity_halves.csv', index=False)
    cols = ['model', 'layer_name', 'family', 'strength', 'gamma', 'amp_rel', 'asr', 'class', 'd_std_median',
            'd_mean_median', 'noise_band', 'threshold', 'flag_rate_thr', 'flag_rate_noise_band', 'campaign_d_std_pct',
            'campaign_d_mean_std_pct']
    det[det['class'].str.startswith('stealthy')].sort_values(['model', 'asr'], ascending=[True, False])[cols].to_csv(
        tables / '4_stealthy_pairs.csv', index=False)
    tcols = ['model', 'tensor', 'threshold', 'fpr_thr', 'either_q99', 'order_either', 'order_flag_thr', 'drift_calibration',
             'n_calibration', 'n_evaluation', 'calibration_source', 'half']
    nat[[c for c in tcols if c in nat]].rename(columns={
        'fpr_thr': 'false_alarm_measured', 'either_q99': 'natural_noise_q99', 'order_either': 'drift_worst_window',
        'order_flag_thr': 'drift_beyond_threshold', 'half': 'window'}).to_csv(tables / '7_thresholds.csv', index=False)

    fig_natural(models, nat, figs, grid_name)
    fig_footprint(models, nat, det, figs, grid_name)
    fig_layer_maps(models, nat, det, figs, grid_name)
    fig_bands(models, bt, figs, grid_name, {n: list(g.threshold) for n, g in nat.groupby('model')})
    fig_stealthy(models, det, figs, grid_name)

    lts, gts = [], []
    for m in models:
        lt, gt = fragility(m, det, pareto(m))
        lts.append(lt), gts.append(gt)
    lt, gt = pd.concat(lts, ignore_index=True), pd.concat(gts, ignore_index=True)
    lt.assign(pareto_genes=lt['pareto_genes'].map(lambda x: ' '.join(map(str, x)))).to_csv(
        tables / '5_fragile_layers.csv', index=False)
    gt.to_csv(tables / '6_pareto_deployments.csv', index=False)
    fig_fragile_layers(models, lt, figs)
    fig_pareto_fragility(models, gt, figs)
    for col in ROBUSTNESS:
        fig_pareto_3d(models, gt, figs, col)
    if calibrated():
        fig_thresholds(models, nat, figs)
    txt = summarize(models, nat, det, bt, grid_name) + '\n' + summarize_fragility(models, lt, gt)
    (out / 'summary.txt').write_text(txt)
    return txt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', type=Path, help='runs folder: <model>/search_results.json and the attack results (ASR)')
    ap.add_argument('--data', type=Path, help='feature-map statistics (default RUNS/report/stealth/data)')
    ap.add_argument('-o', '--out', type=Path, help='output folder (default RUNS/report/stealth)')
    ap.add_argument('--models', nargs='+', help='default: every model with statistics in --data')
    ap.add_argument('--band', type=float, default=10.0, help='stealth tolerance in %% of the clean std (default 10)')
    ap.add_argument('--success-asr', type=float, default=50.0,
                    help='ASR (%%) from which an attack is successful: fragile layer = stealthy AND successful (50)')
    ap.add_argument('--calibrated', action='store_true', help='one threshold per model and cut (see the docstring)')
    ap.add_argument('--window', type=int, default=1000, help='calibrated: inferences per window (default 1000)')
    ap.add_argument('--false-alarm', type=float, default=0.01, help='calibrated: target false-alarm rate (0.01)')
    ap.add_argument('--attack-dirs', nargs='+', default=['attack'],
                    help='attack output folders of each model, merged in this order (a later one wins)')
    args = ap.parse_args()
    out = args.out or args.runs / 'report' / 'stealth'
    th = dict(mode='calibrated' if args.calibrated else 'fixed', window=args.window, false_alarm=args.false_alarm)
    print(generate(args.runs, args.data or out / 'data', out, args.models, args.band, args.attack_dirs, threshold=th,
                   success=args.success_asr))


if __name__ == '__main__':
    main()
