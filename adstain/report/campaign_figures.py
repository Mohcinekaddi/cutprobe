#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
BLUES = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
         '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
CMAP_MARK = LinearSegmentedColormap.from_list('seq_blue_marks', BLUES[3:])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'text.color': INK,
    'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8.5,
    'legend.frameon': False, 'lines.linewidth': 1.6, 'lines.markersize': 5.5,
})

FAMILIES = {
    'mfgsm': ('mFGSM (sign step to dissimilar images)', 'eps_rel', 'ε'),
    'mgna': ('mGNA (Gaussian noise)', 'sigma_rel', 'σ'),
    'mjsma': ('mJSMA (zero strongest activations)', 'theta', 'θ'),
}
FAM_MARKER = {'mfgsm': 'o', 'mgna': 's', 'mjsma': 'D'}
PRETTY = {
    'vgg19_cifar10': 'VGG19 · CIFAR-10', 'yolo11n_coco': 'YOLO11n · COCO', 'yolo26n_coco': 'YOLO26n · COCO',
    'resnet152_imagenet': 'ResNet152 · ImageNet', 'vgg19_imagenet_caffe2': 'VGG19-Caffe2 · ImageNet',
    'densenet_imagenet': 'DenseNet · ImageNet',
}
STEALTH_SHARE = 0.5
STEALTH_STATS = {
    'std': ('d_std_pct', 'Δstd', 'change of feature-map std under attack (% of clean std)'),
    'mean': ('d_mean_std_pct', 'Δmean', 'shift of feature-map mean under attack (% of clean std)'),
}
CRITERIA = list(STEALTH_STATS) + ['both']


def pretty(name):
    return PRETTY.get(name, name.replace('_', ' '))


def short_layer(name, width=34):
    s = re.sub(r'_output_0', '', name).lstrip('_')
    s = re.sub(r'^model\.', '', s)
    s = re.sub(r'^resnetv\d+_', '', s)
    return s if len(s) <= width else s[:width // 2 - 1] + '…' + s[-(width // 2 - 1):]


def fmt_dur(hours):
    if hours is None or np.isnan(hours):
        return '–'
    mins = int(round(hours * 60))
    return f'{mins} min' if mins < 60 else f'{mins // 60} h {mins % 60:02d}'


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def label_points(ax, xs, ys, labels, min_px=16, **kw):
    ax.autoscale_view()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    groups = []
    for i in np.argsort(pts[:, 0]):
        for grp in groups:
            if np.hypot(*(pts[i] - pts[grp[0]])) < min_px:
                grp.append(i)
                break
        else:
            groups.append([i])
    style = dict(dict(fontsize=6.5, color=MUTED, xytext=(4, 3), textcoords='offset points'), **kw)
    for grp in groups:
        names = [labels[i] for i in grp]
        txt = names[0] if len(names) == 1 else 'g' + ','.join(n.lstrip('g') for n in names)
        ax.annotate(txt, (xs[grp[0]], ys[grp[0]]), **style)


def family_info(fam):
    return FAMILIES.get(fam, (fam, None, 'strength'))


def family_color(fam, fams):
    return CAT[fams.index(fam) % len(CAT)]


def crossing(amp, asr, level=50.0):
    if len(asr) == 0:
        return np.nan, 'not_reached'
    if asr[0] >= level:
        return amp[0], 'upper'
    for i in range(1, len(asr)):
        if asr[i] >= level:
            t = (level - asr[i - 1]) / (asr[i] - asr[i - 1])
            return amp[i - 1] + t * (amp[i] - amp[i - 1]), 'reached'
    return np.nan, 'not_reached'


STATUS_FILES = ('full_pipeline_status.txt', 'pipeline_status.txt')


def parse_status(paths):
    pat = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\S+) (search|attack) (start|done|FAILED)')
    events = []
    for path in paths:
        if path.exists():
            events += sorted((mm.groups() for mm in map(pat.match, path.read_text().splitlines()) if mm),
                             key=lambda e: e[0])
    ev = {}
    for t, model, phase, what in events:
        ph = ev.setdefault(model, {}).setdefault(phase, {})
        if what == 'start':
            ph.clear()
        ph[what] = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
    return ev


def load_model(d, attack_dirs=('attack',)):
    sr = d / 'search_results.json'
    if not sr.exists():
        return None
    s = json.loads(sr.read_text())
    m = {'name': d.name, 'pretty': pretty(d.name), 'task': s.get('task', 'classification'), 'search': s}
    am = d / 'analytical_model.json'
    m['model_len'] = json.loads(am.read_text()).get('model_len') if am.exists() else None

    key = lambda e: (tuple(e['cuts']), tuple(e['placement']))
    reps = {key(r): r for r in s.get('repeats', [])}
    vals = {key(v): v for v in s.get('validation', [])}
    infos = {}
    for sub in attack_dirs:
        for f in sorted((d / sub).glob('gene_*/gene_info.json')):
            g = json.loads(f.read_text())
            infos.setdefault(g['gene_id'], g)
    rows = []
    for i, (cuts, place, obj) in enumerate(zip(s['best_cuts'], s['best_placements'], s['best_objectives']), 1):
        r, v = reps.get((tuple(cuts), tuple(place)), {}), vals.get((tuple(cuts), tuple(place)), {})
        rows.append(dict(
            gene=i, cuts=list(cuts), placement=list(place), taps=infos.get(i, {}).get('taps', []),
            tp_first=obj[0], mem_first=obj[1],
            tp=r.get('throughput_mean', obj[0]), tp_std=r.get('throughput_std', 0.0),
            mem=r.get('memory_mean', obj[1]), mem_std=r.get('memory_std', 0.0),
            tp_pred=v.get('throughput_predicted', np.nan), mem_pred=v.get('memory_predicted', np.nan)))
    m['genes'] = pd.DataFrame(rows)
    m['explored'] = pd.DataFrame([dict(tp=h['throughput'], mem=h['max_rank_memory'])
                                  for h in s.get('raw_history', [])])

    layer_cut = {}
    for g in infos.values():
        for k, t in enumerate(g['taps']):
            layer_cut.setdefault(t, g['gene_values'][k + 1])

    frames = [pd.read_csv(c) for c in (d / sub / 'attack_summary_gpu.csv' for sub in attack_dirs) if c.exists()]
    m['attack'] = None
    if frames:
        a = pd.concat(frames, ignore_index=True).drop_duplicates(['layer_name', 'config_tag'], keep='last')
        order = list(dict.fromkeys(frames[0]['layer_name']))
        a = a.assign(_o=a['layer_name'].map({l: i for i, l in enumerate(order)})).sort_values(
            '_o', kind='stable').drop(columns='_o').reset_index(drop=True)
        a['family'] = a['attack']
        a['strength'] = [row[family_info(f)[1]] if family_info(f)[1] in a.columns else np.nan
                         for f, (_, row) in zip(a['family'], a.iterrows())]
        a['asr'] = a['attack_success_rate_pct']
        a['d_mean_std_pct'] = 100 * a['d_mean_sigma'] if 'd_mean_sigma' in a else np.nan
        a['cut'] = a['layer_name'].map(layer_cut)
        a['depth'] = a['cut'] / m['model_len'] if m['model_len'] else np.nan
        m['attack'] = a
        m['layers'] = list(dict.fromkeys(a['layer_name']))
        fams = list(dict.fromkeys(a['family']))
        m['families'] = [f for f in FAMILIES if f in fams] + [f for f in fams if f not in FAMILIES]
        if m['task'] == 'detection':
            m['baseline'] = f"mAP@[.5:.95] {a['map50_95_baseline'].iloc[0]:.2f} %, mAP@.5 {a['map50_baseline'].iloc[0]:.2f} %"
        else:
            m['baseline'] = f"top-1 {a['accuracy_baseline_pct'].iloc[0]:.2f} %"
        m['layer_table'] = layer_table(m)
        m['genes'] = gene_security(m)
    return m


def layer_table(m):
    a = m['attack']
    rows = []
    for layer in m['layers']:
        g = a[a.layer_name == layer]
        row = dict(model=m['name'], layer=layer, cut=g['cut'].iloc[0], depth=g['depth'].iloc[0],
                   mean_asr=g['asr'].mean(), max_asr=g['asr'].max())
        for fam in m['families']:
            gf = g[g.family == fam].sort_values('strength')
            a50, state = crossing(gf['amp_rel'].to_numpy(), gf['asr'].to_numpy())
            row.update({f'{fam}_mean_asr': gf['asr'].mean(), f'{fam}_a50': a50, f'{fam}_a50_state': state,
                        f'{fam}_amp_min': gf['amp_rel'].min(), f'{fam}_amp_max': gf['amp_rel'].max()})
        rows.append(row)
    return pd.DataFrame(rows).set_index('layer', drop=False)


def runs_of(values, flags):
    out, prev = [], False
    for v, f in zip(values, flags):
        if f:
            if prev:
                out[-1][1] = v
            else:
                out.append([v, v])
        prev = f
    return [tuple(r) for r in out]


def fmt_intervals(iv):
    return ' ∪ '.join(f'{{{lo:g}}}' if lo == hi else f'[{lo:g}, {hi:g}]' for lo, hi in iv) or 'none'


def within(df, band, crit):
    cols = [STEALTH_STATS[c][0] for c in (STEALTH_STATS if crit == 'both' else [crit])]
    return (df[cols].abs() <= band).all(axis=1)


def stealth_table(m, bands):
    a, rows = m['attack'], []
    for band in bands:
        for crit in CRITERIA:
            for fam in m['families']:
                for s, g in a[a.family == fam].groupby('strength'):
                    ok = within(g, band, crit)
                    row = dict(model=m['name'], band_pct=band, criterion=crit, family=fam, strength=s,
                               n_tensors=len(g), stealthy_share=ok.mean())
                    for stat, (col, _, _) in STEALTH_STATS.items():
                        row.update({f'd_{stat}_median': g[col].median(), f'd_{stat}_min': g[col].min(),
                                    f'd_{stat}_max': g[col].max()})
                    row.update(asr_median=g['asr'].median(), asr_stealthy_median=g.loc[ok, 'asr'].median(),
                               asr_stealthy_max=g.loc[ok, 'asr'].max())
                    rows.append(row)
    t = pd.DataFrame(rows)
    t['stealthy'] = t['stealthy_share'] >= STEALTH_SHARE
    return t


def stealth_windows(m, band, crit='both'):
    st = m['stealth']
    st = st[(st.band_pct == band) & (st.criterion == crit)]
    return {f: runs_of(st.loc[st.family == f, 'strength'], st.loc[st.family == f, 'stealthy']) for f in m['families']}


def stealth_per_tensor(m, bands):
    a, lt, rows = m['attack'], m['layer_table'], []
    for band in bands:
        for crit in CRITERIA:
            for layer in m['layers']:
                for fam in m['families']:
                    g = a[(a.layer_name == layer) & (a.family == fam)].sort_values('strength')
                    ok = within(g, band, crit)
                    best = g.loc[g.loc[ok, 'asr'].idxmax()] if ok.any() else None
                    rows.append(dict(model=m['name'], band_pct=band, criterion=crit, layer=layer,
                                     cut=lt.loc[layer, 'cut'], family=fam,
                                     stealthy_strengths=fmt_intervals(runs_of(g['strength'], ok)),
                                     best_stealthy_asr=best['asr'] if best is not None else np.nan,
                                     best_stealthy_strength=best['strength'] if best is not None else np.nan))
    return pd.DataFrame(rows)


def gene_security(m):
    lt, genes = m['layer_table'], m['genes'].copy()
    vuln, mean_vuln, worst, a50 = [], [], [], {f: [] for f in m['families']}
    for taps in genes['taps']:
        taps = [t for t in taps if t in lt.index]
        if not taps:
            vuln.append(np.nan), mean_vuln.append(np.nan), worst.append('')
            for f in m['families']:
                a50[f].append(np.nan)
            continue
        sc = lt.loc[taps, 'mean_asr']
        vuln.append(sc.max())
        mean_vuln.append(sc.mean())
        worst.append(sc.idxmax())
        for f in m['families']:
            a50[f].append(lt.loc[taps, f'{f}_a50'].min())
    genes['vulnerability'] = vuln
    genes['worst_tap'] = worst
    genes['robustness_mean'] = 100 - np.asarray(mean_vuln)
    genes['robustness_worst'] = 100 - genes['vulnerability']
    for f in m['families']:
        genes[f'{f}_min_a50'] = a50[f]
    return genes


def fig_timeline(models, status, out):
    names = [m['name'] for m in models if m['name'] in status]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(names) + 1.4))
    hours = {}
    for y, n in enumerate(names):
        left = 0.0
        for k, phase in enumerate(['search', 'attack']):
            ph = status[n].get(phase, {})
            end = ph.get('done') or ph.get('FAILED')
            h = (end - ph['start']).total_seconds() / 3600 if end and 'start' in ph else np.nan
            hours[(n, phase)] = h
            if not np.isnan(h):
                ax.barh(y, h, left=left, height=0.62, color=CAT[k], edgecolor=SURFACE, linewidth=2,
                        label=phase if y == 0 else None)
                left += h
        lbl = f"{fmt_dur(hours[(n, 'search')])} + {fmt_dur(hours[(n, 'attack')])}"
        if any('FAILED' in status[n].get(p, {}) for p in ('search', 'attack')):
            lbl += '  (FAILED step)'
        ax.text(left + 0.05, y, lbl, va='center', fontsize=8, color=INK2)
    ax.set_yticks(range(len(names)), [pretty(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel('wall time (hours)')
    ax.grid(axis='y', visible=False)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.18)
    total = np.nansum(list(hours.values()))
    ax.set_title(f'Pipeline wall time per model (total {fmt_dur(total)})', loc='left')
    ax.legend(loc='lower right', ncol=2)
    save(fig, out / 'overview' / '00_pipeline_timeline.png')


GUIDE = """How to read the outputs
-----------------------
overview/00_pipeline_timeline        wall time of search and attack per model.
tables/layer_vulnerability           per attacked tensor: mean / max ASR, and per family the perturbation size (RMS
                                     in units of the feature map's std, amp_rel) needed for 50 % ASR. Low = fragile.
tables/gene_security                 per Pareto deployment: throughput, memory (re-measured and predicted), the
                                     weakest exposed tensor, robustness = 100 - mean ASR averaged over the exposed
                                     tensors (robustness_mean) or of the weakest one (robustness_worst).
tables/model_summary                 the key findings above, one row per model.
tables/stealth_by_strength           per stealth band x criterion x family x tested strength: share of the attacked
                                     tensors whose Δstd / Δmean (in % of the clean std) stay inside the band.
tables/stealth_per_tensor            the stealthy strengths of every attacked tensor.
"""


def write_tables_and_summary(models, status, bands, out):
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    lt = pd.concat([m['layer_table'] for m in models], ignore_index=True)
    lt.to_csv(out / 'tables' / 'layer_vulnerability.csv', index=False)
    pd.concat([m['stealth'] for m in models], ignore_index=True).to_csv(
        out / 'tables' / 'stealth_by_strength.csv', index=False)
    pd.concat([stealth_per_tensor(m, bands) for m in models], ignore_index=True).to_csv(
        out / 'tables' / 'stealth_per_tensor.csv', index=False)
    gcols = ['gene', 'cuts', 'placement', 'taps', 'tp', 'tp_std', 'tp_first', 'tp_pred', 'mem', 'mem_std',
             'mem_first', 'mem_pred', 'vulnerability', 'worst_tap', 'robustness_mean', 'robustness_worst']
    gt = pd.concat([m['genes'].assign(model=m['name'])[['model'] + gcols +
                    [c for c in m['genes'] if c.endswith('_min_a50')]] for m in models], ignore_index=True)
    gt.to_csv(out / 'tables' / 'gene_security.csv', index=False)

    lines, rows = [], []
    for m in models:
        g, lt_m, s = m['genes'], m['layer_table'], m['search']
        st = status.get(m['name'], {})
        dur = {p: ((st[p].get('done') or st[p].get('FAILED')) - st[p]['start']).total_seconds() / 3600
               if p in st and 'start' in st[p] and (st[p].get('done') or st[p].get('FAILED')) else np.nan
               for p in ('search', 'attack')}
        v = g.dropna(subset=['tp_pred'])
        mape_tp = (np.abs(v['tp_pred'] - v['tp']) / v['tp']).mean() * 100
        mape_mem = (np.abs(v['mem_pred'] - v['mem']) / v['mem']).mean() * 100
        fast, light = g.loc[g['tp'].idxmax()], g.loc[g['mem'].idxmin()]
        frag, robust = lt_m.loc[lt_m['mean_asr'].idxmax()], lt_m.loc[lt_m['mean_asr'].idxmin()]
        safe = g.loc[g['vulnerability'].idxmin()]
        rho = g['tp'].corr(g['vulnerability'], method='spearman')
        a50 = {}
        for f in m['families']:
            reached = lt_m[f'{f}_a50_state'] != 'not_reached'
            a50[f] = (lt_m.loc[reached, f'{f}_a50'].median(), int((~reached).sum()))
        lines += [
            f"{m['pretty']} ({m['task']}) - baseline {m['baseline']}",
            f"  pipeline     search {fmt_dur(dur['search'])}, attack {fmt_dur(dur['attack'])}",
            f"  search       {len(m['explored'])} deployments measured, {len(g)} on the Pareto front; cost-model "
            f"error on the front: throughput {mape_tp:.0f} %, memory {mape_mem:.0f} %",
            f"  fastest      g{fast.gene}: {fast.tp:.1f} ± {fast.tp_std:.1f} img/s, {fast.mem:.0f} MB "
            f"(search reported {fast.tp_first:.1f} from one run), vulnerability {fast.vulnerability:.1f} %",
            f"  lightest     g{light.gene}: {light.mem:.0f} MB, {light.tp:.1f} img/s, vulnerability {light.vulnerability:.1f} %",
            f"  safest       g{safe.gene}: vulnerability {safe.vulnerability:.1f} % (worst tensor {short_layer(safe.worst_tap)}), "
            f"{safe.tp:.1f} img/s = {100 * safe.tp / fast.tp:.0f} % of the fastest",
            f"  speed vs vulnerability on the front: Spearman rho = {rho:+.2f}",
            f"  most fragile tensor  {short_layer(frag.layer)} [cut {frag.cut:.0f}], mean ASR {frag.mean_asr:.1f} %",
            f"  most robust tensor   {short_layer(robust.layer)} [cut {robust.cut:.0f}], mean ASR {robust.mean_asr:.1f} %",
            "  amp_rel for 50 % ASR (median over tensors):  " + ', '.join(
                f"{f} {a50[f][0]:.2f}" + (f" ({a50[f][1]}/{len(lt_m)} never reach 50 %)" if a50[f][1] else '')
                for f in m['families']),
        ]
        if 'mfgsm' in a50 and 'mgna' in a50 and a50['mfgsm'][0] > 0:
            lines.append(f"  random noise needs {a50['mgna'][0] / a50['mfgsm'][0]:.1f}x the amplitude of mFGSM "
                         f"for the same 50 % ASR")
        stealthy = {}
        lines.append(f"  stealthy strengths (>= {100 * STEALTH_SHARE:.0f} % of the tensors inside the band; "
                     f"Δstd and Δmean in % of the clean std):")
        for band in bands:
            for crit in CRITERIA:
                win, parts = stealth_windows(m, band, crit), []
                for f in m['families']:
                    sym, iv = family_info(f)[2], win[f]
                    stealthy[f'{f}_stealthy_{crit}_pm{band:g}'] = fmt_intervals(iv)
                    if not iv:
                        parts.append(f"{f} none")
                        continue
                    sf = m['stealth']
                    sf = sf[(sf.band_pct == band) & (sf.criterion == crit) & (sf.family == f) & sf.stealthy]
                    best = sf.loc[sf['asr_stealthy_median'].idxmax()]
                    rng = f"range {iv[0][0]:g}-{iv[-1][1]:g}, " if len(iv) > 1 else ''
                    parts.append(f"{f} {sym} in {fmt_intervals(iv)} ({rng}best median ASR "
                                 f"{best.asr_stealthy_median:.1f} % at {sym} = {best.strength:g})")
                label = 'std AND mean' if crit == 'both' else f'{crit} only'
                lines.append(f"    ±{band:g} % {label:12s} " + '; '.join(parts))
        lines.append('')
        rows.append(dict(model=m['name'], task=m['task'], baseline=m['baseline'], search_h=dur['search'],
                         attack_h=dur['attack'], n_explored=len(m['explored']), n_pareto=len(g),
                         mape_tp_pct=mape_tp, mape_mem_pct=mape_mem, fastest_gene=fast.gene, fastest_tp=fast.tp,
                         safest_gene=safe.gene, safest_vuln=safe.vulnerability, safest_tp=safe.tp, rho_tp_vuln=rho,
                         **{f'{f}_median_a50': a50[f][0] for f in m['families']}, **stealthy))
    pd.DataFrame(rows).to_csv(out / 'tables' / 'model_summary.csv', index=False)
    text = 'Key findings\n============\n\n' + '\n'.join(lines) + '\n' + GUIDE
    (out / 'summary.txt').write_text(text)
    print('\n' + text)


def generate(runs, out, models=None, bands=(10.0,), attack_dirs=('attack',)):
    bands = sorted(set(float(b) for b in bands))
    runs, out = Path(runs).resolve(), Path(out).resolve()
    status = parse_status([runs / f for f in STATUS_FILES])
    order = list(status) + sorted(p.name for p in runs.iterdir() if p.is_dir() and p.name not in status)
    selected = models
    models = []
    for name in order:
        if selected and name not in selected:
            continue
        if not (runs / name).is_dir():
            continue
        m = load_model(runs / name, attack_dirs)
        if m is None:
            continue
        if m['attack'] is None:
            print(f'  {name}: no attack_summary_gpu.csv in {", ".join(attack_dirs)}, skipped')
            continue
        models.append(m)
    if not models:
        raise RuntimeError(f'no model folder with search_results.json + attack results under {runs}')
    print(f"models: {', '.join(m['name'] for m in models)}\nwriting to {out}")
    for m in models:
        m['stealth'] = stealth_table(m, bands)

    fig_timeline(models, status, out)
    write_tables_and_summary(models, status, bands, out)
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', type=Path, help='folder holding one sub-folder per model')
    ap.add_argument('-o', '--out', type=Path, help='output folder (default: RUNS/report/campaign)')
    ap.add_argument('--models', nargs='+', help='only these model folders')
    ap.add_argument('--attack-dirs', nargs='+', default=['attack'],
                    help='attack output folders of each model, merged in this order (a later one wins)')
    ap.add_argument('--stealth-bands', type=float, nargs='+', default=[10.0],
                    help='stealth tolerances: an attack is stealthy at tolerance b when it changes the feature-map std '
                         'and shifts its mean by at most b %% of the clean std (default 10; one set of rows per value '
                         'in the stealth tables)')
    args = ap.parse_args()
    generate(args.runs, args.out or args.runs / 'report' / 'campaign', args.models, args.stealth_bands,
             args.attack_dirs)


if __name__ == '__main__':
    main()
