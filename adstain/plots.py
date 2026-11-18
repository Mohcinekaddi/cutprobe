import os

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    MPL_OK, MPL_ERROR = True, None
except Exception as _e:
    MPL_OK, MPL_ERROR = False, _e

ATTACK_COLORS = {"mfgsm": "firebrick", "mgna": "steelblue", "mjsma": "darkgreen"}


def fmt_eps(e):
    if e is None or (isinstance(e, float) and not np.isfinite(e)):
        return "solved"
    return f"{float(e):g}"


def _log_if_spread(ax, values, ratio=50.0):
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size and (vals > 0).all() and vals.max() / max(vals.min(), 1e-300) > ratio:
        ax.set_yscale("log")


def _colors(layers):
    cmap = plt.get_cmap("tab10")
    return {lay: cmap(k % 10) for k, lay in enumerate(layers)}


def _save(fig, path, dpi=200, top=None):
    plt.tight_layout(rect=(0, 0, 1, top) if top else None)
    if path.lower().endswith((".jpg", ".jpeg")):
        plt.savefig(path, dpi=dpi, pil_kwargs={"quality": 85})
    else:
        plt.savefig(path, dpi=dpi)
    plt.close(fig)
    return True


def plot_pareto(results, out_png):
    if not MPL_OK:
        return False
    fig, ax = plt.subplots(figsize=(9, 6))
    for st in results.get("stages", []):
        for level, marker in (("level1", "o"), ("level2", "s")):
            front = st.get(level)
            if not front or not front["objectives"]:
                continue
            obj = np.asarray(front["objectives"])
            order = np.argsort(obj[:, 0])
            ax.plot(obj[order, 0], obj[order, 1], marker=marker, ls="--" if level == "level1" else "-",
                    alpha=0.85, label=f"stage {st['stage']} {level} ({len(st['resources'])} PE)")
    val = results.get("validation") or []
    for v in val:
        ax.annotate("", xy=(v["throughput_measured"], v["memory_measured"]),
                    xytext=(v["throughput_predicted"], v["memory_predicted"]),
                    arrowprops=dict(arrowstyle="->", color="grey", alpha=0.5))
    ax.set_xlabel("throughput (img/s)")
    ax.set_ylabel("max rank memory (MB)")
    ax.set_title("Pareto front: throughput vs memory"
                 + ("  (arrows: predicted -> measured)" if val else ""))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, out_png)


def plot_layer_config(out_png, tap, label, gt, top1_clean, top1_atk, valid, names, pm,
                      acc_full_pct, top_classes=20, top_pairs=20, engine="GPU",
                      clean_label="clean suffix"):
    if not MPL_OK:
        return False
    v = np.asarray(valid, dtype=bool)
    if not v.any():
        return False
    idx = np.nonzero(v)[0]
    gt_v = np.asarray(gt)[idx]
    ok_clean = np.asarray(top1_clean)[idx] == gt_v
    atk_v = np.asarray(top1_atk)[idx]
    ok_atk = atk_v == gt_v
    n = idx.size

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"StAIn ({engine}): cut {tap}, {label}", fontsize=14, fontweight="bold")
    ax_cum, ax_cls, ax_pairs, ax_grid = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    k = np.arange(1, n + 1)
    acc_c, acc_a = ok_clean.mean() * 100, ok_atk.mean() * 100
    ax_cum.plot(idx, ok_clean.cumsum() / k * 100, color="steelblue", lw=1.0,
                label=f"{clean_label} ({acc_c:.1f} %), n={n}")
    ax_cum.axhline(acc_c, color="steelblue", ls="--", alpha=0.45, lw=1)
    ax_cum.plot(idx, ok_atk.cumsum() / k * 100, color="crimson", lw=1.0,
                label=f"{label} ({acc_a:.1f} %), n={n}")
    ax_cum.axhline(acc_a, color="crimson", ls="--", alpha=0.45, lw=1)
    if np.isfinite(acc_full_pct):
        ax_cum.axhline(acc_full_pct, color="black", ls=":", lw=1.0,
                       label=f"full model ({acc_full_pct:.1f} %)")
    ax_cum.set_xlabel("image index")
    ax_cum.set_ylabel("cumulative accuracy (%)")
    ax_cum.set_xlim(0, max(int(idx.max()), 1))
    ax_cum.set_title(f"Running accuracy ({n} test images)")
    ax_cum.legend(fontsize=8)
    ax_cum.grid(True, alpha=0.3)
    ax_cum.yaxis.set_major_formatter(mtick.PercentFormatter())

    mc = pd.DataFrame({"gt": gt_v[~ok_atk], "pred": atk_v[~ok_atk]})
    name = lambda c: str(names.get(int(c), c))
    if len(mc):
        cls = mc.groupby("gt").size().sort_values(ascending=False).head(top_classes)
        y = np.arange(len(cls))
        ax_cls.barh(y, cls.values, color="orange", edgecolor="black", linewidth=0.5)
        ax_cls.set_yticks(y)
        ax_cls.set_yticklabels([f"{name(c)[:22]} ({c})" for c in cls.index], fontsize=8)
        ax_cls.invert_yaxis()
        ax_cls.set_xlabel("# misclassified")
        ax_cls.set_title(f"Top {len(cls)} / {mc['gt'].nunique()} affected GT classes")
        pairs = mc.groupby(["gt", "pred"]).size().sort_values(ascending=False).head(top_pairs)
        y2 = np.arange(len(pairs))
        ax_pairs.barh(y2, pairs.values, color="firebrick", edgecolor="black", linewidth=0.5)
        ax_pairs.set_yticks(y2)
        ax_pairs.set_yticklabels([f"{name(g)[:14]} -> {name(p)[:14]}" for g, p in pairs.index],
                                 fontsize=7.5)
        ax_pairs.invert_yaxis()
        ax_pairs.set_xlabel("# occurrences")
        ax_pairs.set_title(f"Top {len(pairs)} / {mc.groupby(['gt', 'pred']).ngroups} "
                           f"GT -> predicted confusions")
    else:
        for ax, title in ((ax_cls, "Errors per GT class"), (ax_pairs, "GT -> predicted")):
            ax.text(0.5, 0.5, "no error", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
    ax_cls.grid(True, axis="x", alpha=0.3)
    ax_pairs.grid(True, axis="x", alpha=0.3)

    npair = pm["n_paired_images"]
    if npair > 0:
        grid = np.array([[pm["base_correct_atk_correct"], pm["base_correct_atk_wrong"]],
                         [pm["base_wrong_atk_correct"], pm["base_wrong_atk_wrong"]]])
        gpct = grid / npair * 100
        ax_grid.imshow(gpct, cmap="YlOrRd", vmin=0, vmax=max(gpct.max(), 1))
        ax_grid.set_xticks([0, 1])
        ax_grid.set_xticklabels(["attacked OK", "attacked KO"], fontsize=10)
        ax_grid.set_yticks([0, 1])
        ax_grid.set_yticklabels(["clean OK", "clean KO"], fontsize=10)
        for r in range(2):
            for c in range(2):
                ax_grid.text(c, r, f"{grid[r, c]}\n({gpct[r, c]:.1f}%)", ha="center", va="center",
                             fontsize=11, fontweight="bold")
        asr, acr = pm["attack_success_rate_pct"], pm["accidental_correction_rate_pct"]
        ax_grid.set_title(f"clean vs attacked (n={npair}): "
                          f"ASR={'n/a' if not np.isfinite(asr) else f'{asr:.1f}%'}, "
                          f"ACR={'n/a' if not np.isfinite(acr) else f'{acr:.1f}%'}")
    else:
        ax_grid.text(0.5, 0.5, "no paired data", ha="center", va="center",
                     transform=ax_grid.transAxes)
    return _save(fig, out_png)


def plot_scores(df, out_png, cut_names=None):
    if not MPL_OK or df is None or df.empty:
        return False
    cut_names = cut_names or {}
    df = df.dropna(subset=["layer"]).copy()
    df["layer"] = df["layer"].astype(int)
    df = df.drop_duplicates(subset=["layer", "tag"], keep="last")
    layers = sorted(df["layer"].unique())
    tags = list(dict.fromkeys(df["tag"].astype(str)))
    x = np.arange(len(layers), dtype=float)
    width = 0.8 / max(len(tags), 1)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    fig.suptitle("StAIn on the AutoDiCE deployment (ncnn + MPI)", fontsize=14, fontweight="bold")
    if "map50_attack" in df.columns:
        panels = (("attack_success_rate_pct", "ASR (%)", "Attack success rate (objects lost)"),
                  ("vanished_pct", "vanished (%)", "Clean detections vanished"),
                  ("map50_attack", "mAP@0.5 (%)", "mAP@0.5 under attack"))
        clean_col = "map50_clean"
    elif "vanished_pct" in df.columns:
        panels = (("attack_success_rate_pct", "ASR (%)", "Attack success rate (detections lost)"),
                  ("ghost_pct", "ghost (%)", "New (ghost) detections"),
                  ("image_changed_pct", "images (%)", "Images whose detections changed"))
        clean_col = None
    else:
        panels = (("attack_success_rate_pct", "ASR (%)", "Attack success rate"),
                  ("accidental_correction_rate_pct", "ACR (%)", "Accidental correction rate"),
                  ("accuracy_attack_pct", "accuracy (%)", "Accuracy under attack"))
        clean_col = "accuracy_clean_pct"
    for k, tag in enumerate(tags):
        sub = df[df["tag"].astype(str) == tag]
        sub = sub.set_index("layer")
        attack = tag.split("_")[0]
        same = [t for t in tags if t.split("_")[0] == attack]
        shade = 0.45 + 0.55 * (same.index(tag) + 1) / len(same)
        off = (k - (len(tags) - 1) / 2.0) * width
        for ax, (col, _, _) in zip(axes, panels):
            vals = [float(sub[col].get(l, np.nan)) if col in sub else np.nan for l in layers]
            ax.bar(x + off, vals, width, color=ATTACK_COLORS.get(attack, "grey"), alpha=shade,
                   edgecolor="black", linewidth=0.4, label=tag)
    clean = df[clean_col].dropna() if clean_col and clean_col in df else pd.Series(dtype=float)
    if not clean.empty:
        axes[2].axhline(float(clean.iloc[-1]), color="black", ls="--", lw=1.2,
                        label=f"clean ({float(clean.iloc[-1]):.1f} %)")
    ticks = [f"L{l}" + (f"\n{cut_names[l]}" if l in cut_names else "") for l in layers]
    for ax, (_, ylabel, title) in zip(axes, panels):
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("ATTACK_TARGET_LAYER (cut tensor)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    return _save(fig, out_png)


def plot_attack_comparison(df, out_png, gene_id, quality=None):
    if not MPL_OK or df is None or df.empty:
        return False
    q_col, b_col, q_label = quality or ("accuracy_attack_pct", "accuracy_baseline_pct",
                                        "accuracy under attack (%)")
    layers = list(dict.fromkeys(df.sort_values("split_layer")["layer_name"].tolist()))
    tags = list(dict.fromkeys(df["config_tag"].tolist()))
    x = np.arange(len(layers), dtype=float)
    width = 0.8 / max(len(tags), 1)

    fig, axes = plt.subplots(2, 2, figsize=(17, 11))
    fig.suptitle(f"Gene {gene_id}: StAIn attacks per split layer", fontsize=14, fontweight="bold")
    ax_asr, ax_acc, ax_var, ax_plane = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    def vals(tag, col):
        sub = df[df["config_tag"] == tag].set_index("layer_name")
        if col not in sub.columns:
            return np.full(len(layers), np.nan)
        return np.array([float(sub[col].get(lay, np.nan)) for lay in layers], dtype=float)

    for k, tag in enumerate(tags):
        rows = df[df["config_tag"] == tag]
        attack = str(rows["attack"].iloc[0])
        label = str(rows["config_label"].iloc[0])
        col = ATTACK_COLORS.get(attack, "grey")
        same = [t for t in tags if str(df[df["config_tag"] == t]["attack"].iloc[0]) == attack]
        shade = 0.45 + 0.55 * (same.index(tag) + 1) / max(len(same), 1)
        off = (k - (len(tags) - 1) / 2.0) * width
        kw = dict(color=col, alpha=shade, edgecolor="black", linewidth=0.4, label=label)
        ax_asr.bar(x + off, vals(tag, "attack_success_rate_pct"), width, **kw)
        ax_acc.bar(x + off, vals(tag, q_col), width, **kw)
        ax_var.bar(x + off, np.abs(vals(tag, "d_var_pct")), width, **kw)
        ax_plane.scatter(np.abs(vals(tag, "d_var_pct")), vals(tag, "attack_success_rate_pct"),
                         s=70, color=col, alpha=shade, edgecolor="black", linewidth=0.5,
                         label=label)

    base = df[b_col].dropna() if b_col and b_col in df.columns else pd.Series(dtype=float)
    if not base.empty:
        ax_acc.axhline(float(base.iloc[0]), color="black", ls="--", lw=1.2,
                       label=f"baseline ({float(base.iloc[0]):.2f} %)")
    ax_asr.set_ylabel("ASR (%)")
    ax_asr.set_title("Attack success rate (over initially correct images)" if q_col.startswith("accuracy")
                     else "Attack success rate (objects found by the clean run, lost under attack)",
                     fontsize=10)
    ax_acc.set_ylabel(q_label)
    title = q_label.split(" (")[0]
    ax_acc.set_title(title if title.startswith("mAP") else title[0].upper() + title[1:], fontsize=10)
    ax_var.set_ylabel("|variance change vs clean FM| (%)")
    ax_var.set_title("Second-order signature", fontsize=10)
    ax_var.set_yscale("symlog", linthresh=1e-6)
    ax_plane.set_xlabel("|d_var| (%)")
    ax_plane.set_ylabel("ASR (%)")
    ax_plane.set_xscale("symlog", linthresh=1e-6)
    ax_plane.set_title("Effectiveness / detectability plane", fontsize=10)
    for ax in (ax_asr, ax_acc, ax_var):
        ax.set_xticks(x)
        ax.set_xticklabels(layers, fontsize=8, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
    ax_plane.grid(True, alpha=0.3)
    for ax in (ax_asr, ax_acc, ax_var, ax_plane):
        ax.legend(fontsize=7, ncol=max(1, min(len(tags), 3)))
    return _save(fig, out_png)


def plot_fm_noise_impact(noise_df, out_png, gene_id):
    noise_df = noise_df.dropna(subset=["eps_rel"])
    if not MPL_OK or noise_df.empty:
        return False
    layers = list(dict.fromkeys(noise_df["layer_name"].tolist()))
    colors = _colors(layers)
    eps_values = sorted(noise_df["eps_rel"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Gene {gene_id}: feature-map statistics under attack "
                 f"(eps_rel = {', '.join(fmt_eps(e) for e in eps_values)})",
                 fontsize=14, fontweight="bold")
    for ax, kc, ka, title in ((axes[0, 0], "mean_clean", "mean_attacked", "mean"),
                              (axes[0, 1], "std_clean", "std_attacked", "std"),
                              (axes[1, 0], "var_clean", "var_attacked", "variance")):
        plotted = []
        for lay in layers:
            sub = noise_df[noise_df["layer_name"] == lay].sort_values("eps_rel")
            if sub.empty:
                continue
            clean = float(sub[kc].iloc[0])
            xs = np.concatenate(([0.0], sub["eps_rel"].to_numpy(dtype=float)))
            ys = np.concatenate(([clean], sub[ka].to_numpy(dtype=float)))
            ax.plot(xs, ys, "-o", color=colors[lay], lw=1.7, ms=5, label=lay)
            ax.axhline(clean, color=colors[lay], ls=":", lw=1.0, alpha=0.55)
            plotted.extend(ys.tolist())
        ax.set_xlabel("eps_rel (eps = eps_rel x std(FM), per image)")
        ax.set_ylabel(title)
        ax.set_title(f"FM {title} (eps=0: clean)", fontsize=10)
        ax.set_xticks([0.0] + [float(e) for e in eps_values])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        _log_if_spread(ax, plotted)
    ax = axes[1, 1]
    for lay in layers:
        sub = noise_df[noise_df["layer_name"] == lay].sort_values("eps_rel")
        for key, ls, lbl in (("d_mean_pct", ":", "mean"), ("d_std_pct", "--", "std"),
                             ("d_var_pct", "-", "variance")):
            ax.plot(sub["eps_rel"], sub[key], ls=ls, marker="o", ms=4, color=colors[lay], lw=1.5,
                    label=f"{lay} {lbl}")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("eps_rel")
    ax.set_ylabel("change vs clean FM (%)")
    ax.set_title("Relative change of mean / std / variance", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=max(1, min(len(layers), 3)))
    return _save(fig, out_png)


def plot_noise_magnitude(noise_df, out_png, gene_id):
    noise_df = noise_df.dropna(subset=["eps_rel"])
    if not MPL_OK or noise_df.empty:
        return False
    layers = list(dict.fromkeys(noise_df["layer_name"].tolist()))
    colors = _colors(layers)
    eps_values = sorted(noise_df["eps_rel"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Gene {gene_id}: magnitude of the injected perturbation",
                 fontsize=14, fontweight="bold")
    ax_l2, ax_snr, ax_std, ax_neg = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    for lay in layers:
        sub = noise_df[noise_df["layer_name"] == lay].sort_values("eps_rel")
        c = colors[lay]
        xs = sub["eps_rel"].to_numpy(dtype=float)
        ax_l2.plot(xs, sub["l2_rel_pct"], "-o", color=c, lw=1.7, ms=5, label=lay)
        ax_snr.plot(xs, sub["snr_db"], "-o", color=c, lw=1.7, ms=5, label=lay)
        ax_std.plot(xs, sub["std_delta"], "-o", color=c, lw=1.7, ms=5, label=f"{lay} std(delta)")
        ax_std.plot(xs, sub["linf_max"], "--s", color=c, lw=1.2, ms=4, alpha=0.7,
                    label=f"{lay} max|delta|")
        ax_neg.plot(xs, sub["neg_frac_attacked_pct"], "-o", color=c, lw=1.7, ms=5, label=lay)
        ax_neg.axhline(float(sub["neg_frac_clean_pct"].iloc[0]), color=c, ls=":", lw=1.0, alpha=0.55)
    ax_l2.set_ylabel("||delta||2 / ||FM||2 (%)")
    ax_snr.set_ylabel("SNR (dB)")
    ax_std.set_ylabel("delta amplitude")
    ax_neg.set_ylabel("negative values in FM (%)")
    for ax in (ax_l2, ax_snr, ax_std, ax_neg):
        ax.set_xlabel("eps_rel")
        ax.set_xticks([float(e) for e in eps_values])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    return _save(fig, out_png)


def plot_attack_vs_eps(summary_df, out_png, gene_id):
    summary_df = summary_df.dropna(subset=["eps_rel"])
    if not MPL_OK or summary_df.empty:
        return False
    layers = list(dict.fromkeys(summary_df["layer_name"].tolist()))
    colors = _colors(layers)
    eps_values = sorted(summary_df["eps_rel"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f"Gene {gene_id}: attack effect vs eps", fontsize=14, fontweight="bold")
    ax_acc, ax_asr, ax_acr, ax_l2 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    base = summary_df["accuracy_baseline_pct"].dropna()
    if not base.empty:
        ax_acc.axhline(float(base.iloc[0]), color="black", ls="--", lw=1.2,
                       label=f"full model ({float(base.iloc[0]):.2f} %)")
    for lay in layers:
        sub = summary_df[summary_df["layer_name"] == lay].sort_values("eps_rel")
        c = colors[lay]
        xs = sub["eps_rel"].to_numpy(dtype=float)
        ax_acc.plot(xs, sub["accuracy_attack_pct"], "-o", color=c, lw=1.7, ms=5, label=lay)
        ax_asr.plot(xs, sub["attack_success_rate_pct"], "-o", color=c, lw=1.7, ms=5, label=lay)
        ax_acr.plot(xs, sub["accidental_correction_rate_pct"], "-o", color=c, lw=1.7, ms=5,
                    label=f"{lay} ACR")
        ax_acr.plot(xs, sub["net_effect_pct"], "--s", color=c, lw=1.2, ms=4, alpha=0.75,
                    label=f"{lay} net effect")
        ax_l2.plot(sub["l2_rel_pct"], sub["attack_success_rate_pct"], "-o", color=c, lw=1.7, ms=5,
                   label=lay)
    ax_acc.set_ylabel("accuracy under attack (%)")
    ax_asr.set_ylabel("ASR (%)")
    ax_acr.set_ylabel("%")
    ax_l2.set_xlabel("||delta||2 / ||FM||2 (%)")
    ax_l2.set_ylabel("ASR (%)")
    for ax in (ax_acc, ax_asr, ax_acr):
        ax.set_xlabel("eps_rel")
        ax.set_xticks([float(e) for e in eps_values])
    for ax in (ax_acc, ax_asr, ax_acr, ax_l2):
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    return _save(fig, out_png)


def _quality(df):
    if "map50_attack" in df.columns:
        return ("map50_attack", "map50_baseline", "mAP@0.5 under attack (%)")
    if "vanished_pct" in df.columns:
        return ("vanished_pct", None, "vanished detections (%)")
    return None


def gene_figures(summary_g, noise_g, gdir, gene_id, kind="classification"):
    made = []
    if not MPL_OK or summary_g.empty:
        return made
    q = _quality(summary_g) if kind == "detection" else None
    jobs = [(lambda d, p, g: plot_attack_comparison(d, p, g, q), summary_g, "attack_comparison.png")]
    mf = summary_g[summary_g["attack"] == "mfgsm"]
    if mf["eps_rel"].dropna().nunique() > 1 and kind != "detection":
        mfn = noise_g[noise_g["attack"] == "mfgsm"]
        jobs += [(plot_fm_noise_impact, mfn, "fm_noise_impact.png"),
                 (plot_noise_magnitude, mfn, "fm_noise_magnitude.png"),
                 (plot_attack_vs_eps, mf, "attack_vs_eps.png")]
    for fn, df, name in jobs:
        path = os.path.join(gdir, name)
        try:
            if fn(df, path, gene_id):
                made.append(path)
        except Exception as e:
            print(f"  [figure] gene {gene_id} {name}: {e}")
    return made


def plot_pareto_attacks(df, out_png):
    if not MPL_OK or df is None or df.empty:
        return False
    tags = list(dict.fromkeys(df["tag"].astype(str)))
    worst = (df.groupby(["gene", "tag"], as_index=False)
             .agg(asr=("attack_success_rate_pct", "max"), throughput=("throughput", "first"),
                  memory=("memory_mb", "first")))
    fig, axes = plt.subplots(2, len(tags), figsize=(6 * len(tags), 10), squeeze=False)
    fig.suptitle("Pareto solutions on the real deployment: worst-case ASR over each solution's cuts",
                 fontsize=14, fontweight="bold")
    for j, tag in enumerate(tags):
        color = ATTACK_COLORS.get(tag.split("_")[0], "grey")
        w = worst[worst["tag"] == tag]
        cuts = df[df["tag"].astype(str) == tag]
        for row, (col, xlabel) in enumerate((("throughput", "throughput (img/s)"),
                                             ("memory", "max rank memory (MB)"))):
            ax = axes[row, j]
            src = "throughput" if col == "throughput" else "memory_mb"
            ax.scatter(cuts[src], cuts["attack_success_rate_pct"], s=18, color=color, alpha=0.3,
                       label="each cut")
            ax.scatter(w[col], w["asr"], s=70, color=color, edgecolor="black", label="worst cut")
            for _, r in w.iterrows():
                ax.annotate(f"g{int(r['gene'])}", (r[col], r["asr"]), fontsize=8,
                            textcoords="offset points", xytext=(4, 4))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("ASR (%)")
            ax.set_ylim(0, max(5.0, float(cuts["attack_success_rate_pct"].max()) * 1.1))
            ax.set_title(tag, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    return _save(fig, out_png)


def plot_det_config(out_png, tap, label, pm, base, clean, atk, lost, base_cnt, cname, img,
                    engine="GPU", top_classes=20):
    if not MPL_OK:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"StAIn ({engine}): cut {tap}, {label}", fontsize=14, fontweight="bold")
    ax_q, ax_cls, ax_grid, ax_img = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    has_gt = "base_hit_atk_hit" in pm
    if has_gt:
        keys = (("map50_95", "mAP@[.5:.95]"), ("map50", "mAP@.5"), ("recall_pct", "recall"),
                ("precision_pct", "precision"))
        x = np.arange(len(keys))
        w = 0.27
        for k, (src, name, col) in enumerate(((base, "full model", "grey"), (clean, "clean suffix", "steelblue"),
                                              (atk, label, "crimson"))):
            vals = [float(src.get(key, np.nan)) for key, _ in keys]
            ax_q.bar(x + (k - 1) * w, vals, w, color=col, edgecolor="black", linewidth=0.5, label=name)
            for xi, v in zip(x + (k - 1) * w, vals):
                if np.isfinite(v):
                    ax_q.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        ax_q.set_xticks(x)
        ax_q.set_xticklabels([n for _, n in keys])
        ax_q.set_ylabel("%")
        ax_q.set_title("Detection quality vs ground truth", fontsize=11)
    else:
        keys = (("vanished_pct", "vanished"), ("ghost_pct", "ghost"), ("relabeled_pct", "relabeled"),
                ("image_changed_pct", "images changed"))
        vals = [float(pm.get(k, np.nan)) for k, _ in keys]
        ax_q.bar(np.arange(len(keys)), vals, color="crimson", edgecolor="black", linewidth=0.5)
        ax_q.set_xticks(np.arange(len(keys)))
        ax_q.set_xticklabels([n for _, n in keys])
        ax_q.set_ylabel("% of clean detections (images for the last bar)")
        ax_q.set_title("Attacked vs clean detections (no ground truth)", fontsize=11)
    ax_q.grid(True, axis="y", alpha=0.3)
    ax_q.legend(fontsize=8)

    if lost:
        top = sorted(lost, key=lambda c: -lost[c])[:top_classes]
        y = np.arange(len(top))
        ax_cls.barh(y, [lost[c] for c in top], color="orange", edgecolor="black", linewidth=0.5)
        ax_cls.set_yticks(y)
        ax_cls.set_yticklabels([f"{str(cname(c))[:22]} ({lost[c]}/{base_cnt.get(c, 0)})" for c in top],
                               fontsize=8)
        ax_cls.invert_yaxis()
        ax_cls.set_xlabel("# objects lost (found by the clean run, missed under attack)")
        ax_cls.set_title(f"Top {len(top)} / {len(lost)} classes losing objects", fontsize=11)
    else:
        ax_cls.text(0.5, 0.5, "no object lost" if has_gt else "no ground truth", ha="center",
                    va="center", transform=ax_cls.transAxes)
        ax_cls.set_title("Objects lost per class", fontsize=11)
    ax_cls.grid(True, axis="x", alpha=0.3)

    if has_gt and pm.get("n_gt_objects"):
        grid = np.array([[pm["base_hit_atk_hit"], pm["base_hit_atk_miss"]],
                         [pm["base_miss_atk_hit"], pm["base_miss_atk_miss"]]])
        n = grid.sum()
        gpct = grid / max(n, 1) * 100
        ax_grid.imshow(gpct, cmap="YlOrRd", vmin=0, vmax=max(gpct.max(), 1))
        ax_grid.set_xticks([0, 1])
        ax_grid.set_xticklabels(["found under attack", "missed under attack"], fontsize=10)
        ax_grid.set_yticks([0, 1])
        ax_grid.set_yticklabels(["found clean", "missed clean"], fontsize=10)
        for r in range(2):
            for c in range(2):
                ax_grid.text(c, r, f"{grid[r, c]}\n({gpct[r, c]:.1f}%)", ha="center", va="center",
                             fontsize=11, fontweight="bold")
        asr, acr = pm["attack_success_rate_pct"], pm["accidental_correction_rate_pct"]
        ax_grid.set_title(f"{n} ground-truth objects: ASR={asr:.1f}%, ACR={acr:.1f}%, "
                          f"images hit {pm.get('image_attack_success_rate_pct', float('nan')):.1f}%",
                          fontsize=10)
    else:
        ref = pm.get("ref_detections", 0)
        ax_grid.bar(["kept", "vanished", "ghost"],
                    [ref * (1 - pm.get("vanished_pct", 0) / 100), ref * pm.get("vanished_pct", 0) / 100,
                     ref * pm.get("ghost_pct", 0) / 100], color=["seagreen", "crimson", "orange"])
        ax_grid.set_title(f"{ref} clean detections: ASR (vanished) {pm.get('vanished_pct', float('nan')):.1f}%",
                          fontsize=10)

    if img is not None and len(img):
        a = "hit_clean" if has_gt else "n_det_clean"
        b = "hit_attack" if has_gt else "n_det_attack"
        xs, ys = img[a].to_numpy(), img[b].to_numpy()
        lim = max(1, int(max(xs.max(), ys.max())) + 1)
        ax_img.scatter(xs + np.random.RandomState(0).uniform(-0.2, 0.2, len(xs)),
                       ys + np.random.RandomState(1).uniform(-0.2, 0.2, len(ys)), s=6, alpha=0.35,
                       color="crimson")
        ax_img.plot([0, lim], [0, lim], color="black", lw=1, ls="--")
        ax_img.set_xlim(-0.5, lim)
        ax_img.set_ylim(-0.5, lim)
        ax_img.set_xlabel("objects found by the clean run" if has_gt else "clean detections")
        ax_img.set_ylabel("objects found under attack" if has_gt else "attacked detections")
        ax_img.set_title("Per image (below the diagonal: objects lost)", fontsize=11)
        ax_img.grid(True, alpha=0.3)
    return _save(fig, out_png)


def _draw_dets(ax, boxes, labels, scores, color, cname, ls="-"):
    for b, l, s in zip(boxes, labels, scores):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=color,
                                   lw=1.4, ls=ls))
        if s is not None:
            ax.text(b[0], b[1] - 2, f"{str(cname(l))[:14]} {s:.2f}", color="white", fontsize=6,
                    bbox=dict(facecolor=color, alpha=0.7, pad=0.5, lw=0))


def plot_det_examples(out_png, tap, label, img, paths, clean, atk, gts, conf, cname, k=4):
    if not MPL_OK or img is None or not len(img):
        return False
    import cv2
    col = "lost" if gts is not None else "vanished"
    pick = img.sort_values([col, "ghost"], ascending=False)
    pick = pick[(pick[col] > 0) | (pick["ghost"] > 0)].head(k)
    if pick.empty:
        return False
    fig, axes = plt.subplots(len(pick), 2, figsize=(14, 5.2 * len(pick)), squeeze=False)
    fig.suptitle(f"StAIn: cut {tap}, {label} -- most affected images (score >= {conf:g})",
                 fontsize=13, fontweight="bold")
    for row, (_, r) in enumerate(pick.iterrows()):
        i = int(r["index"])
        m = cv2.imread(paths[i], cv2.IMREAD_COLOR)
        if m is None:
            continue
        m = m[:, :, ::-1]
        for j, (store, color, title) in enumerate(((clean, "limegreen", "clean"), (atk, "red", "attacked"))):
            ax = axes[row, j]
            ax.imshow(m)
            if gts is not None:
                gb, gl, gc = gts[i]
                _draw_dets(ax, gb[~gc], gl[~gc], [None] * int((~gc).sum()), "white", cname, ls="--")
            b, s, l = store.at(i, conf)
            _draw_dets(ax, b, l, s, color, cname)
            ax.set_title(f"#{i} {os.path.basename(paths[i])} {title}: {len(s)} detection(s)"
                         + (f", {int(r['lost'])} object(s) lost" if j and gts is not None else ""),
                         fontsize=9)
            ax.axis("off")
    return _save(fig, out_png, dpi=100, top=1.0 - 0.45 / (5.2 * len(pick)))
