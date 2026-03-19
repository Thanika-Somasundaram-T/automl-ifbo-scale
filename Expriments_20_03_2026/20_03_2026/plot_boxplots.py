"""
Boxplot visualizations: cross-scale and per-source-HP boxplots.
Supports both NLL and MSE metrics.
"""
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from config import (
    BASE_SCALES, OBS_EPOCHS, SOURCE_CONFIGS, LOSS_MAX,
    HP_MARKERS, HP_COLORS, SCALE_COLORS,
    sorted_target_hps, short_target_label, source_hp_label, clean_scale,
)


def plot_cross_scale_boxplot(all_rows, output_dir, metric="nll", log_scale=False):
    """
    Cross-scale comparison: all 5 scales side-by-side per epoch.
    Aggregates over all 12 source HPs. Markers show target HPs.
    """
    metric_label = metric.upper()
    target_hps = sorted_target_hps(all_rows)
    hp_marker_map = {hp: HP_MARKERS[i % len(HP_MARKERS)] for i, hp in enumerate(target_hps)}
    hp_color_map = {hp: HP_COLORS[i % len(HP_COLORS)] for i, hp in enumerate(target_hps)}

    # y-axis bounds: universal for NLL (show trend), local per-epoch for MSE (show detail)
    all_vals = [r[metric] for r in all_rows if r["scale"] != "baseline" and not np.isnan(r[metric])]
    use_universal_y = not log_scale and metric == "nll"
    if use_universal_y:
        y_lo = np.percentile(all_vals, 1) - 0.2
        y_hi = np.percentile(all_vals, 99) + 0.2

    fig, axes = plt.subplots(len(OBS_EPOCHS), 1,
                             figsize=(14, 5 * len(OBS_EPOCHS)),
                             sharey=use_universal_y, squeeze=False)

    fig.suptitle(
        f"Cross-Scale {metric_label} Comparison (Aggregated Over All Source HPs)\n"
        f"(Target: hd=128, loss_max=log₂(10)≈{LOSS_MAX:.2f})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row_idx, epoch in enumerate(OBS_EPOCHS):
        ax = axes[row_idx][0]

        vals_by_scale = []
        rows_by_scale = []
        for scale in BASE_SCALES:
            cell = [r for r in all_rows
                    if r["scale"] == scale and r["epoch"] == epoch
                    and not np.isnan(r[metric])]
            vals_by_scale.append([r[metric] for r in cell] if cell else [np.nan])
            rows_by_scale.append(cell)

        positions = np.arange(len(BASE_SCALES))
        bp = ax.boxplot(vals_by_scale, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(SCALE_COLORS[i])
            box.set_alpha(0.6)
            box.set_edgecolor("#555555")
        for m in bp["medians"]:
            m.set_color("black")
            m.set_linewidth(1.5)

        # HP markers (mean metric per target HP across all source HPs)
        for scale_idx, (scale, cell) in enumerate(zip(BASE_SCALES, rows_by_scale)):
            target_means = defaultdict(list)
            for r in cell:
                target_means[r["target_hp"]].append(r[metric])
            for hp, vals in target_means.items():
                mean_val = np.mean(vals)
                jitter = (hash(hp + str(epoch) + scale) % 11 - 5) * 0.02
                ax.scatter(
                    scale_idx + jitter, mean_val,
                    marker=hp_marker_map[hp], color=hp_color_map[hp],
                    s=55, alpha=0.85, edgecolors="#222", linewidths=0.4, zorder=5,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels([clean_scale(s) for s in BASE_SCALES], fontsize=11)
        ax.set_xlabel("Base Scale (Hidden Dim)", fontsize=11)
        ax.set_ylabel(f"{metric_label} ← Lower is Better", fontsize=10)
        ax.set_title(f"T = {epoch} Epochs Observed", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        if log_scale:
            ax.set_yscale('log')
        elif use_universal_y:
            ax.set_ylim(y_lo, y_hi)

    # Legend
    handles = [plt.scatter([], [], marker=hp_marker_map[hp], color=hp_color_map[hp],
                           s=50, edgecolors="#222", linewidths=0.4,
                           label=short_target_label(hp))
               for hp in target_hps]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               title=f"Target HP (hd=128) — mean {metric_label} across source HPs",
               title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = os.path.join(output_dir, f"cross_scale_{metric}_aggregated.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {out}")
    plt.close()


def plot_source_hp_boxplots(all_rows, output_dir, metric="nll", log_scale=False):
    """
    For each (scale, epoch): 12 boxplots on x-axis (one per source HP),
    each showing metric distribution across 12 target HPs, with markers.
    """
    metric_label = metric.upper()
    target_hps = sorted_target_hps(all_rows)
    hp_marker_map = {hp: HP_MARKERS[i % len(HP_MARKERS)] for i, hp in enumerate(target_hps)}
    hp_color_map = {hp: HP_COLORS[i % len(HP_COLORS)] for i, hp in enumerate(target_hps)}

    # y-axis bounds: universal for NLL (show trend), local per-epoch for MSE (show detail)
    all_vals = [r[metric] for r in all_rows if r["scale"] != "baseline" and not np.isnan(r[metric])]
    use_universal_y = not log_scale and metric == "nll"
    if use_universal_y:
        y_lo = np.percentile(all_vals, 1) - 0.2
        y_hi = np.percentile(all_vals, 99) + 0.2

    for scale in BASE_SCALES:
        fig, axes = plt.subplots(len(OBS_EPOCHS), 1,
                                 figsize=(18, 5 * len(OBS_EPOCHS)),
                                 sharey=use_universal_y, squeeze=False)
        fig.suptitle(
            f"Source HP Comparison ({metric_label}) — Base Scale = {clean_scale(scale)}\n"
            f"(Each bar = one source HP as context, markers = target HPs)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for row_idx, epoch in enumerate(OBS_EPOCHS):
            ax = axes[row_idx][0]

            data_per_src = []
            rows_per_src = []
            labels = []
            for src_cfg in SOURCE_CONFIGS:
                cell = [r for r in all_rows
                        if r["scale"] == scale
                        and r["source_config"] == src_cfg
                        and r["epoch"] == epoch
                        and not np.isnan(r[metric])]
                data_per_src.append([r[metric] for r in cell] if cell else [np.nan])
                rows_per_src.append(cell)
                labels.append(source_hp_label(src_cfg))

            positions = np.arange(len(SOURCE_CONFIGS))
            bp = ax.boxplot(data_per_src, positions=positions, widths=0.55,
                            patch_artist=True, showfliers=False)
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor("#B3E5FC")
                box.set_alpha(0.5)
                box.set_edgecolor("#555555")
            for m in bp["medians"]:
                m.set_color("black")
                m.set_linewidth(1.5)

            # Target HP markers
            for src_idx, cell in enumerate(rows_per_src):
                for r in cell:
                    hp = r["target_hp"]
                    jitter = (hash(hp + str(epoch) + str(src_idx)) % 11 - 5) * 0.02
                    ax.scatter(
                        src_idx + jitter, r[metric],
                        marker=hp_marker_map[hp], color=hp_color_map[hp],
                        s=45, alpha=0.85, edgecolors="#222", linewidths=0.4, zorder=5,
                    )

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Source HP (context)", fontsize=10)
            ax.set_ylabel(f"{metric_label} ← Lower is Better", fontsize=10)
            ax.set_title(f"T = {epoch} Epochs Observed", fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.25)
            if log_scale:
                ax.set_yscale('log')
            elif use_universal_y:
                ax.set_ylim(y_lo, y_hi)

        # Legend
        handles = [plt.scatter([], [], marker=hp_marker_map[hp], color=hp_color_map[hp],
                               s=50, edgecolors="#222", linewidths=0.4,
                               label=short_target_label(hp))
                   for hp in target_hps]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
                   title="Target HP (hd=128)", title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        out = os.path.join(output_dir, f"source_hp_{metric}_scale_{clean_scale(scale)}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"✅ Saved: {out}")
        plt.close()
