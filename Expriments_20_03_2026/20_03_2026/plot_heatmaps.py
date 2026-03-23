"""
Heatmap visualizations: 12×12 source→target metric heatmaps.
Supports both NLL and MSE metrics, plus baseline-diff mode.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from config import (
    BASE_SCALES, OBS_EPOCHS, SOURCE_CONFIGS,
    sorted_target_hps, short_target_label, source_hp_label, clean_scale,
)


def _build_baseline_lookup(all_rows, metric):
    """Build dict: (epoch, target_hp) → baseline metric value."""
    lookup = {}
    for r in all_rows:
        if r["scale"] == "baseline" and not np.isnan(r[metric]):
            lookup[(r["epoch"], r["target_hp"])] = r[metric]
    return lookup


def plot_heatmaps(all_rows, output_dir, metric="nll"):
    """One heatmap per (scale, epoch). Rows=source HPs, Cols=target HPs, Color=metric."""
    metric_label = metric.upper()
    target_hps = sorted_target_hps(all_rows)
    target_labels = [short_target_label(hp) for hp in target_hps]
    source_labels = [source_hp_label(c) for c in SOURCE_CONFIGS]

    # Compute UNIVERSAL color bounds across all heatmaps
    all_vals = [r[metric] for r in all_rows
                if r["scale"] != "baseline" and not np.isnan(r[metric])]
    global_vmin = np.percentile(all_vals, 2)
    global_vmax = np.percentile(all_vals, 98)
    print(f"  Universal color scale: [{global_vmin:.2f}, {global_vmax:.2f}]")

    for scale in BASE_SCALES:
        for epoch in OBS_EPOCHS:
            matrix = np.full((12, 12), np.nan)

            for r in all_rows:
                if r["scale"] != scale or r["epoch"] != epoch or r["scale"] == "baseline":
                    continue
                src_idx = r["source_config"] - 1
                tgt_idx = target_hps.index(r["target_hp"]) if r["target_hp"] in target_hps else -1
                if 0 <= src_idx < 12 and 0 <= tgt_idx < 12:
                    matrix[src_idx, tgt_idx] = r[metric]

            fig, ax = plt.subplots(figsize=(12, 10))

            valid = matrix[~np.isnan(matrix)]
            if len(valid) == 0:
                plt.close()
                continue

            im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                           vmin=global_vmin, vmax=global_vmax)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label(f"{metric_label} (lower = better)", fontsize=10)

            # Annotate cells
            for i in range(12):
                for j in range(12):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = "white" if val > (global_vmin + global_vmax) / 2 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, fontweight="bold", color=color)

            ax.set_xticks(range(12))
            ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(12))
            ax.set_yticklabels(source_labels, fontsize=8)
            ax.set_xlabel("Target HP (hd=128)", fontsize=11)
            ax.set_ylabel("Source HP (context)", fontsize=11)
            ax.set_title(
                f"Source→Target {metric_label} — Scale={clean_scale(scale)}, T={epoch}\n"
                f"(Green=good, Red=bad)",
                fontsize=12, fontweight="bold",
            )

            plt.tight_layout()
            out = os.path.join(output_dir,
                               f"heatmap_{metric}_scale_{clean_scale(scale)}_T{epoch}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"✅ Saved: {out}")
            plt.close()


def plot_heatmaps_baseline_diff(all_rows, output_dir, metric="nll"):
    """Heatmap showing metric(source) - metric(baseline).

    Negative = source context helps vs no context (green).
    Positive = baseline is better than using this source (red).
    Skips epochs with no baseline (e.g., ep0).
    """
    metric_label = metric.upper()
    target_hps = sorted_target_hps(all_rows)
    target_labels = [short_target_label(hp) for hp in target_hps]
    source_labels = [source_hp_label(c) for c in SOURCE_CONFIGS]
    baseline = _build_baseline_lookup(all_rows, metric)

    if not baseline:
        print("  ⚠️  No baseline data found — skipping baseline-diff heatmaps")
        return

    # Collect all diff values for universal color bounds
    all_diffs = []
    for scale in BASE_SCALES:
        for epoch in OBS_EPOCHS:
            for r in all_rows:
                if r["scale"] != scale or r["epoch"] != epoch or r["scale"] == "baseline":
                    continue
                bl_key = (epoch, r["target_hp"])
                if bl_key in baseline and not np.isnan(r[metric]):
                    all_diffs.append(r[metric] - baseline[bl_key])

    if not all_diffs:
        print("  ⚠️  No overlapping epochs between source and baseline")
        return

    # Symmetric bounds centered at 0
    abs_max = np.percentile(np.abs(all_diffs), 98)
    print(f"  Baseline-diff color scale: [-{abs_max:.2f}, +{abs_max:.2f}]")
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

    for scale in BASE_SCALES:
        for epoch in OBS_EPOCHS:
            # Check if baseline exists for this epoch
            if not any((epoch, hp) in baseline for hp in target_hps):
                continue

            matrix = np.full((12, 12), np.nan)

            for r in all_rows:
                if r["scale"] != scale or r["epoch"] != epoch or r["scale"] == "baseline":
                    continue
                src_idx = r["source_config"] - 1
                tgt_idx = (target_hps.index(r["target_hp"])
                           if r["target_hp"] in target_hps else -1)
                bl_key = (epoch, r["target_hp"])
                if 0 <= src_idx < 12 and 0 <= tgt_idx < 12 and bl_key in baseline:
                    matrix[src_idx, tgt_idx] = r[metric] - baseline[bl_key]

            fig, ax = plt.subplots(figsize=(12, 10))

            valid = matrix[~np.isnan(matrix)]
            if len(valid) == 0:
                plt.close()
                continue

            im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", norm=norm)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label(
                f"Δ{metric_label}  (source − baseline)\n"
                f"← negative = source helps  |  positive = baseline better →",
                fontsize=9,
            )

            # Annotate cells
            for i in range(12):
                for j in range(12):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = "white" if abs(val) > abs_max * 0.6 else "black"
                        sign = "+" if val > 0 else ""
                        ax.text(j, i, f"{sign}{val:.2f}", ha="center", va="center",
                                fontsize=7, fontweight="bold", color=color)

            ax.set_xticks(range(12))
            ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(12))
            ax.set_yticklabels(source_labels, fontsize=8)
            ax.set_xlabel("Target HP (hd=128)", fontsize=11)
            ax.set_ylabel("Source HP (context)", fontsize=11)
            ax.set_title(
                f"Δ{metric_label} vs Baseline — Scale={clean_scale(scale)}, T={epoch}\n"
                f"(Green = source helps, Red = baseline better)",
                fontsize=12, fontweight="bold",
            )

            plt.tight_layout()
            out = os.path.join(
                output_dir,
                f"heatmap_{metric}_baseline_diff_scale_{clean_scale(scale)}_T{epoch}.png",
            )
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"✅ Saved: {out}")
            plt.close()
