"""
Heatmap visualizations: 12×12 source→target metric heatmaps.
Supports both NLL and MSE metrics.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from config import (
    BASE_SCALES, OBS_EPOCHS, SOURCE_CONFIGS,
    sorted_target_hps, short_target_label, source_hp_label, clean_scale,
)


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
