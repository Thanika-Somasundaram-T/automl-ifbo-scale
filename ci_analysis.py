import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── load data ─────────────────────────────
DIR = "ablation2_scales/t610"
with open(f"{DIR}/analysis_nested.json", "r") as f:
    data = json.load(f)

contexts = ["32270848", "77124608", "134561280", "287183360"]
obs_levels = ["0", "20", "50", "90"]

cmap = plt.cm.RdYlGn_r

# ── loop over runs ───────────────────────
for run_key, ctx_dict in data.items():

    matrix = np.full((len(obs_levels), len(contexts)), np.nan)
    ci_matrix = np.zeros_like(matrix, dtype=bool)

    # ── fill matrix ───────────────────────
    for i, obs in enumerate(obs_levels):
        for j, ctx in enumerate(contexts):

            ctx_key = f"context_{ctx}"

            if ctx_key in ctx_dict and obs in ctx_dict[ctx_key]:

                rel_err = float(ctx_dict[ctx_key][obs]["rel_err"])
                matrix[i, j] = round(rel_err, 2)

                ci_matrix[i, j] = ctx_dict[ctx_key][obs]["true_in_ci"]

    # ================================
    # 1. GLOBAL HEATMAP
    # ================================
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    valid_vals = matrix[~np.isnan(matrix)]
    vmin = np.percentile(valid_vals, 5)
    vmax = np.percentile(valid_vals, 95)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    im0 = ax[0].imshow(matrix, cmap=cmap, norm=norm)

    ax[0].set_title("Global Rel Error (red = worse, green = better)")
    ax[0].set_xticks(range(len(contexts)))
    ax[0].set_xticklabels([f"{int(c)//1_000_000}M" for c in contexts])
    ax[0].set_yticks(range(len(obs_levels)))
    ax[0].set_yticklabels(obs_levels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            val = matrix[i, j]
            if np.isnan(val):
                continue

            ci_text = "True" if ci_matrix[i, j] else "False"
            ci_color = "black" if ci_matrix[i, j] else "white"

            ax[0].text(
                j, i,
                f"{val:.2f}\n{ci_text}",
                ha="center",
                va="center",
                fontsize=8,
                color=ci_color
            )

    fig.colorbar(im0, ax=ax[0], label="Rel Error (lower = better)")

    # ================================
    # 2. ROW-WISE RANK HEATMAP
    # ================================
    rank_matrix = np.full_like(matrix, np.nan)

    for i in range(matrix.shape[0]):
        row = matrix[i]

        valid_idx = np.where(~np.isnan(row))[0]
        if len(valid_idx) == 0:
            continue

        sorted_idx = valid_idx[np.argsort(row[valid_idx])]

        for rank, j in enumerate(sorted_idx):
            rank_matrix[i, j] = rank

    n_ctx = len(contexts)

    im1 = ax[1].imshow(rank_matrix, cmap=plt.cm.RdYlGn_r, vmin=0, vmax=n_ctx - 1)

    ax[1].set_title("Row-wise ranking (per obs)")
    ax[1].set_xticks(range(len(contexts)))
    ax[1].set_xticklabels([f"{int(c)//1_000_000}M" for c in contexts])
    ax[1].set_yticks(range(len(obs_levels)))
    ax[1].set_yticklabels(obs_levels)

    # ── annotate ──
    for i in range(matrix.shape[0]):

        row = matrix[i]
        if np.all(np.isnan(row)):
            continue

        for j in range(matrix.shape[1]):

            val = matrix[i, j]
            if np.isnan(val):
                continue

            ci_text = "True" if ci_matrix[i, j] else "False"
            ci_color = "black" if ci_matrix[i, j] else "white"

            ax[1].text(
                j, i,
                f"{val:.2f}\n{ci_text}",
                ha="center",
                va="center",
                fontsize=8,
                color=ci_color
            )

    plt.tight_layout()

    output_dir = f"{DIR}/heatmap"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f"{output_dir}/{run_key}.png", dpi=150)
    plt.close()

print("Saved corrected heatmaps.")