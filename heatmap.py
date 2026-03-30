import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

# -----------------------------
# Define experiment labels + files
# -----------------------------
file_A = "table_ohd.json"
file_B = "table_pn.json"

label_A = "With-HD"
label_B = "Per-curve"

# -----------------------------
# Load JSON files
# -----------------------------
with open(file_A) as f:
    data_A = json.load(f)

with open(file_B) as f:
    data_B = json.load(f)

# -----------------------------
# Config
# -----------------------------
widths = [32, 24, "32_24"]
epochs = ["ep0", "ep5", "ep20", "ep50", "ep90"]
n = 12

os.makedirs("diff_plots", exist_ok=True)

# -----------------------------
# STEP 1: Collect diffs for scaling
# -----------------------------
all_mse_diffs = []
all_nll_diffs = []

eps = 1e-8

for width in widths:
    for epoch in epochs:
        for i in range(n):
            for j in range(n):
                key = f"{width}_config_{i+1}_obs_target_{epoch}"
                col = f"config_{j+1}"

                try:
                    a_mse = data_A[key][col]["mse"]
                    b_mse = data_B[key][col]["mse"]

                    a_nll = data_A[key][col]["nll"]
                    b_nll = data_B[key][col]["nll"]

                    mse_diff = (a_mse - b_mse) / (abs(a_mse) + abs(b_mse) + eps)
                    nll_diff = (a_nll - b_nll) / (abs(a_nll) + abs(b_nll) + eps)

                    all_mse_diffs.append(mse_diff)
                    all_nll_diffs.append(nll_diff)

                except KeyError:
                    continue

all_mse_diffs = np.array(all_mse_diffs)
all_nll_diffs = np.array(all_nll_diffs)

# symmetric → already bounded [-1,1], but clip for robustness
mse_abs_max = np.percentile(np.abs(all_mse_diffs), 95)
nll_abs_max = np.percentile(np.abs(all_nll_diffs), 95)

print("Color limits:")
print(f"MSE ±{mse_abs_max:.3f}")
print(f"NLL ±{nll_abs_max:.3f}")

# -----------------------------
# Stats
# -----------------------------
def compute_stats(diff):
    valid = diff[~np.isnan(diff)]

    mean = np.mean(valid)
    A_win = np.sum(valid < 0) / valid.size * 100
    B_win = np.sum(valid > 0) / valid.size * 100
    tie = np.sum(valid == 0) / valid.size * 100

    return mean, A_win, B_win, tie


# -----------------------------
# STEP 2: MAIN LOOP
# -----------------------------
for width in widths:
    for epoch in epochs:

        mse_A = np.zeros((n, n))
        mse_B = np.zeros((n, n))

        nll_A = np.zeros((n, n))
        nll_B = np.zeros((n, n))

        # -----------------------------
        # Fill matrices
        # -----------------------------
        for i in range(n):
            for j in range(n):
                key = f"{width}_config_{i+1}_obs_target_{epoch}"
                col = f"config_{j+1}"

                try:
                    mse_A[i, j] = data_A[key][col]["mse"]
                    mse_B[i, j] = data_B[key][col]["mse"]

                    nll_A[i, j] = data_A[key][col]["nll"]
                    nll_B[i, j] = data_B[key][col]["nll"]

                except KeyError:
                    mse_A[i, j] = np.nan
                    mse_B[i, j] = np.nan
                    nll_A[i, j] = np.nan
                    nll_B[i, j] = np.nan

        # -----------------------------
        # Normalized differences
        # -----------------------------
        mse_diff = (mse_A - mse_B) / (np.abs(mse_A) + np.abs(mse_B) + eps)
        nll_diff = (nll_A - nll_B) / (np.abs(nll_A) + np.abs(nll_B) + eps)

        # -----------------------------
        # Stats
        # -----------------------------
        mse_mean, mse_Awin, mse_Bwin, mse_tie = compute_stats(mse_diff)
        nll_mean, nll_Awin, nll_Bwin, nll_tie = compute_stats(nll_diff)

        print(f"\nWidth={width}, Epoch={epoch}")
        print(f"MSE mean: {mse_mean:.4f}, {label_A}: {mse_Awin:.2f}%, {label_B}: {mse_Bwin:.2f}%")
        print(f"NLL mean: {nll_mean:.4f}, {label_A}: {nll_Awin:.2f}%, {label_B}: {nll_Bwin:.2f}%")

        # -----------------------------
        # Plot
        # -----------------------------
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # ---- MSE ----
        sns.heatmap(
            mse_diff,
            cmap="RdYlGn_r",
            center=0,
            vmin=-mse_abs_max,
            vmax=mse_abs_max,
            ax=axes[0],
            cbar_kws={"label": f"Normalized ({label_A} - {label_B})"}
        )
        axes[0].set_title(
            f"MSE (Normalized)\n"
            f"Mean={mse_mean:.3f}\n"
            f"{label_A}:{mse_Awin:.1f}% | {label_B}:{mse_Bwin:.1f}%"
        )
        axes[0].set_xlabel("Target Config")
        axes[0].set_ylabel("Source Config")

        # ---- NLL ----
        sns.heatmap(
            nll_diff,
            cmap="RdYlGn_r",
            center=0,
            vmin=-nll_abs_max,
            vmax=nll_abs_max,
            ax=axes[1],
            cbar_kws={"label": f"Normalized ({label_A} - {label_B})"}
        )
        axes[1].set_title(
            f"NLL (Normalized)\n"
            f"Mean={nll_mean:.3f}\n"
            f"{label_A}:{nll_Awin:.1f}% | {label_B}:{nll_Bwin:.1f}%"
        )
        axes[1].set_xlabel("Target Config")
        axes[1].set_ylabel("Source Config")

        fig.suptitle(f"Width={width}, Epoch={epoch}", fontsize=14)

        plt.tight_layout()
        plt.savefig(f"diff_plots/diff_w{width}_{epoch}.png", dpi=300)
        plt.close()

print("\n✅ Done. All plots saved in 'diff_plots/'")