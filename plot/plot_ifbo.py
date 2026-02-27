import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import normalize_log_loss_curve  # <-- use SAME function as training

# ===================================================
# USER INPUT
# ===================================================
RESULTS_DIR = "./results"
PRED_DIR = "./results/norm 2.0"
CURVES_FILE = os.path.join(RESULTS_DIR, "results_metrics.json")
OUTPUT_DIR = "ablation_plots_normalized"
os.makedirs(OUTPUT_DIR, exist_ok=True)

epochs_observed_list = [0, 5, 10, 20]

PRED_PREFIXES = [
    "ifbo_pred_[64_32_24]_configs_1_ep",
    "ifbo_pred_[64_32_24]_configs_5_ep",
    "ifbo_pred_[64_32_24]_configs_12_ep",
    "ifbo_pred_[64]_configs_1_ep",
    "ifbo_pred_[64]_configs_5_ep",
    "ifbo_pred_[64]_configs_12_ep",
    "ifbo_pred_[64_32]_configs_1_ep",
    "ifbo_pred_[64_32]_configs_5_ep",
    "ifbo_pred_[64_32]_configs_12_ep",
    "ifbo_pred_[32]_configs_1_ep",
    "ifbo_pred_[32]_configs_5_ep",
    "ifbo_pred_[32]_configs_12_ep",
    "ifbo_pred_[32_24]_configs_1_ep",
    "ifbo_pred_[32_24]_configs_5_ep",
    "ifbo_pred_[32_24]_configs_12_ep",
    "ifbo_pred_[24]_configs_1_ep",
    "ifbo_pred_[24]_configs_5_ep",
    "ifbo_pred_[24]_configs_12_ep",
]

# ===================================================
# NePS SEARCH SPACE
# ===================================================
NEPS_SPACE = {
    "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 0.01],
    "lr_schedule": ["cosine"],
    "num_layers": [4],
    "hidden_dim": [128],
}

# ===================================================
# LOAD METRICS
# ===================================================
with open(CURVES_FILE) as f:
    metrics = json.load(f)

# ===================================================
# EXTRACT PATTERNS
# ===================================================
patterns = []
run_lookup = {}

for run_key, v in metrics.items():
    if (
        v.get("lr") in NEPS_SPACE["lr"]
        and v.get("weight_decay") in NEPS_SPACE["weight_decay"]
        and v.get("lr_schedule") in NEPS_SPACE["lr_schedule"]
        and v.get("num_layers") in NEPS_SPACE["num_layers"]
        and v.get("hidden_dim") in NEPS_SPACE["hidden_dim"]
    ):
        pattern = (
            v["lr"],
            v["weight_decay"],
            v["lr_schedule"],
            v["num_layers"],
            v["hidden_dim"],
        )
        patterns.append(pattern)
        run_lookup[pattern] = run_key

patterns = sorted(set(patterns))
print(f"✅ Found {len(patterns)} patterns")

# ===================================================
# MAIN LOOP
# ===================================================
colors = plt.cm.tab20(np.linspace(0, 1, len(PRED_PREFIXES)))

for pattern in patterns:
    lr_val, wd_val, sched_val, layers_val, hd_val = pattern
    run_key = run_lookup[pattern]
    run_data = metrics[run_key]

    full_curve_raw = np.asarray(run_data.get("val_loss_curve", []), dtype=float)
    if len(full_curve_raw) == 0:
        continue

    # 🔥 Normalize FULL curve using same function as training
    full_curve_norm = normalize_log_loss_curve(full_curve_raw)

    for obs_epoch in epochs_observed_list:

        n_cols = 3
        n_rows = int(np.ceil(len(PRED_PREFIXES) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), dpi=200)
        axes = axes.flatten()

        for idx, (prefix, color) in enumerate(zip(PRED_PREFIXES, colors)):
            if idx >= len(axes):
                break

            ax = axes[idx]

            pred_filename = f"{prefix}{obs_epoch}.json"
            pred_path = os.path.join(PRED_DIR, pred_filename)

            if not os.path.exists(pred_path):
                continue

            with open(pred_path) as f:
                pred_file_data = json.load(f)

            key = f"layer{layers_val}_lr{lr_val}_hd{hd_val}_wd{wd_val}_{sched_val}_{obs_epoch}"

            if key not in pred_file_data:
                continue

            pred = pred_file_data[key]

            # -----------------------------
            # Ground Truth (normalized)
            # -----------------------------
            ax.plot(
                np.arange(1, len(full_curve_norm) + 1),
                full_curve_norm,
                color=color,
                linewidth=2,
                alpha=0.8,
                label="Ground Truth",
            )

            # -----------------------------
            # Observed portion
            # -----------------------------
            val_obs_norm = full_curve_norm[:obs_epoch]
            ax.plot(
                np.arange(1, len(val_obs_norm) + 1),
                val_obs_norm,
                color=color,
                linewidth=4,
                label="Observed",
            )

            # -----------------------------
            # Prediction (already normalized)
            # -----------------------------
            mean_pred_norm = np.asarray(pred["point"])
            lower_pred_norm = np.asarray(pred["quantiles"]["0.05"])
            upper_pred_norm = np.asarray(pred["quantiles"]["0.95"])

            epochs_pred = np.arange(
                len(val_obs_norm),
                len(val_obs_norm) + len(mean_pred_norm),
            )

            ax.plot(
                epochs_pred,
                mean_pred_norm,
                color=color,
                linestyle="--",
                linewidth=2,
                label="Prediction",
            )

            ax.fill_between(
                epochs_pred,
                lower_pred_norm,
                upper_pred_norm,
                color=color,
                alpha=0.2,
            )

            # -----------------------------
            # Formatting
            # -----------------------------
            pred_name_part = prefix.split("ifbo_pred_")[1].replace("_ep", "")

            ax.set_title(
                f"{pred_name_part} | lr={lr_val}, wd={wd_val} obs_epoch={obs_epoch}",
                fontsize=9,
            )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Normalized Val Score (1 - log-loss)")
            y_min, y_max = 0.0, 1.0       # fixed y-axis range
            ax.set_ylim(y_min, y_max)
            ax.grid(True)

        # Remove unused axes
        for j in range(len(PRED_PREFIXES), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()

        filename = f"lr{lr_val}_wd{wd_val}_obs{obs_epoch}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)

        plt.savefig(out_path)
        plt.close()

        print(f"✅ Saved: {out_path}")