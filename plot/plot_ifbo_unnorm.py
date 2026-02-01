import os
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import fixed_range_normalize, min_max_normalize

# ===================================================
# USER INPUT
# ===================================================
RESULTS_DIR = "./results"
PRED_DIR = "./results"
CURVES_FILE = os.path.join(RESULTS_DIR, "results_metrics.json")
OUTPUT_DIR = "new_ablation_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Observed epochs
epochs_observed_list = [0, 5, 10, 20]

# Prediction file prefixes (15 files)
PRED_PREFIXES = [
    "ifbo_pred_[4]_ep",
    "ifbo_pred_[8]_ep",
    "ifbo_pred_[16]_ep",
    "ifbo_pred_[24]_ep",
    "ifbo_pred_[32]_ep",
    "ifbo_pred_[64]_ep",
    "ifbo_pred_[64_32]_ep",
    "ifbo_pred_[64_32_24]_ep",
    "ifbo_pred_[64_32_24_16]_ep",
    "ifbo_pred_[64_32_24_16_8]_ep",
    "ifbo_pred_[64_32_24_16_8_4]_ep",
    "ifbo_pred_[64_24]_ep",
    "ifbo_pred_[64_16]_ep",
    "ifbo_pred_[64_8]_ep",
    "ifbo_pred_[64_4]_ep",
    "ifbo_pred_[32_24_16_8_4]_ep",
    "ifbo_pred_[24_16_8_4]_ep",
    "ifbo_pred_[16_8_4]_ep",
    "ifbo_pred_[8_4]_ep",
    "ifbo_pred_[64_24_16_8_4]_ep",
    "ifbo_pred_[64_32_16_8_4]_ep",
    "ifbo_pred_[64_32_24_8_4]_ep",
    "ifbo_pred_[64_32_24_16_4]_ep",
    "ifbo_pred_[64_8_4]_ep",
    "ifbo_pred_[32_24_16]_ep",
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
run_lookup = {}  # maps pattern -> run_key
for run_key, v in metrics.items():
    if (
        v.get("lr") in NEPS_SPACE["lr"] and
        v.get("weight_decay") in NEPS_SPACE["weight_decay"] and
        v.get("lr_schedule") in NEPS_SPACE["lr_schedule"] and
        v.get("num_layers") in NEPS_SPACE["num_layers"] and
        v.get("hidden_dim") in NEPS_SPACE["hidden_dim"]
    ):
        pattern = (v["lr"], v["weight_decay"], v["lr_schedule"], v["num_layers"], v["hidden_dim"])
        patterns.append(pattern)
        run_lookup[pattern] = run_key

patterns = sorted(set(patterns))
print(f"✅ Found {len(patterns)} patterns")

# ===================================================
# MAIN LOOP: one pattern at a time
# ===================================================
colors = plt.cm.tab20(np.linspace(0, 1, len(PRED_PREFIXES)))

for pat_idx, pattern in enumerate(patterns):
    lr_val, wd_val, sched_val, layers_val, hd_val = pattern
    run_key = run_lookup[pattern]
    run_data = metrics[run_key]

    # Load full curve once
    full_curve_raw = np.asarray(run_data.get("val_acc_curve", []), dtype=float)
    if len(full_curve_raw) == 0:
        continue
    y_min, y_max = 0.0, 1.0

    for obs_epoch in epochs_observed_list:
        n_rows, n_cols = 5, 5  # 3 plots horizontally, 5 vertically
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 25), dpi=200)
        axes = axes.flatten()

        for idx, (prefix, color) in enumerate(zip(PRED_PREFIXES, colors)):
            if idx >= n_rows * n_cols:
                break
            ax = axes[idx]

            pred_filename = f"{prefix}{obs_epoch}.json"
            pred_path = os.path.join(PRED_DIR, pred_filename)

            if not os.path.exists(pred_path):
                print(f"❌ Missing IFBO file: {pred_filename}")
                continue

            with open(pred_path) as f:
                pred_file_data = json.load(f)

            key = f"layer{layers_val}_lr{lr_val}_hd{hd_val}_wd{wd_val}_{sched_val}_{obs_epoch}"
            if key not in pred_file_data:
                print(f"❌ Missing IFBO key in {pred_filename}: {key}")
                continue

            pred = pred_file_data[key]
            print(prefix, "prefix")

            # -----------------------------
            # Full ground truth
            # -----------------------------
            ax.plot(
                np.arange(1, len(full_curve_raw)+1),
                full_curve_raw,
                color=color,
                linewidth=2,
                alpha=0.8,
                label='Ground Truth'
            )

            # -----------------------------
            # Observed curve
            # -----------------------------
            val_obs_raw = full_curve_raw[:obs_epoch]
            epochs_obs = np.arange(1, len(val_obs_raw)+1)
            ax.plot(
                epochs_obs,
                val_obs_raw,
                color=color,
                linewidth=4,
                label='Observed'
            )

            # -----------------------------
            # Prediction
            # -----------------------------
            mean_pred_norm = np.asarray(pred["point"])
            lower_pred_norm = np.asarray(pred["quantiles"]["0.05"])
            upper_pred_norm = np.asarray(pred["quantiles"]["0.95"])

            mean_pred_raw = mean_pred_norm * (y_max - y_min) + y_min
            lower_pred_raw = lower_pred_norm * (y_max - y_min) + y_min
            upper_pred_raw = upper_pred_norm * (y_max - y_min) + y_min

            epochs_pred = np.arange(len(val_obs_raw), len(val_obs_raw)+len(mean_pred_raw))
            ax.plot(epochs_pred, mean_pred_raw, color=color, linestyle='--', linewidth=2)
            ax.fill_between(epochs_pred, lower_pred_raw, upper_pred_raw, color=color, alpha=0.2)

            # -----------------------------
            # Formatting
            # -----------------------------
            pred_name_part = prefix.split("ifbo_pred_")[1].replace("_ep","")
            ax.set_title(f"{pred_name_part} | lr={lr_val}, wd={wd_val}", fontsize=10)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Val Acc")
            ax.set_ylim(y_min, y_max)
            ax.grid(True)

        # Remove any extra subplots
        for j in range(len(PRED_PREFIXES), n_rows*n_cols):
            fig.delaxes(axes[j])

        plt.tight_layout()

        # -----------------------------
        # Save figure using pred file + pattern
        # -----------------------------
        filename = f"lr{lr_val}_wd{wd_val}_obs{obs_epoch}.png"
        # filename = filename.replace(".", "_")  # optional: replace dots with 'p'

        out_path = os.path.join(OUTPUT_DIR, filename)
        plt.savefig(out_path)
        plt.close()
        print(f"✅ Saved: {out_path}")
