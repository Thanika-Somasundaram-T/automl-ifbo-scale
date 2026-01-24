import os
import json
import matplotlib.pyplot as plt
import numpy as np

from utils import fixed_range_normalize, min_max_normalize

# ---------------------------------------------------
# Hardcoded observed epochs
# ---------------------------------------------------
epochs_observed_list = [15, 50, 90]

for obs_epoch in epochs_observed_list:

    # ---------------------------------------------------
    # Paths
    # ---------------------------------------------------
    results_dir = "./results"
    pred_dir = "./results"
    output_dir = "baseline_128_lr0.00001"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------
    # Selected patterns
    # ---------------------------------------------------
    patterns = [(0.00001, 0.0, "cosine", 4, 128), (0.00001, 0.01, "cosine", 4, 128)]
    colors = ["red", "blue", "green", "orange", "purple", "yellow"]

    # ---------------------------------------------------
    # Load training metrics
    # ---------------------------------------------------
    metrics_file = os.path.join(results_dir, "results_metrics.json")
    with open(metrics_file) as f:
        metrics = json.load(f)

    plt.figure(figsize=(16, 10), dpi=200)
    plt.title(f"Observed + IFBO Predicted Validation Accuracy (Observed {obs_epoch} epochs)", fontsize=22)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Validation Accuracy", fontsize=18)

    # ---------------------------------------------------
    # Loop patterns
    # ---------------------------------------------------
    for (lr_val, wd_val, sched_val, layers_val, hd_val), color in zip(patterns, colors):

        # Find matching run
        run_key = None
        for k, v in metrics.items():
            if (
                v.get("lr") == lr_val and
                v.get("weight_decay") == wd_val and
                v.get("lr_schedule") == sched_val and
                v.get("num_layers") == layers_val and
                v.get("hidden_dim") == hd_val
            ):
                run_key = k
                break

        if not run_key:
            print(f"No match for lr={lr_val}, wd={wd_val}, sched={sched_val}, L={layers_val}, hd={hd_val}")
            continue

        run_data = metrics[run_key]

        # ---------------------------------------------------
        # Full raw curve
        # ---------------------------------------------------
        full_curve_raw = np.array(run_data.get("val_acc_curve", []), dtype=float)
        if len(full_curve_raw) == 0:
            continue

        # Normalize full curve for plotting comparison (optional)
        # full_curve_norm = fixed_range_normalize(full_curve_raw)
        y_min, y_max = 0.0, 1.0

        # Plot full raw curve (light shade)
        plt.plot(
            np.arange(1, len(full_curve_raw) + 1),
            full_curve_raw,
            color=color,
            linewidth=2,
            alpha=0.5,
        )

        # ---------------------------------------------------
        # Observed curve (raw)
        # ---------------------------------------------------
        val_obs_raw = full_curve_raw[:obs_epoch]
        epochs_obs = np.arange(1, len(val_obs_raw) + 1)

        plt.plot(
            epochs_obs,
            val_obs_raw,
            color=color,
            linewidth=3,
            label=f"{sched_val}, lr={lr_val}, wd={wd_val}, L={layers_val}, hd={hd_val} (obs {obs_epoch})"
        )

        # ---------------------------------------------------
        # Load IFBO prediction
        # ---------------------------------------------------
        pred_filename = f"baseline_{obs_epoch}.json"
        pred_path = os.path.join(pred_dir, pred_filename)
        if not os.path.exists(pred_path):
            print(f"❌ Missing IFBO prediction: {pred_filename}")
            continue

        with open(pred_path) as f:
            pred_file_data = json.load(f)

        key = f"layer{layers_val}_lr{lr_val}_hd{hd_val}_wd{wd_val}_{sched_val}_{obs_epoch}"

        if key not in pred_file_data:
            print(f"❌ Missing prediction key in {pred_filename}: {key}")
            continue

        pred = pred_file_data[key]

        mean_pred_norm = np.array(pred["point"])
        lower_pred_norm = np.array(pred["quantiles"]["0.05"])
        upper_pred_norm = np.array(pred["quantiles"]["0.95"])

        # ---------------------------------------------------
        # Unnormalize predicted values
        # ---------------------------------------------------
        mean_pred_raw = mean_pred_norm * (y_max - y_min) + y_min
        lower_pred_raw = lower_pred_norm * (y_max - y_min) + y_min
        upper_pred_raw = upper_pred_norm * (y_max - y_min) + y_min

        # Epoch numbering for predicted tail
        epochs_pred = np.arange(len(val_obs_raw), len(val_obs_raw) + len(mean_pred_raw))

        # Plot unnormalized predicted curve
        plt.plot(epochs_pred, mean_pred_raw, color=color, linewidth=2, linestyle="--")
        plt.fill_between(epochs_pred, lower_pred_raw, upper_pred_raw, color=color, alpha=0.2)

    plt.grid(True)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ifbo_val_acc_obs{obs_epoch}_unnorm.png"))
    print(f"✅ Saved: {output_dir}/ifbo_val_acc_obs{obs_epoch}_unnorm.png")
