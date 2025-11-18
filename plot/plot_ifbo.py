import os
import json
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------
# Hardcoded observed epochs
# ---------------------------------------------------
epochs_observed_list = [0, 10, 15, 30, 45, 50, 70, 90, 100]   # <-- You change this anytime
obs_epoch = 45                                       # <-- The run you want to visualize
assert obs_epoch in epochs_observed_list

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
results_dir = "./results"
pred_dir = "./results"
output_dir = "ifbo_plots_both64_context_128"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------
# Selected patterns: (lr, wd, schedule, layers, hidden_dim)
# ---------------------------------------------------
patterns = [
    (0.01, 0.0001, "cosine", 4, 128),
    (0.001, 0.0, "cosine", 4, 128),
    (0.001, 0.0001, "cosine", 4, 128),
    (0.0001, 0.001, "cosine", 4, 128),
    (0.0003, 0.001, "cosine", 4, 128),
    (0.0003, 0.0, "cosine", 4, 128),
]

colors = ["red", "blue", "green", "orange", "purple", "brown"]

# ---------------------------------------------------
# Load training metrics
# ---------------------------------------------------
metrics_file = os.path.join(results_dir, "results_metrics.json")
with open(metrics_file) as f:
    metrics = json.load(f)

plt.figure(figsize=(16, 10), dpi=200)
plt.title(f"Observed + IFBO Predicted Validation Loss (Observed {obs_epoch} epochs)", fontsize=22)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Validation Loss", fontsize=18)

# ---------------------------------------------------
# Loop patterns
# ---------------------------------------------------
for (lr_val, wd_val, sched_val, layers_val, hd_val), color in zip(patterns, colors):

    # ---------------------------------------------------
    # Find matching run in result_metrics.json
    # ---------------------------------------------------
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
    # Observed curve: first obs_epoch points
    # ---------------------------------------------------
    full_curve = np.array(run_data.get("val_loss_curve", []))
    val_obs = full_curve[:obs_epoch]

    epochs_obs = np.arange(1, len(val_obs) + 1)

    plt.plot(
        epochs_obs,
        val_obs,
        color=color,
        linewidth=3,
        label=f"{sched_val}, lr={lr_val}, wd={wd_val}, L={layers_val}, hd={hd_val} (obs {obs_epoch})"
    )

    # ---------------------------------------------------
    # Load IFBO prediction for this observed epoch
    # ---------------------------------------------------
    pred_filename = f"ifbo_pred_both64_context_{obs_epoch}.json"
    pred_path = os.path.join(pred_dir, pred_filename)
    if not os.path.exists(pred_path):
        print(f"❌ Missing IFBO prediction: {pred_filename}")
        continue

    with open(pred_path) as f:
        pred_file_data = json.load(f)

    # Key format inside JSON:
    #   layer4_lr0.0001_hd64_wd0.0_cosine_30
    key = f"layer{layers_val}_lr{lr_val}_hd{hd_val}_wd{wd_val}_{sched_val}_{obs_epoch}"

    if key not in pred_file_data:
        print(f"❌ Missing prediction key in {pred_filename}: {key}")
        continue

    pred = pred_file_data[key]

    mean_pred = np.array(pred["point"])
    lower_pred = np.array(pred["quantiles"]["0.05"])
    upper_pred = np.array(pred["quantiles"]["0.95"])

    # Epoch numbering for predicted tail
    epochs_pred = np.arange(
        len(val_obs),
        len(val_obs) + len(mean_pred)
    )

    # ---------------------------------------------------
    # Plot predicted curve
    # ---------------------------------------------------
    plt.plot(epochs_pred, mean_pred, color=color, linewidth=2, linestyle='--')
    plt.fill_between(epochs_pred, lower_pred, upper_pred, color=color, alpha=0.2)

plt.grid(True)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"ifbo_val_loss_obs{obs_epoch}.png"))
print(f"✅ Saved: {output_dir}/ifbo_val_loss_obs{obs_epoch}.png")
