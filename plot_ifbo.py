# import os, json
# import matplotlib.pyplot as plt
# import numpy as np

# # Paths
# results_dir = "./results_f"
# pred_dir = "./results_f"
# output_dir = "plots_best"
# os.makedirs(output_dir, exist_ok=True)

# # Updated patterns: (lr, wd, schedule)
# patterns = [
#     (0.01, 0.0001, "cosine"),
#     (0.001, 0.0, "cosine"),
#     (0.001, 0.0001, "warmup_cooldown"),
#     (0.0001, 0.001, "warmup_cooldown"),
#     (0.0003, 0.001, "cooldown"),
#     (0.0003, 0.0, "cooldown"),
# ]

# colors = ["red", "blue", "green", "orange", "purple", "brown"]

# # Load metrics
# with open(os.path.join(results_dir, "result64.json")) as f:
#     metrics = json.load(f)

# plt.figure(figsize=(16, 10), dpi=200)
# plt.title("Selected Runs: Observed + IFBO Predicted Validation Loss", fontsize=22)
# plt.xlabel("Epoch", fontsize=18)
# plt.ylabel("Validation Loss", fontsize=18)

# for (lr_val, wd_val, sched_val), color in zip(patterns, colors):
#     # Find matching run by JSON fields
#     run_key = None
#     for k, v in metrics.items():
#         if (v.get("lr") == lr_val and
#             v.get("weight_decay") == wd_val and
#             v.get("lr_schedule") == sched_val):
#             run_key = k
#             break

#     if not run_key:
#         print(f"No match for lr={lr_val}, wd={wd_val}, sched={sched_val}")
#         continue

#     run_data = metrics[run_key]
#     val_curve = run_data.get("val_loss_curve", [])[:10]
#     epochs_obs = np.arange(1, len(val_curve) + 1)

#     # Plot observed curve
#     plt.plot(epochs_obs, val_curve, color=color, linewidth=3,
#              label=f"{sched_val}, lr={lr_val}, wd={wd_val}")

#     # Plot IFBO prediction if exists
#     pred_path = os.path.join(pred_dir, f"pred_{run_key}.json")
#     if os.path.exists(pred_path):
#         with open(pred_path) as f:
#             pred = json.load(f)
#         mean_pred = np.array(pred["point"])
#         lower_pred = np.array(pred["quantiles"]["0.05"])
#         upper_pred = np.array(pred["quantiles"]["0.95"])
#         epochs_pred = np.arange(len(val_curve) + 1, len(val_curve) + 1 + len(mean_pred))

#         plt.plot(epochs_pred, mean_pred, color=color, linewidth=2)
#         plt.fill_between(epochs_pred, lower_pred, upper_pred, color=color, alpha=0.2)

# plt.grid(True)
# plt.legend(fontsize=14)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "ifbo_val_loss.png"))
# print("✅ Saved: plots_best/ifbo_val_loss.png")


import os
import json
import matplotlib.pyplot as plt
import numpy as np

# --- Paths ---
results_dir = "./results_f"
pred_dir = "./results_f"
output_dir = "plots_best"
os.makedirs(output_dir, exist_ok=True)

# --- Six selected patterns: (lr, wd, schedule) ---
patterns = [
    (0.01, 0.0001, "cosine"),
    (0.001, 0.0, "cosine"),
    (0.001, 0.0001, "warmup_cooldown"),
    (0.0001, 0.001, "warmup_cooldown"),
    (0.0003, 0.001, "cooldown"),
    (0.0003, 0.0, "cooldown"),
]

colors = ["red", "blue", "green", "orange", "purple", "brown"]

# --- Load metrics ---
with open(os.path.join(results_dir, "result64.json")) as f:
    metrics = json.load(f)

plt.figure(figsize=(16, 10), dpi=200)
plt.title("Selected Runs: Observed + IFBO Predicted Validation Loss", fontsize=22)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Validation Loss", fontsize=18)

for (lr_val, wd_val, sched_val), color in zip(patterns, colors):
    # Find matching run by JSON fields
    run_key = None
    for k, v in metrics.items():
        if (v.get("lr") == lr_val and
            v.get("weight_decay") == wd_val and
            v.get("lr_schedule") == sched_val):
            run_key = k
            break

    if not run_key:
        print(f"No match for lr={lr_val}, wd={wd_val}, sched={sched_val}")
        continue

    run_data = metrics[run_key]
    val_curve_obs = np.array(run_data.get("val_loss_curve", [])) # first 10 epochs observed
    epochs_obs = np.arange(1, len(val_curve_obs) + 1)

    # Plot observed solid curve
    plt.plot(epochs_obs, val_curve_obs, color=color, linewidth=3,
             label=f"{sched_val}, lr={lr_val}, wd={wd_val} (obs)")

    # Plot IFBO prediction if exists
    pred_path = os.path.join(pred_dir, f"pred_{run_key}.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            pred = json.load(f)
        mean_pred = np.array(pred["point"])
        lower_pred = np.array(pred["quantiles"]["0.05"])
        upper_pred = np.array(pred["quantiles"]["0.95"])

        # Combine observed + predicted into single arrays
        combined_val = np.concatenate([val_curve_obs, mean_pred])
        combined_lower = np.concatenate([val_curve_obs, lower_pred])
        combined_upper = np.concatenate([val_curve_obs, upper_pred])
        combined_epochs = np.arange(1, len(combined_val) + 1)

        # Plot predicted dashed curve
        plt.plot(combined_epochs, combined_val, color=color, linewidth=2, linestyle='--',
                 label=f"{sched_val}, lr={lr_val}, wd={wd_val} (pred)")
        # Shaded uncertainty
        plt.fill_between(combined_epochs, combined_lower, combined_upper, color=color, alpha=0.2)

plt.ylim(0.1, 0.7)
plt.grid(True)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ifbo_val_loss.png"))
print("✅ Saved: plots_best/ifbo_val_loss.png")
