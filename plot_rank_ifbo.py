import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
results_dir = "./results_f"
pred_dir = "./results_f"
output_dir = "plots_best"
os.makedirs(output_dir, exist_ok=True)

# --- Configs: (lr, wd, schedule) ---
patterns = [
    (0.01, 0.0001, "cosine"),
    (0.001, 0.0, "cosine"),
    (0.001, 0.0001, "warmup_cooldown"),
    (0.0001, 0.001, "warmup_cooldown"),
    (0.0003, 0.001, "cooldown"),
    (0.0003, 0.0, "cooldown"),
]

colors = ["red", "blue", "green", "orange", "purple", "brown"]

plt.figure(figsize=(14, 7))
plt.title("IFBO Rank Correlation vs Final Epoch")
plt.xlabel("Epoch")
plt.ylabel("Rank Correlation (rho)")
plt.grid(True)

for (lr_val, wd_val, sched_val), color in zip(patterns, colors):
    # Load metrics
    run_key = None
    with open(os.path.join(results_dir, "result64.json")) as f:
        metrics = json.load(f)
    for k, v in metrics.items():
        if (v.get("lr") == lr_val and
            v.get("weight_decay") == wd_val and
            v.get("lr_schedule") == sched_val):
            run_key = k
            break
    if run_key is None:
        print(f"No match for lr={lr_val}, wd={wd_val}, sched={sched_val}")
        continue

    run_data = metrics[run_key]
    val_curve_obs = np.array(run_data.get("val_loss_curve", []))
    pred_path = os.path.join(pred_dir, f"pred_{run_key}.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            pred = json.load(f)
        mean_pred = np.array(pred["point"])
        full_curve = np.concatenate([val_curve_obs, mean_pred])
    else:
        full_curve = val_curve_obs

    # Compute rank correlation every 5 epochs
    rho_list = []
    epochs_list = []
    final_value = full_curve[-1]
    for epoch_idx in range(10, len(full_curve)+1, 10):
        current_values = full_curve[:epoch_idx]
        rho = 1 - np.abs(current_values[-1] - final_value) / final_value
        rho_list.append(rho)
        epochs_list.append(epoch_idx)

    plt.plot(epochs_list, rho_list, color=color, label=f"{sched_val}, lr={lr_val}")

plt.ylim(0.7, 1.0)  # focus on 0.7–1.0
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ifbo_rank_correlation_within_10epochs_zoomed.png"))
plt.show()
