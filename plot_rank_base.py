import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --- Paths ---
results_dir = "./results_f"
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
with open(os.path.join(results_dir, "results_metrics.json")) as f:
    metrics = json.load(f)

# plt.figure(figsize=(16, 10), dpi=200)
# plt.title("Baseline Validation Loss + Rank Correlation", fontsize=22)
# plt.xlabel("Epoch", fontsize=18)
# plt.ylabel("Validation Loss / Rank Correlation", fontsize=18)

# --- Plot baseline curves ---
full_curves = []
for (lr_val, wd_val, sched_val), color in zip(patterns, colors):
    run_key = None
    for k, v in metrics.items():
        if (v.get("num_layers") == 4 and
            v.get("hidden_dim") == 64 and
            v.get("lr") == lr_val and
            v.get("weight_decay") == wd_val and
            v.get("lr_schedule") == sched_val):
            run_key = k
            break

    if not run_key:
        print(f"No match for lr={lr_val}, wd={wd_val}, schedule={sched_val}, layer=4, hd=64")
        continue

    val_curve = np.array(metrics[run_key].get("val_loss_curve", []))
    epochs = np.arange(1, len(val_curve)+1)
    
    # plt.plot(epochs, val_curve, color=color, linewidth=3,
    #          label=f"{sched_val}, lr={lr_val}, wd={wd_val}")
    
    full_curves.append(val_curve)

# plt.ylim(0.3, 0.6)
# plt.grid(True)
# plt.legend(fontsize=14)
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "baseline_val_loss_layer4_hd64.png"))
# print("✅ Saved: plots_best/baseline_val_loss_layer4_hd64.png")

# --- Rank correlation within each run ---
plt.figure(figsize=(14, 7))
plt.title("Baseline Rank Correlation vs Final Epoch")
plt.xlabel("Epoch")
plt.ylabel("Rank Correlation (rho)")
plt.grid(True)

for val_curve, (lr_val, wd_val, sched_val), color in zip(full_curves, patterns, colors):
    rho_list = []
    epochs_list = []
    final_value = val_curve[-1]

    for epoch_idx in range(1, len(val_curve)+1, 1):
        current_values = val_curve[:epoch_idx]
        rho = 1 - np.abs(current_values[-1] - final_value) / final_value
        rho_list.append(rho)
        epochs_list.append(epoch_idx)

    plt.plot(epochs_list, rho_list, color=color, label=f"{sched_val}, lr={lr_val}")

# plt.ylim(0.7, 1.0)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "baseline_rank_correlation_within_10epochs_zoomed.png"))
plt.show()
