import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
results_dir = "./results"
pred_dir = "./results"
output_dir = "plots_best"
os.makedirs(output_dir, exist_ok=True)

# --- Load metrics ---
metrics_path = os.path.join(results_dir, "result64.json")
with open(metrics_path) as f:
    metrics = json.load(f)

# ============================================================
#   GROUP ALL RUNS BY MODEL SIZE (num_layers, hidden_dim)
# ============================================================
grouped = {}

for run_key, info in metrics.items():
    nl = info.get("num_layers")
    hd = info.get("hidden_dim")

    # observed validation loss
    val_curve_obs = np.array(info.get("val_loss_curve", []))
    if len(val_curve_obs) == 0:
        continue

    # predicted curve
    pred_path = os.path.join(pred_dir, f"pred_{run_key}.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            pred = json.load(f)
        mean_pred = np.array(pred.get("point", []))
        full_curve = np.concatenate([val_curve_obs, mean_pred])
    else:
        full_curve = val_curve_obs

    grouped.setdefault((nl, hd), []).append(full_curve)

# Sort for stable consistent blue shades
sorted_groups = sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1]))

# ============================================================
#   PLOT RANK CORRELATION FOR EACH MODEL SIZE
# ============================================================
plt.figure(figsize=(14, 7))
plt.title("IFBO Rank Correlation vs Final Epoch (Grouped by Model Size)")
plt.xlabel("Epoch")
plt.ylabel("Rank Correlation (rho)")
plt.grid(True)

# Shades of blue
num_groups = len(sorted_groups)
blues = plt.cm.Blues(np.linspace(0.4, 0.95, num_groups))

for idx, ((nl, hd), curves) in enumerate(sorted_groups):
    # Pad curves so all equal length
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        if len(c) < max_len:
            c = np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
        padded.append(c)
    padded = np.array(padded)

    # Mean full curve (observed + predicted)
    mean_curve = np.nanmean(padded, axis=0)

    # Compute rank correlation every 10 epochs
    final_value = mean_curve[-1]
    rho_list = []
    epochs_list = []
    for epoch_idx in range(10, len(mean_curve) + 1, 10):
        curr_val = mean_curve[epoch_idx - 1]
        rho = 1 - abs(curr_val - final_value) / final_value
        rho_list.append(rho)
        epochs_list.append(epoch_idx)

    plt.plot(epochs_list, rho_list, linewidth=3, color=blues[idx],
             label=f"{nl} layers, {hd} hidden dim ({len(curves)} runs)")

plt.ylim(0.7, 1.0)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ifbo_rank_corr_grouped_by_model_size.png"))
plt.show()

print("✅ Saved: ifbo_rank_corr_grouped_by_model_size.png")
