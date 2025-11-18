import os
import json
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
results_dir = "./results"
output_dir = "rank_plots"
os.makedirs(output_dir, exist_ok=True)

# --- Load metrics ---
with open(os.path.join(results_dir, "results_metrics.json")) as f:
    metrics = json.load(f)

# ============================================================
#   GROUP ALL RUNS BY MODEL SIZE (num_layers, hidden_dim)
# ============================================================
grouped = {}

for run_key, info in metrics.items():
    nl = info.get("num_layers")
    hd = info.get("hidden_dim")
    val_curve = np.array(info.get("val_loss_curve", []))

    if len(val_curve) == 0:
        continue

    grouped.setdefault((nl, hd), []).append(val_curve)

# Sort groups for stable color generation
sorted_groups = sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1]))

# ============================================================
#   VALIDATION LOSS (MEAN PER MODEL SIZE)
# ============================================================
plt.figure(figsize=(16, 10), dpi=200)
plt.title("Validation Loss Grouped by Model Size", fontsize=22)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Validation Loss", fontsize=18)
plt.grid(True)

# Generate shades of blue
num_groups = len(sorted_groups)
blues = plt.cm.Blues(np.linspace(0.4, 0.95, num_groups))

for idx, ((nl, hd), curves) in enumerate(sorted_groups):
    # Normalize lengths by padding
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        if len(c) < max_len:
            c = np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
        padded.append(c)

    padded = np.array(padded)

    # Mean validation curve
    mean_curve = np.nanmean(padded, axis=0)
    epochs = np.arange(1, len(mean_curve) + 1)

    plt.plot(epochs, mean_curve, linewidth=3, color=blues[idx],
             label=f"{nl} layers, {hd} hidden dim ({len(curves)} runs)")

plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "val_loss_grouped_by_model_size.png"))
plt.show()

print("✅ Saved: val_loss_grouped_by_model_size.png")

# ============================================================
#   RANK CORRELATION vs FINAL EPOCH PER MODEL SIZE
# ============================================================
plt.figure(figsize=(14, 7))
plt.title("Rank Correlation vs Final Epoch (Grouped by Model Size)")
plt.xlabel("Epoch")
plt.ylabel("Rank Correlation (rho)")
plt.grid(True)

for idx, ((nl, hd), curves) in enumerate(sorted_groups):
    # Normalize lengths
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        if len(c) < max_len:
            c = np.pad(c, (0, max_len - len(c)), constant_values=np.nan)
        padded.append(c)

    padded = np.array(padded)

    # Mean validation curve
    mean_curve = np.nanmean(padded, axis=0)

    # Rank correlation vs final epoch
    final_value = mean_curve[-1]
    rho_list = []
    epochs = np.arange(1, len(mean_curve) + 1)

    for e in range(1, len(mean_curve) + 1):
        curr_val = mean_curve[e - 1]
        rho = 1 - abs(curr_val - final_value) / final_value
        rho_list.append(rho)

    plt.plot(epochs, rho_list, linewidth=3, color=blues[idx],
             label=f"{nl} layers, {hd} hd")

plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "rank_corr_grouped_by_model_size.png"))
plt.show()

print("✅ Saved: rank_corr_grouped_by_model_size.png")
