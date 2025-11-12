import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- Load metrics ---
with open(os.path.join(results_dir, "results_metrics.json")) as f:
    metrics = json.load(f)

plt.figure(figsize=(14, 8))
plt.title("IFBO Prediction Distribution vs Baseline (Layer4, HD64)", fontsize=20)
plt.xlabel("Configuration", fontsize=16)
plt.ylabel("Validation Loss", fontsize=16)

all_distributions = []
baseline_values = []
labels = []

for (lr_val, wd_val, sched_val), color in zip(patterns, colors):
    # Find run
    run_key = None
    for k, v in metrics.items():
        if (v.get("num_layers") == 4 and
            v.get("hidden_dim") == 64 and
            v.get("lr") == lr_val and
            v.get("weight_decay") == wd_val and
            v.get("lr_schedule") == sched_val):
            run_key = k
            break
    if run_key is None:
        print(f"No match for {lr_val}, {wd_val}, {sched_val}")
        continue

    # Baseline last epoch
    val_curve_obs = np.array(metrics[run_key].get("val_loss_curve", []))
    final_baseline = val_curve_obs[-1]

    # IFBO prediction
    pred_path = os.path.join(pred_dir, f"pred_{run_key}.json")
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            pred = json.load(f)
        # IFBO prediction distribution (z-like)
        mean_pred = np.array(pred["point"])
        lower_pred = np.array(pred["quantiles"]["0.05"])
        upper_pred = np.array(pred["quantiles"]["0.95"])
        # Generate z-distribution samples using mean ± sigma approximation
        sigma = (upper_pred - lower_pred) / (2*1.645)  # approximate std from 5%-95%
        samples = [np.random.normal(mu, s, 100) for mu, s in zip(mean_pred, sigma)]
        samples = np.concatenate(samples)

        all_distributions.append(samples)
        baseline_values.append(final_baseline)
        labels.append(f"{sched_val}, lr={lr_val}")

# --- Plot violin plot ---
sns.violinplot(data=all_distributions, palette=colors, cut=0)
plt.xticks(range(len(labels)), labels, rotation=45, ha='right')

# Overlay baseline as stars
for i, base_val in enumerate(baseline_values):
    plt.scatter(i, base_val, color='white', marker='*', s=150, zorder=10, label="Baseline" if i==0 else "")

plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "ifbo_distribution_layer4_hd64_violin.png"))
plt.show()
print("✅ Saved: plots_best/ifbo_distribution_layer4_hd64_violin.png")
