import os, json
import matplotlib.pyplot as plt
import numpy as np

# Paths
results_dir = "./results_f"
output_dir = "plots_best"
os.makedirs(output_dir, exist_ok=True)

# Patterns: lr, wd, schedule
patterns = [
    (0.01, 0.0001, "cosine"),
    (0.001, 0.0, "cosine"),
    (0.001, 0.0001, "warmup_cooldown"),
    (0.0001, 0.001, "warmup_cooldown"),
    (0.0003, 0.001, "cooldown"),
    (0.0003, 0.0, "cooldown"),
]

colors = ["red", "blue", "green", "orange", "purple", "brown"]

# Load metrics
with open(os.path.join(results_dir, "results_metrics.json")) as f:
    metrics = json.load(f)

plt.figure(figsize=(16, 10), dpi=200)
plt.title("Top Runs: Baseline (layer4, hd64)", fontsize=22)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Validation Loss", fontsize=18)

for (lr_val, wd_val, sched_val), color in zip(patterns, colors):
    # Filter runs by JSON fields, not by regex key
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

    val_curve = metrics[run_key].get("val_loss_curve", [])
    epochs = np.arange(1, len(val_curve) + 1)

    plt.plot(epochs, val_curve, color=color, linewidth=3,
             label=f"{sched_val}, lr={lr_val}, wd={wd_val}, layer=4, hd=64")
plt.ylim(0.1, 0.7)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "baseline_val_loss.png"))
print("✅ Saved: plots_best/baseline_val_loss_layer4_hd64.png")
