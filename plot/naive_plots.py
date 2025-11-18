import json
import matplotlib.pyplot as plt
import os
import re

# Load JSON data
with open("results/results_metrics.json", "r") as f:
    data = json.load(f)

# Metrics
metrics = [
    ("train_loss_curve", "Train Loss"),
    ("val_loss_curve", "Validation Loss"),
    ("train_acc_curve", "Train Accuracy"),
    ("val_acc_curve", "Validation Accuracy"),
    ("train_auc_curve", "Train AUC"),
    ("val_auc_curve", "Validation AUC"),
]

# Output folder
output_dir = "plots_naive_l6"
os.makedirs(output_dir, exist_ok=True)

# Explicit lr_schedule order
lr_schedules = ["cosine"]

# Define colors for learning rates
lr_colors = {1e-4: "blue", 3e-4: "green", 1e-3: "red", 1e-2: "purple"}

# Define alpha/shade for weight decay
wd_alpha = {0.0: 1.0, 1e-4: 0.7, 1e-3: 0.4}

# Parse key to extract hd, lr_schedule, lr, wd
def parse_key(key):
    lr_match = re.search(r"lr([0-9.e-]+)", key)
    hd_match = re.search(r"hd([0-9]+)", key)
    wd_match = re.search(r"wd([0-9.e-]+)", key)
    
    # lr_schedule: everything after wd<number>_
    lr_schedule_match = re.search(r"wd[0-9.e-]+_(.+)$", key)
    lr_schedule = lr_schedule_match.group(1) if lr_schedule_match else None

    lr = float(lr_match.group(1)) if lr_match else None
    hd = int(hd_match.group(1)) if hd_match else None
    wd = float(wd_match.group(1)) if wd_match else None
    return hd, lr_schedule, lr, wd

# Detect hidden_dims
hidden_dims = sorted({parse_key(k)[0] for k in data.keys()})

# Group runs by hidden_dim and lr_schedule
runs_by_hd_lr = {}
for key, value in data.items():
    hd, lr_schedule, lr, wd = parse_key(key)
    runs_by_hd_lr.setdefault(hd, {}).setdefault(lr_schedule, []).append(
        (key, lr, wd, value)
    )

# Generate plots
for hd in hidden_dims:
    for metric_key, metric_name in metrics:
        plt.figure(figsize=(60, 25), dpi=200)  # max size
        plt.suptitle(f"6-layer, HD={hd}, Metric: {metric_name}", fontsize=36)

        for i, lr_schedule in enumerate(lr_schedules):
            # plt.subplot(2, 3, i + 1)
            runs = runs_by_hd_lr.get(hd, {}).get(lr_schedule, [])
            if not runs:
                plt.title(f"{lr_schedule} (no runs)", fontsize=22)
                plt.axis('off')
                continue

            for run_name, lr, wd, run_data in runs:
                curve = run_data.get(metric_key, [])
                color = lr_colors.get(lr, "black")  # default black if lr not found
                alpha = wd_alpha.get(wd, 1.0)
                plt.plot(curve, label=f"lr={lr}, wd={wd}", color=color, alpha=alpha, linewidth=3)

            plt.title(f"lr_schedule = {lr_schedule}", fontsize=22)
            plt.xlabel("Epoch", fontsize=20)
            plt.ylabel(metric_name, fontsize=20)
            # plt.ylim(0.3, 0.65)
            plt.grid(True)
            if i == len(lr_schedules) - 1:
                plt.legend(fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"HD{hd}_{metric_name.replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

print("All 18 images generated!")
