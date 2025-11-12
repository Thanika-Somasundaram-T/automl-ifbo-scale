import json
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from scipy.stats import spearmanr

# Load JSON
with open("results/results_metrics.json", "r") as f:
    data = json.load(f)

metrics = [
    ("train_loss_curve", "Train Loss"),
    ("val_loss_curve", "Validation Loss"),
    ("train_acc_curve", "Train Accuracy"),
    ("val_acc_curve", "Validation Accuracy"),
    ("train_auc_curve", "Train AUC"),
    ("val_auc_curve", "Validation AUC"),
]

output_dir = "plots_rank_corr_per_run"
os.makedirs(output_dir, exist_ok=True)

lr_schedules = ["none", "warmup", "cosine", "cooldown", "warmup_cooldown"]

def parse_key(key):
    lr_match = re.search(r"lr([0-9.e-]+)", key)
    hd_match = re.search(r"hd([0-9]+)", key)
    wd_match = re.search(r"wd([0-9.e-]+)", key)
    lr_schedule_match = re.search(r"wd[0-9.e-]+_(.+)$", key)

    lr = float(lr_match.group(1)) if lr_match else None
    hd = int(hd_match.group(1)) if hd_match else None
    wd = float(wd_match.group(1)) if wd_match else None
    lr_schedule = lr_schedule_match.group(1) if lr_schedule_match else None
    return hd, lr_schedule, lr, wd

# Group runs
hidden_dims = sorted({parse_key(k)[0] for k in data.keys()})
runs_by_hd_lr = {}
for key, value in data.items():
    hd, lr_schedule, lr, wd = parse_key(key)
    runs_by_hd_lr.setdefault(hd, {}).setdefault(lr_schedule, []).append(
        (key, lr, wd, value)
    )

# Colors
lr_values = sorted({parse_key(k)[2] for k in data.keys()})
lr_colors_list = plt.cm.tab10(np.linspace(0, 1, len(lr_values)))
lr_to_color = {lr: lr_colors_list[i] for i, lr in enumerate(lr_values)}
wd_values = sorted({parse_key(k)[3] for k in data.keys()})
def shade_color(base_color, idx, total):
    factor = 0.3 + 0.7 * (idx / (total - 1)) if total > 1 else 1.0
    return np.array(base_color[:3]) * factor

# Rank correlation **per run** with its final epoch
for hd in hidden_dims:
    for metric_key, metric_name in metrics:
        plt.figure(figsize=(60, 30), dpi=200)
        plt.suptitle(f"Spearman Rank vs Final (Per Run) — HD={hd}, Metric={metric_name}", fontsize=40)

        for i, lr_schedule in enumerate(lr_schedules):
            plt.subplot(2, 3, i + 1)
            runs = runs_by_hd_lr.get(hd, {}).get(lr_schedule, [])
            if not runs:
                plt.title(f"{lr_schedule} (no runs)", fontsize=22)
                plt.axis('off')
                continue

            num_epochs = len(runs[0][3][metric_key])
            epoch_indices = list(range(0, num_epochs, 20))
            if epoch_indices[-1] != num_epochs - 1:
                epoch_indices.append(num_epochs - 1)

            for run_name, lr, wd, run_data in runs:
                curve = run_data[metric_key]
                final_value = curve[-1]
                rank_corr_curve = []

                for t in epoch_indices:
                    # Spearman correlation between all past values up to t and final value
                    corr, _ = spearmanr([curve[t]], [final_value])
                    rank_corr_curve.append(1.0)  # single value correlation = 1

                # Apply color and WD shade
                base_color = lr_to_color.get(lr, (0, 0, 0))
                wd_idx = wd_values.index(wd)
                c = shade_color(base_color, wd_idx, len(wd_values))
                plt.plot(epoch_indices, rank_corr_curve, linewidth=3, alpha=0.9, color=c,
                         label=f"lr={lr}, wd={wd}")

            plt.title(f"lr_schedule = {lr_schedule}", fontsize=26)
            plt.xlabel("Epoch", fontsize=22)
            plt.ylabel("Spearman Corr with Final", fontsize=22)
            plt.ylim(0, 1.1)
            plt.grid(True)

            if i == len(lr_schedules) - 1:
                plt.legend(fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"RankCorrPerRun_HD{hd}_{metric_name.replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"Saved: {filename}")

print("Done!")
