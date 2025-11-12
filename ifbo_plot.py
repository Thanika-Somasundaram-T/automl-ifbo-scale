import json
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# --- Paths ---
train_results_path = "./results_f/result64.json"
pred_dir = "./results_f"   # folder containing pred_*.json files
output_dir = "plots_ifbo"
os.makedirs(output_dir, exist_ok=True)

# --- Metrics ---
metrics = [
    ("val_loss_curve", "Validation Loss"),
    ("val_acc_curve", "Validation Accuracy"),
    ("val_auc_curve", "Validation AUC"),
]

# --- LR schedules and plotting configs ---
lr_schedules = ["none", "warmup", "cosine", "cooldown", "warmup_cooldown"]
lr_colors = {1e-4: "blue", 3e-4: "green", 1e-3: "red", 1e-2: "purple"}
wd_alpha = {0.0: 1.0, 1e-4: 0.7, 1e-3: 0.4}

# --- Helper: parse key into components ---
def parse_key(key):
    lr_match = re.search(r"lr([0-9.e-]+)", key)
    hd_match = re.search(r"hd([0-9]+)", key)
    wd_match = re.search(r"wd([0-9.e-]+)", key)
    lr_schedule_match = re.search(r"wd[0-9.e-]+_(.+)$", key)
    lr_schedule = lr_schedule_match.group(1) if lr_schedule_match else None

    lr = float(lr_match.group(1)) if lr_match else None
    hd = int(hd_match.group(1)) if hd_match else None
    wd = float(wd_match.group(1)) if wd_match else None
    return hd, lr_schedule, lr, wd

# --- Load train data ---
with open(train_results_path, "r") as f:
    all_train_data = json.load(f)

hidden_dims = sorted({parse_key(k)[0] for k in all_train_data.keys()})

# --- Group runs by HD + lr_schedule ---
runs_by_hd_lr = {}
for key, value in all_train_data.items():
    hd, lr_schedule, lr, wd = parse_key(key)
    runs_by_hd_lr.setdefault(hd, {}).setdefault(lr_schedule, []).append(
        (key, lr, wd, value)
    )

# --- Plot observed + predicted curves ---
for hd in hidden_dims:
    for metric_key, metric_name in metrics:
        plt.figure(figsize=(60, 25), dpi=200)
        plt.suptitle(f"IFBO Prediction — HD={hd}, Metric: {metric_name}", fontsize=36)

        for i, lr_schedule in enumerate(lr_schedules):
            plt.subplot(2, 3, i + 1)
            runs = runs_by_hd_lr.get(hd, {}).get(lr_schedule, [])
            if not runs:
                plt.title(f"{lr_schedule} (no runs)", fontsize=22)
                plt.axis('off')
                continue

            for run_name, lr, wd, run_data in runs:
                # --- Observed curve (first 10 epochs) ---
                observed = np.array(run_data.get(metric_key, []))
                if len(observed) == 0:
                    continue
                epochs_obs = np.arange(1, len(observed) + 1)

                # Plot observed
                color = lr_colors.get(lr, "black")
                alpha = wd_alpha.get(wd, 1.0)
                plt.plot(epochs_obs, observed, label=f"lr={lr}, wd={wd}", color=color, alpha=alpha, linewidth=3)

                # --- Try to load prediction file ---
                pred_file = os.path.join(pred_dir, f"pred_{run_name}.json")
                if os.path.exists(pred_file):
                    with open(pred_file, "r") as pf:
                        pred_data = json.load(pf)

                    pred_point = np.array(pred_data["point"])
                    pred_low = np.array(pred_data["quantiles"]["0.05"])
                    pred_high = np.array(pred_data["quantiles"]["0.95"])

                    epochs_pred = np.arange(len(observed) + 1, len(observed) + len(pred_point) + 1)

                    # Plot predictions with shaded span
                    plt.plot(epochs_pred, pred_point, color=color, linestyle="--", alpha=alpha, linewidth=2)
                    plt.fill_between(epochs_pred, pred_low, pred_high, color=color, alpha=0.15)
                else:
                    print(f"⚠️ Prediction file not found for {run_name}")

            plt.title(f"lr_schedule = {lr_schedule}", fontsize=22)
            plt.xlabel("Epoch", fontsize=20)
            plt.ylabel(metric_name, fontsize=20)
            plt.ylim(0.3, 0.65)
            plt.grid(True)

            if i == len(lr_schedules) - 1:
                plt.legend(fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = f"IFBO_HD{hd}_{metric_name.replace(' ', '_')}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        print(f"✅ Saved IFBO plot: {filename}")

print("✅ All IFBO plots generated successfully!")
