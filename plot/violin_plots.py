import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Paths ---
results_dir = "./results"
output_dir = "violin_plots same_axis"
os.makedirs(output_dir, exist_ok=True)

# --- Configs ---
patterns = [
    (0.01, 0.0001, "cosine"),
    (0.001, 0.0, "cosine"),
    (0.001, 0.0001, "cosine"),
    (0.0001, 0.001, "cosine"),
    (0.0003, 0.001, "cosine"),
    (0.0003, 0.0, "cosine"),
]

colors = [
    "#FF9999",  # light red
    "#99BBFF",  # light blue
    "#99EEAA",  # light green
    "#FFCC99",  # light orange
    "#D1B3FF",  # light purple
    "#D2B48C",  # light brown
]
observed_epochs = [10, 15, 30, 45, 50, 70, 90]

# --- Load metrics ---
model_size = 64
json_file = os.path.join(results_dir, f"{model_size}.json")
with open(json_file) as f:
    data = json.load(f)

# --- Load predictions ---
pred_top = {epoch: json.load(open(os.path.join(results_dir, f"ifbo_pred_{epoch}.json"))) 
            for epoch in observed_epochs if os.path.exists(os.path.join(results_dir, f"ifbo_pred_{epoch}.json"))}
pred_bottom = {epoch: json.load(open(os.path.join(results_dir, f"ifbo_pred_both_context_{epoch}.json"))) 
               for epoch in observed_epochs if os.path.exists(os.path.join(results_dir, f"ifbo_pred_both_context_{epoch}.json"))}

# ----------------------------------------------------------------------
#                   ONE PLOT PER PATTERN (TOP & BOTTOM)
# ----------------------------------------------------------------------
for (lr_val, wd_val, sched_val), color in zip(patterns, colors):

    run_key = next((k for k, v in data.items() 
                    if v.get("num_layers")==4 and v.get("hidden_dim")==model_size and
                       v.get("lr")==lr_val and v.get("weight_decay")==wd_val and
                       v.get("lr_schedule")==sched_val), None)
    if not run_key:
        print(f"No match for {lr_val}, {wd_val}, {sched_val}")
        continue

    run_data = data[run_key]
    val_curve = np.array(run_data.get("val_loss_curve", []))
    final_baseline = val_curve[-1]
    best_val_loss = run_data.get("best_val_loss", None)

    top_dists, bottom_dists = [], []
    baseline_vals, best_vals = [], []
    labels = []

    for obs_epoch in observed_epochs:
        predT = pred_top.get(obs_epoch, {}).get(f"{run_key}_{obs_epoch}", None)
        predB = pred_bottom.get(obs_epoch, {}).get(f"{run_key}_{obs_epoch}", None)
        if predT is None or predB is None: continue

        meanT = np.array(predT["point"])
        sigmaT = (np.array(predT["quantiles"]["0.95"]) - np.array(predT["quantiles"]["0.05"])) / (2*1.645)
        samplesT = np.concatenate([np.random.normal(mu, s, 100) for mu, s in zip(meanT, sigmaT)])

        meanB = np.array(predB["point"])
        sigmaB = (np.array(predB["quantiles"]["0.95"]) - np.array(predB["quantiles"]["0.05"])) / (2*1.645)
        samplesB = np.concatenate([np.random.normal(mu, s, 100) for mu, s in zip(meanB, sigmaB)])

        top_dists.append(samplesT)
        bottom_dists.append(samplesB)
        baseline_vals.append(final_baseline)
        best_vals.append(best_val_loss)
        labels.append(f"obs={obs_epoch}")

    # ------------------ Compute consistent y-limits per pattern ------------------
    all_samples = np.concatenate(top_dists + bottom_dists)
    y_min = min(all_samples.min(), final_baseline, best_val_loss) * 0.995
    y_max = max(all_samples.max(), final_baseline, best_val_loss) * 1.005

    # 2-panel figure
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"IFBO Prediction Distribution\nPattern: lr={lr_val}, wd={wd_val}, sched={sched_val}, HD={model_size}", fontsize=18)

    # TOP
    sns.violinplot(data=top_dists, palette=[color]*len(top_dists), cut=0, ax=ax_top)
    ax_top.set_ylim(y_min, y_max)
    for i, base in enumerate(baseline_vals):
        ax_top.scatter(i, base, color='white', marker='*', s=120, edgecolor='black', linewidth=1.0, zorder=20)
    for i, best in enumerate(best_vals):
        if best is not None:
            ax_top.scatter(i, best, color='#FFD700', marker='.', s=100, edgecolor='black', linewidth=1.0, zorder=21)
    ax_top.set_title("IFBO Prediction (per model Context)")
    ax_top.set_ylabel("Validation Loss")
    ax_top.grid(True, axis='y')

    # BOTTOM
    sns.violinplot(data=bottom_dists, palette=[color]*len(bottom_dists), cut=0, ax=ax_bottom)
    ax_bottom.set_ylim(y_min, y_max)
    for i, base in enumerate(baseline_vals):
        ax_bottom.scatter(i, base, color='white', marker='*', s=120, edgecolor='black', linewidth=1.0, zorder=20)
    for i, best in enumerate(best_vals):
        if best is not None:
            ax_bottom.scatter(i, best, color='#FFD700', marker='.', s=100, edgecolor='black', linewidth=1.0, zorder=21)
    ax_bottom.set_title("IFBO Prediction (With Per Model and 32 Context)")
    ax_bottom.set_ylabel("Validation Loss")
    ax_bottom.set_xlabel("Observed Epoch")
    ax_bottom.set_xticks(range(len(labels)))
    ax_bottom.set_xticklabels(labels)
    ax_bottom.grid(True, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"2panel_lr{lr_val}_wd{wd_val}_{sched_val}_hd{model_size}.png")
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Saved: {save_path}")
