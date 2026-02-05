"""
Plot NLL/MSE breakdown by hyperparameter (LR and WD) for specified contexts.
Uses the correct rsplit matching logic.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
RESULTS_DIR = "../results"
OUTPUT_DIR = "."
EPOCHS_LIST = [0, 5, 10, 20]
EPOCH_COLORS = {0: '#e57373', 5: '#ffb74d', 10: '#64b5f6', 20: '#81c784'}

# Contexts to analyze
CONTEXTS_TO_PLOT = [
    {"name": "[64, 32, 24, 16, 8]", "ablation": "[64_32_24_16_8]"},
    {"name": "Baseline", "ablation": "Baseline"},
]

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def calculate_nll(y_true_curve, y_pred_dist, epoch_idx):
    y_true_future = np.array(y_true_curve[epoch_idx:])
    pred_mean = np.array(y_pred_dist["point"])
    q05 = np.array(y_pred_dist["quantiles"]["0.05"])
    q95 = np.array(y_pred_dist["quantiles"]["0.95"])
    
    min_len = min(len(y_true_future), len(pred_mean))
    if min_len == 0:
        return np.nan
        
    y_true = y_true_future[:min_len]
    y_mean = pred_mean[:min_len]
    sigma = (q95[:min_len] - q05[:min_len]) / 3.29
    sigma = np.maximum(sigma, 1e-6)
    
    residual = y_true - y_mean
    nll_per_step = 0.5 * np.log(2 * np.pi * sigma**2) + (residual**2) / (2 * sigma**2)
    
    return np.mean(nll_per_step)

def calculate_mse(y_true_curve, y_pred_dist, epoch_idx):
    y_true_future = np.array(y_true_curve[epoch_idx:])
    pred_mean = np.array(y_pred_dist["point"])
    
    min_len = min(len(y_true_future), len(pred_mean))
    if min_len == 0:
        return np.nan
        
    return np.mean((y_true_future[:min_len] - pred_mean[:min_len]) ** 2)

def collect_data(ablation, true_curves, config_info):
    """Collect NLL/MSE data for a given ablation context."""
    data = defaultdict(list)
    
    for epoch in EPOCHS_LIST:
        if ablation == "Baseline":
            pred_path = os.path.join(RESULTS_DIR, f"baseline_{epoch}.json")
        else:
            pred_path = os.path.join(RESULTS_DIR, f"ifbo_pred_{ablation}_ep{epoch}.json")
        
        pred_data = load_json(pred_path)
        if not pred_data:
            continue
            
        for pred_key, pred_dist in pred_data.items():
            metrics_key = pred_key.rsplit('_', 1)[0]
            if metrics_key in true_curves:
                nll = calculate_nll(true_curves[metrics_key], pred_dist, epoch)
                mse = calculate_mse(true_curves[metrics_key], pred_dist, epoch)
                if not np.isnan(nll) and not np.isnan(mse):
                    data[epoch].append({
                        "nll": nll, 
                        "mse": mse, 
                        "lr": config_info[metrics_key]["lr"], 
                        "wd": config_info[metrics_key]["wd"]
                    })
    return data

def plot_breakdown(ctx_name, data, metric):
    """Plot the hyperparameter breakdown for a given metric."""
    # Group by LR and WD
    lr_data = defaultdict(lambda: defaultdict(list))
    wd_data = defaultdict(lambda: defaultdict(list))
    
    for epoch in EPOCHS_LIST:
        for d in data.get(epoch, []):
            lr_data[d["lr"]][epoch].append(d[metric])
            wd_data[d["wd"]][epoch].append(d[metric])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    label = "NLL" if metric == "nll" else "MSE"
    fig.suptitle(f'{label} Breakdown by Hyperparameter (Context: {ctx_name})', 
                 fontsize=14, fontweight='bold')
    
    # LR plot
    lrs = sorted(lr_data.keys())
    x = np.arange(len(lrs))
    width = 0.2
    
    for i, epoch in enumerate(EPOCHS_LIST):
        means = [np.mean(lr_data[lr].get(epoch, [np.nan])) for lr in lrs]
        ax1.bar(x + i*width, means, width, label=f'{epoch} epochs', 
                color=EPOCH_COLORS[epoch], alpha=0.8)
    
    ax1.set_ylabel(f'Mean {label}')
    ax1.set_title(f'{label} by Learning Rate')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([f'LR={lr}' for lr in lrs], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # WD plot
    wds = sorted(wd_data.keys())
    x = np.arange(len(wds))
    
    for i, epoch in enumerate(EPOCHS_LIST):
        means = [np.mean(wd_data[wd].get(epoch, [np.nan])) for wd in wds]
        ax2.bar(x + i*width, means, width, label=f'{epoch} epochs', 
                color=EPOCH_COLORS[epoch], alpha=0.8)
    
    ax2.set_ylabel(f'Mean {label}')
    ax2.set_title(f'{label} by Weight Decay')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([f'WD={wd}' for wd in wds], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    safe_name = ctx_name.replace(", ", "_").replace("[", "").replace("]", "")
    output_path = os.path.join(OUTPUT_DIR, f"{metric}_by_hyperparameter_{safe_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

def main():
    print("Loading Ground Truth Metrics...")
    gt_path = os.path.join(RESULTS_DIR, "results_metrics.json")
    gt_data = load_json(gt_path)
    if not gt_data:
        print("Error: Could not load results_metrics.json")
        return
    
    # Build lookups
    true_curves = {k: v.get("val_acc_curve", []) for k, v in gt_data.items() if v.get("val_acc_curve")}
    config_info = {k: {"lr": v.get("lr"), "wd": v.get("weight_decay")} for k, v in gt_data.items()}
    
    # Generate plots for each context
    for ctx in CONTEXTS_TO_PLOT:
        print(f"\nProcessing: {ctx['name']}")
        data = collect_data(ctx["ablation"], true_curves, config_info)
        
        # Generate both NLL and MSE plots
        plot_breakdown(ctx["name"], data, "nll")
        plot_breakdown(ctx["name"], data, "mse")

if __name__ == "__main__":
    main()
