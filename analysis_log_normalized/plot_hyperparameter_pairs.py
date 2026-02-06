"""
Plot NLL/MSE breakdown by LR+WD pairs (12 combinations) for specified contexts.
Uses log-normalized experiment epochs: 0, 5, 10, 20
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
                    lr = config_info[metrics_key]["lr"]
                    wd = config_info[metrics_key]["wd"]
                    pair_key = f"LR={lr}\nWD={wd}"
                    data[epoch].append({
                        "nll": nll, 
                        "mse": mse, 
                        "pair": pair_key,
                        "lr": lr,
                        "wd": wd
                    })
    return data

def plot_by_pairs(ctx_name, data, metric):
    """Plot the hyperparameter breakdown by LR+WD pairs."""
    # Group by pair
    pair_data = defaultdict(lambda: defaultdict(list))
    
    for epoch in EPOCHS_LIST:
        for d in data.get(epoch, []):
            pair_data[d["pair"]][epoch].append(d[metric])
    
    # Sort pairs by LR then WD
    def sort_key(pair):
        lines = pair.split('\n')
        lr = float(lines[0].split('=')[1])
        wd = float(lines[1].split('=')[1])
        return (lr, wd)
    
    pairs = sorted(pair_data.keys(), key=sort_key)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    label = "NLL" if metric == "nll" else "MSE"
    fig.suptitle(f'{label} Breakdown by LR+WD Pairs (Context: {ctx_name})', 
                 fontsize=14, fontweight='bold')
    
    x = np.arange(len(pairs))
    width = 0.2
    
    for i, epoch in enumerate(EPOCHS_LIST):
        means = [np.mean(pair_data[pair].get(epoch, [np.nan])) for pair in pairs]
        ax.bar(x + i*width, means, width, label=f'{epoch} epochs', 
               color=EPOCH_COLORS[epoch], alpha=0.8)
    
    ax.set_ylabel(f'Mean {label}')
    ax.set_xlabel('LR + WD Configuration')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(pairs, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    safe_name = ctx_name.replace(", ", "_").replace("[", "").replace("]", "")
    output_path = os.path.join(OUTPUT_DIR, f"{metric}_by_lr_wd_pairs_{safe_name}.png")
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
        plot_by_pairs(ctx["name"], data, "nll")
        plot_by_pairs(ctx["name"], data, "mse")

if __name__ == "__main__":
    main()
