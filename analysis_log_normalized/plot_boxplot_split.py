"""
Boxplot comparison of NLL and MSE across context strategies.
Split version: 13 contexts per NLL row.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import LogNorm

# Configuration
RESULTS_DIR = "results"
EPOCHS_LIST = [0, 5, 10, 20]
ABLATIONS_LIST = [
    "[4]", "[8]", "[16]", "[24]", "[32]", "[64]",
    "[8_4]", "[16_8_4]", "[24_16_8_4]", "[32_24_16]", "[32_24_16_8_4]",
    "[64_4]", "[64_8]", "[64_8_4]", "[64_16]", "[64_24]", "[64_32]",
    "[64_24_16_8_4]", "[64_32_16_8_4]", "[64_32_24]", "[64_32_24_8_4]",
    "[64_32_24_16]", "[64_32_24_16_4]", "[64_32_24_16_8]", "[64_32_24_16_8_4]"
]

EPOCH_COLORS = {0: '#e57373', 5: '#ffb74d', 10: '#64b5f6', 20: '#81c784'}

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

def clean_context_name(s):
    return s.replace("[","").replace("]","").replace("_", ", ")

def main():
    print("Loading Ground Truth Metrics...")
    metrics = load_json(os.path.join(RESULTS_DIR, "results_metrics.json"))
    if not metrics:
        print("Error: Could not load results_metrics.json")
        return

    true_curves = {k: data.get("val_acc_curve", []) for k, data in metrics.items() if data.get("val_acc_curve")}

    nll_data = defaultdict(lambda: defaultdict(list))
    mse_data = defaultdict(lambda: defaultdict(list))
    
    for epoch in EPOCHS_LIST:
        print(f"Processing Epoch {epoch}...")
        
        for context_name in ABLATIONS_LIST:
            pred_filename = f"ifbo_pred_{context_name}_ep{epoch}.json"
            pred_path = os.path.join(RESULTS_DIR, pred_filename)
            predictions = load_json(pred_path)
            
            if not predictions:
                continue
            
            clean_ctx = clean_context_name(context_name)
            
            for pred_key, pred_data in predictions.items():
                metrics_key = pred_key.rsplit('_', 1)[0]
                
                if metrics_key in true_curves:
                    nll = calculate_nll(true_curves[metrics_key], pred_data, epoch)
                    mse = calculate_mse(true_curves[metrics_key], pred_data, epoch)
                    
                    if not np.isnan(nll):
                        nll_data[clean_ctx][epoch].append(nll)
                    if not np.isnan(mse):
                        mse_data[clean_ctx][epoch].append(mse)
        
        baseline_filename = f"baseline_{epoch}.json"
        baseline_path = os.path.join(RESULTS_DIR, baseline_filename)
        baseline_preds = load_json(baseline_path)
        
        if baseline_preds:
            for pred_key, pred_data in baseline_preds.items():
                metrics_key = pred_key.rsplit('_', 1)[0]
                
                if metrics_key in true_curves:
                    nll = calculate_nll(true_curves[metrics_key], pred_data, epoch)
                    mse = calculate_mse(true_curves[metrics_key], pred_data, epoch)
                    
                    if not np.isnan(nll):
                        nll_data["Baseline"][epoch].append(nll)
                    if not np.isnan(mse):
                        mse_data["Baseline"][epoch].append(mse)

    # Prepare context order and split
    context_order = ["Baseline"] + [clean_context_name(ab) for ab in ABLATIONS_LIST]
    mid = len(context_order) // 2
    contexts_top = context_order[:mid]
    contexts_bottom = context_order[mid:]
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [1.5, 1.5, 1]})
    fig.suptitle('FT-PFN Performance Comparison (Log-Normalized Width)\n(Observation Epochs: 0, 5, 10, 20)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    legend_patches = [plt.Rectangle((0,0),1,1, facecolor=EPOCH_COLORS[e], alpha=0.7, label=f'{e} Epochs Observed') 
                      for e in EPOCHS_LIST]
    fig.legend(handles=legend_patches, loc='upper right', title='Observation Horizon', 
               bbox_to_anchor=(0.99, 0.96), fontsize=9)
    
    box_width = 0.18
    
    # NLL Part 1
    for i, epoch in enumerate(EPOCHS_LIST):
        positions = np.arange(len(contexts_top)) + (i - 1.5) * box_width
        data_to_plot = [nll_data[ctx].get(epoch, [np.nan]) for ctx in contexts_top]
        bp = ax1.boxplot(data_to_plot, positions=positions, widths=box_width, patch_artist=True, showfliers=True)
        for patch in bp['boxes']:
            patch.set_facecolor(EPOCH_COLORS[epoch])
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
    
    ax1.set_ylabel('NLL\n← Lower is Better', fontsize=10)
    ax1.set_title('Uncertainty Quality (NLL) - Part 1', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticks(np.arange(len(contexts_top)))
    ax1.set_xticklabels(contexts_top, rotation=45, ha='right', fontsize=9)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
    ymin1, ymax1 = ax1.get_ylim()
    
    # NLL Part 2
    for i, epoch in enumerate(EPOCHS_LIST):
        positions = np.arange(len(contexts_bottom)) + (i - 1.5) * box_width
        data_to_plot = [nll_data[ctx].get(epoch, [np.nan]) for ctx in contexts_bottom]
        bp = ax2.boxplot(data_to_plot, positions=positions, widths=box_width, patch_artist=True, showfliers=True)
        for patch in bp['boxes']:
            patch.set_facecolor(EPOCH_COLORS[epoch])
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
    
    ax2.set_ylabel('NLL\n← Lower is Better', fontsize=10)
    ax2.set_title('Uncertainty Quality (NLL) - Part 2', fontsize=12, fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(np.arange(len(contexts_bottom)))
    ax2.set_xticklabels(contexts_bottom, rotation=45, ha='right', fontsize=9)
    ax2.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    # Share y-axis limits
    ymin2, ymax2 = ax2.get_ylim()
    shared_ymin = min(ymin1, ymin2)
    shared_ymax = max(ymax1, ymax2)
    ax1.set_ylim(shared_ymin, shared_ymax)
    ax2.set_ylim(shared_ymin, shared_ymax)
    
    # MSE Heatmap
    mse_matrix = []
    for epoch in EPOCHS_LIST:
        row = [np.mean(mse_data[ctx].get(epoch, [np.nan])) if mse_data[ctx].get(epoch, []) else np.nan 
               for ctx in context_order]
        mse_matrix.append(row)
    mse_matrix = np.array(mse_matrix)
    
    valid_vals = mse_matrix[~np.isnan(mse_matrix)]
    norm = LogNorm(vmin=max(np.nanmin(valid_vals), 1e-10), vmax=np.nanmax(valid_vals)) if len(valid_vals) > 0 else None
    
    im = ax3.imshow(mse_matrix, cmap='viridis', aspect='auto', norm=norm)
    cbar = fig.colorbar(im, ax=ax3, shrink=0.8, pad=0.02)
    cbar.set_label('MSE (log scale)', fontsize=10)
    
    ax3.set_yticks(np.arange(len(EPOCHS_LIST)))
    ax3.set_yticklabels([f'{e}' for e in EPOCHS_LIST], fontsize=10)
    ax3.set_ylabel('Observation Epochs', fontsize=11)
    ax3.set_xticks(np.arange(len(context_order)))
    ax3.set_xticklabels(context_order, rotation=45, ha='right', fontsize=8)
    ax3.set_xlabel('Hidden Dimension Context (Log-Normalized)', fontsize=11)
    ax3.set_title('Prediction Accuracy (MSE Heatmap)', fontsize=12, fontweight='bold', pad=10)
    
    for i in range(len(EPOCHS_LIST)):
        for j in range(len(context_order)):
            val = mse_matrix[i, j]
            if not np.isnan(val):
                ax3.text(j, i, f'{val:.2e}', ha='center', va='center', fontsize=6, fontweight='bold', color='white')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    output_path = "analysis_log_normalized/nll_mse_boxplot_split.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
