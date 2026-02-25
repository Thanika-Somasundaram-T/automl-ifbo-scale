import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuration
RESULTS_DIR = "results"
EPOCHS_LIST = [5]
ABLATIONS_LIST = [
    "[64]", "[32]", "[24]", "[16]", "[8]", "[4]",
    "[64_32]", "[64_24]", "[64_16]", "[64_8]", "[64_4]",
    "[64_32_24]", "[64_8_4]", "[16_8_4]",
    "[64_32_24_16]",
    "[64_32_24_16_8]", "[64_32_24_16_4]", "[64_32_24_8_4]",
    "[64_32_24_16_8_4]"
]

# Colors for observation horizons
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
        
    y_true = y_true_future[min_len - 1]
    y_mean = pred_mean[min_len - 1]
    sigma = (q95[min_len - 1] - q05[min_len - 1]) / 3.29
    sigma = max(sigma, 1e-6)
    
    residual = y_true - y_mean
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + (residual**2) / (2 * sigma**2)
    
    return float(nll)

def calculate_mse(y_true_curve, y_pred_dist, epoch_idx):
    y_true_future = np.array(y_true_curve[epoch_idx:])
    pred_mean = np.array(y_pred_dist["point"])
    
    min_len = min(len(y_true_future), len(pred_mean))
    if min_len == 0:
        return np.nan
    
    y_true = y_true_future[min_len - 1]
    y_pred = pred_mean[min_len - 1]
    
    return float((y_true - y_pred) ** 2)

def clean_context_name(s):
    """Convert [64_32_24] to '64, 32, 24' for display."""
    return s.replace("[","").replace("]","").replace("_", ", ")

def main():
    print("Loading Ground Truth Metrics...")
    metrics = load_json(os.path.join(RESULTS_DIR, "results_hd128.json"))
    if not metrics:
        print("Error: Could not load results_metrics.json")
        return

    true_curves = {k: data.get("val_acc_curve", []) for k, data in metrics.items() if data.get("val_acc_curve")}

    # data[context_name][epoch] = list of NLL/MSE values across all configs
    nll_data = defaultdict(lambda: defaultdict(list))
    mse_data = defaultdict(lambda: defaultdict(list))
    
    # Collect Data
    for epoch in EPOCHS_LIST:
        print(f"Processing Epoch {epoch}...")
        
        # Process ablation contexts
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
        
        # Process baseline (no context)
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

    # Prepare x-axis labels in order (Baseline first for reference)
    context_order = ["Baseline"] + [clean_context_name(ab) for ab in ABLATIONS_LIST]
    
    # Create figure with two subplots - give heatmap more height
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 12), gridspec_kw={'height_ratios': [6, 1]})
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle('FT-PFN Performance Comparison Across Context Strategies\n(Distribution over 12 Target Configurations)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Box width and positions
    n_epochs = len(EPOCHS_LIST)
    box_width = 0.5
    
    # Plot NLL
    for i, epoch in enumerate(EPOCHS_LIST):
        positions = np.arange(len(context_order)) + (i - 1) * box_width
        data_to_plot = [nll_data[ctx].get(epoch, [np.nan]) for ctx in context_order]
        
        bp = ax1.boxplot(data_to_plot, positions=positions, widths=box_width,
                         patch_artist=True, showfliers=False)
        
        for patch in bp['boxes']:
            patch.set_facecolor(EPOCH_COLORS[epoch])
            patch.set_alpha(0.7)
        for whisker in bp['whiskers']:
            whisker.set_color('gray')
        for cap in bp['caps']:
            cap.set_color('gray')
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
        #for flier in bp['fliers']:
        #    flier.set_markerfacecolor('gray')
        #    flier.set_markersize(4)
        #    flier.set_alpha(0.5)
    baseline_vals = nll_data["Baseline"][EPOCHS_LIST[0]]
    baseline_median = np.median(baseline_vals)
    ax1.set_ylabel('NLL (Negative Log Likelihood)\n← Lower is Better', fontsize=11)
    ax1.set_title('Uncertainty Quality (NLL)', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=baseline_median, color='#ffb74d', linestyle='--', linewidth=2)
    ax1.set_ylim(-4.5, -0)
    # Add x-axis labels to NLL plot (no xlabel text - only on bottom plot)
    ax1.set_xticks(np.arange(len(context_order)))
    ax1.set_xticklabels(context_order, rotation=45, ha='right', fontsize=10)
    ax1.tick_params(axis='both', which='major', labelsize=14, width=3)
    
    # NEW: MSE Heatmap with scientific notation
    # Build matrix: rows = epochs, cols = contexts
    mse_matrix = []
    for epoch in EPOCHS_LIST:
        row = []
        for ctx in context_order:
            mse_vals = mse_data[ctx].get(epoch, [])
            if mse_vals:
                mse = np.mean(mse_vals)  # Mean MSE
            else:
                mse = np.nan
            row.append(mse)
        mse_matrix.append(row)
    
    mse_matrix = np.array(mse_matrix)
    
    # Plot heatmap with LOG scale normalization (spreads out small differences)
    from matplotlib.colors import LogNorm
    vmin = np.nanmin(mse_matrix)
    vmax = np.nanmax(mse_matrix)
    im = ax2.imshow(mse_matrix, cmap='viridis', aspect='auto', norm=LogNorm(vmin=vmin, vmax=vmax))
    
    # Add colorbar with scientific notation
    cbar = fig.colorbar(im, ax=ax2, shrink=0.8, pad=0.02, format='%.0e')
    cbar.set_label('MSE (log scale)', fontsize=10)
    
    # Labels
    ax2.set_yticks(np.arange(len(EPOCHS_LIST)))
    ax2.set_yticklabels([f'{e}' for e in EPOCHS_LIST], fontsize=10)
    ax2.set_ylabel('Observation Epochs', fontsize=11)
    ax2.set_xticks(np.arange(len(context_order)))
    ax2.set_xticklabels(context_order, rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('Hidden Dimension Context (Ablation)', fontsize=11)
    ax2.set_title('Prediction Accuracy (MSE Heatmap)', fontsize=12, fontweight='bold', pad=10)
    
    # Add text annotations on heatmap with larger font
    for i in range(len(EPOCHS_LIST)):
        for j in range(len(context_order)):
            val = mse_matrix[i, j]
            if not np.isnan(val):
                # White text on dark cells, black on light
                text_color = 'white' if val < np.nanmedian(mse_matrix) else 'white'
                ax2.text(j, i, f'{val:.2e}', rotation=50, ha='center', va='center', 
                        fontsize=8, fontweight='bold', color=text_color)
    
    # Shared x-axis labels
    ax2.set_xticks(np.arange(len(context_order)))
    ax2.set_xticklabels(context_order, rotation=45, ha='right', fontsize=9)
    
    '''
    # Add legend
    legend_patches = [plt.Rectangle((0,0),1,1, facecolor=EPOCH_COLORS[e], alpha=0.7, label=f'{e} Epochs Observed')
                      for e in EPOCHS_LIST]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.95),
               title='Observation Horizon', title_fontsize=10, fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    '''

    # Save
    output_path = "analysis_128/nll_mse_boxplot_ep5.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
