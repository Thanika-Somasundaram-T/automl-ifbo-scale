"""
Boxplot comparison of NLL and MSE across context strategies.
Generates two subplots (NLL and MSE) with clear legends.
Uses unnormalized predictions from utils.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.colors import LogNorm

from utils import unnormalize_pred  # <- using unnormalized predictions

# -----------------------------
# Configuration
# -----------------------------
RESULTS_DIR = "results"
PRED_DIR = "results/norm 3.0"
EPOCHS_LIST = [0, 5, 10, 20]
ABLATIONS_LIST = ["[64_32_24]", "[64]", "[64_32]", "[32]", "[32_24]", "[24]"]  # final set
EPOCH_COLORS = {0: "#1f77b4", 5: "#ff7f0e", 10: "#2ca02c", 20: "#d62728"}

# -----------------------------
# Utility functions
# -----------------------------
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def calculate_nll(y_true_curve, y_pred_dist):
    y_true_curve = np.array(y_true_curve)
    pred_mean = np.array(y_pred_dist["point"])
    q05 = np.array(y_pred_dist["quantiles"]["0.05"])
    q95 = np.array(y_pred_dist["quantiles"]["0.95"])

    if len(y_true_curve) == 0 or len(pred_mean) == 0:
        return np.nan

    # Unnormalize predictions
    pred_mean = unnormalize_pred(pred_mean)
    q05 = unnormalize_pred(q05)
    q95 = unnormalize_pred(q95)

    y_true_final = y_true_curve[-1]
    y_mean_final = pred_mean[-1]

    sigma = max((q95[-1] - q05[-1]) / 3.29, 1e-6)
    residual = y_true_final - y_mean_final

    nll = 0.5 * np.log(2 * np.pi * sigma ** 2) + (residual ** 2) / (2 * sigma ** 2)
    return nll

def calculate_mse(y_true_curve, y_pred_dist):
    y_true_curve = np.array(y_true_curve)
    pred_mean = np.array(y_pred_dist["point"])

    if len(y_true_curve) == 0 or len(pred_mean) == 0:
        return np.nan

    # Unnormalize prediction
    pred_mean = unnormalize_pred(pred_mean)

    y_true_final = y_true_curve[-1]
    y_mean_final = pred_mean[-1]

    return (y_true_final - y_mean_final) ** 2

def clean_context_name(s):
    return s.replace("[","").replace("]","").replace("_", ", ")

# -----------------------------
# Main plotting script
# -----------------------------
def main():
    print("Loading Ground Truth Metrics...")
    metrics = load_json(os.path.join(RESULTS_DIR, "results_metrics.json"))
    if not metrics:
        print("Error: Could not load results_metrics.json")
        return

    true_curves = {k: data.get("val_loss_curve", []) for k, data in metrics.items() if data.get("val_loss_curve")}

    nll_data = defaultdict(lambda: defaultdict(list))
    mse_data = defaultdict(lambda: defaultdict(list))

    # -----------------------------
    # Collect NLL & MSE
    # -----------------------------
    for epoch in EPOCHS_LIST:
        print(f"Processing Epoch {epoch}...")

        # Ablation contexts
        for context_name in ABLATIONS_LIST:
            pred_filename = f"ifbo_pred_{context_name}_configs_12_ep{epoch}.json"
            pred_path = os.path.join(PRED_DIR, pred_filename)
            predictions = load_json(pred_path)
            if not predictions:
                continue

            clean_ctx = clean_context_name(context_name)
            for pred_key, pred_data in predictions.items():
                metrics_key = pred_key.rsplit('_', 1)[0]
                if metrics_key in true_curves:
                    nll = calculate_nll(true_curves[metrics_key], pred_data)
                    mse = calculate_mse(true_curves[metrics_key], pred_data)
                    if not np.isnan(nll):
                        nll_data[clean_ctx][epoch].append(nll)
                    if not np.isnan(mse):
                        mse_data[clean_ctx][epoch].append(mse)

        # Baseline
        baseline_filename = f"baseline_ep{epoch}.json"
        baseline_path = os.path.join(PRED_DIR, baseline_filename)
        baseline_preds = load_json(baseline_path)
        if baseline_preds:
            for pred_key, pred_data in baseline_preds.items():
                metrics_key = pred_key.rsplit('_', 1)[0]
                if metrics_key in true_curves:
                    nll = calculate_nll(true_curves[metrics_key], pred_data)
                    mse = calculate_mse(true_curves[metrics_key], pred_data)
                    if not np.isnan(nll):
                        nll_data["baseline"][epoch].append(nll)
                    if not np.isnan(mse):
                        mse_data["baseline"][epoch].append(mse)

    # -----------------------------
    # Plot
    # -----------------------------
    context_order = ["baseline"] + [clean_context_name(ab) for ab in ABLATIONS_LIST]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))
    fig.suptitle('FT-PFN Performance Comparison Across Context Strategies\n(Distribution over 12 Target Configurations)',
                 fontsize=14, fontweight='bold', y=0.98)

    # --- NLL Boxplot ---
    positions_base = np.arange(len(context_order))
    box_width = 0.25

    for i, epoch in enumerate(EPOCHS_LIST):
        positions = positions_base + (i - 1) * box_width
        data_to_plot = [nll_data[ctx].get(epoch, [np.nan]) for ctx in context_order]

        bp = ax1.boxplot(data_to_plot, positions=positions, widths=box_width,
                         patch_artist=True, showfliers=True)
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
        for flier in bp['fliers']:
            flier.set_markerfacecolor('gray')
            flier.set_markersize(4)
            flier.set_alpha(0.5)

        # Baseline median + shaded
        baseline_vals = nll_data.get("baseline", {}).get(epoch, [])
        if baseline_vals:
            median_baseline = np.median(baseline_vals)
            min_baseline = np.min(baseline_vals)
            max_baseline = np.max(baseline_vals)
            ax1.axhline(median_baseline, color=EPOCH_COLORS[epoch], linestyle='--', alpha=0.7,
                        label=f'Baseline median (epoch {epoch})')
            ax1.fill_between([-1, len(context_order)], min_baseline, max_baseline,
                             color=EPOCH_COLORS[epoch], alpha=0.1)

    # Reference line
    ax1.axhline(y=-4.0, color='gray', linestyle='--', alpha=0.5, label='Reference: -4.0')

    ax1.set_xticks(positions_base)
    ax1.set_xticklabels(context_order, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('NLL (Negative Log Likelihood)\n← Lower is Better', fontsize=11)
    ax1.set_title('Uncertainty Quality (NLL)', fontsize=12, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3)

    # Deduplicate legend
    handles, labels = ax1.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), fontsize=9, title='Baseline & Observation Horizon', title_fontsize=10)

    # # --- Force NLL y-axis ---
    # all_nll_values = []
    # for ctx in context_order:
    #     for epoch in EPOCHS_LIST:
    #         all_nll_values.extend(nll_data[ctx].get(epoch, []))
    # data_max = max(all_nll_values) if all_nll_values else 1.0
    # ax1.set_ylim(-4, data_max + 0.1 * abs(data_max))
    # ax1.set_autoscaley_on(False)  # prevent autoscaling from overriding

    # --- MSE Heatmap ---
    mse_matrix = []
    for epoch in EPOCHS_LIST:
        row = []
        for ctx in context_order:
            mse_vals = mse_data[ctx].get(epoch, [])
            row.append(np.mean(mse_vals) if mse_vals else np.nan)
        mse_matrix.append(row)
    mse_matrix = np.array(mse_matrix)

    im = ax2.imshow(mse_matrix, cmap='viridis', aspect='auto', norm=LogNorm(vmin=1e-5, vmax=1e-1))
    cbar = fig.colorbar(im, ax=ax2, shrink=0.8, pad=0.02, format='%.0e')
    cbar.set_label('MSE (log scale)', fontsize=10)

    ax2.set_yticks(np.arange(len(EPOCHS_LIST)))
    ax2.set_yticklabels([f'{e}' for e in EPOCHS_LIST], fontsize=10)
    ax2.set_ylabel('Observation Epochs', fontsize=11)
    ax2.set_xticks(np.arange(len(context_order)))
    ax2.set_xticklabels(context_order, rotation=45, ha='right', fontsize=9)
    ax2.set_xlabel('Hidden Dimension Context (Ablation)', fontsize=11)
    ax2.set_title('Prediction Accuracy (MSE Heatmap)', fontsize=12, fontweight='bold', pad=10)

    for i in range(len(EPOCHS_LIST)):
        for j in range(len(context_order)):
            val = mse_matrix[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f'{val:.2e}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    # Legend patches for epochs
    legend_patches = [plt.Rectangle((0,0),1,1, facecolor=EPOCH_COLORS[e], alpha=0.7,
                                    label=f'{e} Epochs Observed') for e in EPOCHS_LIST]
    fig.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(0.98, 0.95),
               title='Observation Horizon', title_fontsize=10, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs("box_plots", exist_ok=True)
    output_path = "box_plots/nll_mse_boxplot_pred_unnormalized.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()