import os
import json
import numpy as np
from collections import defaultdict
import re

# ---------------------------------------------------
# Configuration
# ---------------------------------------------------
RESULTS_DIR = "results"
EPOCHS_LIST = [15, 50, 90]
ABLATIONS_LIST = [
    "[16]", "[18]", "[24]", "[32]", "[64]",
    "[64_16]", "[64_18]", "[64_24]", "[64_32]",
    "[32_24_18]", "[64_32_24]", "[64_32_24_18]",
    "[64_32_24_18_16]", "[64_32_24_18_16_8]", "[64_32_24_18_16_8_4]"
]

# ---------------------------------------------------
# Helper Functions
# ---------------------------------------------------
def calculate_nll(y_true_curve, y_pred_dist, epoch_idx):
    """Calculate Gaussian NLL (approximation using quantiles)."""
    y_true_future = np.array(y_true_curve[epoch_idx:])
    pred_mean = np.array(y_pred_dist["point"])
    q05 = np.array(y_pred_dist["quantiles"]["0.05"])
    q95 = np.array(y_pred_dist["quantiles"]["0.95"])
    
    min_len = min(len(y_true_future), len(pred_mean))
    if min_len == 0:
        return np.nan
        
    y_true = y_true_future[:min_len]
    y_mean = pred_mean[:min_len]
    y_q05 = q05[:min_len]
    y_q95 = q95[:min_len]
    
    sigma = (y_q95 - y_q05) / 3.29
    sigma = np.maximum(sigma, 1e-6)
    
    residual = y_true - y_mean
    nll_per_step = 0.5 * np.log(2 * np.pi * sigma**2) + (residual**2) / (2 * sigma**2)
    
    return np.mean(nll_per_step)

def calculate_mse(y_true_curve, y_pred_dist, epoch_idx):
    """Calculate Mean Squared Error between true curve and predicted mean."""
    y_true_future = np.array(y_true_curve[epoch_idx:])
    pred_mean = np.array(y_pred_dist["point"])
    
    min_len = min(len(y_true_future), len(pred_mean))
    if min_len == 0:
        return np.nan
        
    y_true = y_true_future[:min_len]
    y_mean = pred_mean[:min_len]
    
    return np.mean((y_true - y_mean) ** 2)

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def clean_context_name(s):
    if s == "Baseline": return "Baseline (No Context)"
    return s.replace("[","").replace("]","").replace("_", ", ")

# ---------------------------------------------------
# Main Analysis
# ---------------------------------------------------
def main():
    print("Loading Ground Truth Metrics...")
    metrics = load_json(os.path.join(RESULTS_DIR, "results_metrics.json"))
    if not metrics:
        print("Error: Could not load results_metrics.json")
        return

    true_curves = {k: data.get("val_acc_curve", []) for k, data in metrics.items() if data.get("val_acc_curve")}

    # detailed_data[config_id][context_name][epoch] = {"nll": ..., "mse": ...}
    detailed_data = defaultdict(lambda: defaultdict(dict))
    
    # 1. Collect Data
    for epoch in EPOCHS_LIST:
        print(f"Processing Epoch {epoch}...")
        
        # Sources: Standard Ablations + Baseline
        sources = [(ab, f"ifbo_pred_{ab}_ep{epoch}.json") for ab in ABLATIONS_LIST]
        baseline_file = f"baseline_{epoch}.json"
        if os.path.exists(os.path.join(RESULTS_DIR, baseline_file)):
            sources.append(("Baseline", baseline_file))
        else:
            print(f"Warning: {baseline_file} not found.")
        
        for context_name, pred_filename in sources:
            pred_path = os.path.join(RESULTS_DIR, pred_filename)
            predictions = load_json(pred_path)
            
            if not predictions:
                continue
                
            clean_ctx = clean_context_name(context_name)

            # --- NLL and MSE Calculation for all predictions in this context ---
            for pred_key, pred_data in predictions.items():
                metrics_key = pred_key.rsplit('_', 1)[0]
                
                if metrics_key in true_curves:
                    nll = calculate_nll(true_curves[metrics_key], pred_data, epoch)
                    mse = calculate_mse(true_curves[metrics_key], pred_data, epoch)
                    if not np.isnan(nll) and not np.isnan(mse):
                        detailed_data[metrics_key][clean_ctx][epoch] = {"nll": nll, "mse": mse}

    # 2. Generate HTML Report
    html_path = "analysis_outputs/analysis_report_detailed.html"
    print(f"Generating report: {html_path}...")
    
    style = """
    <style>
        body { font-family: sans-serif; margin: 40px; background-color: #f9f9f9; }
        .container { max-width: 1000px; margin: auto; background: white; padding: 40px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 0.85em; }
        th, td { border: 1px solid #ddd; padding: 6px; text-align: center; }
        th { background-color: #f1f1f1; font-weight: bold; }
        td:first-child { text-align: left; font-weight: bold; min-width: 140px; max-width: 180px; }
        tbody tr:nth-child(odd) { background-color: #fff; }
        tbody tr:nth-child(even) { background-color: #f5f5f5; }
        .nll-col { font-weight: bold; }
        .mse-col { color: #555; }
        .best { background-color: #198754 !important; color: white; font-weight: bold; }
        .worst { background-color: #dc3545 !important; color: white; font-weight: bold; }
    </style>
    """
    
    html_content = [f"<html><head>{style}</head><body><div class='container'>"]
    html_content.append(f"<h1>FT-PFN Detailed Analysis (Per Configuration)</h1>")
    html_content.append(f"<p>Structure: For each Config, we show <strong>NLL</strong> (Gaussian approx, lower=better) and <strong>MSE</strong> (lower=better).</p>")
    
    # Sort configs
    all_configs = sorted(detailed_data.keys())
    
    # Context sorter
    def context_sorter(c):
        if "Baseline" in c: return -1
        c_raw = c.replace(", ", "_")
        c_raw = f"[{c_raw}]"
        for idx, ab in enumerate(ABLATIONS_LIST):
            if ab == c_raw: return idx
        return 999
        
    for config_id in all_configs:
        try:
             lr = re.search(r"lr([\d\.e-]+)_", config_id).group(1)
             wd = re.search(r"wd([\d\.e-]+)_", config_id).group(1)
             title = f"Config: LR={lr}, WD={wd}"
        except:
             title = f"Config: {config_id}"
             
        html_content.append(f"<h2>{title}</h2>")
        html_content.append(f"<div class='config-meta'>ID: {config_id}</div>")
        
        html_content.append("<table>")
        html_content.append("<thead><tr><th rowspan='2'>Context</th>")
        
        # Header Row 1
        for e in EPOCHS_LIST: 
            html_content.append(f"<th colspan='2'>Epoch {e}</th>")
        html_content.append("</tr><tr>")
        
        # Header Row 2
        for _ in EPOCHS_LIST:
            html_content.append("<th>NLL</th><th>MSE</th>")
        html_content.append("</tr></thead><tbody>")
        
        contexts = sorted(detailed_data[config_id].keys(), key=context_sorter)
        
        # Collect all NLLs and MSEs for coloring
        all_nlls = []
        all_mses = []
        for c in contexts:
            for e in EPOCHS_LIST:
                val = detailed_data[config_id][c].get(e)
                if val:
                    all_nlls.append(val["nll"])
                    all_mses.append(val["mse"])
        
        nll_min = min(all_nlls) if all_nlls else -999
        nll_max = max(all_nlls) if all_nlls else 999
        mse_min = min(all_mses) if all_mses else 0
        mse_max = max(all_mses) if all_mses else 1
        
        for ctx in contexts:
            html_content.append(f"<tr><td>{ctx}</td>")
            for epoch in EPOCHS_LIST:
                data = detailed_data[config_id][ctx].get(epoch)
                
                if data:
                    nll = data["nll"]
                    mse = data["mse"]
                    
                    # NLL Cell
                    nll_class = "nll-col"
                    if len(all_nlls) > 1:
                        if np.isclose(nll, nll_min): nll_class += " best"
                        elif np.isclose(nll, nll_max): nll_class += " worst"
                    html_content.append(f"<td class='{nll_class}'>{nll:.4f}</td>")
                    
                    # MSE Cell - use exact comparison since values are very small
                    mse_class = "mse-col"
                    if len(all_mses) > 1:
                        if mse == mse_min: mse_class += " best"
                        elif mse == mse_max: mse_class += " worst"
                    html_content.append(f"<td class='{mse_class}'>{mse:.2e}</td>")
                else:
                    html_content.append("<td>-</td><td>-</td>")
                
            html_content.append("</tr>")
            
        html_content.append("</tbody></table>")

    html_content.append("</div></body></html>")
    
    with open(html_path, "w") as f:
        f.write("\n".join(html_content))
        
    print(f"Saved detailed report to {html_path}")

    # ===================================================
    # SUMMARY LEADERBOARD
    # ===================================================
    print("\n" + "="*70)
    print(f" SUMMARY LEADERBOARD (Across {len(detailed_data)} Configs)")
    print("="*70)
    print(f"{'Context':<30} | {'Mean NLL':<12} | {'Mean MSE':<12}")
    print("-" * 70)
    
    leaderboard = []
    
    first_cfg = list(detailed_data.keys())[0]
    all_contexts = list(detailed_data[first_cfg].keys())
    
    for ctx in all_contexts:
        nll_values = []
        mse_values = []
        
        for epoch in EPOCHS_LIST:
            for config_id in detailed_data:
                val = detailed_data[config_id].get(ctx, {}).get(epoch)
                if val:
                    nll_values.append(val["nll"])
                    mse_values.append(val["mse"])
        
        if nll_values:
            avg_nll = sum(nll_values) / len(nll_values)
            avg_mse = sum(mse_values) / len(mse_values)
            leaderboard.append({"name": ctx, "nll": avg_nll, "mse": avg_mse})
            
    # Sort by NLL (Lower is Better)
    leaderboard.sort(key=lambda x: x["nll"])
    
    for item in leaderboard:
        print(f"{item['name']:<30} | {item['nll']:<12.4f} | {item['mse']:<12.6f}")
    print("="*70)

if __name__ == "__main__":
    main()
