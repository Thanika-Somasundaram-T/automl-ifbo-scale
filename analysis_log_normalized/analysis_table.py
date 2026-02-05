"""
HTML analysis tables for log-normalized hidden dimension.
Generates tables for epochs 0, 5, 10, 20.
Two-tier highlighting: light colors for column best/worst, dark for table best/worst.
"""
import os
import json
import numpy as np
from collections import defaultdict

# Configuration
RESULTS_DIR = "../results"
OUTPUT_DIR = "."
EPOCHS_LIST = [0, 5, 10, 20]
ABLATIONS_LIST = [
    "[4]", "[8]", "[16]", "[24]", "[32]", "[64]",
    "[8_4]", "[16_8_4]", "[24_16_8_4]", "[32_24_16]", "[32_24_16_8_4]",
    "[64_4]", "[64_8]", "[64_8_4]", "[64_16]", "[64_24]", "[64_32]",
    "[64_24_16_8_4]", "[64_32_16_8_4]", "[64_32_24]", "[64_32_24_8_4]",
    "[64_32_24_16]", "[64_32_24_16_4]", "[64_32_24_16_8]", "[64_32_24_16_8_4]"
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

def clean_context_name(ctx):
    return ctx.strip("[]").replace("_", ", ")

def main():
    print("Loading Ground Truth Metrics...")
    gt_path = os.path.join(RESULTS_DIR, "results_metrics.json")
    gt_data = load_json(gt_path)
    if not gt_data:
        print("Error: Could not load results_metrics.json")
        return
    
    true_curves = {k: data.get("val_acc_curve", []) for k, data in gt_data.items() if data.get("val_acc_curve")}
    
    hd128_configs = {}
    for run_key, run_data in gt_data.items():
        if run_data.get("hidden_dim") == 128:
            lr = run_data.get("lr")
            wd = run_data.get("weight_decay")
            hd128_configs[run_key] = {"lr": lr, "wd": wd}
    
    detailed_data = {}
    
    for config_key, config_info in hd128_configs.items():
        metrics_display = f"LR={config_info['lr']}, WD={config_info['wd']}"
        detailed_data[metrics_display] = defaultdict(lambda: defaultdict(dict))
        
        for epoch in EPOCHS_LIST:
            baseline_path = os.path.join(RESULTS_DIR, f"baseline_{epoch}.json")
            baseline_data = load_json(baseline_path)
            
            if baseline_data:
                for pred_key, pred_dist in baseline_data.items():
                    metrics_key = pred_key.rsplit('_', 1)[0]
                    if metrics_key == config_key:
                        nll = calculate_nll(true_curves[config_key], pred_dist, epoch)
                        mse = calculate_mse(true_curves[config_key], pred_dist, epoch)
                        if not np.isnan(nll) and not np.isnan(mse):
                            detailed_data[metrics_display]["Baseline"][epoch] = {"nll": nll, "mse": mse}
                        break
            
            for ablation in ABLATIONS_LIST:
                pred_path = os.path.join(RESULTS_DIR, f"ifbo_pred_{ablation}_ep{epoch}.json")
                pred_data = load_json(pred_path)
                
                if not pred_data:
                    continue
                
                clean_ctx = clean_context_name(ablation)
                
                for pred_key, pred_dist in pred_data.items():
                    metrics_key = pred_key.rsplit('_', 1)[0]
                    if metrics_key == config_key:
                        nll = calculate_nll(true_curves[config_key], pred_dist, epoch)
                        mse = calculate_mse(true_curves[config_key], pred_dist, epoch)
                        if not np.isnan(nll) and not np.isnan(mse):
                            detailed_data[metrics_display][clean_ctx][epoch] = {"nll": nll, "mse": mse}
                        break

    html_path = os.path.join(OUTPUT_DIR, "analysis_report.html")
    print(f"Generating report: {html_path}...")
    
    # Three-tier color scheme: light for column, dark for table, black border for global
    style = """
    <style>
        body { font-family: sans-serif; margin: 40px; background-color: #f9f9f9; }
        .container { max-width: 1200px; margin: auto; background: white; padding: 40px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 30px; font-size: 0.85em; }
        th, td { border: 1px solid #ddd; padding: 6px; text-align: center; }
        th { background-color: #f1f1f1; font-weight: bold; }
        td:first-child { text-align: left; font-weight: bold; min-width: 140px; max-width: 180px; }
        tbody tr:nth-child(odd) { background-color: #fff; }
        tbody tr:nth-child(even) { background-color: #f5f5f5; }
        .nll-col { font-weight: bold; }
        .mse-col { color: #555; }
        /* Light colors for column best/worst */
        .col-best { background-color: #c8e6c9 !important; }
        .col-worst { background-color: #ffcdd2 !important; }
        /* Dark colors for table best/worst */
        .table-best { background-color: #198754 !important; color: white; font-weight: bold; }
        .table-worst { background-color: #dc3545 !important; color: white; font-weight: bold; }
        /* Black border for global best/worst across ALL tables */
        .global-best { border: 3px solid black !important; }
        .global-worst { border: 3px solid black !important; }
        /* Black border on entire row for global best/worst */
        .global-best-row td { border-top: 3px solid black !important; border-bottom: 3px solid black !important; }
        .global-best-row td:first-child { border-left: 3px solid black !important; }
        .global-best-row td:last-child { border-right: 3px solid black !important; }
        .global-worst-row td { border-top: 3px solid black !important; border-bottom: 3px solid black !important; }
        .global-worst-row td:first-child { border-left: 3px solid black !important; }
        .global-worst-row td:last-child { border-right: 3px solid black !important; }
    </style>
    """
    
    # First pass: collect all values across all configs to find global best/worst
    global_nlls = []
    global_mses = []
    for metrics_key, ctx_data in detailed_data.items():
        contexts = ["Baseline"] + [clean_context_name(ab) for ab in ABLATIONS_LIST]
        contexts = [c for c in contexts if c in ctx_data]
        for ctx in contexts:
            for epoch in EPOCHS_LIST:
                if epoch in ctx_data[ctx]:
                    global_nlls.append((metrics_key, ctx, epoch, ctx_data[ctx][epoch]["nll"]))
                    global_mses.append((metrics_key, ctx, epoch, ctx_data[ctx][epoch]["mse"]))
    
    global_best_nll = min(global_nlls, key=lambda x: x[3]) if global_nlls else None
    global_worst_nll = max(global_nlls, key=lambda x: x[3]) if global_nlls else None
    global_best_mse = min(global_mses, key=lambda x: x[3]) if global_mses else None
    global_worst_mse = max(global_mses, key=lambda x: x[3]) if global_mses else None
    
    print(f"Global best NLL: {global_best_nll}")
    print(f"Global worst NLL: {global_worst_nll}")
    print(f"Global best MSE: {global_best_mse}")
    print(f"Global worst MSE: {global_worst_mse}")
    
    html_content = [f"<html><head><title>Log-Normalized Analysis</title>{style}</head><body><div class='container'>"]
    html_content.append("<h1>FT-PFN Analysis (Log-Normalized Hidden Dim)</h1>")
    html_content.append("<p>Observation epochs: 0, 5, 10, 20. <b>NLL</b> and <b>MSE</b> (lower=better).</p>")
    html_content.append("<p><small>🟩 Dark green = table best | 🟥 Dark red = table worst | Light green = column best | Light red = column worst</small></p>")
    
    for metrics_key, ctx_data in sorted(detailed_data.items()):
        html_content.append(f"<h2>Config: {metrics_key}</h2>")
        html_content.append("<table><thead><tr><th rowspan='2'>Context</th>")
        for epoch in EPOCHS_LIST:
            html_content.append(f"<th colspan='2'>Epoch {epoch}</th>")
        html_content.append("</tr><tr>")
        for _ in EPOCHS_LIST:
            html_content.append("<th>NLL</th><th>MSE</th>")
        html_content.append("</tr></thead><tbody>")
        
        contexts = ["Baseline"] + [clean_context_name(ab) for ab in ABLATIONS_LIST]
        contexts = [c for c in contexts if c in ctx_data]
        
        # Find column best/worst and table best/worst
        col_best_worst = {}
        all_nlls = []
        all_mses = []
        
        for epoch in EPOCHS_LIST:
            nlls = [(c, ctx_data[c][epoch]["nll"]) for c in contexts if epoch in ctx_data[c]]
            mses = [(c, ctx_data[c][epoch]["mse"]) for c in contexts if epoch in ctx_data[c]]
            
            if nlls:
                col_best_worst[(epoch, "nll", "best")] = min(nlls, key=lambda x: x[1])[0]
                col_best_worst[(epoch, "nll", "worst")] = max(nlls, key=lambda x: x[1])[0]
                all_nlls.extend(nlls)
            if mses:
                col_best_worst[(epoch, "mse", "best")] = min(mses, key=lambda x: x[1])[0]
                col_best_worst[(epoch, "mse", "worst")] = max(mses, key=lambda x: x[1])[0]
                all_mses.extend(mses)
        
        # Table-wide best/worst
        table_best_nll = min(all_nlls, key=lambda x: x[1]) if all_nlls else None
        table_worst_nll = max(all_nlls, key=lambda x: x[1]) if all_nlls else None
        table_best_mse = min(all_mses, key=lambda x: x[1]) if all_mses else None
        table_worst_mse = max(all_mses, key=lambda x: x[1]) if all_mses else None
        
        for ctx in contexts:
            # Check if this row contains global best/worst
            row_class = ""
            if global_best_nll and global_best_nll[0] == metrics_key and global_best_nll[1] == ctx:
                row_class = " global-best-row"
            elif global_worst_nll and global_worst_nll[0] == metrics_key and global_worst_nll[1] == ctx:
                row_class = " global-worst-row"
            elif global_best_mse and global_best_mse[0] == metrics_key and global_best_mse[1] == ctx:
                row_class = " global-best-row"
            elif global_worst_mse and global_worst_mse[0] == metrics_key and global_worst_mse[1] == ctx:
                row_class = " global-worst-row"
            
            html_content.append(f"<tr class='{row_class.strip()}'><td>{ctx}</td>")
            for epoch in EPOCHS_LIST:
                if epoch in ctx_data[ctx]:
                    nll = ctx_data[ctx][epoch]["nll"]
                    mse = ctx_data[ctx][epoch]["mse"]
                    
                    # NLL class - check global first (black border), then table (dark), then column (light)
                    nll_class = "nll-col"
                    # Check global best/worst first (adds black border)
                    if global_best_nll and global_best_nll[0] == metrics_key and global_best_nll[1] == ctx and global_best_nll[2] == epoch:
                        nll_class += " global-best table-best"
                    elif global_worst_nll and global_worst_nll[0] == metrics_key and global_worst_nll[1] == ctx and global_worst_nll[2] == epoch:
                        nll_class += " global-worst table-worst"
                    elif table_best_nll and table_best_nll[0] == ctx and abs(table_best_nll[1] - nll) < 1e-10:
                        nll_class += " table-best"
                    elif table_worst_nll and table_worst_nll[0] == ctx and abs(table_worst_nll[1] - nll) < 1e-10:
                        nll_class += " table-worst"
                    elif col_best_worst.get((epoch, "nll", "best")) == ctx:
                        nll_class += " col-best"
                    elif col_best_worst.get((epoch, "nll", "worst")) == ctx:
                        nll_class += " col-worst"
                    
                    # MSE class
                    mse_class = "mse-col"
                    if global_best_mse and global_best_mse[0] == metrics_key and global_best_mse[1] == ctx and global_best_mse[2] == epoch:
                        mse_class += " global-best table-best"
                    elif global_worst_mse and global_worst_mse[0] == metrics_key and global_worst_mse[1] == ctx and global_worst_mse[2] == epoch:
                        mse_class += " global-worst table-worst"
                    elif table_best_mse and table_best_mse[0] == ctx and abs(table_best_mse[1] - mse) < 1e-15:
                        mse_class += " table-best"
                    elif table_worst_mse and table_worst_mse[0] == ctx and abs(table_worst_mse[1] - mse) < 1e-15:
                        mse_class += " table-worst"
                    elif col_best_worst.get((epoch, "mse", "best")) == ctx:
                        mse_class += " col-best"
                    elif col_best_worst.get((epoch, "mse", "worst")) == ctx:
                        mse_class += " col-worst"
                    
                    html_content.append(f"<td class='{nll_class}'>{nll:.4f}</td>")
                    html_content.append(f"<td class='{mse_class}'>{mse:.2e}</td>")
                else:
                    html_content.append("<td>-</td><td>-</td>")
            html_content.append("</tr>")
        
        html_content.append("</tbody></table>")
    
    html_content.append("</div></body></html>")
    
    with open(html_path, "w") as f:
        f.write("\n".join(html_content))
    
    print(f"✅ Saved: {html_path}")

if __name__ == "__main__":
    main()
