import json
import os
import numpy as np
from utils import generate_keys, normalize_log_loss_curve, normalize_log_loss_curve_per_curve

def calculate_nll(y_true_curve, y_pred_dist):
    y_true_final = y_true_curve[-1]
    y_mean_final = y_pred_dist["point"][-1]
    q05 = np.array(y_pred_dist["quantiles"]["0.05"])
    q95 = np.array(y_pred_dist["quantiles"]["0.95"])
    sigma = (q95[-1] - q05[-1]) / 3.29
    sigma = max(sigma, 1e-6)
    residual = y_true_final - y_mean_final
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + (residual**2) / (2 * sigma**2)
    return residual, sigma, nll

def calculate_mse(y_true_curve, y_pred_dist):
    y_true_final = y_true_curve[-1]
    y_mean_final = y_pred_dist["point"][-1]
    mse = (y_true_final - y_mean_final) ** 2
    return mse

# --- Load ground truth ---
with open("results/results_metrics.json", "r") as f:
    result_metrics = json.load(f)

# --- Settings ---
widths = [32, 24, "32_24"]
configs = range(1, 13)
eps = [0, 5, 10, 20, 50, 90]
keys_128 = generate_keys(64)
pred_dir = "results/only_hd"

# --- Table dict for JSON ---
table_dict = {}

# --- Function to process prediction files ---
def process_file(file_path, row_name, table_dict):
    table_dict[row_name] = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            pred_data = json.load(f)
        for col_c in configs:
            base_key = keys_128[col_c]
            pred_key = f"{base_key}_{row_name.split('_ep')[-1]}"  # epoch from row_name
            if base_key in result_metrics and pred_key in pred_data:
                y_true_curve = result_metrics[base_key]["val_loss_curve"]
                y_pred_dist = pred_data[pred_key]
                residual, sigma, nll = calculate_nll(normalize_log_loss_curve(y_true_curve), y_pred_dist)
                mse = calculate_mse(normalize_log_loss_curve(y_true_curve), y_pred_dist)
                table_dict[row_name][f"config_{col_c}"] = {
                    "residual": residual,
                    "sigma": sigma,
                    "nll": nll,
                    "mse": mse,
                    
                }
            else:
                table_dict[row_name][f"config_{col_c}"] = None
    else:
        for col_c in configs:
            table_dict[row_name][f"config_{col_c}"] = None

# --- Process IFBO predictions ---
for w in widths:
    for row_c in configs:
        for ep in eps:
            row_name = f"{w}_config_{row_c}_obs_target_ep{ep}"
            pred_file = os.path.join(pred_dir, f"ifbo_pred_[{w}]_configs_{row_c}_ep{ep}.json")
            process_file(pred_file, row_name, table_dict)

# --- Process Baseline predictions ---
baseline_eps = [5, 10, 20, 50, 90]
for ep in baseline_eps:
    row_name = f"baseline_ep{ep}"
    baseline_file = os.path.join(pred_dir, f"baseline_ep{ep}.json")
    process_file(baseline_file, row_name, table_dict)

# --- Save JSON ---
with open("table_ohd.json", "w") as f:
    json.dump(table_dict, f, indent=2)

print("JSON updated with baseline predictions!")