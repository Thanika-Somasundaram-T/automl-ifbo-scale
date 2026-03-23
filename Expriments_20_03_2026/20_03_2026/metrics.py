"""
Metrics: normalization, NLL/MSE calculations, ground truth loading, and data computation.
"""
import json
import math
import os

import numpy as np

from config import (
    LOSS_MIN, LOSS_MAX, PRED_DIR, RESULTS_METRICS,
    BASE_SCALES, OBS_EPOCHS, SOURCE_CONFIGS,
    source_hp_label, short_target_label,
)


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────
def normalize_log_loss_curve(curve_values, loss_min=LOSS_MIN, loss_max=LOSS_MAX):
    curve_values = np.array(curve_values, dtype=float)
    log_losses = np.log(np.maximum(curve_values, 1e-12))
    log_min = math.log(loss_min)
    log_max = math.log(loss_max)
    norm = (log_losses - log_min) / (log_max - log_min)
    y = 1.0 - norm
    return np.clip(y, 0.0, 1.0)


def unnormalize_log_loss_curve(norm_values, loss_min=LOSS_MIN, loss_max=LOSS_MAX):
    """Convert normalized performance values back to raw loss space."""
    norm_values = np.clip(np.array(norm_values, dtype=float), 0.0, 1.0)
    log_min = math.log(loss_min)
    log_max = math.log(loss_max)
    log_losses = log_min + (1.0 - norm_values) * (log_max - log_min)
    return np.exp(log_losses)


# ─────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def parse_metrics_key(pred_key):
    """Strip trailing _epoch suffix from prediction keys."""
    return pred_key.rsplit("_", 1)[0]


# ─────────────────────────────────────────────
# NLL / MSE calculations
# ─────────────────────────────────────────────
def calculate_nll(y_true, pred_data, epoch_idx):
    y_true_future = np.array(y_true[epoch_idx:])
    pred_mean = np.array(pred_data["point"])
    q05 = np.array(pred_data["quantiles"]["0.05"])
    q95 = np.array(pred_data["quantiles"]["0.95"])
    min_len = min(len(y_true_future), len(pred_mean))
    if min_len == 0:
        return np.nan
    y_t = y_true_future[:min_len]
    y_m = pred_mean[:min_len]
    sigma = (q95[:min_len] - q05[:min_len]) / 3.29
    sigma = np.maximum(sigma, 1e-6)
    nll = 0.5 * np.log(2 * np.pi * sigma**2) + (y_t - y_m) ** 2 / (2 * sigma**2)
    return float(np.mean(nll))


def calculate_mse(y_true, pred_data, epoch_idx):
    y_true_future = np.array(y_true[epoch_idx:])
    pred_mean = np.array(pred_data["point"])
    min_len = min(len(y_true_future), len(pred_mean))
    if min_len == 0:
        return np.nan
    return float(np.mean((y_true_future[:min_len] - pred_mean[:min_len]) ** 2))


def calculate_nll_final(y_true_norm, pred_data):
    """NLL at final point in normalized space (no unnormalization)."""
    pred_mean = np.array(pred_data["point"])
    q05 = np.array(pred_data["quantiles"]["0.05"])
    q95 = np.array(pred_data["quantiles"]["0.95"])
    if len(pred_mean) == 0:
        return np.nan
    y_true_final = y_true_norm[-1]
    y_mean_final = pred_mean[-1]
    sigma = max((q95[-1] - q05[-1]) / 3.29, 1e-6)
    residual = y_true_final - y_mean_final
    return float(0.5 * np.log(2 * np.pi * sigma**2) + (residual**2) / (2 * sigma**2))


def calculate_mse_final(y_true_norm, pred_data):
    """MSE at final point in normalized space (no unnormalization)."""
    pred_mean = np.array(pred_data["point"])
    if len(pred_mean) == 0:
        return np.nan
    return float((y_true_norm[-1] - pred_mean[-1]) ** 2)


def calculate_nll_raw(y_true_raw, pred_data):
    """NLL at final point in raw loss space (unnormalize preds first)."""
    pred_mean = unnormalize_log_loss_curve(np.array(pred_data["point"]))
    q05 = unnormalize_log_loss_curve(np.array(pred_data["quantiles"]["0.05"]))
    q95 = unnormalize_log_loss_curve(np.array(pred_data["quantiles"]["0.95"]))
    if len(pred_mean) == 0:
        return np.nan
    y_true_final = y_true_raw[-1]
    y_mean_final = pred_mean[-1]
    sigma = max((q95[-1] - q05[-1]) / 3.29, 1e-6)
    residual = y_true_final - y_mean_final
    return float(0.5 * np.log(2 * np.pi * sigma**2) + (residual**2) / (2 * sigma**2))


def calculate_mse_raw(y_true_raw, pred_data):
    """MSE at final point in raw loss space (unnormalize preds first)."""
    pred_mean = unnormalize_log_loss_curve(np.array(pred_data["point"]))
    if len(pred_mean) == 0:
        return np.nan
    return float((y_true_raw[-1] - pred_mean[-1]) ** 2)


# ─────────────────────────────────────────────
# Load ground truth
# ─────────────────────────────────────────────
def load_ground_truth(normalized=True):
    metrics = load_json(RESULTS_METRICS)
    if not metrics:
        print("Error: results_metrics.json not found")
        return None
    true_curves = {}
    for key, data in metrics.items():
        if data.get("hidden_dim") != 128:
            continue
        loss_curve = data.get("val_loss_curve", [])
        if len(loss_curve) > 0:
            if normalized:
                try:
                    true_curves[key] = normalize_log_loss_curve(loss_curve).tolist()
                except (ValueError, ZeroDivisionError):
                    continue
            else:
                true_curves[key] = list(loss_curve)  # raw
    mode_str = "normalized" if normalized else "raw"
    print(f"Loaded {len(true_curves)} ground truth curves (hd=128, {mode_str})")
    return true_curves


# ─────────────────────────────────────────────
# Compute all metrics
# ─────────────────────────────────────────────
def compute_all_metrics(true_curves, raw_mode=False, final_point=False):
    """Returns list of dicts with (scale, source_config, epoch, target_hp, nll, mse)."""
    rows = []
    for scale in BASE_SCALES:
        for src_cfg in SOURCE_CONFIGS:
            for epoch in OBS_EPOCHS:
                fname = f"ifbo_pred_{scale}_configs_{src_cfg}_ep{epoch}.json"
                preds = load_json(os.path.join(PRED_DIR, fname))
                if not preds:
                    continue
                for pred_key, pred_data in preds.items():
                    target_hp = parse_metrics_key(pred_key)
                    if target_hp not in true_curves:
                        continue
                    if raw_mode:
                        nll = calculate_nll_raw(true_curves[target_hp], pred_data)
                        mse = calculate_mse_raw(true_curves[target_hp], pred_data)
                    elif final_point:
                        nll = calculate_nll_final(true_curves[target_hp], pred_data)
                        mse = calculate_mse_final(true_curves[target_hp], pred_data)
                    else:
                        nll = calculate_nll(true_curves[target_hp], pred_data, epoch)
                        mse = calculate_mse(true_curves[target_hp], pred_data, epoch)
                    rows.append({
                        "scale": scale,
                        "source_config": src_cfg,
                        "source_hp": source_hp_label(src_cfg),
                        "epoch": epoch,
                        "target_hp": target_hp,
                        "target_hp_short": short_target_label(target_hp),
                        "nll": nll,
                        "mse": mse,
                    })

    # Baseline
    for epoch in OBS_EPOCHS:
        baseline = load_json(os.path.join(PRED_DIR, f"baseline_ep{epoch}.json"))
        if not baseline:
            continue
        for pred_key, pred_data in baseline.items():
            target_hp = parse_metrics_key(pred_key)
            if target_hp not in true_curves:
                continue
            if raw_mode:
                nll = calculate_nll_raw(true_curves[target_hp], pred_data)
                mse = calculate_mse_raw(true_curves[target_hp], pred_data)
            elif final_point:
                nll = calculate_nll_final(true_curves[target_hp], pred_data)
                mse = calculate_mse_final(true_curves[target_hp], pred_data)
            else:
                nll = calculate_nll(true_curves[target_hp], pred_data, epoch)
                mse = calculate_mse(true_curves[target_hp], pred_data, epoch)
            rows.append({
                "scale": "baseline",
                "source_config": 0,
                "source_hp": "baseline",
                "epoch": epoch,
                "target_hp": target_hp,
                "target_hp_short": short_target_label(target_hp),
                "nll": nll,
                "mse": mse,
            })

    print(f"Computed {len(rows)} metric rows")
    return rows
