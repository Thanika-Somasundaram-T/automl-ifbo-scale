"""
Aggregates results_diverse_lower_k/*.json into the same
(policy_variant, k, rel_err %, NLL, WIS, WIS90, cov90) table format used for
the real-world-data ablation.

Metric definitions (final predicted point only, one row per (target config,
observation epoch)):
  - rel_err %: |pred_median - true| / |true| * 100, in raw val_loss space.
  - NLL:       Gaussian NLL in normalized [0,1] IFBO-score space, sigma from
               the 90% interval width (q95-q05)/3.29.
  - WIS:       Weighted Interval Score (K=3: 50/80/90% central intervals),
               in normalized score space.
  - WIS90:     WIS restricted to just the 90% interval (K=1).
  - cov90:     empirical coverage of the 90% interval (fraction of points
               where true value falls within [q05, q95]).

Run from the repo root, after run_diverse_lower_k.py has produced
predictions:
    python compute_diverse_table.py
"""
import json
import math
import os
from collections import defaultdict

import numpy as np

RESULTS_JSON = "./results_***/results_metrics.json"
PRED_DIR = "./results_diverse_lower_k"
TARGET_EPOCHS = 100

K_VALUES = [4, 8, 12, 16, 24, 32]
SUBSAMPLE_MODES = ["geometric", "uniform"]
NON_DIVERSE_POLICIES = ["all_lower", "nearest_lower", "baseline", "interpolation"]


# ---------------------------------------------------------------------
# Normalization (mirrors utils.normalize_log_loss_curve / compute_min_max)
# ---------------------------------------------------------------------
def compute_min_max(all_results, buffer=0.05, min_value=1e-8):
    all_losses = []
    for entry in all_results.values():
        curve = entry.get("val_loss_curve", [])[:TARGET_EPOCHS]
        if curve:
            all_losses.append(np.asarray(curve, dtype=np.float64))

    all_losses = np.concatenate(all_losses)
    log_others = np.log(np.clip(all_losses, 1e-8, None))
    lo, hi = float(log_others.min()), float(log_others.max())
    margin = (hi - lo) * buffer

    lo_adj = max(lo - margin, math.log(min_value))
    hi_adj = hi + margin
    return lo_adj, hi_adj


def normalize_log_loss(val_loss, log_min, log_max, eps=1e-8):
    val_loss = np.asarray(val_loss, dtype=np.float64)
    log_curve = np.log(val_loss + eps)
    norm = (log_curve - log_min) / (log_max - log_min)
    norm = np.clip(norm, 0.0, 1.0)
    return 1.0 - norm


def unnormalize_log_loss(y_norm, log_min, log_max):
    y_norm = min(max(y_norm, 0.0), 1.0)
    log_loss = log_min + (1.0 - y_norm) * (log_max - log_min)
    return math.exp(log_loss)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def interval_score(l, u, y, alpha):
    score = u - l
    if y < l:
        score += (2.0 / alpha) * (l - y)
    elif y > u:
        score += (2.0 / alpha) * (y - u)
    return score


def gaussian_nll(y, mean, q05, q95):
    sigma = max((q95 - q05) / 3.29, 1e-6)
    return 0.5 * math.log(2 * math.pi * sigma ** 2) + (y - mean) ** 2 / (2 * sigma ** 2)


def wis_k3(y, q):
    median = q["0.50"]
    intervals = [
        (q["0.05"], q["0.95"], 0.10),
        (q["0.10"], q["0.90"], 0.20),
        (q["0.25"], q["0.75"], 0.50),
    ]
    total = 0.5 * abs(y - median)
    for l, u, alpha in intervals:
        total += (alpha / 2.0) * interval_score(l, u, y, alpha)
    return total / (3 + 0.5)


def wis90(y, q):
    median = q["0.50"]
    l, u, alpha = q["0.05"], q["0.95"], 0.10
    total = 0.5 * abs(y - median) + (alpha / 2.0) * interval_score(l, u, y, alpha)
    return total / (1 + 0.5)


def coverage90(y, q):
    return 1.0 if q["0.05"] <= y <= q["0.95"] else 0.0


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def compute_table(pred_dir, excluded_target_scales):
    """
    Aggregates predictions in pred_dir into the 2-table (geometric/uniform)
    format, skipping any prediction whose TARGET scale is in
    excluded_target_scales (used for scales with no lower-scale context to
    draw from, e.g. the smallest scale in whatever ladder is being tested).

    Ground truth / normalization always come from the FULL results_metrics.json
    (not whatever scale subset a given ladder run restricted its context to),
    since that's what FT-PFN's normalize_log_loss_curve used internally when
    the predictions were generated.
    """
    with open(RESULTS_JSON) as f:
        all_results = json.load(f)

    log_min, log_max = compute_min_max(all_results)

    gt_raw_final = {}
    gt_norm_final = {}
    for run_key, run_data in all_results.items():
        curve = run_data.get("val_loss_curve", [])[:TARGET_EPOCHS]
        if not curve:
            continue
        gt_raw_final[run_key] = curve[-1]
        gt_norm_final[run_key] = float(
            normalize_log_loss(curve, log_min, log_max)[-1]
        )

    rows = defaultdict(lambda: defaultdict(list))

    def _accumulate(policy_name, preds):
        for pred_key, pred_data in preds.items():
            run_key = pred_key.rsplit("_", 1)[0]
            if run_key not in gt_raw_final:
                continue
            if all_results[run_key].get("hidden_dim") in excluded_target_scales:
                continue

            quantiles = pred_data.get("quantiles", {})
            if not quantiles or len(quantiles.get("0.50", [])) == 0:
                continue

            q_final = {level: vals[-1] for level, vals in quantiles.items()}

            y_norm = gt_norm_final[run_key]
            y_raw = gt_raw_final[run_key]
            pred_median_raw = unnormalize_log_loss(q_final["0.50"], log_min, log_max)

            rel_err = abs(pred_median_raw - y_raw) / (abs(y_raw) + 1e-12) * 100.0
            nll = gaussian_nll(y_norm, q_final["0.50"], q_final["0.05"], q_final["0.95"])
            wis = wis_k3(y_norm, q_final)
            w90 = wis90(y_norm, q_final)
            cov = coverage90(y_norm, q_final)

            rows[policy_name]["rel_err"].append(rel_err)
            rows[policy_name]["nll"].append(nll)
            rows[policy_name]["wis"].append(wis)
            rows[policy_name]["wis90"].append(w90)
            rows[policy_name]["cov90"].append(cov)

    for subsample_mode in SUBSAMPLE_MODES:
        for k in K_VALUES:
            policy_name = f"diverse_lower_k_{subsample_mode}_k{k}"
            pred_path = os.path.join(pred_dir, f"{policy_name}.json")
            if not os.path.exists(pred_path):
                continue
            with open(pred_path) as f:
                _accumulate(policy_name, json.load(f))

    for policy_prefix in NON_DIVERSE_POLICIES:
        for subsample_mode in SUBSAMPLE_MODES:
            policy_name = f"{policy_prefix}_{subsample_mode}"
            pred_path = os.path.join(pred_dir, f"{policy_name}.json")
            if not os.path.exists(pred_path):
                continue
            with open(pred_path) as f:
                _accumulate(policy_name, json.load(f))

    def _print_row(label, k, d):
        if not d or not d["rel_err"]:
            print(f"{label:<32}{k:>4}{'(no data)':>12}")
            return
        print(
            f"{label:<32}{k:>4}"
            f"{np.mean(d['rel_err']):>12.2f}"
            f"{np.mean(d['nll']):>8.2f}"
            f"{np.mean(d['wis']):>8.3f}"
            f"{np.mean(d['wis90']):>8.3f}"
            f"{np.mean(d['cov90']):>8.3f}"
        )

    header = f"{'policy_variant':<32}{'k':>4}{'rel_err %':>12}{'NLL':>8}{'WIS':>8}{'WIS90':>8}{'cov90':>8}"

    for subsample_mode in SUBSAMPLE_MODES:
        print(f"\n{subsample_mode.capitalize()}\n")
        print(header)
        for k in K_VALUES:
            policy_name = f"diverse_lower_k_{subsample_mode}_k{k}"
            _print_row(policy_name, k, rows.get(policy_name))
        for policy_prefix in NON_DIVERSE_POLICIES:
            policy_name = f"{policy_prefix}_{subsample_mode}"
            _print_row(policy_name, "-", rows.get(policy_name))


def main():
    # hidden_dim=4 is the smallest scale in the full 7-scale ladder -- it has
    # no lower scale to draw context from, so every non-baseline policy
    # silently falls back to baseline-only behavior there. Excluded so it
    # doesn't dilute the other policies' numbers.
    compute_table(PRED_DIR, excluded_target_scales={4})


if __name__ == "__main__":
    main()
