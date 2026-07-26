"""
Same two tables as compute_diverse_table.py (Geometric / Uniform), but
broken out per observation epoch instead of pooling all epochs together --
one row per (policy_variant, epoch).

Run from the repo root, after run_diverse_lower_k.py has produced
predictions:
    python compute_diverse_table_by_epoch.py
"""
import json
import os
from collections import defaultdict

import numpy as np

from compute_diverse_table import (
    RESULTS_JSON,
    PRED_DIR,
    TARGET_EPOCHS,
    K_VALUES,
    SUBSAMPLE_MODES,
    NON_DIVERSE_POLICIES,
    compute_min_max,
    normalize_log_loss,
    unnormalize_log_loss,
    gaussian_nll,
    wis_k3,
    wis90,
    coverage90,
)

# hidden_dim=4 has no lower scale to draw context from (see compute_diverse_table.py)
EXCLUDED_TARGET_SCALES = {4}


def main():
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
        gt_norm_final[run_key] = float(normalize_log_loss(curve, log_min, log_max)[-1])

    # rows[(policy_name, epoch)][metric] = [values...]
    rows = defaultdict(lambda: defaultdict(list))
    epochs_seen = set()

    def _accumulate(policy_name, preds):
        for pred_key, pred_data in preds.items():
            run_key, epoch_str = pred_key.rsplit("_", 1)
            if run_key not in gt_raw_final:
                continue
            if all_results[run_key].get("hidden_dim") in EXCLUDED_TARGET_SCALES:
                continue

            quantiles = pred_data.get("quantiles", {})
            if not quantiles or len(quantiles.get("0.50", [])) == 0:
                continue

            epoch = int(epoch_str)
            epochs_seen.add(epoch)

            q_final = {level: vals[-1] for level, vals in quantiles.items()}

            y_norm = gt_norm_final[run_key]
            y_raw = gt_raw_final[run_key]
            pred_median_raw = unnormalize_log_loss(q_final["0.50"], log_min, log_max)

            rel_err = abs(pred_median_raw - y_raw) / (abs(y_raw) + 1e-12) * 100.0
            nll = gaussian_nll(y_norm, q_final["0.50"], q_final["0.05"], q_final["0.95"])
            wis = wis_k3(y_norm, q_final)
            w90 = wis90(y_norm, q_final)
            cov = coverage90(y_norm, q_final)

            key = (policy_name, epoch)
            rows[key]["rel_err"].append(rel_err)
            rows[key]["nll"].append(nll)
            rows[key]["wis"].append(wis)
            rows[key]["wis90"].append(w90)
            rows[key]["cov90"].append(cov)

    for subsample_mode in SUBSAMPLE_MODES:
        for k in K_VALUES:
            policy_name = f"diverse_lower_k_{subsample_mode}_k{k}"
            pred_path = os.path.join(PRED_DIR, f"{policy_name}.json")
            if os.path.exists(pred_path):
                with open(pred_path) as f:
                    _accumulate(policy_name, json.load(f))

        for policy_prefix in NON_DIVERSE_POLICIES:
            policy_name = f"{policy_prefix}_{subsample_mode}"
            pred_path = os.path.join(PRED_DIR, f"{policy_name}.json")
            if os.path.exists(pred_path):
                with open(pred_path) as f:
                    _accumulate(policy_name, json.load(f))

    epochs_sorted = sorted(epochs_seen)

    header = (
        f"{'policy_variant':<32}{'k':>4}{'ep':>5}"
        f"{'rel_err %':>12}{'NLL':>8}{'WIS':>8}{'WIS90':>8}{'cov90':>8}"
    )

    def _print_row(label, k, epoch, d):
        if not d or not d["rel_err"]:
            print(f"{label:<32}{k:>4}{epoch:>5}{'(no data)':>12}")
            return
        print(
            f"{label:<32}{k:>4}{epoch:>5}"
            f"{np.mean(d['rel_err']):>12.2f}"
            f"{np.mean(d['nll']):>8.2f}"
            f"{np.mean(d['wis']):>8.3f}"
            f"{np.mean(d['wis90']):>8.3f}"
            f"{np.mean(d['cov90']):>8.3f}"
        )

    for subsample_mode in SUBSAMPLE_MODES:
        print(f"\n{subsample_mode.capitalize()}\n")
        print(header)
        for k in K_VALUES:
            policy_name = f"diverse_lower_k_{subsample_mode}_k{k}"
            for epoch in epochs_sorted:
                _print_row(policy_name, k, epoch, rows.get((policy_name, epoch)))
        for policy_prefix in NON_DIVERSE_POLICIES:
            policy_name = f"{policy_prefix}_{subsample_mode}"
            for epoch in epochs_sorted:
                _print_row(policy_name, "-", epoch, rows.get((policy_name, epoch)))


if __name__ == "__main__":
    main()
