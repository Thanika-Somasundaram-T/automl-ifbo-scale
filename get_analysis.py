import json
import os
import glob
import numpy as np
import re

from utils import compute_loo, load_data

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
EXPERIMENTS_PATH = "all_curves.json"
BASE_DIR = "ablation2_scales/t134"

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with open(EXPERIMENTS_PATH, "r") as f:
    experiments = json.load(f)

df = load_data(EXPERIMENTS_PATH)

# ─────────────────────────────────────────────
# DENORMALIZATION FUNCTION
# ─────────────────────────────────────────────
def denormalize(y_norm: np.ndarray) -> np.ndarray:
    log_min, log_max = compute_loo(df=df, target_N=None, buffer=0.05)
    y_norm = np.clip(y_norm, 0.0, 1.0)
    log_val = y_norm * (log_max - log_min) + log_min
    return np.exp(log_val) - 1e-8

# ─────────────────────────────────────────────
# FILE SEARCH
# ─────────────────────────────────────────────
prediction_files = glob.glob(
    os.path.join(BASE_DIR, "**", "target*_obs*.json"),
    recursive=True
)

print(f"Found {len(prediction_files)} files")

# ─────────────────────────────────────────────
# OUTPUT STRUCTURE
# ─────────────────────────────────────────────
all_results = {}

# ─────────────────────────────────────────────
# PARSE RUN KEY (IMPORTANT FIX)
# ─────────────────────────────────────────────
def parse_run_key(key: str):
    patterns = {
        "base_N": r"base(\d+)",
        "target_N": r"target(\d+)",
        "lr": r"lr([0-9.]+)",
        "shrink": r"shrink([0-9.]+)",
        "tkpm": r"tkpm([0-9.]+)",
        "emb": r"emb(\d+)",
        "head": r"head(\d+)",
        "gw": r"gw([0-9.]+)",
        "gn": r"gn([0-9.]+)",
    }

    out = {}
    for k, pat in patterns.items():
        m = re.search(pat, key)
        out[k] = float(m.group(1)) if m else None

    # cast correct types
    if out["base_N"] is not None:
        out["base_N"] = int(out["base_N"])
    if out["target_N"] is not None:
        out["target_N"] = int(out["target_N"])
    if out["emb"] is not None:
        out["emb"] = int(out["emb"])
    if out["head"] is not None:
        out["head"] = int(out["head"])

    return out

# ─────────────────────────────────────────────
# SAFE KEY BUILDER
# ─────────────────────────────────────────────
def build_exp_key(exp):
    return (
        f"base{exp.get('base_N', 'NA')}_"
        f"target{exp.get('target_N', 'NA')}_"
        f"lr{exp.get('lr', 'NA')}_"
        f"shrink{exp.get('shrink', 'NA')}_"
        f"tkpm{exp.get('tkpm', 'NA')}_"
        f"emb{exp.get('emb', 'NA')}_"
        f"head{exp.get('head', 'NA')}_"
        f"gw{exp.get('gw', 'NA')}_"
        f"gn{exp.get('gn', 'NA')}"
    )

# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────
for pred_path in prediction_files:

    context = int(os.path.basename(os.path.dirname(pred_path)))

    with open(pred_path, "r") as f:
        predictions = json.load(f)

    print(f"Processing: {pred_path}")

    for run_key, pred in predictions.items():

        # ── FIX: parse from run_key instead of hyperparameters
        exp = parse_run_key(run_key)

        exp_key = build_exp_key(exp)

        if exp_key not in all_results:
            all_results[exp_key] = {}

        obs_pct = int(pred["observe_fraction"] * 100)

        # ── ground truth
        gt = experiments[run_key]["curve"]
        gt_loss = np.array(gt["val_loss"])

        # ── prediction
        median = denormalize(np.array(pred["median"]))
        q05 = denormalize(np.array(pred["q05"]))
        q95 = denormalize(np.array(pred["q95"]))

        true_final = float(gt_loss[-1])
        pred_final = float(median[-1])

        abs_err = abs(pred_final - true_final)
        rel_err = abs_err / (true_final + 1e-8) * 100

        ci_width = float(q95[-1] - q05[-1])
        true_in_ci = bool(q05[-1] <= true_final <= q95[-1])

        # ── keys
        ctx_key = f"context_{context}"
        obs_key = str(obs_pct)

        if ctx_key not in all_results[exp_key]:
            all_results[exp_key][ctx_key] = {}

        all_results[exp_key][ctx_key][obs_key] = {
            "context": context,
            "obs_pct": obs_pct,

            "true_final": true_final,
            "pred_final": pred_final,

            "abs_err": abs_err,
            "rel_err": rel_err,

            "ci_width": ci_width,
            "true_in_ci": true_in_ci
        }

# ─────────────────────────────────────────────
# SORT (IMPORTANT FOR CLEAN ANALYSIS)
# ─────────────────────────────────────────────
all_results = dict(sorted(all_results.items()))
for k in all_results:
    all_results[k] = dict(sorted(all_results[k].items()))

# ─────────────────────────────────────────────
# SAVE OUTPUT
# ─────────────────────────────────────────────
out_file = os.path.join(BASE_DIR, "analysis_nested.json")

with open(out_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nSaved → {out_file}")