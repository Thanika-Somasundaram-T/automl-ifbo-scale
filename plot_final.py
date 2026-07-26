"""
Plot IFBO scaling-law predictions vs ground truth.

This version combines all 5 target (lr, wd) configurations into a SINGLE
plot per (hd, context, ep) combination, instead of one plot per run.

Visual improvements requested:
  - Predictions are now drawn as SOLID lines (not dashed).
  - Ground truth and predictions are clearly distinguished by:
      * GT   -> thin, semi-transparent line (no markers)
      * Pred -> thick, fully-opaque line with markers, plus a shaded
                uncertainty band (q05-q95) in the same color.
  - Each (lr, wd) target keeps a fixed, consistent color across GT/pred
    so it's easy to trace which curve pair belongs to which config.
"""

import glob
import json
import math
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from utils import compute_min_max


# ── CONFIG ───────────────────────────────────────────────────────
IFBO_ROOT = "ifbo_pred"
GT_PATH = "results/results_metrics.json"
OUTPUT_DIR = "ifbo_plots_combined"

NORM_HD_MIN = 4
NORM_HD_MAX = 128
NORM_BUFFER = 0.05
NORM_MIN_VALUE = 1e-8
NORM_EPS = 1e-8

RUN_KEY_RE = re.compile(
    r"layer(?P<layer>\d+)_lr(?P<lr>[0-9eE.+-]+)_hd(?P<hd>\d+)"
    r"_wd(?P<wd>[0-9.]+)_(?P<schedule>[a-zA-Z]+)_(?P<epoch_tag>\d+)"
)

TARGETS = [
    # {"lr": 0.00003, "wd": 0.0,  "color": "red",     "label": "lr=3e-5, wd=0.0"},
    # {"lr": 0.0001,  "wd": 0.0,  "color": "blue",    "label": "lr=1e-4, wd=0.0"},
    {"lr": 0.00003, "wd": 0.01, "color": "red",   "label": "lr=3e-5, wd=0.01"},
    {"lr": 0.00001, "wd": 0.0,  "color": "skyblue", "label": "lr=1e-5, wd=0.0"},
    # {"lr": 0.003,   "wd": 0.0,  "color": "brown",   "label": "lr=3e-3, wd=0.0"},
]


# ────────────────────────────────────────────────────────────────
# STABLE DENORMALIZER
# ────────────────────────────────────────────────────────────────
def make_denormalizer(log_min, log_max, eps=1e-8):

    def denormalize(score):
        score = np.asarray(score, dtype=np.float64)
        score = np.clip(score, 0.0, 1.0)

        norm = 1.0 - score
        log_loss = log_min + norm * (log_max - log_min)

        # correct inverse of log(val_loss + eps)
        return np.exp(log_loss) - eps

    return denormalize


# ────────────────────────────────────────────────────────────────
# RUN KEY PARSER
# ────────────────────────────────────────────────────────────────
def parse_run_key(run_key):
    m = RUN_KEY_RE.match(run_key)
    if not m:
        return {}

    d = m.groupdict()
    d["layer"] = int(d["layer"])
    d["hd"] = int(d["hd"])
    d["wd"] = float(d["wd"])
    d["lr"] = float(d["lr"])
    return d


def match_target(run_key, tol=1e-9):
    """Return the TARGETS entry whose (lr, wd) matches this run_key, or None."""
    parsed = parse_run_key(run_key)
    if not parsed:
        return None

    for target in TARGETS:
        if math.isclose(parsed["lr"], target["lr"], rel_tol=1e-6, abs_tol=tol) and \
           math.isclose(parsed["wd"], target["wd"], rel_tol=1e-6, abs_tol=tol):
            return target
    return None


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():

    with open(GT_PATH, "r") as f:
        gt_metrics = json.load(f)

    log_min, log_max = compute_min_max(NORM_HD_MIN, NORM_HD_MAX)
    denormalize = make_denormalizer(log_min, log_max)

    print(f"[INFO] log-range = [{log_min:.4f}, {log_max:.4f}]")

    pred_files = glob.glob(os.path.join(IFBO_ROOT, "*", "*", "ep*.json"))
    print(f"[INFO] found {len(pred_files)} prediction files")

    # group prediction files by (hd_dir, context_dir, ep_file) so that each
    # group becomes exactly one combined plot
    groups = defaultdict(list)
    for pred_path in pred_files:
        parts = pred_path.split(os.sep)
        hd_dir = parts[-3]
        context_dir = parts[-2]
        ep_file = parts[-1]
        groups[(hd_dir, context_dir, ep_file)].append(pred_path)

    for (hd_dir, context_dir, ep_file), paths in groups.items():

        obs_pct = int(ep_file.replace("ep", "").replace(".json", ""))
        out_dir = os.path.join(OUTPUT_DIR, hd_dir, context_dir)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nProcessing hd={hd_dir}, ctx={context_dir}, ep={obs_pct}")

        fig, ax = plt.subplots(figsize=(11, 6))
        t = np.linspace(0, 1, 100)
        plotted_any = False
        last_n_obs = 0

        for pred_path in paths:
            with open(pred_path, "r") as f:
                predictions = json.load(f)

            for run_key, pred in predictions.items():

                target = match_target(run_key)
                if target is None:
                    continue

                gt_key = run_key.rsplit("_", 1)[0]
                if gt_key not in gt_metrics:
                    continue

                gt = np.array(gt_metrics[gt_key]["val_loss_curve"], dtype=float)
                gt = gt[:100]

                median = denormalize(pred["quantiles"]["0.5"])
                q05 = denormalize(pred["quantiles"]["0.05"])
                q95 = denormalize(pred["quantiles"]["0.95"])

                lo = np.minimum(q05, q95)
                hi = np.maximum(q05, q95)

                n_pred = len(median)
                n_obs = max(0, len(gt) - n_pred)
                print(f"n_pred: {n_pred}, n_obs: {n_obs}")

                obs_t = t[:n_obs]
                fut_t = t[n_obs:]

                color = target["color"]
                label = target["label"]

                # ── Ground truth: thin, semi-transparent, no markers ──
                ax.plot(
                    t, gt,
                    color=color, linewidth=1.4, alpha=0.45,
                    linestyle="-", zorder=2,
                    label=f"{label} (GT)",
                )

                # ── Prediction (future part): thick, solid, markers ──
                pred_full = np.concatenate([gt[:n_obs], median])
                ax.plot(
                    fut_t, pred_full[n_obs:],
                    color=color, linewidth=2.4, alpha=0.95,
                    linestyle="-",
                    markevery=max(1, len(fut_t) // 15),
                    zorder=4,
                    label=f"{label} (Pred)",
                )

                # connect GT -> pred visually at the boundary
                # if n_obs > 0:
                #     ax.plot(
                #         [t[n_obs - 1], fut_t[0]] if len(fut_t) else [t[n_obs - 1]],
                #         [gt[n_obs - 1], pred_full[n_obs]] if len(fut_t) else [gt[n_obs - 1]],
                #         color=color, linewidth=2.4, alpha=0.95, zorder=4,
                #     )

                # ── uncertainty band ──
                # ax.fill_between(fut_t, lo, hi, color=color, alpha=0.12, zorder=1)

                plotted_any = True
                last_n_obs = n_obs

        if not plotted_any:
            plt.close(fig)
            continue

        ax.axvline(t[last_n_obs], linestyle=":", color="gray", alpha=0.6)

        ax.set_title(f"IFBO predictions vs GT — hd={hd_dir}, ctx={context_dir}, ep{obs_pct}")
        ax.set_xlabel("Normalized epoch")
        ax.set_ylim(bottom=0.2, top=0.9)
        ax.set_ylabel("Validation loss")
        ax.grid(alpha=0.25)

        # de-duplicate legend, keep GT/Pred grouped per target
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2, loc="best")

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"combined_ep{obs_pct}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()

        print(f"saved -> {save_path}")


if __name__ == "__main__":
    main()