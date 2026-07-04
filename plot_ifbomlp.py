"""
Plot IFBO scaling-law predictions vs ground truth.

Fixed:
- correct path parsing (hd / context / ep)
- correct ep extraction
- stable denormalization
- clean directory traversal

Visual updates:
- Ground truth and prediction are both solid lines, but clearly distinct
  (color + line width), instead of a dashed hybrid line.
- The prediction line is drawn only over the actually-predicted horizon,
  connected smoothly to the point where observation ends.
- Context curves: `context_dir` (e.g. "64", "64_32", "64_32_24") lists the
  hidden-dim values used as context to predict the target hd in `hd_dir`
  (e.g. "128"). For each of those context hd values, we look up the SAME
  run_key's ground-truth curve at that hd and draw it as a light,
  low-opacity reference line behind the main curves.
"""

import glob
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colormaps

from utils import compute_min_max


# ── CONFIG ───────────────────────────────────────────────────────
IFBO_ROOT = "ifbo_pred_fixed"
GT_PATH = "results/results_metrics.json"
OUTPUT_DIR = "ifbo_plots_fixed"

NORM_HD_MIN = 32
NORM_HD_MAX = 128
NORM_BUFFER = 0.05
NORM_MIN_VALUE = 1e-8
NORM_EPS = 1e-8

RUN_KEY_RE = re.compile(
    r"layer(?P<layer>\d+)_lr(?P<lr>[0-9eE.+-]+)_hd(?P<hd>\d+)"
    r"_wd(?P<wd>[0-9.]+)_(?P<schedule>[a-zA-Z]+)_(?P<epoch_tag>\d+)"
)

# used to swap out the hd of a run_key/gt_key to look up a context curve
HD_SUB_RE = re.compile(r"hd\d+")

# ── VISUAL STYLE ─────────────────────────────────────────────────
GT_COLOR = "#1a1a1a"          # near-black, solid, thick -> ground truth
PRED_COLOR = "#e4572e"        # warm orange-red, solid -> prediction
BAND_COLOR = "#e4572e"        # matches prediction, low alpha fill
SPLIT_LINE_COLOR = "#888888"
CONTEXT_CMAP = plt.get_cmap("Blues")  # context curves shade from light->mid blue


# ────────────────────────────────────────────────────────────────
# STABLE DENORMALIZER
# ────────────────────────────────────────────────────────────────
def make_denormalizer(log_min, log_max, eps=1e-8):

    def denormalize(score):
        score = np.asarray(score, dtype=np.float64)
        score = np.clip(score, 0.0, 1.0)

        norm = 1.0 - score
        log_loss = log_min + norm * (log_max - log_min)

        # IMPORTANT: correct inverse of log(val_loss + eps)
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


def context_key_for_hd(gt_key, ctx_hd):
    """Swap the hd### token in gt_key for the given context hidden-dim."""
    return HD_SUB_RE.sub(f"hd{ctx_hd}", gt_key, count=1)


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────
def main():

    with open(GT_PATH, "r") as f:
        gt_metrics = json.load(f)

    # log_min, log_max = compute_min_max(NORM_HD_MIN, NORM_HD_MAX)
    # denormalize = make_denormalizer(log_min, log_max)

    # print(f"[INFO] log-range = [{log_min:.4f}, {log_max:.4f}]")

    # correct recursive structure: ifbo_pred/<hd_dir>/<context_dir>/ep*.json
    pred_files = glob.glob(os.path.join(IFBO_ROOT, "*", "*", "ep*.json"))

    print(f"[INFO] found {len(pred_files)} prediction files")

    for pred_path in pred_files:

        # ── PATH PARSING ─────────────────────────────
        parts = pred_path.split(os.sep)

        hd_dir = parts[-3]        # target hd, e.g. "128"
        context_dir = parts[-2]   # context hd(s), e.g. "64" or "64_32_24"
        ep_file = parts[-1]       # ep20.json

        obs_pct = int(ep_file.replace("ep", "").replace(".json", ""))
        context_hds = [c for c in context_dir.split("_") if c]

        print(f"\nProcessing {pred_path} (hd={hd_dir}, ctx={context_dir}, ep={obs_pct})")

        with open(pred_path, "r") as f:
            predictions = json.load(f)

        out_dir = os.path.join(OUTPUT_DIR, hd_dir, context_dir, f"ep{obs_pct}")
        os.makedirs(out_dir, exist_ok=True)

        for run_key, pred in predictions.items():

            gt_key = run_key.rsplit("_", 1)[0]

            if gt_key not in gt_metrics:
                continue

            # ── ground truth (target hd) ────────────────────
            gt = np.array(gt_metrics[gt_key]["val_loss_curve"], dtype=float)
            denormalize = make_denormalizer(np.min(np.log(gt)), np.max(np.log(gt)))
            gt = gt[:100]

            # ── prediction (denormalize) ────────────────────
            
            
            median = denormalize(pred["quantiles"]["0.5"])
            q05 = denormalize(pred["quantiles"]["0.05"])
            q95 = denormalize(pred["quantiles"]["0.95"])

            lo = np.minimum(q05, q95)
            hi = np.maximum(q05, q95)

            n_pred = len(median)
            n_obs = max(0, len(gt) - n_pred)

            t = np.linspace(0, 1, 100)
            fut_t = t[n_obs:]

            # connect the prediction smoothly to the last observed GT point
            if n_obs > 0:
                fut_t_plot = np.concatenate(([t[n_obs - 1]], fut_t))
                median_plot = np.concatenate(([gt[n_obs - 1]], median))
                lo_plot = np.concatenate(([gt[n_obs - 1]], lo))
                hi_plot = np.concatenate(([gt[n_obs - 1]], hi))
            else:
                fut_t_plot, median_plot, lo_plot, hi_plot = fut_t, median, lo, hi

            # ── plot ────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(10, 5.5))

            # context curves: same run_key, other hidden dims used as context
            n_ctx = len(context_hds)
            for i, ctx_hd in enumerate(context_hds):
                ctx_key = context_key_for_hd(gt_key, ctx_hd)
                if ctx_key == gt_key or ctx_key not in gt_metrics:
                    continue
                ctx_curve = np.array(gt_metrics[ctx_key]["val_loss_curve"], dtype=float)[:100]
                ctx_t = np.linspace(0, 1, len(ctx_curve))
                # shade progressively (later context entries drawn a bit darker)
                shade = 0.35 + 0.4 * ((i + 1) / max(n_ctx, 1))
                ax.plot(
                    ctx_t, ctx_curve,
                    color=CONTEXT_CMAP(shade),
                    linewidth=1.6,
                    alpha=0.55,
                    zorder=1,
                    label=f"context hd{ctx_hd}",
                )

            # ground truth (solid, thick, black)
            ax.plot(t, gt, color=GT_COLOR, linewidth=2.5, solid_capstyle="round",
                     label=f"Ground truth (hd{hd_dir})", zorder=3)

            # prediction interval + median (solid, distinct color)
            ax.fill_between(fut_t_plot, lo_plot, hi_plot, color=BAND_COLOR,
                             alpha=0.18, zorder=2, label="90% interval")
            ax.plot(fut_t_plot, median_plot, color=PRED_COLOR, linewidth=2.5,
                     solid_capstyle="round", label=f"IFBO prediction (ep{obs_pct})",
                     zorder=4)

            # marker + guide line at the observation cutoff
            ax.axvline(t[n_obs] if n_obs < len(t) else t[-1],
                       linestyle=":", color=SPLIT_LINE_COLOR, linewidth=1.2, zorder=0)
            if 0 < n_obs <= len(gt):
                ax.scatter([t[n_obs - 1]], [gt[n_obs - 1]], color=GT_COLOR,
                           s=35, zorder=5, edgecolor="white", linewidth=0.8)

            ax.set_title(run_key, fontsize=12, fontweight="bold")
            ax.set_xlabel("Normalized epoch")
            ax.set_ylabel("Validation loss")
            ax.set_ylim(bottom=0.2, top=1.0)

            ax.legend(frameon=False, fontsize=9, loc="best")
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            save_path = os.path.join(out_dir, f"{run_key}.png")
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

            print(f"saved -> {save_path}")


if __name__ == "__main__":
    main()