"""
Box plots of NLL and predictive uncertainty vs. OBSERVED PERCENTAGE
(0, 20, 50, 90), for a FIXED context ablation, side by side.

This is a variant of the pooled-by-ablation script: instead of pooling
across all ep* checkpoints per ablation, it pools across hidden-dim
targets but keeps the observed-percentage checkpoints (ep0/20/50/90)
separate, so you can see how NLL / uncertainty evolve as more of the
curve is observed, for a given context ablation.

One figure is produced per ablation (ABLATIONS below), each with two
subplots (NLL, Uncertainty), and four boxes per subplot: 0%, 20%, 50%, 90%.

Same data model / NLL assumption as the original script:
  - results_metrics.json[gt_key]["val_loss_curve"] = raw loss curve (len 100).
  - ifbo_per_10/{hd_dir}/{context_dir}/ep{obs_pct}.json maps
    run_key -> {"point": [...], "quantiles": {"0.05":[...], "0.5":[...], "0.95":[...]}}
    in NORMALIZED score space; run_key = "<gt_key>_<epoch_tag>".
  - mean = denormalize(q0.5), std = (denormalize(q0.95) - denormalize(q0.05)) / (2*1.645)
  - NLL_t = 0.5*log(2*pi*std_t^2) + (gt_t - mean_t)^2 / (2*std_t^2), averaged over t >= n_obs.
  - UNCERTAINTY = mean width of the denormalized 90% interval over the predicted horizon.
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from utils import compute_min_max

# ── CONFIG ───────────────────────────────────────────────────────

IFBO_ROOT = "ifbo_per_10"
GT_PATH = "results_***/results_metrics.json"
OUTPUT_DIR = "ifbo_per_10"

NORM_HD_MIN = 4
NORM_HD_MAX = 128
NORM_EPS = 1e-8

ABLATIONS = ["32", "32_24", "64_32_24"]
OBS_PCTS = [0, 20, 50, 90]  # edit if your ep*.json files use different values

Z_90 = 1.6448536269514722  # z-score for the 90% central interval of N(0,1)

BOX_COLOR = "#e4572e"
MEDIAN_COLOR = "#1a1a1a"

EP_RE = re.compile(r"ep(\d+)\.json$")

# ────────────────────────────────────────────────────────────────


def make_denormalizer(log_min, log_max, eps=1e-8):
    def denormalize(score):
        score = np.asarray(score, dtype=np.float64)
        score = np.clip(score, 0.0, 1.0)
        norm = 1.0 - score
        log_loss = log_min + norm * (log_max - log_min)
        return np.exp(log_loss) - eps

    return denormalize


def boxplot_compat(ax, data, labels, **kwargs):
    """matplotlib >=3.9 renamed boxplot's ⁠ labels ⁠ kwarg to ⁠ tick_labels ⁠."""
    try:
        return ax.boxplot(data, tick_labels=labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kwargs)


def gaussian_nll(gt_vals, mean_vals, std_vals):
    """Per-point Gaussian negative log likelihood."""
    gt_vals = np.asarray(gt_vals, dtype=float)
    mean_vals = np.asarray(mean_vals, dtype=float)
    std_vals = np.asarray(std_vals, dtype=float)

    nll = np.full_like(gt_vals, np.nan, dtype=float)

    valid = std_vals > 0

    nll[valid] = (
        0.5 * np.log(2 * np.pi * std_vals[valid] ** 2)
        + (gt_vals[valid] - mean_vals[valid]) ** 2
        / (2 * std_vals[valid] ** 2)
    )

    return nll

def collect_for_ablation(ablation, gt_metrics, denormalize):
    """Pool NLL / uncertainty across all hidden-dim dirs, grouped by obs pct."""
    nll_by_pct = {p: [] for p in OBS_PCTS}
    unc_by_pct = {p: [] for p in OBS_PCTS}

    degenerate_std_count = 0
    skipped_no_gt = 0
    skipped_len_mismatch = 0
    skipped_unknown_pct = 0

    pred_files = glob.glob(
    os.path.join(IFBO_ROOT, "*", ablation, "ep*.json")
)
    print(f"[INFO] ablation={ablation}: found {len(pred_files)} ep*.json files")

    for pred_path in pred_files:
        m = EP_RE.search(os.path.basename(pred_path))
        if not m:
            continue
        obs_pct = int(m.group(1))
        if obs_pct not in nll_by_pct:
            skipped_unknown_pct += 1
            continue

        with open(pred_path, "r") as f:
            predictions = json.load(f)

        for run_key, pred in predictions.items():
            gt_key = run_key.rsplit("_", 1)[0]

            if gt_key not in gt_metrics:
                skipped_no_gt += 1
                continue

            gt = np.asarray(gt_metrics[gt_key]["val_loss_curve"], dtype=float)[:100]

            median = denormalize(pred["quantiles"]["0.5"])
            q05 = denormalize(pred["quantiles"]["0.05"])
            q95 = denormalize(pred["quantiles"]["0.95"])

            lo = np.minimum(q05, q95)
            hi = np.maximum(q05, q95)

            n_pred = len(median)
            n_obs = max(0, len(gt) - n_pred)
            pred_t_gt = gt[n_obs:]

            if len(pred_t_gt) != len(median):
                skipped_len_mismatch += 1
                continue

            # uncertainty: mean 90% interval width over predicted horizon
            width = np.abs(hi - lo)
            unc_by_pct[obs_pct].append(float(np.nanmean(width)))

            # NLL: Gaussian approx from quantiles, vs actual ground truth
            std = width / (2 * Z_90)
            nll_vals = gaussian_nll(pred_t_gt, median, std)

            degenerate_std_count += int(np.sum(np.isnan(nll_vals)))

            if np.all(np.isnan(nll_vals)):
                continue

            nll_by_pct[obs_pct].append(float(np.nanmean(nll_vals)))

    if skipped_no_gt:
        print(f"[WARN] ablation={ablation}: {skipped_no_gt} runs skipped (gt_key not found)")
    if skipped_len_mismatch:
        print(f"[WARN] ablation={ablation}: {skipped_len_mismatch} runs skipped (length mismatch)")
    if skipped_unknown_pct:
        print(f"[WARN] ablation={ablation}: {skipped_unknown_pct} files skipped (obs pct not in OBS_PCTS={OBS_PCTS})")
    if degenerate_std_count:
        print(f"[WARN] ablation={ablation}: {degenerate_std_count} timesteps had zero-width interval, excluded from NLL")

    return nll_by_pct, unc_by_pct


def plot_ablation(ablation, nll_by_pct, unc_by_pct):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    labels = [f"{p}%" for p in OBS_PCTS]

    for ax, data_by_pct, title, ylabel in [
        (axes[0], nll_by_pct, f"NLL vs. observed % — ablation {ablation}", "Mean NLL (Gaussian approx)"),
        (axes[1], unc_by_pct, f"Uncertainty vs. observed % — ablation {ablation}", "Mean 90% interval width"),
    ]:
        data = [data_by_pct[p] for p in OBS_PCTS]

        boxplot_compat(
            ax,
            data,
            labels,
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(marker="D", markerfacecolor="white",
                            markeredgecolor=MEDIAN_COLOR, markersize=6),
            medianprops=dict(color=MEDIAN_COLOR, linewidth=2),
            boxprops=dict(facecolor=BOX_COLOR, alpha=0.35, edgecolor=MEDIAN_COLOR),
            whiskerprops=dict(color=MEDIAN_COLOR),
            capprops=dict(color=MEDIAN_COLOR),
            flierprops=dict(marker="o", markersize=4, markerfacecolor=BOX_COLOR,
                             markeredgecolor="none", alpha=0.5),
        )

        for i, p in enumerate(OBS_PCTS):
            n = len(data_by_pct[p])
            ax.text(i + 1, ax.get_ylim()[0], f"n={n}",
                    ha="center", va="bottom", fontsize=8, color="#666666")

        ax.set_title(title)
        ax.set_xlabel("Observed %")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"nll_uncertainty_by_pct_{ablation}.svg")
    plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"saved -> {save_path}")


def main():
    with open(GT_PATH, "r") as f:
        gt_metrics = json.load(f)

    log_min, log_max = compute_min_max(NORM_HD_MIN, NORM_HD_MAX, path=GT_PATH)
    denormalize = make_denormalizer(log_min, log_max)

    for ablation in ABLATIONS:
        nll_by_pct, unc_by_pct = collect_for_ablation(ablation, gt_metrics, denormalize)
        plot_ablation(ablation, nll_by_pct, unc_by_pct)


if __name__ == "__main__":
    main()