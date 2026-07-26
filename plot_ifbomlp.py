"""
Plot IFBO scaling-law predictions vs ground truth.

Fixes:
- correct path parsing (hd / context / ep)
- correct ep extraction
- correct global denormalization
- prediction aligned with observed horizon
- uncertainty interval correctly positioned
- context curves shown for hidden-dim references
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

# MUST match normalization during IFBO training
NORM_HD_MIN = 4
NORM_HD_MAX = 128

NORM_EPS = 1e-8

RUN_KEY_RE = re.compile(
    r"layer(?P<layer>\d+)_lr(?P<lr>[0-9eE.+-]+)_hd(?P<hd>\d+)"
    r"_wd(?P<wd>[0-9.]+)_(?P<schedule>[a-zA-Z]+)_(?P<epoch_tag>\d+)"
)

HD_SUB_RE = re.compile(r"hd\d+")

# ── VISUAL STYLE ────────────────────────────────────────────────

GT_COLOR = "#1a1a1a"
PRED_COLOR = "#e4572e"
BAND_COLOR = "#e4572e"
SPLIT_LINE_COLOR = "#888888"

CONTEXT_CMAP = plt.get_cmap("Blues")

# ────────────────────────────────────────────────────────────────
# DENORMALIZER
# ────────────────────────────────────────────────────────────────

def make_denormalizer(log_min, log_max, eps=1e-8):
    """
    Inverse of:

        score = 1 - (log(loss)-log_min)/(log_max-log_min)
    """

    def denormalize(score):
        score = np.asarray(score, dtype=np.float64)
        score = np.clip(score, 0.0, 1.0)
        norm = 1.0 - score
        log_loss = log_min + norm * (log_max - log_min)
        return np.exp(log_loss) - eps

    return denormalize


# ────────────────────────────────────────────────────────────────
# RUN KEY HELPERS
# ────────────────────────────────────────────────────────────────

def parse_run_key(run_key):
    m = RUN_KEY_RE.match(run_key)

    if not m:
        return {}

    d = m.groupdict()
    d["layer"] = int(d["layer"])
    d["hd"] = int(d["hd"])
    d["lr"] = float(d["lr"])
    d["wd"] = float(d["wd"])

    return d


def context_key_for_hd(gt_key, ctx_hd):
    return HD_SUB_RE.sub(f"hd{ctx_hd}", gt_key, count=1)


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    with open(GT_PATH, "r") as f:
        gt_metrics = json.load(f)

    # IMPORTANT:
    # same normalization range used in normalize_log_loss_curve()
    log_min, log_max = compute_min_max(
        NORM_HD_MIN,
        NORM_HD_MAX,
        path=GT_PATH,
    )

    denormalize = make_denormalizer(log_min, log_max)

    print(f"[INFO] log range: {log_min:.5f} -> {log_max:.5f}")

    pred_files = glob.glob(
        os.path.join(
            IFBO_ROOT,
            "*",
            "*",
            "ep*.json",
        )
    )

    print(f"[INFO] found {len(pred_files)} prediction files")

    for pred_path in pred_files:
        parts = pred_path.split(os.sep)

        hd_dir = parts[-3]
        context_dir = parts[-2]
        ep_file = parts[-1]

        obs_pct = int(ep_file.replace("ep", "").replace(".json", ""))

        context_hds = [c for c in context_dir.split("_") if c]

        print(
            f"\nProcessing {pred_path}"
            f" hd={hd_dir}"
            f" ctx={context_dir}"
            f" ep={obs_pct}"
        )

        with open(pred_path, "r") as f:
            predictions = json.load(f)

        out_dir = os.path.join(
            OUTPUT_DIR,
            hd_dir,
            context_dir,
            f"ep{obs_pct}",
        )

        os.makedirs(out_dir, exist_ok=True)

        for run_key, pred in predictions.items():
            gt_key = run_key.rsplit("_", 1)[0]

            if gt_key not in gt_metrics:
                continue

            # ground truth
            gt = np.asarray(
                gt_metrics[gt_key]["val_loss_curve"],
                dtype=float,
            )[:100]

            # IFBO prediction
            median = denormalize(pred["quantiles"]["0.5"])
            q05 = denormalize(pred["quantiles"]["0.05"])
            q95 = denormalize(pred["quantiles"]["0.95"])

            lo = np.minimum(q05, q95)
            hi = np.maximum(q05, q95)

            n_pred = len(median)

            # FIX: clip to 0 so a malformed prediction file (n_pred > len(gt))
            # can't send n_obs negative and silently produce garbage slices
            n_obs = max(0, len(gt) - n_pred)

            t = np.linspace(0, 1, len(gt))
            pred_t = t[n_obs:]

            # sanity check
            if len(pred_t) != len(median):
                print("WARNING length mismatch", len(pred_t), len(median))
                continue

            # ───────────────────────────────────────────
            # PLOT
            # ───────────────────────────────────────────

            fig, ax = plt.subplots(figsize=(10, 5.5))

            # Context curves
            n_ctx = len(context_hds)

            for i, ctx_hd in enumerate(context_hds):
                ctx_key = context_key_for_hd(gt_key, ctx_hd)

                if ctx_key == gt_key or ctx_key not in gt_metrics:
                    continue

                ctx_curve = np.asarray(
                    gt_metrics[ctx_key]["val_loss_curve"],
                    dtype=float,
                )[:100]

                ctx_t = np.linspace(0, 1, len(ctx_curve))
                shade = 0.35 + 0.4 * ((i + 1) / max(n_ctx, 1))

                ax.plot(
                    ctx_t,
                    ctx_curve,
                    color=CONTEXT_CMAP(shade),
                    linewidth=1.6,
                    alpha=0.55,
                    zorder=1,
                    label=f"context hd{ctx_hd}",
                )

            # GT observed (solid)
            if n_obs > 0:
                ax.plot(
                    t[:n_obs],
                    gt[:n_obs],
                    color=GT_COLOR,
                    linewidth=2.5,
                    label=f"Ground truth (hd{hd_dir})",
                    zorder=3,
                )

            # GT future (dashed)
            if n_obs < len(t):
                ax.plot(
                    t[max(n_obs - 1, 0):],
                    gt[max(n_obs - 1, 0):],
                    color=GT_COLOR,
                    linewidth=1.6,
                    linestyle=(0, (5, 3)),
                    zorder=3,
                    label=None if n_obs > 0 else f"Ground truth (hd{hd_dir})",
                )

            # uncertainty
            ax.fill_between(
                pred_t,
                lo,
                hi,
                color=BAND_COLOR,
                alpha=0.18,
                label="90% interval",
                zorder=2,
            )

            # prediction
            ax.plot(
                pred_t,
                median,
                color=PRED_COLOR,
                linewidth=2.5,
                solid_capstyle="round",
                label=f"IFBO prediction ep{obs_pct}",
                zorder=4,
            )

            # split marker
            if n_obs < len(t):
                ax.axvline(
                    t[n_obs],
                    linestyle=":",
                    color=SPLIT_LINE_COLOR,
                    linewidth=1.2,
                )

                ax.scatter(
                    [t[n_obs]],
                    [gt[n_obs]],
                    color=GT_COLOR,
                    s=35,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )

            ax.set_xlabel("Normalized epoch")
            ax.set_ylabel("Validation loss")

            ax.set_ylim(
                bottom=0.2,
                top=1.0,
            )

            # ax.legend(
            #     frameon=False,
            #     fontsize=9,
            #     loc="best"
            # )

            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            save_path = os.path.join(
                out_dir,
                f"{run_key}.svg",
            )

            plt.tight_layout()
            plt.savefig(
                save_path,
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.close()

            print(f"saved -> {save_path}")


if __name__ == "__main__":
    main()