"""
Boxplot comparison of NLL and MSE across context strategies.

Evaluation:
- IFBO predictions are normalized scores (higher = better)
- Predictions are denormalized back to validation loss
- Metrics are computed on the predicted future trajectory
- Ground truth is aligned using suffix matching

Example:
true curve:
    [loss_0, loss_1, ..., loss_100]

prediction after observation:
    [pred_50, ..., pred_100]

comparison:
    true_curve[-len(pred):] vs pred
"""

import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = "results_***"
PRED_DIR = "./ifbo_pred"
OUTPUT_DIR = "box_plots_tn"

EPOCHS_LIST = [20, 50, 90]

TARGET_SCALES = ["128"]

ABLATIONS_LIST = [
    "24",
    "32",
    "64",
    "32_24",
    "64_32",
    "64_32_24",
]

BASELINE_SUBDIR = "baseline"

EPOCH_COLORS = {
    0: "#1f77b4",
    20: "#ff7f0e",
    50: "#2ca02c",
    90: "#d62728",
}

# NLL sanity bound: values with |nll| beyond this are dropped as
# numerically degenerate (e.g. near-zero sigma blow-ups), not because
# they are IQR outliers -- IQR outliers are handled separately per-plot.
NLL_SANITY_BOUND = 50


# ---------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Normalization range
# ---------------------------------------------------------------------------

def compute_min_max(hidden_dim_min, hidden_dim_max, path, buffer=0.05, min_value=1e-8):
    with open(path) as f:
        results = json.load(f)

    selected_losses = []

    for key, entry in results.items():
        match = re.search(r"_hd(\d+)_", key)
        if match is None:
            continue

        hd = int(match.group(1))
        if hidden_dim_min <= hd <= hidden_dim_max:
            curve = entry.get("val_loss_curve", [])
            if curve:
                selected_losses.append(np.asarray(curve))

    if not selected_losses:
        raise ValueError("No matching curves found")

    all_losses = np.concatenate(selected_losses)
    log_losses = np.log(np.clip(all_losses, min_value, None))

    log_min = float(log_losses.min())
    log_max = float(log_losses.max())

    margin = (log_max - log_min) * buffer
    log_min = max(log_min - margin, np.log(min_value))
    log_max = log_max + margin

    if log_max <= log_min:
        log_max = log_min + 1e-6

    return log_min, log_max


# ---------------------------------------------------------------------------
# Denormalizer
# ---------------------------------------------------------------------------

def make_denormalizer(log_min, log_max, eps=1e-8):
    """
    Inverse of:

        score = 1 - (log(loss) - log_min) / (log_max - log_min)
    """

    def denormalize(score):
        score = np.asarray(score, dtype=np.float64)
        score = np.clip(score, 0.0, 1.0)

        norm = 1.0 - score
        log_loss = log_min + norm * (log_max - log_min)

        return np.exp(log_loss) - eps

    return denormalize


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_mse(y_true_curve, y_pred_dist, denormalize):
    y_true_curve = np.asarray(y_true_curve, dtype=np.float64)

    # IFBO normalized prediction -> loss
    pred_score = np.asarray(y_pred_dist["point"], dtype=np.float64)
    pred_loss = denormalize(pred_score)

    if len(pred_loss) == 0:
        return np.nan

    # Prediction is future continuation: align on the suffix
    y_true = y_true_curve[len(y_true_curve) - len(pred_loss):]

    return np.mean((y_true - pred_loss) ** 2)


def calculate_nll(y_true_curve, y_pred_dist, denormalize):
    y_true_curve = np.asarray(y_true_curve, dtype=np.float64)

    # Normalized IFBO outputs -> loss space
    mu = denormalize(np.asarray(y_pred_dist["point"], dtype=np.float64))
    q05 = denormalize(np.asarray(y_pred_dist["quantiles"]["0.05"], dtype=np.float64))
    q95 = denormalize(np.asarray(y_pred_dist["quantiles"]["0.95"], dtype=np.float64))

    if len(mu) == 0:
        return np.nan

    # Future suffix alignment
    y_true = y_true_curve[len(y_true_curve) - len(mu):]

    # Convert 90% CI (q05..q95) to std, assuming Gaussian
    sigma = (q95 - q05) / 3.29
    sigma = np.maximum(sigma, 1e-8)

    nll = 0.5 * np.log(2 * np.pi * sigma ** 2) + ((y_true - mu) ** 2 / (2 * sigma ** 2))

    return np.mean(nll)


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def remove_outliers_iqr(values):
    """
    Remove IQR outliers and return (clean_values, outliers).
    Leaves small samples (<4 points) untouched since IQR is unreliable there.
    """
    values = np.asarray(values, dtype=np.float64)

    if len(values) < 4:
        return values, np.array([])

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (values >= lower) & (values <= upper)

    return values[mask], values[~mask]


# ---------------------------------------------------------------------------
# Context name cleanup
# ---------------------------------------------------------------------------

def clean_context_name(s):
    return s.replace("[", "").replace("]", "").replace("_", ", ")


# ---------------------------------------------------------------------------
# Load data for one target scale
# ---------------------------------------------------------------------------

def load_scale_data(true_curves, scale, denormalize):
    nll_data = defaultdict(lambda: defaultdict(list))
    mse_data = defaultdict(lambda: defaultdict(list))

    scale_dir = os.path.join(PRED_DIR, scale)
    print(f"Looking in {scale_dir}")

    for epoch in EPOCHS_LIST:

        # -----------------------------
        # Ablations
        # -----------------------------
        for context_name in ABLATIONS_LIST:
            pred_path = os.path.join(scale_dir, context_name, f"ep{epoch}.json")
            predictions = load_json(pred_path)

            if not predictions:
                print(f"MISSING {pred_path}")
                continue

            print(f"Loaded {len(predictions)} entries from {pred_path}")

            ctx = clean_context_name(context_name)
            matched = 0

            for pred_key, pred_data in predictions.items():
                metrics_key = pred_key.rsplit("_", 1)[0]

                if metrics_key not in true_curves:
                    continue

                matched += 1

                nll = calculate_nll(true_curves[metrics_key], pred_data, denormalize)
                mse = calculate_mse(true_curves[metrics_key], pred_data, denormalize)

                if not np.isnan(nll):
                    nll_data[ctx][epoch].append(nll)

                if not np.isnan(mse):
                    mse_data[ctx][epoch].append(mse)

            if matched == 0:
                print("WARNING: no matching keys")

        # -----------------------------
        # Baseline
        # -----------------------------
        baseline_path = os.path.join(scale_dir, BASELINE_SUBDIR, f"ep{epoch}.json")
        baseline_preds = load_json(baseline_path)

        if not baseline_preds:
            continue

        for pred_key, pred_data in baseline_preds.items():
            metrics_key = pred_key.rsplit("_", 1)[0]

            if metrics_key not in true_curves:
                continue

            nll = calculate_nll(true_curves[metrics_key], pred_data, denormalize)
            mse = calculate_mse(true_curves[metrics_key], pred_data, denormalize)

            # Sanity bound guards against numerically degenerate NLLs
            # (e.g. near-zero sigma). Applied to the baseline bucket only.
            if not np.isnan(nll) and abs(nll) < NLL_SANITY_BOUND:
                nll_data["baseline"][epoch].append(nll)

            if not np.isnan(mse):
                mse_data["baseline"][epoch].append(mse)

    return nll_data, mse_data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_scale(scale, nll_data, mse_data):
    has_baseline = any(
        nll_data.get("baseline", {}).get(epoch) for epoch in EPOCHS_LIST
    )

    context_order = ["baseline"] if has_baseline else []
    context_order += [clean_context_name(x) for x in ABLATIONS_LIST]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(min(14, 1.4 * len(context_order) + 3), 14),
    )

    fig.suptitle(
        f"FT-PFN Performance Comparison (Target Scale {scale})\n"
        "Trajectory Evaluation After Observation",
        fontsize=14,
        fontweight="bold",
    )

    positions_base = np.arange(len(context_order))
    box_width = 0.6 / max(len(EPOCHS_LIST), 1)

    # -------------------------
    # NLL boxplot with outliers shown as points
    # -------------------------
    seen_outlier_labels = set()

    for i, epoch in enumerate(EPOCHS_LIST):
        offset = (i - (len(EPOCHS_LIST) - 1) / 2) * box_width
        positions = positions_base + offset

        values_clean = []
        outlier_points = []

        for idx, ctx in enumerate(context_order):
            vals = nll_data[ctx].get(epoch, [])
            clean, outliers = remove_outliers_iqr(vals)

            # Empty list rather than [nan]: matplotlib handles empty
            # groups cleanly and just leaves that position blank.
            values_clean.append(clean if len(clean) > 0 else [])

            for o in outliers:
                outlier_points.append((positions[idx], o))

        bp = ax1.boxplot(
            values_clean,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
        )

        for box in bp["boxes"]:
            box.set_facecolor(EPOCH_COLORS[epoch])
            box.set_alpha(0.7)

        if outlier_points:
            xs, ys = zip(*outlier_points)
            label = f"{epoch} outliers"
            ax1.scatter(
                xs, ys,
                marker="o",
                s=25,
                color=EPOCH_COLORS[epoch],
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
                label=label if label not in seen_outlier_labels else None,
            )
            seen_outlier_labels.add(label)

    ax1.set_xticks(positions_base)
    ax1.set_xticklabels(context_order, rotation=45, ha="right")
    ax1.set_ylabel("NLL")
    ax1.set_title("Trajectory NLL by Context and Observed Epoch")
    if seen_outlier_labels:
        ax1.legend(fontsize=8, loc="best")

    # -------------------------
    # MSE heatmap
    # -------------------------
    mse_matrix = []
    for epoch in EPOCHS_LIST:
        row = [
            np.mean(mse_data[ctx].get(epoch, [])) if mse_data[ctx].get(epoch) else np.nan
            for ctx in context_order
        ]
        mse_matrix.append(row)

    mse_matrix = np.asarray(mse_matrix)

    im = ax2.imshow(
        mse_matrix,
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=1e-8, vmax=1e-1),
    )

    fig.colorbar(im, ax=ax2, label="Trajectory MSE")

    ax2.set_yticks(np.arange(len(EPOCHS_LIST)))
    ax2.set_yticklabels(EPOCHS_LIST)
    ax2.set_xticks(np.arange(len(context_order)))
    ax2.set_xticklabels(context_order, rotation=45, ha="right")
    ax2.set_xlabel("Context")
    ax2.set_ylabel("Observed Epoch")

    for i in range(len(EPOCHS_LIST)):
        for j in range(len(context_order)):
            value = mse_matrix[i, j]
            if not np.isnan(value):
                ax2.text(
                    j, i, f"{value:.2e}",
                    ha="center", va="center",
                    fontsize=8, color="white",
                )

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, f"nll_mse_boxplot_comparison_scale{scale}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    metrics_path = os.path.join(RESULTS_DIR, "results_metrics.json")
    metrics = load_json(metrics_path)

    if metrics is None:
        raise RuntimeError(f"Cannot load {metrics_path}")

    # Keep original (non-normalized) ground-truth loss curves
    true_curves = {
        k: data["val_loss_curve"]
        for k, data in metrics.items()
        if data.get("val_loss_curve")
    }

    print(f"Loaded {len(true_curves)} ground truth curves")

    log_min, log_max = compute_min_max(4, 128, metrics_path)
    print("log range:", log_min, log_max)

    denormalize = make_denormalizer(log_min, log_max)

    for scale in TARGET_SCALES:
        print(f"\nProcessing scale {scale}")

        nll_data, mse_data = load_scale_data(true_curves, scale, denormalize)
        plot_scale(scale, nll_data, mse_data)


if __name__ == "__main__":
    main()