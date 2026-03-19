"""
Fixed supervisor extension analysis.

Correctly interprets Thanika's new prediction data (norm_with_classes_log/):
  - configs_N means "source HP #N" (single specific HP as context),
    NOT "N configs combined."
  - Uses generate_keys() mapping: config 1..12 → specific (lr, wd) combo.

Produces:
  1) Cross-scale boxplots with per-target-HP markers  (one per epoch)
  2) Per-scale source-HP boxplots  (which source HP is best per scale)
  3) 12×12 source→target NLL heatmaps  (per scale × epoch)
  4) Cross-ranking HTML tables
"""
import csv
import html as html_mod
import json
import math
import os
from collections import defaultdict, OrderedDict

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PRED_DIR = os.path.join(ROOT, "analysis_loss_normalized", "norm_with_classes_log")
RESULTS_METRICS = os.path.join(ROOT, "results", "results_metrics.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
NUM_CLASSES = 10
LOSS_MIN = 1e-3
LOSS_MAX = math.log2(NUM_CLASSES)  # ≈ 3.3219

BASE_SCALES = ["[8]", "[16]", "[24]", "[32]", "[64]"]
OBS_EPOCHS = [0, 5, 10, 20, 50, 90]
SOURCE_CONFIGS = list(range(1, 13))  # all 12 source HPs

# ─────────────────────────────────────────────
# Source HP mapping (mirrors Thanika's generate_keys)
# ─────────────────────────────────────────────
LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
WDS = [0.0, 0.01]


def generate_keys(hd):
    """Mirrors Thanika's generate_keys — maps config 1..12 to run keys."""
    keys = {}
    idx = 1
    for lr in LRS:
        for wd in WDS:
            keys[idx] = f"layer4_lr{lr}_hd{hd}_wd{wd}_cosine"
            idx += 1
    return keys


def source_hp_label(config_num):
    """Human-readable label for source config number."""
    idx = 1
    for lr in LRS:
        for wd in WDS:
            if idx == config_num:
                lr_s = f"{lr:.0e}" if lr < 1e-3 else f"{lr}"
                return f"lr={lr_s}, wd={wd}"
            idx += 1
    return f"config_{config_num}"


def short_target_label(metrics_key):
    """Convert 'layer4_lr0.001_hd128_wd0.0_cosine' to 'lr=0.001,wd=0.0'."""
    parts = metrics_key.split("_")
    lr_part = [p for p in parts if p.startswith("lr")][0]
    wd_part = [p for p in parts if p.startswith("wd")][0]
    return f"{lr_part},{wd_part}"


def _parse_hp_numerics(metrics_key):
    """Extract (lr_float, wd_float) from a metrics key for numerical sorting."""
    parts = metrics_key.split("_")
    lr_val = float([p for p in parts if p.startswith("lr")][0][2:])
    wd_val = float([p for p in parts if p.startswith("wd")][0][2:])
    return (lr_val, wd_val)


def sorted_target_hps(all_rows):
    """Return target HPs sorted in ascending numerical order (lr, then wd)."""
    unique = {r["target_hp"] for r in all_rows if r["scale"] != "baseline"}
    return sorted(unique, key=_parse_hp_numerics)


# ─────────────────────────────────────────────
# Normalization (matches Thanika's new function)
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
# Helpers
# ─────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def parse_metrics_key(pred_key):
    """Strip trailing _epoch suffix from prediction keys."""
    return pred_key.rsplit("_", 1)[0]


def clean_scale(s):
    return s.replace("[", "").replace("]", "")


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
def compute_all_metrics(true_curves, raw_mode=False):
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


# ─────────────────────────────────────────────
# Plot 1: Cross-scale boxplots (aggregated over source HPs)
# ─────────────────────────────────────────────
HP_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "d"]
HP_COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#6D4C41", "#F06292",
    "#7CB342", "#5C6BC0", "#26A69A", "#FF7043",
]
SCALE_COLORS = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD0", "#D1C4E9"]


def plot_cross_scale_boxplot(all_rows, output_dir, log_scale=False):
    """
    Cross-scale comparison: all 5 scales side-by-side per epoch.
    Aggregates over all 12 source HPs. Markers show target HPs.
    """
    target_hps = sorted_target_hps(all_rows)
    hp_marker_map = {hp: HP_MARKERS[i % len(HP_MARKERS)] for i, hp in enumerate(target_hps)}
    hp_color_map = {hp: HP_COLORS[i % len(HP_COLORS)] for i, hp in enumerate(target_hps)}

    # Universal y-axis bounds
    all_nll = [r["nll"] for r in all_rows if r["scale"] != "baseline" and not np.isnan(r["nll"])]
    if not log_scale:
        y_lo = np.percentile(all_nll, 1) - 0.2
        y_hi = np.percentile(all_nll, 99) + 0.2

    fig, axes = plt.subplots(len(OBS_EPOCHS), 1,
                             figsize=(14, 5 * len(OBS_EPOCHS)),
                             sharey=True, squeeze=False)

    fig.suptitle(
        "Cross-Scale NLL Comparison (Aggregated Over All Source HPs)\n"
        f"(Target: hd=128, loss_max=log₂(10)≈{LOSS_MAX:.2f})",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row_idx, epoch in enumerate(OBS_EPOCHS):
        ax = axes[row_idx][0]

        nll_by_scale = []
        rows_by_scale = []
        for scale in BASE_SCALES:
            cell = [r for r in all_rows
                    if r["scale"] == scale and r["epoch"] == epoch
                    and not np.isnan(r["nll"])]
            nll_by_scale.append([r["nll"] for r in cell] if cell else [np.nan])
            rows_by_scale.append(cell)

        positions = np.arange(len(BASE_SCALES))
        bp = ax.boxplot(nll_by_scale, positions=positions, widths=0.55,
                        patch_artist=True, showfliers=False)
        for i, box in enumerate(bp["boxes"]):
            box.set_facecolor(SCALE_COLORS[i])
            box.set_alpha(0.6)
            box.set_edgecolor("#555555")
        for m in bp["medians"]:
            m.set_color("black")
            m.set_linewidth(1.5)

        # HP markers (mean NLL per target HP across all source HPs)
        for scale_idx, (scale, cell) in enumerate(zip(BASE_SCALES, rows_by_scale)):
            target_means = defaultdict(list)
            for r in cell:
                target_means[r["target_hp"]].append(r["nll"])
            for hp, vals in target_means.items():
                mean_nll = np.mean(vals)
                jitter = (hash(hp + str(epoch) + scale) % 11 - 5) * 0.02
                ax.scatter(
                    scale_idx + jitter, mean_nll,
                    marker=hp_marker_map[hp], color=hp_color_map[hp],
                    s=55, alpha=0.85, edgecolors="#222", linewidths=0.4, zorder=5,
                )

        ax.set_xticks(positions)
        ax.set_xticklabels([clean_scale(s) for s in BASE_SCALES], fontsize=11)
        ax.set_xlabel("Base Scale (Hidden Dim)", fontsize=11)
        ax.set_ylabel("NLL ← Lower is Better", fontsize=10)
        ax.set_title(f"T = {epoch} Epochs Observed", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        if log_scale:
            ax.set_yscale('log')
        else:
            ax.set_ylim(y_lo, y_hi)

    # Legend
    handles = [plt.scatter([], [], marker=hp_marker_map[hp], color=hp_color_map[hp],
                           s=50, edgecolors="#222", linewidths=0.4,
                           label=short_target_label(hp))
               for hp in target_hps]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               title="Target HP (hd=128) — mean NLL across source HPs",
               title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = os.path.join(output_dir, "cross_scale_nll_aggregated.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"✅ Saved: {out}")
    plt.close()


# ─────────────────────────────────────────────
# Plot 2: Per-scale source HP boxplots
# ─────────────────────────────────────────────
def plot_source_hp_boxplots(all_rows, output_dir, log_scale=False):
    """
    For each (scale, epoch): 12 boxplots on x-axis (one per source HP),
    each showing NLL distribution across 12 target HPs, with markers.
    """
    target_hps = sorted_target_hps(all_rows)
    hp_marker_map = {hp: HP_MARKERS[i % len(HP_MARKERS)] for i, hp in enumerate(target_hps)}
    hp_color_map = {hp: HP_COLORS[i % len(HP_COLORS)] for i, hp in enumerate(target_hps)}

    # Universal y-axis bounds
    all_nll = [r["nll"] for r in all_rows if r["scale"] != "baseline" and not np.isnan(r["nll"])]
    if not log_scale:
        y_lo = np.percentile(all_nll, 1) - 0.2
        y_hi = np.percentile(all_nll, 99) + 0.2

    for scale in BASE_SCALES:
        fig, axes = plt.subplots(len(OBS_EPOCHS), 1,
                                 figsize=(18, 5 * len(OBS_EPOCHS)),
                                 sharey=True, squeeze=False)
        fig.suptitle(
            f"Source HP Comparison — Base Scale = {clean_scale(scale)}\n"
            f"(Each bar = one source HP as context, markers = target HPs)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for row_idx, epoch in enumerate(OBS_EPOCHS):
            ax = axes[row_idx][0]

            data_per_src = []
            rows_per_src = []
            labels = []
            for src_cfg in SOURCE_CONFIGS:
                cell = [r for r in all_rows
                        if r["scale"] == scale
                        and r["source_config"] == src_cfg
                        and r["epoch"] == epoch
                        and not np.isnan(r["nll"])]
                data_per_src.append([r["nll"] for r in cell] if cell else [np.nan])
                rows_per_src.append(cell)
                labels.append(source_hp_label(src_cfg))

            positions = np.arange(len(SOURCE_CONFIGS))
            bp = ax.boxplot(data_per_src, positions=positions, widths=0.55,
                            patch_artist=True, showfliers=False)
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor("#B3E5FC")
                box.set_alpha(0.5)
                box.set_edgecolor("#555555")
            for m in bp["medians"]:
                m.set_color("black")
                m.set_linewidth(1.5)

            # Target HP markers
            for src_idx, cell in enumerate(rows_per_src):
                for r in cell:
                    hp = r["target_hp"]
                    jitter = (hash(hp + str(epoch) + str(src_idx)) % 11 - 5) * 0.02
                    ax.scatter(
                        src_idx + jitter, r["nll"],
                        marker=hp_marker_map[hp], color=hp_color_map[hp],
                        s=45, alpha=0.85, edgecolors="#222", linewidths=0.4, zorder=5,
                    )

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Source HP (context)", fontsize=10)
            ax.set_ylabel("NLL ← Lower is Better", fontsize=10)
            ax.set_title(f"T = {epoch} Epochs Observed", fontsize=11, fontweight="bold")
            ax.grid(axis="y", alpha=0.25)
            if log_scale:
                ax.set_yscale('log')
            else:
                ax.set_ylim(y_lo, y_hi)

        # Legend
        handles = [plt.scatter([], [], marker=hp_marker_map[hp], color=hp_color_map[hp],
                               s=50, edgecolors="#222", linewidths=0.4,
                               label=short_target_label(hp))
                   for hp in target_hps]
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
                   title="Target HP (hd=128)", title_fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        out = os.path.join(output_dir, f"source_hp_nll_scale_{clean_scale(scale)}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"✅ Saved: {out}")
        plt.close()


# ─────────────────────────────────────────────
# Plot 3: 12×12 source→target NLL heatmaps
# ─────────────────────────────────────────────
def plot_heatmaps(all_rows, output_dir):
    """One heatmap per (scale, epoch). Rows=source HPs, Cols=target HPs, Color=NLL."""
    target_hps = sorted_target_hps(all_rows)
    target_labels = [short_target_label(hp) for hp in target_hps]
    source_labels = [source_hp_label(c) for c in SOURCE_CONFIGS]

    # Compute UNIVERSAL color bounds across all heatmaps
    all_nll = [r["nll"] for r in all_rows
               if r["scale"] != "baseline" and not np.isnan(r["nll"])]
    global_vmin = np.percentile(all_nll, 2)
    global_vmax = np.percentile(all_nll, 98)
    print(f"  Universal color scale: [{global_vmin:.2f}, {global_vmax:.2f}]")

    for scale in BASE_SCALES:
        for epoch in OBS_EPOCHS:
            matrix = np.full((12, 12), np.nan)

            for r in all_rows:
                if r["scale"] != scale or r["epoch"] != epoch or r["scale"] == "baseline":
                    continue
                src_idx = r["source_config"] - 1
                tgt_idx = target_hps.index(r["target_hp"]) if r["target_hp"] in target_hps else -1
                if 0 <= src_idx < 12 and 0 <= tgt_idx < 12:
                    matrix[src_idx, tgt_idx] = r["nll"]

            fig, ax = plt.subplots(figsize=(12, 10))

            valid = matrix[~np.isnan(matrix)]
            if len(valid) == 0:
                plt.close()
                continue

            im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                           vmin=global_vmin, vmax=global_vmax)
            cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            cbar.set_label("NLL (lower = better)", fontsize=10)

            # Annotate cells
            for i in range(12):
                for j in range(12):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = "white" if val > (global_vmin + global_vmax) / 2 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, fontweight="bold", color=color)

            ax.set_xticks(range(12))
            ax.set_xticklabels(target_labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(12))
            ax.set_yticklabels(source_labels, fontsize=8)
            ax.set_xlabel("Target HP (hd=128)", fontsize=11)
            ax.set_ylabel("Source HP (context)", fontsize=11)
            ax.set_title(
                f"Source→Target NLL — Scale={clean_scale(scale)}, T={epoch}\n"
                f"(Green=good, Red=bad)",
                fontsize=12, fontweight="bold",
            )

            plt.tight_layout()
            out = os.path.join(output_dir,
                               f"heatmap_nll_scale_{clean_scale(scale)}_T{epoch}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
            print(f"✅ Saved: {out}")
            plt.close()


# ─────────────────────────────────────────────
# Table 4: Cross-ranking HTML
# ─────────────────────────────────────────────
def generate_cross_ranking_html(all_rows, output_dir):
    """
    HTML with 12×12 rank heatmap matrices.
    One table per (scale, epoch): rows=source HPs, cols=target HPs.
    Cell = rank of that (source, target) NLL within its column (per target HP).
    """
    target_hps = sorted_target_hps(all_rows)
    target_labels = [short_target_label(hp) for hp in target_hps]
    source_labels = [source_hp_label(c) for c in SOURCE_CONFIGS]

    sections = []
    table_idx = 0

    for scale in BASE_SCALES:
        for epoch in OBS_EPOCHS:
            table_idx += 1

            # Build NLL matrix
            nll_matrix = {}
            for r in all_rows:
                if r["scale"] != scale or r["epoch"] != epoch or r["scale"] == "baseline":
                    continue
                nll_matrix[(r["source_config"], r["target_hp"])] = r["nll"]

            if not nll_matrix:
                continue

            # Rank within each target HP column (rank 1 = lowest NLL = best)
            rank_matrix = {}
            for tgt_hp in target_hps:
                col_vals = [(src, nll_matrix.get((src, tgt_hp), np.nan))
                            for src in SOURCE_CONFIGS]
                col_vals.sort(key=lambda x: x[1] if not np.isnan(x[1]) else float("inf"))
                for rank, (src, _) in enumerate(col_vals, 1):
                    rank_matrix[(src, tgt_hp)] = rank

            max_rank = 12

            def rank_to_style(rank_val):
                t = (rank_val - 1) / (max_rank - 1) if max_rank > 1 else 0
                # Green (good) → Red (bad)
                r = int(60 + 180 * t)
                g = int(180 - 120 * t)
                b = int(60 - 30 * t)
                bg = f"rgb({r},{g},{b})"
                txt = "#ffffff" if t > 0.4 else "#1d2430"
                return f"background:{bg};color:{txt};"

            header = "".join(
                f"<th style='font-size:10px'>{html_mod.escape(tl)}</th>"
                for tl in target_labels
            )
            body = []
            for src_cfg in SOURCE_CONFIGS:
                cells = []
                for tgt_hp in target_hps:
                    rank = rank_matrix.get((src_cfg, tgt_hp), "")
                    style = rank_to_style(rank) if rank != "" else ""
                    cells.append(f"<td style='{style};text-align:center;font-weight:700'>{rank}</td>")
                body.append(
                    f"<tr><th style='text-align:left;font-size:10px'>"
                    f"{html_mod.escape(source_hp_label(src_cfg))}</th>"
                    f"{''.join(cells)}</tr>"
                )

            section = (
                f"<h2>Table {table_idx}: Scale={html_mod.escape(clean_scale(scale))}, T={epoch}</h2>"
                f"<p style='font-size:12px;color:#666'>Rank 1 (green) = best source HP for that target. "
                f"Rank 12 (red) = worst.</p>"
                f"<table><thead><tr>"
                f"<th style='min-width:150px'>Source HP \\ Target HP</th>"
                f"{header}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>"
            )
            sections.append(section)

    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Source→Target HP Cross-Ranking</title>"
        "<style>"
        "body{font-family:'Segoe UI',Arial,sans-serif;margin:20px;background:#f8f9fa;color:#1d2430;}"
        "h1{margin:0 0 8px 0;} h2{margin:28px 0 8px 0;font-size:16px;}"
        "table{border-collapse:collapse;margin-bottom:20px;}"
        "th,td{border:1px solid #d5dbe3;padding:4px 6px;font-size:11px;}"
        "th{background:#e9eef5;}"
        "tbody tr:hover{outline:2px solid #1E88E5;}"
        "</style></head><body>"
        "<h1>Source→Target HP Cross-Ranking</h1>"
        "<p>For each (scale, T): which source HP produces the best prediction "
        "for each target HP? Rank 1 = lowest NLL (best). "
        "Green cells = good, red = bad.</p>"
        f"{''.join(sections)}"
        "</body></html>"
    )

    out = os.path.join(output_dir, "cross_ranking_source_target.html")
    os.makedirs(output_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"✅ Saved: {out}")


# ─────────────────────────────────────────────
# CSV rank tables
# ─────────────────────────────────────────────
def generate_csv_tables(all_rows, output_dir):
    """Raw CSV with all metrics + rank per target HP."""
    os.makedirs(output_dir, exist_ok=True)

    # Full data CSV
    csv_path = os.path.join(output_dir, "all_metrics.csv")
    fieldnames = ["scale", "source_config", "source_hp", "epoch",
                   "target_hp", "target_hp_short", "nll", "mse"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"✅ Saved: {csv_path}")

    # Rank pivot: for each (scale, epoch, target_hp), rank source HPs by NLL
    target_hps = sorted_target_hps(all_rows)
    pivot_rows = []

    for scale in BASE_SCALES:
        for epoch in OBS_EPOCHS:
            for tgt_hp in target_hps:
                relevant = [(r["source_config"], r["nll"])
                            for r in all_rows
                            if r["scale"] == scale and r["epoch"] == epoch
                            and r["target_hp"] == tgt_hp
                            and r["scale"] != "baseline"
                            and not np.isnan(r["nll"])]
                relevant.sort(key=lambda x: x[1])
                for rank, (src_cfg, nll_val) in enumerate(relevant, 1):
                    pivot_rows.append({
                        "scale": scale,
                        "epoch": epoch,
                        "target_hp": short_target_label(tgt_hp),
                        "source_config": src_cfg,
                        "source_hp": source_hp_label(src_cfg),
                        "nll": round(nll_val, 4),
                        "rank": rank,
                    })

    pivot_path = os.path.join(output_dir, "rank_source_per_target.csv")
    with open(pivot_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "scale", "epoch", "target_hp", "source_config", "source_hp", "nll", "rank",
        ])
        writer.writeheader()
        for r in pivot_rows:
            writer.writerow(r)
    print(f"✅ Saved: {pivot_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run_analysis(true_curves, all_rows, output_dir, label, log_scale=False):
    """Run all analysis steps for one mode."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    print("\n─── Cross-scale boxplots ───")
    plot_cross_scale_boxplot(all_rows, output_dir, log_scale=log_scale)

    print("\n─── Source HP boxplots ───")
    plot_source_hp_boxplots(all_rows, output_dir, log_scale=log_scale)

    print("\n─── 12×12 heatmaps ───")
    plot_heatmaps(all_rows, output_dir)

    print("\n─── CSV tables ───")
    generate_csv_tables(all_rows, output_dir)

    print("\n─── Cross-ranking HTML ───")
    generate_cross_ranking_html(all_rows, output_dir)

    print(f"\n✅ {label} done. Outputs in: {output_dir}")


def main():
    # Mode 1: Normalized space, all-future averaged
    true_norm = load_ground_truth(normalized=True)
    if true_norm:
        rows_norm = compute_all_metrics(true_norm, raw_mode=False)
        if rows_norm:
            run_analysis(true_norm, rows_norm, OUTPUT_DIR, "NORMALIZED (all-future avg)")

    # Mode 2: Raw loss space, final-point only (log-scale boxplots)
    raw_dir = OUTPUT_DIR.rstrip("/") + "_raw"
    true_raw = load_ground_truth(normalized=False)
    if true_raw:
        rows_raw = compute_all_metrics(true_raw, raw_mode=True)
        if rows_raw:
            run_analysis(true_raw, rows_raw, raw_dir, "RAW LOSS (final-point only)", log_scale=True)


if __name__ == "__main__":
    main()
