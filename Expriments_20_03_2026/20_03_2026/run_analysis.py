"""
Entry point: runs the full cross-evaluation analysis pipeline.

Generates outputs organized as:
    outputs/nll/        — Normalized NLL plots + ranking
    outputs/mse/        — Normalized MSE ranking (HTML only)
    outputs_raw/nll/    — Raw NLL plots + ranking (log-scale boxplots)
    outputs_raw/mse/    — Raw MSE ranking (HTML only)

Usage:
    python run_analysis.py
"""
import os

from config import OUTPUT_DIR
from metrics import load_ground_truth, compute_all_metrics
from plot_boxplots import plot_cross_scale_boxplot, plot_source_hp_boxplots
from plot_heatmaps import plot_heatmaps
from generate_html import generate_cross_ranking_html
from generate_csv import generate_csv_tables


def run_analysis(all_rows, output_dir, label, log_scale=False):
    """Run all analysis steps for one mode (normalized or raw)."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")

    # CSV tables (both metrics, saved at mode level)
    print("\n─── CSV tables ───")
    generate_csv_tables(all_rows, output_dir)

    # ── NLL: full suite (boxplots + heatmaps + HTML ranking) ──
    nll_dir = os.path.join(output_dir, "nll")
    os.makedirs(nll_dir, exist_ok=True)

    print("\n─── NLL: Cross-scale boxplots ───")
    plot_cross_scale_boxplot(all_rows, nll_dir, metric="nll", log_scale=log_scale)

    print("\n─── NLL: Source HP boxplots ───")
    plot_source_hp_boxplots(all_rows, nll_dir, metric="nll", log_scale=log_scale)

    print("\n─── NLL: 12×12 heatmaps ───")
    plot_heatmaps(all_rows, nll_dir, metric="nll")

    print("\n─── NLL: Cross-ranking HTML ───")
    generate_cross_ranking_html(all_rows, nll_dir, metric="nll")

    # ── MSE: ranking table only ──
    mse_dir = os.path.join(output_dir, "mse")
    os.makedirs(mse_dir, exist_ok=True)

    print("\n─── MSE: Cross-ranking HTML ───")
    generate_cross_ranking_html(all_rows, mse_dir, metric="mse")

    print(f"\n✅ {label} done. Outputs in: {output_dir}")


def main():
    # Mode 1: Normalized space, all-future averaged
    true_norm = load_ground_truth(normalized=True)
    if true_norm:
        rows_norm = compute_all_metrics(true_norm, raw_mode=False)
        if rows_norm:
            run_analysis(rows_norm, OUTPUT_DIR, "NORMALIZED (all-future avg)")

    # Mode 2: Normalized space, final-point only (matches Thanika's table.py)
    final_dir = OUTPUT_DIR.rstrip("/") + "_final"
    if true_norm:
        rows_final = compute_all_metrics(true_norm, final_point=True)
        if rows_final:
            run_analysis(rows_final, final_dir, "NORMALIZED (final-point only)")

    # Mode 3: Raw loss space, final-point only (log-scale boxplots)
    raw_dir = OUTPUT_DIR.rstrip("/") + "_raw"
    true_raw = load_ground_truth(normalized=False)
    if true_raw:
        rows_raw = compute_all_metrics(true_raw, raw_mode=True)
        if rows_raw:
            run_analysis(rows_raw, raw_dir, "RAW LOSS (final-point only)", log_scale=True)


if __name__ == "__main__":
    main()
