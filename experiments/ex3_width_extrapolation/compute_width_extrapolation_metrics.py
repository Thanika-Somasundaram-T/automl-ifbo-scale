"""Experiment 3 — Width Extrapolation metrics wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments._lib.metrics import main as metrics_main

DEFAULT_OUTPUT_DIR = Path("experiments/ex3_width_extrapolation")
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "results" / "predictions.parquet"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_DIR / "results"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots" / "_summary"
POLICY_COLORS = {
    "prefix_only": "#6b7280",
    "all_lower_only": "#2563eb",
    "nearest_lower_only": "#16a34a",
    "random_lower_k": "#f59e0b",
    "lower_shrink_lt_1": "#9333ea",
    "lower_high_tkpm_cold_start": "#dc2626",
    "all_context_oracle": "#000000",
}


if __name__ == "__main__":
    metrics_main(
        description="Ex3: width extrapolation metrics.",
        default_predictions=DEFAULT_PREDICTIONS,
        default_output_dir=DEFAULT_RESULTS_DIR,
        default_plots_dir=DEFAULT_PLOTS_DIR,
        policy_colors=POLICY_COLORS,
    )
