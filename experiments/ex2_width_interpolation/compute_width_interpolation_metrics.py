"""Experiment 2 — Width Interpolation metrics wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments._lib.metrics import main as metrics_main

DEFAULT_OUTPUT_DIR = Path("experiments/ex2_width_interpolation")
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "results" / "predictions.parquet"
DEFAULT_RESULTS_DIR = DEFAULT_OUTPUT_DIR / "results"
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots" / "_summary"
POLICY_COLORS = {
    "bracket_nearest": "#2563eb",
    "prefix_only": "#6b7280",
    "all_context_oracle": "#000000",
}


if __name__ == "__main__":
    metrics_main(
        description="Ex2: width interpolation metrics.",
        default_predictions=DEFAULT_PREDICTIONS,
        default_output_dir=DEFAULT_RESULTS_DIR,
        default_plots_dir=DEFAULT_PLOTS_DIR,
        policy_colors=POLICY_COLORS,
    )
