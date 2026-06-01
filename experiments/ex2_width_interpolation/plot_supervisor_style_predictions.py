"""Experiment 2 — supervisor-style interpolation prediction plots wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments._lib.plot import main as plot_main

DEFAULT_OUTPUT_DIR = Path("experiments/ex2_width_interpolation")
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "results" / "predictions.parquet"
DEFAULT_RAW_PARQUET = Path("demo_analysis/warmstart_runs_flattened.parquet")
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"
DEFAULT_DIAGNOSTICS = DEFAULT_OUTPUT_DIR / "results" / "uncertainty_diagnostics.csv"
DEFAULT_POLICIES = ["bracket_nearest", "prefix_only", "all_context_oracle"]
POLICY_TITLES = {
    "bracket_nearest": "nearest lower + nearest higher",
    "prefix_only": "prefix only",
    "all_context_oracle": "all non-query context oracle",
}


if __name__ == "__main__":
    plot_main(
        description="Ex2: supervisor-style interpolation prediction plots.",
        default_predictions=DEFAULT_PREDICTIONS,
        default_raw_parquet=DEFAULT_RAW_PARQUET,
        default_output_dir=DEFAULT_PLOTS_DIR,
        default_diagnostics=DEFAULT_DIAGNOSTICS,
        default_policies=DEFAULT_POLICIES,
        policy_titles=POLICY_TITLES,
    )
