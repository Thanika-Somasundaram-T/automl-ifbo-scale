"""Experiment 3 — supervisor-style prediction plots wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments._lib.plot import main as plot_main
from experiments.ex3_width_extrapolation import policies as ex3_policies

DEFAULT_OUTPUT_DIR = Path("experiments/ex3_width_extrapolation")
DEFAULT_PREDICTIONS = DEFAULT_OUTPUT_DIR / "results" / "predictions.parquet"
DEFAULT_RAW_PARQUET = Path("demo_analysis/warmstart_runs_flattened.parquet")
DEFAULT_PLOTS_DIR = DEFAULT_OUTPUT_DIR / "plots"
DEFAULT_DIAGNOSTICS = DEFAULT_OUTPUT_DIR / "results" / "uncertainty_diagnostics.csv"
DEFAULT_POLICIES = [
    "prefix_only",
    "context_ifbo",
    "all_lower_only",
    "nearest_lower_only",
    "lower_geom_sparse_k8",
]
POLICY_TITLES = {
    "prefix_only": "prefix only",
    "context_ifbo": "default iFBO context",
    "all_lower_only": "all lower targets",
    "nearest_lower_only": "nearest lower target",
    "lower_geom_sparse_k8": "lower geometric sparse k=8",
}


if __name__ == "__main__":
    plot_main(
        description="Ex3: supervisor-style prediction plots.",
        default_predictions=DEFAULT_PREDICTIONS,
        default_raw_parquet=DEFAULT_RAW_PARQUET,
        default_output_dir=DEFAULT_PLOTS_DIR,
        default_diagnostics=DEFAULT_DIAGNOSTICS,
        default_policies=[policy for policy in DEFAULT_POLICIES if policy in ex3_policies.POLICIES_ALL],
        policy_titles=POLICY_TITLES,
    )
