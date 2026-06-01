"""Experiment 3 — Width Extrapolation: multi-policy wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments._lib.runner import build_parser, run_policies
from experiments.ex3_width_extrapolation import policies as ex3_policies

DEFAULT_OUTPUT_DIR = Path("experiments/ex3_width_extrapolation")


def main() -> None:
    parser = build_parser("Ex3: width extrapolation predictions.")
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policies", nargs="+", default=ex3_policies.POLICIES_ALL)
    parser.add_argument("--tier", choices=["1", "1+2", "1+2+3", "all"], default=None,
                        help="Run a tier of policies (overrides --policies).")
    args = parser.parse_args()
    policies = ex3_policies.resolve_policies_from_tier(args.tier, args.policies)
    run_policies(policies, args, ex3_policies)


if __name__ == "__main__":
    main()
