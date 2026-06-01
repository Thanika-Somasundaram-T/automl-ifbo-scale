"""Experiment 2 — Width Interpolation: multi-policy wrapper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments._lib.runner import build_parser, run_policies
from experiments.ex2_width_interpolation import policies as ex2_policies

DEFAULT_OUTPUT_DIR = Path("experiments/ex2_width_interpolation")


def main() -> None:
    parser = build_parser("Ex2: width interpolation predictions.")
    parser.set_defaults(output_dir=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policies", nargs="+", default=ex2_policies.POLICIES_ALL)
    parser.add_argument("--tier", choices=["primary", "baselines", "all"], default=None,
                        help="Run a tier of policies (overrides --policies).")
    args = parser.parse_args()
    policies = ex2_policies.resolve_policies_from_tier(args.tier, args.policies)
    run_policies(policies, args, ex2_policies)


if __name__ == "__main__":
    main()
