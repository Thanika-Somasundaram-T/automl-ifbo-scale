"""Ex3 policy: lower_final_pt_only — lower sources contribute only final point."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pathlib import Path

from experiments._lib.runner import main_for_policy
from experiments.ex3_width_extrapolation import policies as ex3_policies

if __name__ == "__main__":
    main_for_policy("lower_final_pt_only", ex3_policies, Path("experiments/ex3_width_extrapolation"), "Ex3: width extrapolation predictions.")
