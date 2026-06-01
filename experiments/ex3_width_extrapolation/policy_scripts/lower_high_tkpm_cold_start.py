"""Ex3 policy: lower_high_tkpm_cold_start — lower sources with tkpm == 30.0."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pathlib import Path

from experiments._lib.runner import main_for_policy
from experiments.ex3_width_extrapolation import policies as ex3_policies

if __name__ == "__main__":
    main_for_policy("lower_high_tkpm_cold_start", ex3_policies, Path("experiments/ex3_width_extrapolation"), "Ex3: width extrapolation predictions.")
