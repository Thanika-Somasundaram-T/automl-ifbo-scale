"""Ex2 policy: bracket_nearest."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pathlib import Path

from experiments._lib.runner import main_for_policy
from experiments.ex2_width_interpolation import policies as ex2_policies

if __name__ == "__main__":
    main_for_policy("bracket_nearest", ex2_policies, Path("experiments/ex2_width_interpolation"), "Ex2: width interpolation predictions.")
