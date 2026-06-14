import torch
import pandas as pd
import numpy as np
import neps
import yaml

from pipeline import neps_wrapper
from train.predict import (
    DATA_PATH,
    SAVE_DIR,
)
from utils import get_device, load_data

def main():
    device = get_device()

    print("Loading all_curves.json...")
    df = load_data(DATA_PATH)

    print(f"✓ Loaded {len(df)} runs")
    print(f"  Base-N values: {sorted(df['base_N'].unique())}")
    print(f"  Target-N values: {sorted(df['target_N'].unique())}")

    # -------------------------
    # FIXED: global loss computation
    # -------------------------


    # -------------------------
    # Load NePS config
    # -------------------------
    with open("./neps_config.yaml", "r") as f:
        neps_config = yaml.safe_load(f)

    # -------------------------
    # IMPORTANT: keep config NePS-safe
    # (avoid DataFrame / torch objects inside YAML config)
    # -------------------------
    neps_config["evaluate_pipeline"] = neps_wrapper(df, device)

    # Pass only primitives (safe)

    print("NePS config loaded:")
    print(neps_config)

    # -------------------------
    # Run NePS
    # -------------------------
    neps.run(**neps_config)

if __name__ == "__main__":
    main()