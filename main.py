import torch
import pandas as pd
import numpy as np
import neps
import yaml

from pipeline import neps_wrapper
from train.predict import (
    DATA_PATH,
    SAVE_DIR,
    get_device,
    load_data,
    predict
)

def main():
    device = get_device()

    print("Loading experiments.json...")
    df = load_data(DATA_PATH)

    print(f"✓ Loaded {len(df)} runs")
    print(f"  Base-N values: {sorted(df['base_N'].unique())}")
    print(f"  Target-N values: {sorted(df['target_N'].unique())}")

    # -------------------------
    # FIXED: global loss computation
    # -------------------------
    all_losses = np.concatenate(df["val_loss"].values)

    global_min = float(np.min(all_losses))
    global_max = float(np.max(all_losses))

    print(f"  Global loss range: [{global_min:.4f}, {global_max:.4f}]")

    # -------------------------
    # Load NePS config
    # -------------------------
    with open("./neps_config.yaml", "r") as f:
        neps_config = yaml.safe_load(f)

    # -------------------------
    # IMPORTANT: keep config NePS-safe
    # (avoid DataFrame / torch objects inside YAML config)
    # -------------------------
    neps_config["evaluate_pipeline"] = neps_wrapper(df, device, global_min, global_max)

    # Pass only primitives (safe)

    print("NePS config loaded:")
    print(neps_config)

    # -------------------------
    # Run NePS
    # -------------------------
    neps.run(**neps_config)

if __name__ == "__main__":
    main()