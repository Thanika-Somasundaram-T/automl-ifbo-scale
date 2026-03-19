"""
Configuration: paths, constants, HP mappings, and helper functions.
"""
import math
import os

import numpy as np

# ─────────────────────────────────────────────
# Paths (self-contained — data lives inside this folder)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PRED_DIR = os.path.join(DATA_DIR, "predictions")
RESULTS_METRICS = os.path.join(DATA_DIR, "results_metrics.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NUM_CLASSES = 10
LOSS_MIN = 1e-3
LOSS_MAX = math.log2(NUM_CLASSES)  # ≈ 3.3219

BASE_SCALES = ["[8]", "[16]", "[24]", "[32]", "[64]"]
OBS_EPOCHS = [0, 5, 10, 20, 50, 90]
SOURCE_CONFIGS = list(range(1, 13))  # all 12 source HPs

# ─────────────────────────────────────────────
# HP definitions
# ─────────────────────────────────────────────
LRS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
WDS = [0.0, 0.01]

# ─────────────────────────────────────────────
# Visual constants
# ─────────────────────────────────────────────
HP_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "d"]
HP_COLORS = [
    "#E53935", "#1E88E5", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#6D4C41", "#F06292",
    "#7CB342", "#5C6BC0", "#26A69A", "#FF7043",
]
SCALE_COLORS = ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD0", "#D1C4E9"]


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def generate_keys(hd):
    """Mirrors Thanika's generate_keys — maps config 1..12 to run keys."""
    keys = {}
    idx = 1
    for lr in LRS:
        for wd in WDS:
            keys[idx] = f"layer4_lr{lr}_hd{hd}_wd{wd}_cosine"
            idx += 1
    return keys


def source_hp_label(config_num):
    """Human-readable label for source config number."""
    idx = 1
    for lr in LRS:
        for wd in WDS:
            if idx == config_num:
                lr_s = f"{lr:.0e}" if lr < 1e-3 else f"{lr}"
                return f"lr={lr_s}, wd={wd}"
            idx += 1
    return f"config_{config_num}"


def short_target_label(metrics_key):
    """Convert 'layer4_lr0.001_hd128_wd0.0_cosine' to 'lr=0.001,wd=0.0'."""
    parts = metrics_key.split("_")
    lr_part = [p for p in parts if p.startswith("lr")][0]
    wd_part = [p for p in parts if p.startswith("wd")][0]
    return f"{lr_part},{wd_part}"


def _parse_hp_numerics(metrics_key):
    """Extract (lr_float, wd_float) from a metrics key for numerical sorting."""
    parts = metrics_key.split("_")
    lr_val = float([p for p in parts if p.startswith("lr")][0][2:])
    wd_val = float([p for p in parts if p.startswith("wd")][0][2:])
    return (lr_val, wd_val)


def sorted_target_hps(all_rows):
    """Return target HPs sorted in ascending numerical order (lr, then wd)."""
    unique = {r["target_hp"] for r in all_rows if r["scale"] != "baseline"}
    return sorted(unique, key=_parse_hp_numerics)


def clean_scale(s):
    return s.replace("[", "").replace("]", "")
