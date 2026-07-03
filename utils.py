import json
import os
import math
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ifbo import Curve

def get_device():
    """
    Determine and return the best available computation device.

    Returns:
        torch.device
            Available device in priority order: MPS, CUDA, CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device

# ============================================================

def safe_bounds(lower, upper, min_value=1e-8):
        # ensures valid positive and non-zero range
        lower = max(lower, min_value)
        upper = max(upper, lower + min_value)
        return lower, upper

def normalize_hyperparameters(row, df, clip=True):
    EPS = 1e-8

    def get_minmax(col):
        return df[col].min(), df[col].max()

    def add_buffer(lower, upper, buffer_pct=0.1):
        width = upper - lower
        return (
            lower - buffer_pct * width,
            upper + buffer_pct * width,
        )



    def minmax_norm(x, lower, upper):
        return (x - lower) / (upper - lower + EPS)

    def clip01(x):
        return np.clip(x, 0.0, 1.0)

    # -----------------------
    # Compute buffered ranges
    # -----------------------
    b_min, b_max = add_buffer(*get_minmax("base_N"), buffer_pct=0.1)
    sh_min, sh_max = add_buffer(*get_minmax("shrinking"), buffer_pct=0.1)
    tk_min, tk_max = add_buffer(*get_minmax("tkpm"), buffer_pct=0.1)
    emb_min, emb_max = add_buffer(*get_minmax("n_embd"), buffer_pct=0.1)

    # critical safety fix for log2
    b_min, b_max = safe_bounds(b_min, b_max, EPS)

    # -----------------------
    # Normalization
    # -----------------------

    base_val = np.log2(np.clip(row["base_N"], b_min, b_max))
    base_norm = minmax_norm(
        base_val,
        np.log2(b_min),
        np.log2(b_max),
    )

    shrink_norm = minmax_norm(row["shrinking"], sh_min, sh_max)
    tkpm_norm = minmax_norm(row["tkpm"], tk_min, tk_max)
    emb_norm = minmax_norm(row["n_embd"], emb_min, emb_max)

    # -----------------------
    # Optional strict clipping
    # -----------------------
    if clip:
        base_norm = clip01(base_norm)
        shrink_norm = clip01(shrink_norm)
        tkpm_norm = clip01(tkpm_norm)
        emb_norm = clip01(emb_norm)

    return torch.tensor(
        # [shrink_norm, tkpm_norm, emb_norm],
        [tkpm_norm, emb_norm, shrink_norm, base_norm],
        dtype=torch.float32,
    )
    
def compute_loo(df, target_N=None, buffer=0.05):
    all_losses = np.concatenate(df["val_loss"].values)
    log_others   = np.log(np.clip(all_losses, 1e-8, None))
    lo           = float(log_others.min())
    hi           = float(log_others.max())
    margin       = (hi - lo) * buffer
    v_min, v_max = safe_bounds(lo - margin, hi + margin, min_value=1e-8)
    return  v_min, v_max


def normalize_log_loss_curve(val_loss_list,  df, target_N=None):

    val_loss = np.asarray(val_loss_list, dtype=np.float64)

    log_curve = np.log(val_loss + 1e-8)
    log_min, log_max = compute_loo(df, target_N, buffer=0.05)
    print("dfff. ", log_min, log_max)

    if abs(log_max - log_min) < 1e-8:
        return np.zeros_like(log_curve)

    norm = (log_curve - log_min) / (log_max - log_min)

    return np.clip(norm, 0.0, 1.0)


def load_data(path):
    """
    Load experiments.json into dataframe.
    One row = one complete training run.
    """

    with open(path, "r") as f:
        data = json.load(f)

    rows = []

    for run_key, run_data in data.items():
        hp = run_data["hyperparameters"]
        curve = run_data["curve"]

        rows.append(
            {
                "run_key": run_key,

                # scale
                "base_N": hp["base_N"],
                "target_N": hp["target_N"],

                # optimization
                "max_lr": hp["max_lr"],
                "shrinking": hp.get("shrink", 1.0),
                "tkpm": hp["tkpm"],

                # architecture
                "n_embd": hp["n_embd"],
                "n_head": hp["n_head"],
                "g_width": hp.get("g_width", 0.0),
                "g_N": hp.get("g_N", 0.0),

                # curves
                "tokens": np.asarray(curve["tokens"]),
                "val_loss": np.asarray(curve["val_loss"]),
                "train_loss": np.asarray(curve["train_loss"]),
                "flops": np.asarray(curve["flops"]),
            }
        )

    return pd.DataFrame(rows)


def subsample_curve(t, y, n_total=10):
    t = np.array(t)
    y = np.array(y)

    if len(t) <= n_total:
        return t, y

    # uniformly spaced indices across the full curve
    idx = np.linspace(0, len(t) - 1, n_total)
    idx = np.round(idx).astype(int)

    # ensure uniqueness (just in case of rounding collisions)
    idx = np.unique(idx)

    print(f"***** Subsampling: {len(idx)} points selected out of {len(t)}")

    return t[idx], y[idx]
