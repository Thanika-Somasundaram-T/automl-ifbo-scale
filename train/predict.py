from math import log
import os
import json

import torch
import pandas as pd
import numpy as np

from ifbo.surrogate import FTPFN
from ifbo import Curve

from utils import normalize_hyperparameters, normalize_log_loss_curve, subsample_curve


# ============================================================
# Config
# ============================================================

DATA_PATH = "./all_curves.json"

SAVE_DIR = "./june/baseline"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# Subsampling
# ============================================================

# def subsample_curve(t, y, n_total=200, warmstart_frac=0.7):
#     t = np.array(t)
#     y = np.array(y)
    
#     if len(t) <= n_total:
#         return t, y
    
#     n_post  = int(n_total * 0.50)
#     n_mid   = int(n_total * 0.25)
#     n_early = n_total - n_post - n_mid

#     def sample_zone(mask, n):
#         idx = np.where(mask)[0]
#         if len(idx) == 0:
#             return np.array([], dtype=int)   # ← handle empty zone
#         if len(idx) <= n:
#             return idx
#         positions = np.round(np.linspace(0, len(idx) - 1, n)).astype(int)
#         return idx[positions]

#     early_idx = sample_zone(t < 0.5,                           n_early)
#     mid_idx   = sample_zone((t >= 0.5) & (t < warmstart_frac), n_mid)
#     post_idx  = sample_zone(t >= warmstart_frac,                n_post)

#     # always assigned now, even if some zones are empty
#     all_idx = np.sort(np.unique(np.concatenate([early_idx, mid_idx, post_idx])))
#     print(f"***** Subsampling: {len(all_idx)} points selected out of {len(t)}")

#     return t[all_idx], y[all_idx]


# def subsample_partial_target(t, y, n_total=200):
#     """
#     For partial target curves — subsample evenly across the observed window.
#     """
#     t = np.array(t)
#     y = np.array(y)

#     if len(t) <= n_total:
#         return t, y

#     positions = np.round(np.linspace(0, len(t) - 1, n_total)).astype(int)
#     positions = np.unique(positions)

#     print(f"  ✓ Partial target subsampling: {len(positions)} points evenly sampled out of {len(t)}")

#     return t[positions], y[positions]


# ============================================================
# Context curves
# ============================================================

def build_base_context_curves(
    df,
    target_n,
    context,
):

    curves = []
    subset = df[(df["target_N"] != target_n) & (df["target_N"] == context)]

    for _, row in subset.iterrows():

        val_loss = np.asarray(
            row["val_loss"],
            dtype=np.float64,
        )

        flops = np.asarray(
            row["flops"],
            dtype=np.float64,
        )

        idx = np.argsort(flops)

        flops = flops[idx]
        val_loss = val_loss[idx]

        y_norm = normalize_log_loss_curve(
            val_loss,
            df,
            target_N=target_n,
        )

        t_raw = flops / (flops[-1] + 1e-8)

        t_sub, y_sub = subsample_curve(
            t_raw,
            y_norm,
        )

        hp = normalize_hyperparameters(
            row,
            df,
        )

        curves.append(
            Curve(
                hyperparameters=hp,
                t=torch.tensor(
                    t_sub,
                    dtype=torch.float32,
                ),
                y=torch.tensor(
                    y_sub,
                    dtype=torch.float32,
                ),
            )
        )

        print(
            f"  ✓ run {row['base_N']}→{row['target_N']}: "
            f"{len(val_loss)} → {len(t_sub)} pts"
        )

    return curves


# ============================================================
# Partial target curve
# ============================================================

def build_partial_target_curve(
    df,
    row,
    observe_fraction,
    target_n,
):

    if observe_fraction <= 0.0:
        print("  ✓ No target observation (0%), skipping partial curve construction")
        return None
    val_loss = np.asarray(
        row["val_loss"],
        dtype=np.float64,
    )

    flops = np.asarray(
        row["flops"],
        dtype=np.float64,
    )

    idx = np.argsort(flops)

    flops = flops[idx]
    val_loss = val_loss[idx]

    t_flops = flops / (flops[-1] + 1e-8)

    n_observe = max(0, int(len(t_flops) * observe_fraction))

    # slice directly — no floating point boundary issues
    t_partial = t_flops[:n_observe]
    y_partial = val_loss[:n_observe]

    print(f"  ✓ Partial observation: {n_observe} points out of {len(t_flops)}")

    y_norm = normalize_log_loss_curve(
        y_partial,
        df,
        target_N=target_n,
    )

    t_sub, y_sub = subsample_curve(
        t_partial,
        y_norm,
    )

    hp = normalize_hyperparameters(
        row,
        df,
    )

    return Curve(
        hyperparameters=hp,
        t=torch.tensor(
            t_sub,
            dtype=torch.float32,
        ),
        y=torch.tensor(
            y_sub,
            dtype=torch.float32,
        ),
    )


# ============================================================
# Config key
# ============================================================

def make_config_key(row):

    return (
        f"base{row['base_N']}_"
        f"target{row['target_N']}_"
        f"lr{row['max_lr']}_"
        f"shrink{row['shrinking']}_"
        f"tkpm{row['tkpm']}_"
        f"emb{row['n_embd']}_"
        f"head{row['n_head']}_"
        f"gw{row['g_width']}_"
        f"gn{row['g_N']}"
    )


# ============================================================
# Save
# ============================================================

def save_prediction(
    path,
    meta,
    key,
    result,
):

    if os.path.exists(path):

        with open(path, "r") as f:
            data = json.load(f)

    else:

        data = {
            "meta": meta,
            "runs": {},
        }

    data["runs"][key] = result

    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
        )


# ============================================================
# Predict
# ============================================================
    

def predict(
    scale_pair,
    observe_fraction,
    df,
    device,
):

    print("\n" + "=" * 60)
    print(
        f"Predicting {scale_pair} @ {observe_fraction*100:.0f}%"
    )
    print("=" * 60)

    if isinstance(scale_pair, str):
        scale_pair = json.loads(scale_pair)

    context, target_n = scale_pair
    context_curves = []

    # context_curves = build_base_context_curves(
    #     df=df,
    #     target_n=target_n,
    #     context=context,
    # )

    print(
        f"  Total context curves: {len(context_curves)}"
    )

    target_subset = df[(df["target_N"] == target_n)
    ]

    if target_subset.empty:
        raise ValueError(
            f"No data for pair {scale_pair}"
        )

    model = FTPFN(
        version="0.0.1",
        device=device,
    )

    result = None

    for _, row in target_subset.iterrows():

        print(
            f"\nProcessing target run: {row['target_N']}"
        )
        base_n, target_n = row["base_N"], row["target_N"]

        partial_target = build_partial_target_curve(
            df,
            row,
            observe_fraction,
            target_n,
        )

        partial_added = context_curves + ([partial_target] if partial_target is not None else [])

        print(
            f"  Total context curves after adding target: "
            f"{len(partial_added)}"
        )

        total_points = len(row["flops"])

        # how many have we already observed
        observed_points = int(total_points * observe_fraction)

        # remaining points to predict = what we haven't seen yet
        remaining_points = total_points - observed_points

        query_hp = normalize_hyperparameters(row, df)
        t_query = torch.linspace(observe_fraction, 1.0, steps=remaining_points)

        query_curve = [
            Curve(
                hyperparameters=query_hp,
                t=t_query,
            )
        ]

        pred = model.predict(
            context=partial_added,
            query=query_curve,
        )[0]

        median = pred.quantile(0.5).tolist()
        q05 = pred.quantile(0.05).tolist()
        q95 = pred.quantile(0.95).tolist()

        final_pred = median[-1]

        config_key = make_config_key(row)

        context_dir = os.path.join(SAVE_DIR, f"{context}")
        os.makedirs(context_dir, exist_ok=True)

        out_file = os.path.join(
            context_dir,
            f"target{target_n}_obs{int(observe_fraction*100)}.json"
        )

        result = {
            "base_N":            base_n,
            "target_N":          target_n,
            "observe_fraction":  observe_fraction,
            "run_key":           config_key,
            "shrink":            row["shrinking"],
            "tkpm":              row["tkpm"],
            "max_lr":            row["max_lr"],
            "final_val_true":    float(row["val_loss"][-1]),   # ground truth
            "final_val_pred":    final_pred,                    # ifbo prediction
            "median":            median,
            "q05":               q05,
            "q95":               q95,
        }

        # save — one result per run inside the file
        if os.path.exists(out_file):
            with open(out_file, "r") as f:
                saved = json.load(f)
        else:
            saved = {}

        saved[config_key] = result

        with open(out_file, "w") as f:
            json.dump(saved, f, indent=2)
            
    return {
        "objective_to_minimize": final_pred,
        }