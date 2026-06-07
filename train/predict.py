from math import log
import os
import json

import torch
import pandas as pd
import numpy as np

from ifbo.surrogate import FTPFN
from ifbo import Curve


# ============================================================
# Config
# ============================================================

DATA_PATH = "./experiments.json"

SAVE_DIR = "./a5_predicting_135_only_feeding_135"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Device
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Load data
# ============================================================

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


# ============================================================
# Loss normalization
# ============================================================

def normalize_log_loss_curve(val_loss_list, global_min, global_max):

    val_loss = np.asarray(val_loss_list, dtype=np.float64)

    log_curve = np.log(val_loss + 1e-8)
    log_min = np.log(global_min + 1e-8)
    log_max = np.log(global_max + 1e-8)

    if abs(log_max - log_min) < 1e-8:
        return np.zeros_like(log_curve)

    norm = (log_curve - log_min) / (log_max - log_min)

    return np.clip(norm, 0.0, 1.0)


# ============================================================
# Hyperparameter normalization
# ============================================================

def normalize_hyperparameters(row, df):

    def get_minmax(col):
        return df[col].min(), df[col].max()

    b_min, b_max = get_minmax("base_N")
    t_min, t_max = get_minmax("target_N")

    lr_min, lr_max = get_minmax("max_lr")

    sh_min, sh_max = get_minmax("shrinking")
    tk_min, tk_max = get_minmax("tkpm")

    emb_min, emb_max = get_minmax("n_embd")
    head_min, head_max = get_minmax("n_head")

    gw_min, gw_max = get_minmax("g_width")
    gn_min, gn_max = get_minmax("g_N")

    base_norm = (
        (np.log2(row["base_N"]) - np.log2(b_min))
        / (np.log2(b_max) - np.log2(b_min) + 1e-8)
    )

    target_norm = (
        (np.log2(row["target_N"]) - np.log2(t_min))
        / (np.log2(t_max) - np.log2(t_min) + 1e-8)
    )

    lr_norm = (
        (np.log10(row["max_lr"]) - np.log10(lr_min))
        / (np.log10(lr_max) - np.log10(lr_min) + 1e-8)
    )

    shrink_norm = (
        (row["shrinking"] - sh_min)
        / (sh_max - sh_min + 1e-8)
    )

    tkpm_norm = (
        (row["tkpm"] - tk_min)
        / (tk_max - tk_min + 1e-8)
    )

    emb_norm = (
        (row["n_embd"] - emb_min)
        / (emb_max - emb_min + 1e-8)
    )

    head_norm = (
        (row["n_head"] - head_min)
        / (head_max - head_min + 1e-8)
    )

    gw_norm = (
        (row["g_width"] - gw_min)
        / (gw_max - gw_min + 1e-8)
    )

    gn_norm = (
        (row["g_N"] - gn_min)
        / (gn_max - gn_min + 1e-8)
    )

    return torch.tensor(
        [
            base_norm,
            target_norm,
            lr_norm,
            shrink_norm,
            tkpm_norm,
            emb_norm,
            head_norm,
            gw_norm,
            gn_norm,
        ],
        dtype=torch.float32,
    )


# ============================================================
# Subsampling
# ============================================================

def subsample_curve(t, y, n_total=200, warmstart_frac=0.7):
    t = np.array(t)
    y = np.array(y)
    
    if len(t) <= n_total:
        return t, y
    
    n_post  = int(n_total * 0.50)
    n_mid   = int(n_total * 0.25)
    n_early = n_total - n_post - n_mid

    def sample_zone(mask, n):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)   # ← handle empty zone
        if len(idx) <= n:
            return idx
        positions = np.round(np.linspace(0, len(idx) - 1, n)).astype(int)
        return idx[positions]

    early_idx = sample_zone(t < 0.5,                           n_early)
    mid_idx   = sample_zone((t >= 0.5) & (t < warmstart_frac), n_mid)
    post_idx  = sample_zone(t >= warmstart_frac,                n_post)

    # always assigned now, even if some zones are empty
    all_idx = np.sort(np.unique(np.concatenate([early_idx, mid_idx, post_idx])))
    print(f"***** Subsampling: {len(all_idx)} points selected out of {len(t)}")

    return t[all_idx], y[all_idx]


def subsample_partial_target(t, y, n_total=200):
    """
    For partial target curves — subsample evenly across the observed window.
    """
    t = np.array(t)
    y = np.array(y)

    if len(t) <= n_total:
        return t, y

    positions = np.round(np.linspace(0, len(t) - 1, n_total)).astype(int)
    positions = np.unique(positions)

    print(f"  ✓ Partial target subsampling: {len(positions)} points evenly sampled out of {len(t)}")

    return t[positions], y[positions]


# ============================================================
# Context curves
# ============================================================

def build_base_context_curves(
    df,
    base_n,
    target_n,
    global_min,
    global_max,
):

    curves = []
    subset = df[(df["target_N"] == target_n)]

    for _, row in subset.iterrows():
        
        print(f"  ✓ Processing run: {row['run_key']}")

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
            global_min,
            global_max,
        )

        t_raw = flops / (flops[-1] + 1e-8)

        print(
            f"  ✓ FLOPs time axis: {t_raw[0]} ... {t_raw[-1]}"
        )

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
    global_min,
    global_max,
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

    # mask = t_flops <= observe_fraction

    # t_partial = t_flops[mask]
    # y_partial = val_loss[mask]

    # print(
    #     f"  ✓ Partial observation: {len(t_partial)} points out of {len(flops)}"
    # )

    # if len(t_partial) < 2:

    #     print(
    #         "  ⚠️ Not enough points in partial curve, using first 2 points as fallback"
    #     )

    #     t_partial = t_flops[:2]
    #     y_partial = val_loss[:2]
        
    # how many points to observe based on fraction
    n_observe = max(0, int(len(t_flops) * observe_fraction))

    # slice directly — no floating point boundary issues
    t_partial = t_flops[:n_observe]
    y_partial = val_loss[:n_observe]

    print(f"  ✓ Partial observation: {n_observe} points out of {len(t_flops)}")

    y_norm = normalize_log_loss_curve(
        y_partial,
        global_min,
        global_max,
    )

    t_sub, y_sub = subsample_partial_target(
        t_partial,
        y_norm,
    )

    hp = normalize_hyperparameters(
        row,
        df,
    )

    print(
        f"  ✓ target_N partial: "
        f"{len(t_partial)}/{len(flops)} points "
        f"({observe_fraction*100:.0f}% FLOPs)"
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
    global_min,
    global_max,
):

    print("\n" + "=" * 60)
    print(
        f"Predicting {scale_pair} @ {observe_fraction*100:.0f}%"
    )
    print("=" * 60)

    if isinstance(scale_pair, str):
        scale_pair = json.loads(scale_pair)

    base_n, target_n = scale_pair

    context_curves = build_base_context_curves(
        df=df,
        base_n=14562560,
        target_n=target_n,
        global_min=global_min,
        global_max=global_max,
    )

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
            "testttttt      ",
            row["run_key"],
        )

        partial_target = build_partial_target_curve(
            df,
            row,
            observe_fraction,
            global_min,
            global_max,
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

        print(
            f"  → Predicted final val loss: {final_pred:.4f}"
        )

        print(
            f"  → 90% CI: [{q05[-1]:.4f}, {q95[-1]:.4f}]"
        )

        config_key = make_config_key(row)

        # cleaner filename — one file per scale pair + observation level
        out_file = os.path.join(
            SAVE_DIR,
            f"base{base_n}_target{target_n}_obs{int(observe_fraction*100)}.json"
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