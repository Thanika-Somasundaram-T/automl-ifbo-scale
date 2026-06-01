"""Single-query FT-PFN prediction and output row construction."""

from __future__ import annotations

import json
from types import ModuleType

import numpy as np
import pandas as pd

from ifbo import Curve
from ifbo.surrogate import FTPFN

from demo_analysis._common.normalization import (
    loo_log_loss_range,
    loss_to_perf,
    perf_to_loss,
)

from experiments._lib.common import tensor_to_numpy


def predict_one(
    model: FTPFN,
    cache: dict[int, dict[str, object]],
    target_id: int,
    obs_frac: float,
    policy: str,
    partner_ids: list[int],
    registry: ModuleType,
) -> pd.DataFrame:
    target = cache[target_id]
    target_df = target["df"]
    row = target["row"]
    t = target["t"]
    y_raw = target["y_raw"]
    hp = target["hp"]

    observed_mask = t <= obs_frac
    if int(observed_mask.sum()) == 0:
        raise ValueError(f"No observed prefix points for trajectory {target_id} at obs_frac={obs_frac}")
    query_mask = t > obs_frac
    if int(query_mask.sum()) == 0:
        raise ValueError(f"No query points for trajectory {target_id} at obs_frac={obs_frac}")

    log_min, log_max = loo_log_loss_range(cache, target_id)
    normalization_scope = "loo_all_non_query_paws"

    context: list[Curve] = []
    for pid in partner_ids:
        partner = cache[pid]
        context.append(registry.context_curve_for_policy(partner, log_min, log_max, policy))

    observed_y = loss_to_perf(y_raw[observed_mask], log_min, log_max)
    context.append(Curve(hyperparameters=hp, t=t[observed_mask], y=observed_y))

    query_t = t[query_mask]
    prediction = model.predict(context=context, query=[Curve(hyperparameters=hp, t=query_t)])[0]

    perf_q05 = tensor_to_numpy(prediction.quantile(0.05))
    perf_q50 = tensor_to_numpy(prediction.quantile(0.5))
    perf_q95 = tensor_to_numpy(prediction.quantile(0.95))
    loss_q05 = perf_to_loss(perf_q05, log_min, log_max)
    loss_q50 = perf_to_loss(perf_q50, log_min, log_max)
    loss_q95 = perf_to_loss(perf_q95, log_min, log_max)
    pred_p05 = np.minimum(loss_q05, loss_q95)
    pred_p50 = loss_q50
    pred_p95 = np.maximum(loss_q05, loss_q95)

    query_target_n = float(row["target_N"])
    partner_rows = [cache[pid]["row"] for pid in partner_ids]
    partner_target_ns = [float(partner_row["target_N"]) for partner_row in partner_rows]
    partner_base_ns = [float(partner_row["base_N"]) for partner_row in partner_rows]
    n_lower_selected = sum(1 for pn in partner_target_ns if pn < query_target_n)
    n_lower_available = sum(
        1
        for tid, item in cache.items()
        if tid != target_id and float(item["row"]["target_N"]) < query_target_n
    )
    nearest_partner_n = max(partner_target_ns) if partner_target_ns else np.nan
    scale_gaps = [query_target_n / pn for pn in partner_target_ns if pn < query_target_n]
    min_gap = min(scale_gaps) if scale_gaps else np.nan
    max_gap = max(scale_gaps) if scale_gaps else np.nan

    query_df = target_df.loc[query_mask.detach().cpu().numpy()].copy()
    n_points = int(query_mask.sum())
    return pd.DataFrame({
        "policy": policy,
        "trajectory_id": target_id,
        "query_target_N": int(row["target_N"]),
        "base_N": int(row["base_N"]),
        "target_N": int(row["target_N"]),
        "G": float(row["G"]),
        "shrink": float(row["shrink"]),
        "tkpm": float(row["tkpm"]),
        "obs_frac": float(obs_frac),
        "context_size": len(context),
        "n_lower_available": n_lower_available,
        "n_lower_selected": n_lower_selected,
        "nearest_partner_target_N": nearest_partner_n,
        "min_scale_gap": min_gap,
        "max_scale_gap": max_gap,
        "partner_trajectory_ids": json.dumps([int(pid) for pid in partner_ids]),
        "partner_target_Ns": json.dumps([int(pn) for pn in partner_target_ns]),
        "partner_base_Ns": json.dumps([int(pn) for pn in partner_base_ns]),
        "target_N_norm": float(row["target_N_norm"]),
        "G_norm": float(row["G_norm"]),
        "shrink_norm": float(row["shrink_norm"]),
        "tkpm_norm": float(row["tkpm_norm"]),
        "hp_is_all_zero": bool(float(hp.abs().sum()) == 0.0),
        "normalization_scope": normalization_scope,
        "loss_log_min": log_min,
        "loss_log_max": log_max,
        "x_idx": query_df["x_idx"].to_numpy(dtype=int),
        "x_norm": query_df["x_norm"].to_numpy(dtype=float),
        "y_true": query_df["val_loss_interp"].to_numpy(dtype=float),
        "pred_p05": pred_p05[:n_points],
        "pred_p50": pred_p50[:n_points],
        "pred_p95": pred_p95[:n_points],
    })
