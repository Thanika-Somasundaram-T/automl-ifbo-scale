"""Shared constants, defaults, and utility helpers for width-transfer experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils import get_device

DEFAULT_OBS_FRACS = [0.05, 0.10, 0.20, 0.50]
DEFAULT_PROCESSED_DIR = Path("experiments/processed_flattened")
DEFAULT_OUTPUT_DIR = Path("experiments/width_transfer")
DEFAULT_MODEL_PATH = Path(".model")
DEFAULT_MODEL_VERSION = "0.0.1"
DEFAULT_RESAMPLED_K = 256

GEOM_SPARSE_K = 8


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return get_device()
    return torch.device(name)


def minmax(values: pd.Series) -> tuple[pd.Series, dict[str, float]]:
    lo = float(values.min())
    hi = float(values.max())
    if math.isclose(lo, hi):
        return pd.Series(np.zeros(len(values)), index=values.index), {"min": lo, "max": hi}
    return (values - lo) / (hi - lo), {"min": lo, "max": hi}


def tensor_to_numpy(values: torch.Tensor | list[float]) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def check_invariants(predictions: pd.DataFrame, lower_only_policies: set[str]) -> None:
    violations = predictions[predictions["x_norm"] <= predictions["obs_frac"]]
    if len(violations) > 0:
        raise RuntimeError(f"INVARIANT VIOLATION: {len(violations)} rows with x_norm <= obs_frac")

    quantile_violations = predictions[
        (predictions["pred_p05"] > predictions["pred_p50"])
        | (predictions["pred_p50"] > predictions["pred_p95"])
    ]
    if len(quantile_violations) > 0:
        raise RuntimeError(
            f"INVARIANT VIOLATION: {len(quantile_violations)} rows with unordered prediction quantiles"
        )

    hp_zero = predictions[predictions["hp_is_all_zero"]]
    if len(hp_zero) > 0:
        raise RuntimeError(f"INVARIANT VIOLATION: {len(hp_zero)} rows with all-zero HP tensor")

    lower_only_preds = predictions[predictions["policy"].isin(lower_only_policies)]
    for _, row in lower_only_preds[["query_target_N", "trajectory_id", "partner_trajectory_ids", "partner_target_Ns"]].drop_duplicates().iterrows():
        partner_ids = json.loads(row["partner_trajectory_ids"])
        partner_targets = json.loads(row["partner_target_Ns"])
        if any(int(pid) == int(row["trajectory_id"]) for pid in partner_ids):
            raise RuntimeError("INVARIANT VIOLATION: lower-only context includes query trajectory_id")
        if any(float(partner_n) >= float(row["query_target_N"]) for partner_n in partner_targets):
            raise RuntimeError("INVARIANT VIOLATION: lower-only context includes non-lower target_N")
