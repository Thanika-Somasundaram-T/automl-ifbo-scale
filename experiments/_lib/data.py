"""data loading, PAWS filtering, HP normalization, and curve cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments._lib.common import minmax


def load_processed(processed_dir: Path, resampled_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = processed_dir / "trajectory_summary.parquet"
    curves_path = processed_dir / f"resampled_flops_k{resampled_k}.parquet"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not curves_path.exists():
        raise FileNotFoundError(f"Missing {curves_path}")
    summary = pd.read_parquet(summary_path)
    curves = pd.read_parquet(curves_path)
    return summary, curves


def filter_paws(summary: pd.DataFrame) -> pd.DataFrame:
    before = len(summary)
    filtered = summary[summary["method"] == "paws"].copy()
    after = len(filtered)
    print(f"PAWS filter: {before} → {after} trajectories")
    filtered["G"] = filtered["target_N"] / filtered["base_N"]
    return filtered.sort_values("trajectory_id").reset_index(drop=True)


def add_normalized_hps(summary: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    out = summary.copy()
    ranges: dict[str, dict[str, float]] = {}
    out["target_N_hp"] = np.log10(out["target_N"])
    out["G_hp"] = np.log10(out["G"])
    out["shrink_hp"] = out["shrink"]
    out["tkpm_hp"] = out["tkpm"]
    for src, dst in [
        ("target_N_hp", "target_N_norm"),
        ("G_hp", "G_norm"),
        ("shrink_hp", "shrink_norm"),
        ("tkpm_hp", "tkpm_norm"),
    ]:
        out[dst], ranges[dst] = minmax(out[src].astype(float))
    return out, ranges


def hp_tensor(row: pd.Series) -> torch.Tensor:
    return torch.tensor(
        [row["target_N_norm"], row["G_norm"], row["shrink_norm"], row["tkpm_norm"]],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)


def build_curve_cache(summary: pd.DataFrame, curves: pd.DataFrame) -> dict[int, dict[str, object]]:
    merged = curves[curves["trajectory_id"].isin(summary["trajectory_id"])].copy()
    summary_by_id = summary.set_index("trajectory_id")
    cache: dict[int, dict[str, object]] = {}
    for trajectory_id, curve_df in merged.groupby("trajectory_id", sort=True):
        curve_df = curve_df.sort_values("x_idx")
        row = summary_by_id.loc[trajectory_id]
        t = torch.tensor(curve_df["x_norm"].to_numpy(dtype=np.float32), dtype=torch.float32)
        y_raw = torch.tensor(curve_df["val_loss_interp"].to_numpy(dtype=np.float32), dtype=torch.float32)
        hp = hp_tensor(row)
        cache[int(trajectory_id)] = {
            "row": row,
            "df": curve_df,
            "t": t,
            "y_raw": y_raw,
            "hp": hp,
        }
    return cache
