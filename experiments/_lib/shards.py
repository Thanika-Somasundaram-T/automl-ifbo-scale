"""Shared parquet shard helpers for experiment prediction outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def shard_tag(obs_frac: float) -> str:
    return f"{int(round(obs_frac * 1000)):03d}"


def merge_shards(partial_dir: Path, results_dir: Path, out_name: str) -> None:
    shards = sorted(partial_dir.glob("*.parquet"))
    if not shards:
        return
    pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True).to_parquet(
        results_dir / out_name,
        index=False,
    )
