"""Shared parquet shard helpers for experiment prediction outputs."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

INPROGRESS_SUFFIX = ".inprogress.parquet"


def shard_tag(obs_frac: float) -> str:
    return f"{int(round(obs_frac * 1000)):03d}"


def inprogress_path(partial_dir: Path, policy: str, tag: str) -> Path:
    return partial_dir / f"{policy}__obs{tag}{INPROGRESS_SUFFIX}"


def load_done_trajectory_ids(inprogress: Path) -> set[int]:
    """Trajectory ids already persisted in a mid-shard in-progress file."""
    if not inprogress.exists():
        return set()
    try:
        done = pd.read_parquet(inprogress, columns=["trajectory_id"])
    except Exception:
        return set()
    return set(int(t) for t in done["trajectory_id"].unique())


def flush_inprogress(frames: list[pd.DataFrame], inprogress: Path) -> None:
    """Atomically write the cumulative completed frames to the in-progress file.

    Writes to a temp path then os.replace so a crash mid-flush never corrupts
    the in-progress shard (worst case: it reflects the previous flush).
    """
    if not frames:
        return
    tmp = inprogress.with_suffix(".tmp")
    pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False)
    os.replace(tmp, inprogress)


def finalize_shard(inprogress: Path, shard: Path) -> None:
    """Promote a completed in-progress file to the final shard, atomically."""
    os.replace(inprogress, shard)


def merge_shards(partial_dir: Path, results_dir: Path, out_name: str) -> None:
    # Only final shards — never partial/in-progress files (would double-count
    # or inject incomplete trajectories into the merged predictions).
    shards = sorted(
        p for p in partial_dir.glob("*.parquet")
        if not p.name.endswith(INPROGRESS_SUFFIX)
    )
    if not shards:
        return
    pd.concat([pd.read_parquet(path) for path in shards], ignore_index=True).to_parquet(
        results_dir / out_name,
        index=False,
    )
