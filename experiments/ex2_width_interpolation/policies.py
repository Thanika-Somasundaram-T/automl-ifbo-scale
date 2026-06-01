"""Ex2 policy registry for width interpolation."""

from __future__ import annotations

import numpy as np

from ifbo import Curve

from demo_analysis._common.normalization import loss_to_perf

POLICIES_PRIMARY = ["bracket_nearest"]
POLICIES_BASELINES = ["prefix_only", "all_context_oracle"]
POLICIES_ALL = POLICIES_PRIMARY + POLICIES_BASELINES
LOWER_ONLY_POLICIES: set[str] = set()


def resolve_policies_from_tier(tier: str | None, explicit: list[str] | None) -> list[str]:
    if tier is None:
        return list(explicit) if explicit else list(POLICIES_ALL)
    if tier == "primary":
        return list(POLICIES_PRIMARY)
    if tier == "baselines":
        return list(POLICIES_BASELINES)
    return list(POLICIES_ALL)


def requires_partners(policy: str) -> bool:
    return policy in {"bracket_nearest", "all_context_oracle"}


def select_context_partners(
    cache: dict[int, dict[str, object]],
    target_id: int,
    target_row,
    policy: str,
    random_k: int | None,
    rng: np.random.RandomState,
) -> list[int]:
    del random_k, rng
    query_target_n = float(target_row["target_N"])

    if policy == "prefix_only":
        return []

    if policy == "all_context_oracle":
        return [tid for tid in cache if tid != target_id]

    if policy == "bracket_nearest":
        lower = [
            tid for tid in cache
            if tid != target_id and float(cache[tid]["row"]["target_N"]) < query_target_n
        ]
        upper = [
            tid for tid in cache
            if tid != target_id and float(cache[tid]["row"]["target_N"]) > query_target_n
        ]
        if not lower or not upper:
            return []
        nearest_below = max(float(cache[tid]["row"]["target_N"]) for tid in lower)
        nearest_above = min(float(cache[tid]["row"]["target_N"]) for tid in upper)
        return (
            [tid for tid in lower if float(cache[tid]["row"]["target_N"]) == nearest_below]
            + [tid for tid in upper if float(cache[tid]["row"]["target_N"]) == nearest_above]
        )

    raise ValueError(f"Unknown policy: {policy}")


def context_curve_for_policy(item: dict[str, object], log_min: float, log_max: float, policy: str) -> Curve:
    del policy
    context_y = loss_to_perf(item["y_raw"], log_min, log_max)
    return Curve(hyperparameters=item["hp"], t=item["t"], y=context_y)
