"""Ex3 policy registry, partner selection, and context-curve shaping."""

from __future__ import annotations

import numpy as np
import torch

from ifbo import Curve

from demo_analysis._common.normalization import loss_to_perf
from experiments._lib.common import GEOM_SPARSE_K

POLICIES_TIER1 = ["prefix_only", "context_ifbo", "all_lower_only"]
POLICIES_TIER2 = ["nearest_lower_only", "lower_final_pt_only", "lower_geom_sparse_k8"]
POLICIES_TIER3 = ["lower_shrink_lt_1", "lower_high_tkpm_cold_start"]
POLICIES_TIER4 = ["random_lower_k", "lower_start_final_pt", "all_context_oracle"]
POLICIES_ALL = POLICIES_TIER1 + POLICIES_TIER2 + POLICIES_TIER3 + POLICIES_TIER4

LOWER_ONLY_POLICIES = {
    "all_lower_only",
    "nearest_lower_only",
    "lower_final_pt_only",
    "lower_geom_sparse_k8",
    "lower_shrink_lt_1",
    "lower_high_tkpm_cold_start",
    "random_lower_k",
    "lower_start_final_pt",
}


def resolve_policies_from_tier(tier: str | None, explicit: list[str] | None) -> list[str]:
    if tier is None:
        return list(explicit) if explicit else list(POLICIES_ALL)
    if tier == "1":
        return list(POLICIES_TIER1)
    if tier == "1+2":
        return POLICIES_TIER1 + POLICIES_TIER2
    if tier == "1+2+3":
        return POLICIES_TIER1 + POLICIES_TIER2 + POLICIES_TIER3
    return list(POLICIES_ALL)


def requires_partners(policy: str) -> bool:
    return policy in LOWER_ONLY_POLICIES


def select_context_partners(
    cache: dict[int, dict[str, object]],
    target_id: int,
    target_row,
    policy: str,
    random_k: int | None,
    rng: np.random.RandomState,
) -> list[int]:
    query_target_n = float(target_row["target_N"])

    if policy == "prefix_only":
        return []

    if policy in ("context_ifbo", "all_context_oracle"):
        return [tid for tid in cache if tid != target_id]

    candidates = [
        tid for tid, item in cache.items()
        if tid != target_id and float(item["row"]["target_N"]) < query_target_n
    ]

    if not candidates:
        return []

    if policy in ("all_lower_only", "lower_final_pt_only", "lower_geom_sparse_k8", "lower_start_final_pt"):
        return candidates

    if policy == "nearest_lower_only":
        max_partner_n = max(float(cache[tid]["row"]["target_N"]) for tid in candidates)
        return [tid for tid in candidates if float(cache[tid]["row"]["target_N"]) == max_partner_n]

    if policy == "random_lower_k":
        max_partner_n = max(float(cache[tid]["row"]["target_N"]) for tid in candidates)
        nearest_count = sum(
            1 for tid in candidates if float(cache[tid]["row"]["target_N"]) == max_partner_n
        )
        k = random_k if random_k is not None else nearest_count
        k = min(k, len(candidates))
        chosen = rng.choice(candidates, size=k, replace=False)
        return [int(x) for x in chosen]

    if policy == "lower_shrink_lt_1":
        return [tid for tid in candidates if float(cache[tid]["row"]["shrink"]) < 1.0]

    if policy == "lower_high_tkpm_cold_start":
        return [tid for tid in candidates if float(cache[tid]["row"]["tkpm"]) == 30.0]

    raise ValueError(f"Unknown policy: {policy}")


def geom_sparse_indices(n_points: int, k: int) -> np.ndarray:
    if n_points <= k:
        return np.arange(n_points)
    log_idx = np.geomspace(1, n_points, num=k)
    idx = np.unique(np.clip(np.round(log_idx).astype(int) - 1, 0, n_points - 1))
    return idx


def context_curve_for_policy(item: dict[str, object], log_min: float, log_max: float, policy: str) -> Curve:
    context_y = loss_to_perf(item["y_raw"], log_min, log_max)
    t = item["t"]
    hp = item["hp"]
    if policy == "lower_final_pt_only":
        return Curve(hyperparameters=hp, t=t[-1:], y=context_y[-1:])
    if policy == "lower_start_final_pt":
        return Curve(
            hyperparameters=hp,
            t=torch.stack([t[0], t[-1]]),
            y=torch.stack([context_y[0], context_y[-1]]),
        )
    if policy == "lower_geom_sparse_k8":
        idx = geom_sparse_indices(len(t), GEOM_SPARSE_K)
        idx_t = torch.tensor(idx, dtype=torch.long)
        return Curve(hyperparameters=hp, t=t[idx_t], y=context_y[idx_t])
    return Curve(hyperparameters=hp, t=t, y=context_y)
