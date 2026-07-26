"""
diverse_lower_k context-selection policy.

For a query (lr, hidden_dim, weight_decay), pick the k lower-scale runs
whose HPs are closest (Euclidean, on normalize_hyperparameters) to the
query, round-robining across lower scales nearest-to-target first so the
context stays diverse across scales instead of piling onto one.
"""
import numpy as np

from utils import normalize_hyperparameters


def lower_scales_nearest_first(all_scales, target_scale):
    """Distinct scales strictly below target_scale, largest (nearest) first."""
    return sorted({s for s in all_scales if s < target_scale}, reverse=True)


def select_diverse_lower_k(all_results, target_hidden_dim, query_lr, query_wd, k):
    """
    Returns up to k (distance, run_key, run_data) tuples, sorted by the
    round-robin selection order (not by distance): nearest-scale-first,
    then next-closest HP match within that scale, cycling through scales.
    """
    all_scales = {run_data.get("hidden_dim") for run_data in all_results.values()}
    lower_scales = lower_scales_nearest_first(all_scales, target_hidden_dim)
    if not lower_scales:
        return []

    query_hp = normalize_hyperparameters(
        query_lr, target_hidden_dim, query_wd
    ).numpy()

    ranked_per_scale = {}
    for scale in lower_scales:
        candidates = []
        for run_key, run_data in all_results.items():
            if run_data.get("hidden_dim") != scale:
                continue
            if len(run_data.get("val_loss_curve", [])) == 0:
                continue

            cand_hp = normalize_hyperparameters(
                run_data["lr"], run_data["hidden_dim"], run_data["weight_decay"]
            ).numpy()
            dist = float(np.linalg.norm(cand_hp - query_hp))
            candidates.append((dist, run_key, run_data))

        candidates.sort(key=lambda c: c[0])
        ranked_per_scale[scale] = candidates

    selected = []
    rank = 0
    while len(selected) < k:
        added_this_round = False
        for scale in lower_scales:
            pool = ranked_per_scale[scale]
            if rank < len(pool):
                selected.append(pool[rank])
                added_this_round = True
                if len(selected) == k:
                    break
        if not added_this_round:
            break
        rank += 1

    return selected


def _has_curve(run_data):
    return len(run_data.get("val_loss_curve", [])) > 0


def select_all_lower(all_results, target_hidden_dim):
    """Every run at every scale strictly below the target (no HP filtering)."""
    return [
        run_data
        for run_data in all_results.values()
        if run_data.get("hidden_dim") is not None
        and run_data["hidden_dim"] < target_hidden_dim
        and _has_curve(run_data)
    ]


def select_nearest_lower(all_results, target_hidden_dim):
    """Every run at just the single nearest scale below the target."""
    all_scales = {run_data.get("hidden_dim") for run_data in all_results.values()}
    lower = [s for s in all_scales if s is not None and s < target_hidden_dim]
    if not lower:
        return []
    nearest = max(lower)
    return [
        run_data
        for run_data in all_results.values()
        if run_data.get("hidden_dim") == nearest and _has_curve(run_data)
    ]


def select_interpolation_context(all_results, target_hidden_dim):
    """
    Every run at every scale other than the target's own scale (lower AND
    higher). Never includes another config's full curve at the target's own
    hidden_dim -- only the query's own partial curve represents that scale.
    """
    return [
        run_data
        for run_data in all_results.values()
        if run_data.get("hidden_dim") != target_hidden_dim and _has_curve(run_data)
    ]
