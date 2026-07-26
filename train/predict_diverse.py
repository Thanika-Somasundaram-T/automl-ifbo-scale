"""
FT-PFN prediction under several context-selection policies, all sharing the
same "always include the query's own partial curve" core:

  - diverse_lower_k: k HP-nearest curves picked round-robin across lower
    scales (train.context_policy.select_diverse_lower_k).
  - all_lower: every run at every scale strictly below the target.
  - nearest_lower: every run at just the single nearest scale below target.
  - baseline: no other-scale context at all -- just the query's own partial
    curve.
  - interpolation: every run at every scale OTHER than the target's own
    (lower and higher), excluding other configs' full curves at the
    target's own scale.

subsample_mode selects how every context curve (other-scale curves and the
query's own partial curve) is reduced to 10 points:
  - "uniform":   evenly spaced in time (utils.subsample_curve_uniform)
  - "geometric": denser near the start of training (utils.subsample_curve_geometric)
"""
import numpy as np
import torch

from train.context_policy import (
    select_diverse_lower_k,
    select_all_lower,
    select_nearest_lower,
    select_interpolation_context,
)
from utils import (
    get_device,
    normalize_hyperparameters,
    normalize_log_loss_curve,
    subsample_curve_uniform,
    subsample_curve_geometric,
)

from ifbo.surrogate import FTPFN
from ifbo import Curve

TARGET_EPOCHS = 100
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

SUBSAMPLERS = {
    "uniform": subsample_curve_uniform,
    "geometric": subsample_curve_geometric,
}


def _build_full_curve(run_data, subsampler):
    val_loss = np.asarray(run_data.get("val_loss_curve", []), dtype=np.float64)[:TARGET_EPOCHS]
    if len(val_loss) == 0:
        return None
    t = np.linspace(0.0, 1.0, len(val_loss))
    y_norm = normalize_log_loss_curve(val_loss)
    return subsampler(t, y_norm)


def _build_partial_curve(run_data, observe_fraction, subsampler):
    val_loss = np.asarray(run_data.get("val_loss_curve", []), dtype=np.float64)[:TARGET_EPOCHS]
    if len(val_loss) == 0 or observe_fraction <= 0:
        return None

    n_observe = int(len(val_loss) * observe_fraction)
    if n_observe == 0:
        return None

    y_partial = val_loss[:n_observe]
    t_partial = np.linspace(0.0, 1.0, len(val_loss))[:n_observe]
    y_norm = normalize_log_loss_curve(y_partial)
    return subsampler(t_partial, y_norm)


def _find_query_run(all_results, lr, hidden_dim, weight_decay):
    for run_data in all_results.values():
        if (
            run_data.get("lr") == lr
            and run_data.get("hidden_dim") == hidden_dim
            and run_data.get("weight_decay") == weight_decay
        ):
            return run_data
    return None


def _predict_with_context_runs(
    all_results,
    lr,
    hidden_dim,
    weight_decay,
    epochs,
    context_runs,
    subsample_mode,
    device=None,
    ft_pfn_model=None,
):
    """Shared core: builds context curves from context_runs (list of run_data
    dicts, full curves) + the query's own partial curve, then predicts.
    Returns {"point": [...], "quantiles": {"0.05": [...], ...}} or None.
    """
    subsampler = SUBSAMPLERS[subsample_mode]
    device = device or get_device()

    context_curves = []

    for run_data in context_runs:
        built = _build_full_curve(run_data, subsampler)
        if built is None:
            continue
        t_sub, y_sub = built
        hp = normalize_hyperparameters(
            run_data["lr"], run_data["hidden_dim"], run_data["weight_decay"]
        )
        context_curves.append(
            Curve(
                hyperparameters=hp,
                t=torch.tensor(t_sub, dtype=torch.float32),
                y=torch.tensor(y_sub, dtype=torch.float32),
            )
        )

    query_run = _find_query_run(all_results, lr, hidden_dim, weight_decay)
    if query_run is None:
        return None

    observe_fraction = float(epochs) / float(TARGET_EPOCHS)
    partial = _build_partial_curve(query_run, observe_fraction, subsampler)
    if partial is not None:
        t_sub, y_sub = partial
        hp = normalize_hyperparameters(lr, hidden_dim, weight_decay)
        context_curves.append(
            Curve(
                hyperparameters=hp,
                t=torch.tensor(t_sub, dtype=torch.float32),
                y=torch.tensor(y_sub, dtype=torch.float32),
            )
        )

    if len(context_curves) == 0:
        return None

    remaining_steps = max(TARGET_EPOCHS - epochs, 1)
    t_target = torch.linspace(observe_fraction, 1.0, steps=remaining_steps)
    query_hp = normalize_hyperparameters(lr, hidden_dim, weight_decay)
    query_curve = [Curve(hyperparameters=query_hp, t=t_target)]

    ft_pfn_model = ft_pfn_model or FTPFN(version="0.0.1", device=device)
    prediction = ft_pfn_model.predict(context=context_curves, query=query_curve)[0]

    quantiles = {}
    for q in QUANTILE_LEVELS:
        val = prediction.quantile(q).squeeze().tolist()
        if isinstance(val, float):
            val = [val]
        quantiles[f"{q:.2f}"] = val

    return {
        "point": quantiles["0.50"],
        "quantiles": quantiles,
    }


def predict_diverse_lower_k(
    all_results, lr, hidden_dim, weight_decay, epochs, k, subsample_mode,
    device=None, ft_pfn_model=None,
):
    context_runs = [
        run_data
        for _dist, _run_key, run_data in select_diverse_lower_k(
            all_results, hidden_dim, lr, weight_decay, k
        )
    ]
    return _predict_with_context_runs(
        all_results, lr, hidden_dim, weight_decay, epochs, context_runs,
        subsample_mode, device, ft_pfn_model,
    )


def predict_all_lower(
    all_results, lr, hidden_dim, weight_decay, epochs, subsample_mode,
    device=None, ft_pfn_model=None,
):
    context_runs = select_all_lower(all_results, hidden_dim)
    return _predict_with_context_runs(
        all_results, lr, hidden_dim, weight_decay, epochs, context_runs,
        subsample_mode, device, ft_pfn_model,
    )


def predict_nearest_lower(
    all_results, lr, hidden_dim, weight_decay, epochs, subsample_mode,
    device=None, ft_pfn_model=None,
):
    context_runs = select_nearest_lower(all_results, hidden_dim)
    return _predict_with_context_runs(
        all_results, lr, hidden_dim, weight_decay, epochs, context_runs,
        subsample_mode, device, ft_pfn_model,
    )


def predict_baseline(
    all_results, lr, hidden_dim, weight_decay, epochs, subsample_mode,
    device=None, ft_pfn_model=None,
):
    return _predict_with_context_runs(
        all_results, lr, hidden_dim, weight_decay, epochs, [],
        subsample_mode, device, ft_pfn_model,
    )


def predict_interpolation(
    all_results, lr, hidden_dim, weight_decay, epochs, subsample_mode,
    device=None, ft_pfn_model=None,
):
    context_runs = select_interpolation_context(all_results, hidden_dim)
    return _predict_with_context_runs(
        all_results, lr, hidden_dim, weight_decay, epochs, context_runs,
        subsample_mode, device, ft_pfn_model,
    )
