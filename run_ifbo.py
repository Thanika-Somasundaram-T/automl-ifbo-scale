"""
Warm-start ifBO for the MLP pipeline_space, targeting hidden_dim=128.

- Context: every run in result_metrics.json with hidden_dim != 128,
  subsampled to 10 points each (uniform linspace over epoch index).
  Raw val_loss values were checked and already lie in [0,1], so no
  normalization is applied -- passed straight into
  objective_to_minimize. encode_ftpfn does its own internal 1-loss
  flip for the maximize-space criterion; that needs no action here.
- Target: hidden_dim == 128 only. evaluate_pipeline does an exact
  lookup into result_metrics.json (all fields are categorical/discrete,
  so every candidate ifBO proposes has a real match -- no nearest-
  neighbor needed).
"""

import json
import os
import neps
import numpy as np

from ifbo_fixed_scale import build_fixed_scale_ifbo

from neps.state.pipeline_eval import UserResultDict

# ============================================================
# Config
# ============================================================

DATA_PATH = "./ground_truth/results_metrics.json"
ROOT_DIRECTORY = "./results_hd128"
TARGET_HIDDEN_DIM = 128
FIDELITY_NAME = "epochs"
N_SUBSAMPLE = 10
MAX_EVALUATIONS_TOTAL = 5


# ============================================================
# Load data
# ============================================================

def load_data(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# NOTE: raw val_loss values were checked and already lie in [0, 1]
# (observed range ~0.57-0.86), so no normalization is applied -- we
# pass val_loss straight into objective_to_minimize. encode_ftpfn
# still does its own internal 1-loss flip for the maximize-space
# criterion; that's unrelated to this and needs no action here.
# If you ever see a value outside [0,1] (e.g. from a different loss
# type or an outlier early-training spike), you'll get an explicit
# RuntimeError from encode_ftpfn -- that's your signal normalization
# is needed again, not something to guard against pre-emptively.
# ============================================================
# Uniform 10-point subsample (by index, matches your existing pattern)
# ============================================================

def subsample_uniform(arr, n=N_SUBSAMPLE):
    if len(arr) <= n:
        return np.arange(len(arr))
    return np.round(np.linspace(0, len(arr) - 1, n)).astype(int)


# ============================================================
# Pipeline space
# ============================================================

def build_pipeline_space():
    return neps.SearchSpace({
        "lr": neps.HPOCategorical(choices=[3e-3, 1e-3, 3e-4, 1e-4]),
        "weight_decay": neps.HPOCategorical(choices=[0.0, 1e-3]),
        "hidden_dim": neps.HPOCategorical(choices=[24, 32, 64, 128, 256]),
        FIDELITY_NAME: neps.HPOInteger(lower=1, upper=100, is_fidelity=True),
        "num_layers": neps.HPOConstant(value=4),
        "lr_schedule": neps.HPOConstant(value="none"),
        "batch_size": neps.HPOConstant(value=64),
    })


# ============================================================
# Build injected context trials (hidden_dim != target)
# ============================================================

def build_context_trials(data, target_hd):
    evaluated_trials = []
    for key, entry in data.items():
        if entry["hidden_dim"] == target_hd:
            continue  # this is target-scale data, not context

        val_loss = np.asarray(entry["val_loss_curve"], dtype=np.float64)
        idx = subsample_uniform(val_loss)
        epochs = idx + 1  # epoch index -> epoch number (1-indexed, matches fidelity domain)
        y = val_loss[idx]  # raw value, already in [0,1] -- no normalization needed

        base_config = {
            "lr": entry["lr"],
            "weight_decay": entry["weight_decay"],
            "hidden_dim": entry["hidden_dim"],
            "num_layers": 4,
            "lr_schedule": "none",
            "batch_size": 64,
        }

        for ep, y_val in zip(epochs, y):
            config = {**base_config, FIDELITY_NAME: int(ep)}
            result = UserResultDict(objective_to_minimize=float(y_val))
            evaluated_trials.append((config, result))

    print(f"  Built {len(evaluated_trials)} context points "
          f"(hidden_dim != {target_hd})")
    return evaluated_trials


# ============================================================
# evaluate_pipeline: exact lookup for target_hd == 128 configs
# ============================================================

def make_evaluate_pipeline(data, target_hd):
    # index by (lr, weight_decay, hidden_dim) -> full val_loss_curve
    lookup = {}
    for entry in data.values():
        if entry["hidden_dim"] != target_hd:
            continue
        key = (entry["lr"], entry["weight_decay"], entry["hidden_dim"])
        lookup[key] = np.asarray(entry["val_loss_curve"], dtype=np.float64)

    def evaluate_pipeline(lr, weight_decay, hidden_dim, epochs, **_):
        key = (lr, weight_decay, hidden_dim)
        if key not in lookup:
            raise KeyError(
                f"No ground-truth run found for {key} at hidden_dim={target_hd}. "
                "This candidate wasn't in the pre-run json -- pipeline_space "
                "choices should make this impossible; check for a mismatch."
            )
        curve = lookup[key]
        raw_loss = curve[epochs - 1]  # epochs is 1-indexed, value already in [0,1]
        return {"objective_to_minimize": float(raw_loss)}

    return evaluate_pipeline


# ============================================================
# Main
# ============================================================

def main():
    print("Loading result_metrics.json...")
    data = load_data(DATA_PATH)
    print(f"  Loaded {len(data)} runs")

    pipeline_space = build_pipeline_space()

    context_trials = build_context_trials(data, TARGET_HIDDEN_DIM)

    print(f"Importing {len(context_trials)} context points into {ROOT_DIRECTORY} ...")
    neps.import_trials(
        context_trials,
        root_directory=ROOT_DIRECTORY,
        pipeline_space=pipeline_space,
        overwrite_root_directory=True,
    )

    evaluate_pipeline = make_evaluate_pipeline(data, TARGET_HIDDEN_DIM)

    optimizer = build_fixed_scale_ifbo(pipeline_space, fixed_hidden_dim=128)

    neps.run(
        pipeline_space=pipeline_space,
        evaluate_pipeline=evaluate_pipeline,
        searcher=optimizer,          # <- instance, not the string "ifbo"
        max_evaluations_total=MAX_EVALUATIONS_TOTAL,
        root_directory=ROOT_DIRECTORY,
    )



if __name__ == "__main__":
    main()