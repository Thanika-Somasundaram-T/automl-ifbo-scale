"""
Batch-runs all context-selection policy ablations over every MLP scale/HP
config/observation-epoch, saving FT-PFN predictions to
results_diverse_lower_k/<policy_variant>.json.

Policies:
  - diverse_lower_k_{geometric,uniform}_k{4,8,12,16,24,32}
  - all_lower_{geometric,uniform}      (every run at every scale below target)
  - nearest_lower_{geometric,uniform}  (every run at just the nearest scale below target)
  - baseline_{geometric,uniform}       (no other-scale context at all)
  - interpolation_{geometric,uniform}  (every run at every OTHER scale, lower + higher)

Run from the repo root:
    python run_diverse_lower_k.py
"""
import json
import os

from train.predict_diverse import (
    predict_diverse_lower_k,
    predict_all_lower,
    predict_nearest_lower,
    predict_baseline,
    predict_interpolation,
)
from utils import get_device

from ifbo.surrogate import FTPFN

RESULTS_JSON = "./results_***/results_metrics.json"
SAVE_DIR = "./results_diverse_lower_k"

OBS_EPOCHS = [0, 5, 10, 20, 50, 90]
K_VALUES = [4, 8, 12, 16, 24, 32]
SUBSAMPLE_MODES = ["geometric", "uniform"]


def _run_policy(policy_name, predict_fn, query_configs, all_preds_cache_path):
    all_preds = {}
    if os.path.exists(all_preds_cache_path):
        with open(all_preds_cache_path) as f:
            all_preds = json.load(f)

    for run_key, lr, hd, wd in query_configs:
        for epochs in OBS_EPOCHS:
            pred_key = f"{run_key}_{epochs}"
            if pred_key in all_preds:
                continue

            result = predict_fn(lr, hd, wd, epochs)
            if result is None:
                continue

            all_preds[pred_key] = result

        # checkpoint after each config so long runs are resumable
        with open(all_preds_cache_path, "w") as f:
            json.dump(all_preds, f, indent=2)

    print(f"[{policy_name}] saved {len(all_preds)} predictions -> {all_preds_cache_path}")


def run_batch(all_results, save_dir):
    device = get_device()
    ft_pfn_model = FTPFN(version="0.0.1", device=device)

    os.makedirs(save_dir, exist_ok=True)

    query_configs = [
        (run_key, run_data["lr"], run_data["hidden_dim"], run_data["weight_decay"])
        for run_key, run_data in all_results.items()
    ]

    # ---------------------------------------------------------------
    # diverse_lower_k
    # ---------------------------------------------------------------
    for subsample_mode in SUBSAMPLE_MODES:
        for k in K_VALUES:
            policy_name = f"diverse_lower_k_{subsample_mode}_k{k}"
            out_path = os.path.join(save_dir, f"{policy_name}.json")

            def predict_fn(lr, hd, wd, epochs, _k=k, _mode=subsample_mode):
                return predict_diverse_lower_k(
                    all_results, lr, hd, wd, epochs, _k, _mode,
                    device=device, ft_pfn_model=ft_pfn_model,
                )

            _run_policy(policy_name, predict_fn, query_configs, out_path)

    # ---------------------------------------------------------------
    # all_lower / nearest_lower / baseline / interpolation
    # ---------------------------------------------------------------
    non_diverse_policies = {
        "all_lower": predict_all_lower,
        "nearest_lower": predict_nearest_lower,
        "baseline": predict_baseline,
        "interpolation": predict_interpolation,
    }

    for policy_prefix, predict_base_fn in non_diverse_policies.items():
        for subsample_mode in SUBSAMPLE_MODES:
            policy_name = f"{policy_prefix}_{subsample_mode}"
            out_path = os.path.join(save_dir, f"{policy_name}.json")

            def predict_fn(lr, hd, wd, epochs, _fn=predict_base_fn, _mode=subsample_mode):
                return _fn(
                    all_results, lr, hd, wd, epochs, _mode,
                    device=device, ft_pfn_model=ft_pfn_model,
                )

            _run_policy(policy_name, predict_fn, query_configs, out_path)


def main():
    with open(RESULTS_JSON) as f:
        all_results = json.load(f)
    run_batch(all_results, SAVE_DIR)


if __name__ == "__main__":
    main()
