import os
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.mlp import MLP4
from utils import (
    get_cifar_loaders,
    get_device,
    get_fashion_mnist_loaders,
    normalize_hyperparameters,
    load_context_curves,
    save_context_curves,
)
from ifbo.surrogate import FTPFN
from ifbo import Curve


def predict(
    lr: float = 1e-3,
    num_layers: int = 4,
    hidden_dim: int = 64,
    weight_decay: float = 0.0,
    lr_schedule: str = "none",
    epochs: int = 15,
    batch_size: int = 96,
    use_context: bool = True,
    trial_id=None,
    log_dir: str = "./runs_ifbo",
    save_dir: str = "./results",
):
    device = get_device()

    # Select model
    model = MLP4(hidden_dim=hidden_dim)
    model.to(device)

    json_path = os.path.join(save_dir, "64.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    target_epochs = 100
    context_curves = []
    for run_key, run_data in all_results.items():
        if run_data.get("num_layers") != 4 or run_data.get("hidden_dim") != 64:
            continue

        curve_values = run_data.get("val_loss_curve", [])
        if len(curve_values) == 0:
            continue

        hp = normalize_hyperparameters(run_data["lr"], run_data["num_layers"],
                                       run_data["hidden_dim"], run_data["weight_decay"])

        t = torch.linspace(0.0, float(len(curve_values)) / float(target_epochs),
                           steps=len(curve_values))

        context_curves.append(
            Curve(hyperparameters=hp, t=t, y=torch.tensor(curve_values, dtype=torch.float32))
        )
    
    json_path = os.path.join(save_dir, "results_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Build FT-PFN context
    target_epochs = 100
    for run_key, run_data in all_results.items():
        if run_data.get("num_layers") != num_layers or run_data.get("hidden_dim") != hidden_dim:
            continue

        curve_values = run_data.get("val_loss_curve", [])[:epochs]
        print(len(curve_values))
        if len(curve_values) == 0:
            continue

        hp = normalize_hyperparameters(run_data["lr"], run_data["num_layers"],
                                       run_data["hidden_dim"], run_data["weight_decay"])

        t = torch.linspace(0.0, float(len(curve_values)) / float(target_epochs),
                           steps=len(curve_values))

        context_curves.append(
            Curve(hyperparameters=hp, t=t, y=torch.tensor(curve_values, dtype=torch.float32))
        )

    print(f"Saved {len(context_curves)} curves to FT-PFN context")

    # 🔥🔥🔥 SKIP FT-PFN IF NO CONTEXT
    if len(context_curves) == 0:
        print("⚠️ Context empty — skipping FT-PFN prediction.")
        return float(-1)

    # --- FT-PFN Prediction ---
    pred_results = {}

    query_hp = normalize_hyperparameters(lr, num_layers, hidden_dim, weight_decay)

    observed_fraction = float(epochs) / float(target_epochs)
    remaining_steps = max(target_epochs - epochs, 1)
    t_target = torch.linspace(observed_fraction, 1.0, steps=remaining_steps)

    query_curve = [Curve(hyperparameters=query_hp, t=t_target)]

    ft_pfn_model = FTPFN(version="0.0.1", device=get_device())
    prediction = ft_pfn_model.predict(context=context_curves, query=query_curve)[0]

    pred_point = prediction.quantile(0.5).tolist()
    pred_q05 = prediction.quantile(0.05).tolist()
    pred_q95 = prediction.quantile(0.95).tolist()

    pred_results = {
        "point": pred_point,
        "quantiles": {"0.05": pred_q05, "0.5": pred_point, "0.95": pred_q95},
    }

    pred_json_path = os.path.join(save_dir, f"ifbo_pred_both64_context_{epochs}.json")

    if os.path.exists(pred_json_path):
        with open(pred_json_path, "r") as f:
            all_preds = json.load(f)
    else:
        all_preds = {}

    pred_key = f"layer{num_layers}_lr{lr}_hd{hidden_dim}_wd{weight_decay}_{lr_schedule}_{epochs}"
    all_preds[pred_key] = pred_results

    with open(pred_json_path, "w") as f:
        json.dump(all_preds, f, indent=4)

    print(f"Saved FT-PFN prediction under key: {pred_key} in {pred_json_path}")

    return float(pred_point[-1])
