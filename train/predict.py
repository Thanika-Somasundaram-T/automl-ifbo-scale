import os
import json
import torch
from models.mlp import MLP4
from utils import fixed_range_normalize, get_device, normalize_hyperparameters, min_max_normalize
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
    model = MLP4(hidden_dim=hidden_dim).to(device)

    # -------------------------
    # Load low-scale context curves (e.g., HD=64)
    # -------------------------
    context_curves = []
    json_path = os.path.join(save_dir, "results_metrics.json")
    print("for hd: ", hidden_dim)
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)

        target_epochs = 150
        for run_key, run_data in all_results.items():
            if run_data.get("hidden_dim") != 22:
                print(run_data.get("hidden_dim"))
                continue

            curve_values = run_data.get("val_acc_curve", [])
            if len(curve_values) == 0:
                continue
            hp = normalize_hyperparameters(
                run_data["lr"], run_data["num_layers"], run_data["hidden_dim"], run_data["weight_decay"]
            )
            t = torch.linspace(0.0, float(len(curve_values)) / float(target_epochs), steps=len(curve_values))
            y_norm = fixed_range_normalize(curve_values)  # normalization for prediction only
            
            print(run_data["hidden_dim"],"added to context")

            context_curves.append(Curve(hyperparameters=hp, t=t, y=torch.tensor(y_norm, dtype=torch.float32)))
        print(f"Saved ----------- {len(context_curves)} curves to FT-PFN context")

    # -------------------------
    # Add same-scale model curves as context
    # -------------------------
    json_path = os.path.join(save_dir, "results_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)

        target_epochs = 150
        for run_key, run_data in all_results.items():
            if run_data.get("num_layers") != num_layers or run_data.get("hidden_dim") != hidden_dim:
                continue
            print(run_data.get("hidden_dim"), "partial added as context")
            curve_values = run_data.get("val_acc_curve", [])[:epochs]
            if len(curve_values) == 0:
                continue

            hp = normalize_hyperparameters(
                run_data["lr"], run_data["num_layers"], run_data["hidden_dim"], run_data["weight_decay"]
            )
            t = torch.linspace(0.0, float(len(curve_values)) / float(target_epochs), steps=len(curve_values))
            y_norm = fixed_range_normalize(curve_values)

            context_curves.append(Curve(hyperparameters=hp, t=t, y=torch.tensor(y_norm, dtype=torch.float32)))

    print(f"Saved +per = {len(context_curves)} curves to FT-PFN context")

    if len(context_curves) == 0:
        print("⚠️ Context empty — skipping FT-PFN prediction.")
        return float(-1)

    # -------------------------
    # FT-PFN Prediction
    # -------------------------
    query_hp = normalize_hyperparameters(lr, num_layers, hidden_dim, weight_decay)
    observed_fraction = float(epochs) / float(target_epochs)
    remaining_steps = max(target_epochs - epochs, 1)
    t_target = torch.linspace(observed_fraction, 1.0, steps=remaining_steps)
    query_curve = [Curve(hyperparameters=query_hp, t=t_target)]

    ft_pfn_model = FTPFN(version="0.0.1", device=device)
    prediction = ft_pfn_model.predict(context=context_curves, query=query_curve)[0]

    pred_point = prediction.quantile(0.5).tolist()
    pred_q05 = prediction.quantile(0.05).tolist()
    pred_q95 = prediction.quantile(0.95).tolist()

    # -------------------------
    # Save prediction only (no y_norm or raw_y)
    # -------------------------
    pred_json_path = os.path.join(save_dir, f"context_par_{epochs}.json")
    if os.path.exists(pred_json_path):
        with open(pred_json_path, "r") as f:
            all_preds = json.load(f)
    else:
        all_preds = {}

    pred_key = f"layer{num_layers}_lr{lr}_hd{hidden_dim}_wd{weight_decay}_{lr_schedule}_{epochs}"

    all_preds[pred_key] = {
        "point": pred_point,
        "quantiles": {"0.05": pred_q05, "0.5": pred_point, "0.95": pred_q95},
    }

    with open(pred_json_path, "w") as f:
        json.dump(all_preds, f, indent=4)

    print(f"Saved FT-PFN prediction under key: {pred_key} in {pred_json_path}")

    return float(pred_point[-1])
