import os
import json
import numpy as np
import torch

from models.mlp import MLP4

from utils import (
    get_device,
    normalize_hyperparameters,
    normalize_log_loss_curve,
    subsample_curve,
)

from ifbo.surrogate import FTPFN
from ifbo import Curve



# ----------------------------------------------------------------------
# Available context combinations
# ----------------------------------------------------------------------

selected_context_curves = [
    (64,),
    (64, 32),
    (64, 32, 24),
    (32,),
    (24,),
    (32, 24),
]



# ----------------------------------------------------------------------
# Build partial observed curve
# ----------------------------------------------------------------------

def build_partial_curve(
        run_data,
        observe_fraction,
        target_epochs=100):


    val_loss = np.asarray(
        run_data.get("val_loss_curve", []),
        dtype=np.float64
    )[:100]


    if len(val_loss) == 0:
        return None

    if observe_fraction <= 0:
        return None

    n_observe = int(len(val_loss) * observe_fraction)

    if n_observe == 0:
        return None


    # Slice observed part only
    y_partial = val_loss[:n_observe]


    t_partial = np.linspace(
        0.0,
        1.0,
        len(val_loss)
    )[:n_observe]


    print(
        f"    partial observation: "
        f"{n_observe} points, {observe_fraction}"
    )
    
    # Normalize loss only for FT-PFN
    y_norm = normalize_log_loss_curve(
        y_partial
    )


    # Reduce curve size
    t_sub, y_sub = subsample_curve(
        t_partial,
        y_norm
    )


    return t_sub, y_sub





# ----------------------------------------------------------------------
# Prediction function
# ----------------------------------------------------------------------

def predict(
    lr: float = 1e-3,
    num_layers: int = 4,
    hidden_dim: int = 64,
    weight_decay: float = 0.0,
    lr_schedule: str = "none",
    epochs: int = 15,
    context_id: int = 0,
    batch_size: int = 96,
    use_context: bool = True,
    trial_id=None,
    log_dir: str = "./runs_ifbo",
    save_dir: str = "./ifbo_per_10",
):

    print(
            "\n\n\n\n\n\n\n\n\n\n\n\n\n************************************************************************Target hidden dim:",
            hidden_dim
        )

    device = get_device()


    model = MLP4(
        hidden_dim=hidden_dim
    ).to(device)



    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    json_path = os.path.join(
        "./results_***",
        "results_metrics.json"
    )


 


    if not os.path.exists(json_path):
        raise FileNotFoundError(
            json_path
        )


    with open(json_path, "r") as f:
        all_results = json.load(f)



    target_epochs = 100
    print(epochs, "  ----- epochs ------")



    if (
        context_id < 0
        or
        context_id >= len(selected_context_curves)
    ):
        raise ValueError(
            f"Invalid context_id {context_id}"
        )



    width_subset = selected_context_curves[
        context_id
    ]


    subset_name = "_".join(
        map(str, width_subset)
    )

    context_curves = []



    # ---------------------------------------------------------------
    # 1. Add lower-scale context curves
    # ---------------------------------------------------------------

    for run_key, run_data in all_results.items():


        hd = run_data.get(
            "hidden_dim"
        )
        rlr = run_data.get("lr")
        rwd = run_data.get("weight_decay")


        if hd not in width_subset:
            continue

        if rlr != lr or rwd != weight_decay:
            continue

        curve_values = run_data.get(
            "val_loss_curve",
            []
        )[:100]


        if len(curve_values) == 0:
            continue



        hp = normalize_hyperparameters(
            run_data["lr"],
            run_data["hidden_dim"],
            run_data["weight_decay"]
        )



        # full curve
        t = np.linspace(
            0.0,
            1.0,
            len(curve_values)
        )


        y_norm = normalize_log_loss_curve(
            curve_values
        )


        t_sub, y_sub = subsample_curve(
            t,
            y_norm
        )

        context_curves.append(
            Curve(
                hyperparameters=hp,

                t=torch.tensor(
                    t_sub,
                    dtype=torch.float32
                ),

                y=torch.tensor(
                    y_sub,
                    dtype=torch.float32
                )
            )
        )



    print(
        f"Low-scale context curves: "
        f"{len(context_curves)}"
    )
    
    # ---------------------------------------------------------------
    # 2. Add same-scale partial observed curve
    # ---------------------------------------------------------------

    observe_fraction = (
        float(epochs)
        /
        float(target_epochs)
    )


    for run_key, run_data in all_results.items():


        if (
            run_data.get("lr") != lr
            or
            run_data.get("hidden_dim") != hidden_dim
            or
            run_data.get("weight_decay") != weight_decay
        ):
            continue



        partial_curve = build_partial_curve(
            run_data,
            observe_fraction,
            target_epochs
        )


        if partial_curve is None:
            continue



        t_sub, y_sub = partial_curve



        hp = normalize_hyperparameters(
            run_data["lr"],
            run_data["hidden_dim"],
            run_data["weight_decay"]
        )

        context_curves.append(
            Curve(
                hyperparameters=hp,

                t=torch.tensor(
                    t_sub,
                    dtype=torch.float32
                ),

                y=torch.tensor(
                    y_sub,
                    dtype=torch.float32
                )
            )
        )



    print(
        f"Total context curves: "
        f"{len(context_curves)}"
    )



    if len(context_curves) == 0:

        print(
            "⚠️ Empty context. Skipping prediction."
        )

        return float(-1)



    # ---------------------------------------------------------------
    # 3. FT-PFN prediction
    # ---------------------------------------------------------------


    query_hp = normalize_hyperparameters(
        lr,
        hidden_dim,
        weight_decay
    )


    observed_fraction = (
        float(epochs)
        /
        float(target_epochs)
    )


    remaining_steps = max(
        target_epochs - epochs,
        1
    )


    # future time points
    t_target = torch.linspace(
        observed_fraction,
        1.0,
        steps=remaining_steps
    )



    query_curve = [
        Curve(
            hyperparameters=query_hp,
            t=t_target
        )
    ]



    ft_pfn_model = FTPFN(
        version="0.0.1",
        device=device
    )



    prediction = ft_pfn_model.predict(
        context=context_curves,
        query=query_curve
    )[0]



    # ---------------------------------------------------------------
    # Extract predictive distribution
    # ---------------------------------------------------------------

    pred_point = (
        prediction
        .quantile(0.5)
        .squeeze()
        .tolist()
    )


    pred_q05 = (
        prediction
        .quantile(0.05)
        .squeeze()
        .tolist()
    )


    pred_q95 = (
        prediction
        .quantile(0.95)
        .squeeze()
        .tolist()
    )



    # Make sure scalar prediction is list
    if isinstance(pred_point, float):
        pred_point = [pred_point]

    if isinstance(pred_q05, float):
        pred_q05 = [pred_q05]

    if isinstance(pred_q95, float):
        pred_q95 = [pred_q95]



    # ---------------------------------------------------------------
    # Save prediction
    # ---------------------------------------------------------------


    pred_json_path = os.path.join(
        save_dir,
        f"{hidden_dim}/{subset_name}/ep{epochs}.json"
    )


    os.makedirs(
        os.path.dirname(pred_json_path),
        exist_ok=True
    )



    if os.path.exists(pred_json_path):

        with open(pred_json_path, "r") as f:
            all_preds = json.load(f)

    else:

        all_preds = {}



    pred_key = (
        f"layer{num_layers}"
        f"_lr{lr}"
        f"_hd{hidden_dim}"
        f"_wd{weight_decay}"
        f"_{lr_schedule}"
        f"_{epochs}"
    )



    all_preds[pred_key] = {

        "point": pred_point,

        "quantiles": {

            "0.05": pred_q05,

            "0.5": pred_point,

            "0.95": pred_q95,
        },
    }



    with open(
        pred_json_path,
        "w"
    ) as f:

        json.dump(
            all_preds,
            f,
            indent=4
        )

    # normalized IFBO score
    return float(
        pred_point[-1]
    )