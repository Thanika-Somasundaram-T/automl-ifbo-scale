"""
FT-PFN prediction with exactly 2 HP context curves from a lower scale.

Context = [anchor_hp @ lower_scale, partner_hp @ lower_scale, target partial curve @ hd=128]

Where:
  anchor_hp = same (lr, wd) as the target query
  partner_hp = a different (lr, wd)

This produces predictions to answer:
  "Does adding a second HP context curve help or hurt vs. just one?"
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import get_device, normalize_hyperparameters, normalize_log_loss_curve
from ifbo.surrogate import FTPFN
from ifbo import Curve


def predict_2hp(
    target_lr: float,
    target_wd: float,
    partner_lr: float,
    partner_wd: float,
    context_scale: int = 64,
    epochs: int = 5,
    save_dir: str = "../results",
):
    """
    Run FT-PFN prediction for a target HP using 2-HP context from a lower scale.

    Context curves:
      1. anchor: (target_lr, target_wd) @ context_scale  (full curve)
      2. partner: (partner_lr, partner_wd) @ context_scale  (full curve)
      3. target partial: (target_lr, target_wd) @ hd=128  (first `epochs` points)

    Returns:
        float: predicted final performance value
    """
    device = get_device()
    hidden_dim = 128
    target_epochs = 150

    json_path = os.path.join(save_dir, "results_metrics.json")
    if not os.path.exists(json_path):
        print(f"⚠️ Results file not found: {json_path}")
        return float(-1)

    with open(json_path, "r") as f:
        all_results = json.load(f)

    context_curves = []

    # ── 1. Anchor curve: same HP as target, from lower scale (full curve) ──
    anchor_added = False
    for run_key, run_data in all_results.items():
        if (run_data.get("hidden_dim") == context_scale
            and run_data.get("lr") == target_lr
            and run_data.get("weight_decay") == target_wd):

            curve_values = run_data.get("val_loss_curve", [])
            if len(curve_values) == 0:
                continue

            y_norm = normalize_log_loss_curve(curve_values)
            hp = normalize_hyperparameters(
                run_data["lr"], run_data["hidden_dim"], run_data["weight_decay"]
            )
            t = torch.linspace(0.0, float(len(y_norm)) / float(target_epochs), steps=len(y_norm))
            context_curves.append(
                Curve(hyperparameters=hp, t=t, y=torch.tensor(y_norm, dtype=torch.float32))
            )
            anchor_added = True
            break

    if not anchor_added:
        print(f"⚠️ Anchor curve not found: lr={target_lr}, wd={target_wd} @ hd={context_scale}")
        return float(-1)

    # ── 2. Partner curve: different HP, from same lower scale (full curve) ──
    partner_added = False
    for run_key, run_data in all_results.items():
        if (run_data.get("hidden_dim") == context_scale
            and run_data.get("lr") == partner_lr
            and run_data.get("weight_decay") == partner_wd):

            curve_values = run_data.get("val_loss_curve", [])
            if len(curve_values) == 0:
                continue

            y_norm = normalize_log_loss_curve(curve_values)
            hp = normalize_hyperparameters(
                run_data["lr"], run_data["hidden_dim"], run_data["weight_decay"]
            )
            t = torch.linspace(0.0, float(len(y_norm)) / float(target_epochs), steps=len(y_norm))
            context_curves.append(
                Curve(hyperparameters=hp, t=t, y=torch.tensor(y_norm, dtype=torch.float32))
            )
            partner_added = True
            break

    if not partner_added:
        print(f"⚠️ Partner curve not found: lr={partner_lr}, wd={partner_wd} @ hd={context_scale}")
        return float(-1)

    # ── 3. Target partial curve: same HP @ hd=128 (first `epochs` points) ──
    if epochs > 0:
        for run_key, run_data in all_results.items():
            if (run_data.get("hidden_dim") == hidden_dim
                and run_data.get("lr") == target_lr
                and run_data.get("weight_decay") == target_wd):

                curve_values = run_data.get("val_loss_curve", [])[:epochs]
                if len(curve_values) == 0:
                    continue

                y_norm = normalize_log_loss_curve(curve_values)
                hp = normalize_hyperparameters(
                    run_data["lr"], run_data["hidden_dim"], run_data["weight_decay"]
                )
                t = torch.linspace(0.0, float(len(y_norm)) / float(target_epochs), steps=len(y_norm))
                context_curves.append(
                    Curve(hyperparameters=hp, t=t, y=torch.tensor(y_norm, dtype=torch.float32))
                )
                break

    print(f"  Context: {len(context_curves)} curves "
          f"(anchor=lr{target_lr}/wd{target_wd}@hd{context_scale}, "
          f"partner=lr{partner_lr}/wd{partner_wd}@hd{context_scale}, "
          f"target partial @hd128, T={epochs})")

    # ── 4. FT-PFN Prediction ──
    query_hp = normalize_hyperparameters(target_lr, hidden_dim, target_wd)
    observed_fraction = float(epochs) / float(target_epochs)
    remaining_steps = max(target_epochs - epochs, 1)

    t_target = torch.linspace(observed_fraction, 1.0, steps=remaining_steps)
    query_curve = [Curve(hyperparameters=query_hp, t=t_target)]

    ft_pfn_model = FTPFN(version="0.0.1", device=device)
    prediction = ft_pfn_model.predict(context=context_curves, query=query_curve)[0]

    pred_point = prediction.quantile(0.5).tolist()
    pred_q05 = prediction.quantile(0.05).tolist()
    pred_q95 = prediction.quantile(0.95).tolist()

    # ── 5. Save prediction ──
    # File naming: ifbo_pred_2hp_[scale]_anchor{cfg}_partner{cfg}_ep{epochs}.json
    # But for simplicity, group all partner predictions for a given anchor+epoch into one file
    pred_json_path = os.path.join(
        save_dir,
        f"ifbo_pred_2hp_[{context_scale}]_ep{epochs}.json"
    )

    if os.path.exists(pred_json_path):
        with open(pred_json_path, "r") as f:
            all_preds = json.load(f)
    else:
        all_preds = {}

    # Key encodes: target HP + partner HP
    pred_key = (
        f"target_lr{target_lr}_wd{target_wd}"
        f"__partner_lr{partner_lr}_wd{partner_wd}"
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

    with open(pred_json_path, "w") as f:
        json.dump(all_preds, f, indent=4)

    print(f"  ✅ Saved: {pred_key} → {pred_json_path}")
    return float(pred_point[-1])
