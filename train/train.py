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
import os
from datetime import datetime
import json


def train(
    lr: float = 1e-3,
    num_layers: int = 4,
    hidden_dim: int = 64,
    weight_decay: float = 0.0,
    lr_schedule: str = "none",
    epochs: int = 15,
    batch_size: int = 96,
    use_context: bool = True,
    trial_id=None,
    log_dir: str = "./runs",
    save_dir: str = "./results_f", # bigger hidden_dim to predict
):
    device = get_device()

    # Select model
    model = MLP4(hidden_dim=hidden_dim)
    model.to(device)

    # Load CIFAR
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # TensorBoard setup
    timestamp = datetime.now().strftime("%H%M%S")
    run_name = f"layer{num_layers}_lr{lr:.0e}_hd{hidden_dim}_wd{weight_decay}_{timestamp}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    # Initialize curves
    train_loss_curve, val_loss_curve = [], []
    train_acc_curve, val_acc_curve = [], []
    train_auc_curve, val_auc_curve = [], []  # placeholders if needed
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss_sum += loss.item() * x.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)

        val_loss = val_loss_sum / total_val
        val_acc = correct_val / total_val

        # Save best val loss
        best_val_loss = min(best_val_loss, val_loss)

        # Append to curves
        train_loss_curve.append(train_loss)
        val_loss_curve.append(val_loss)
        train_acc_curve.append(train_acc)
        val_acc_curve.append(val_acc)
        # AUC can be added if available

        # TensorBoard
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/acc", val_acc, epoch)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    writer.close()

    # --- Save curves ---
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"layers{num_layers}_lr{lr}_hd{hidden_dim}_wd{weight_decay}_{lr_schedule}.pt"
    torch.save(
        {
            "lr": lr,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "lr_schedule": lr_schedule,
            "train_loss_curve": train_loss_curve,
            "val_loss_curve": val_loss_curve,
            "train_acc_curve": train_acc_curve,
            "val_acc_curve": val_acc_curve,
            "train_auc_curve": train_auc_curve,
            "val_auc_curve": val_auc_curve,
            "best_val_loss": best_val_loss,
        },
        os.path.join(save_dir, save_name),
    )

    # --- Update JSON results ---
    json_path = os.path.join(save_dir, "result.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    key = f"layers{num_layers}_lr{lr}_hd{hidden_dim}_wd{weight_decay}_{lr_schedule}"
    all_results[key] = {
        "lr": lr,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "weight_decay": weight_decay,
        "lr_schedule": lr_schedule,
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "train_acc_curve": train_acc_curve,
        "val_acc_curve": val_acc_curve,
        "train_auc_curve": train_auc_curve,
        "val_auc_curve": val_auc_curve,
        "best_val_loss": best_val_loss,
    }

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Stored full metrics in {json_path}")
    target_epochs = 100
    

    # --- Build FT-PFN context from all 4-layer, HD=32 curves ---
    context_curves = []
    for run_key, run_data in all_results.items():
        for metric_name in ["val_loss_curve"]:
            curve_values = run_data[metric_name]
            hp = normalize_hyperparameters(
                run_data["lr"], run_data["hidden_dim"], run_data["weight_decay"]
            )
            t = torch.linspace(0, epochs/target_epochs, steps=len(curve_values))
            context_curves.append(
                Curve(hyperparameters=hp, t=t, y=torch.tensor(curve_values, dtype=torch.float32))
            )

    save_context_curves(context_curves, path="./context_layer.pt")
    print(f"Saved {len(context_curves)} curves to FT-PFN context.")

    # --- Predict for HD=64 using context ---
# --- Predict for HD=64 using context ---
    pred_results = {}
# --- Predict for HD=64 using context and observed 15-epoch curve ---
    if use_context:
        query_hp = normalize_hyperparameters(lr, hidden_dim, weight_decay)
        t_target = torch.linspace(epochs/target_epochs, 1.0, steps=(target_epochs - epochs))  # scale to fraction of target


        query_curve = [Curve(hyperparameters=query_hp, t=t_target)]

        ft_pfn_model = FTPFN(version="0.0.1", device=get_device())
        prediction = ft_pfn_model.predict(context=context_curves, query=query_curve)[0]

        # Save predicted point + quantiles
        pred_point = prediction.quantile(0.5).tolist()
        pred_q05 = prediction.quantile(0.05).tolist()
        pred_q95 = prediction.quantile(0.95).tolist()
        pred_results = {
            "point": pred_point,
            "quantiles": {"0.05": pred_q05, "0.5": pred_point, "0.95": pred_q95},
        }

        pred_path = os.path.join(
            save_dir, f"pred_layer{num_layers}_lr{lr}_hd{hidden_dim}_wd{weight_decay}_{lr_schedule}.json"
        )
        with open(pred_path, "w") as f:
            json.dump(pred_results, f, indent=4)
        print(f"Saved FT-PFN prediction for HD={hidden_dim} (100 epochs) to {pred_path}")



    return float(best_val_loss)
