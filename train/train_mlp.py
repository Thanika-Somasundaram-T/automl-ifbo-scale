import os
import json
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np


from models.mlp import MLP2
from utils import get_device, get_fashion_mnist_loaders


def train_mlp(
    lr: float = 1e-3,
    num_layers: int = 2,
    hidden_dim: int = 256,
    weight_decay: float = 0.0,
    epochs: int = 50,
    batch_size: int = 128,
    trial_id: Optional[str] = None,
    log_dir: str = "./runs",
    save_dir: str = "./results",
    lr_schedule: str = "none",  # NePS can override
) -> float:
    """Train MLP on Fashion-MNIST and return best validation loss (NePS compatible)."""

    device = get_device()
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # --- Model selection ---
    model = MLP2(hidden_dim=hidden_dim)
    model.to(device)

    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Learning rate schedule ---
    if lr_schedule == "none":
        scheduler = None
    elif lr_schedule == "warmup":
        def lr_lambda(epoch):
            warmup_epochs = max(1, int(0.08 * epochs))
            return min(1.0, (epoch + 1) / warmup_epochs)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif lr_schedule == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif lr_schedule == "cooldown":
        def lr_lambda(epoch):
            cooldown_start = int(0.75 * epochs)
            if epoch < cooldown_start:
                return 1.0
            return max(0.1, 1 - (epoch - cooldown_start) / (epochs - cooldown_start))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif lr_schedule == "warmup_cooldown":
        def lr_lambda(epoch):
            warmup_epochs = max(1, int(0.08 * epochs))
            warmup_factor = (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
            cooldown_start = int(0.75 * epochs)
            cooldown_factor = max(0.1, 1 - (epoch - cooldown_start) / (epochs - cooldown_start)) if epoch >= cooldown_start else 1.0
            return warmup_factor * cooldown_factor
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        raise ValueError(f"Unknown lr_schedule: {lr_schedule}")

    # --- Run naming ---
    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"MLP_layers{num_layers}_lr{lr:.0e}_hd{hidden_dim}_wd{weight_decay}_{lr_schedule}_{time_tag}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    # --- Tracking ---
    train_loss_curve, val_loss_curve = [], []
    train_acc_curve, val_acc_curve = [], []
    train_auc_curve, val_auc_curve = [], []

    best_val_loss = float("inf")

    # --- Training Loop ---
    for epoch in range(epochs):
        model.train()
        total_train_loss, total_train_samples = 0.0, 0
        all_train_targets, all_train_probs = [], []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * y.size(0)
            total_train_samples += y.size(0)

            # Collect targets and softmax probabilities for train AUC
            all_train_targets.extend(y.cpu().numpy())
            all_train_probs.extend(F.softmax(outputs, dim=1).detach().cpu().numpy())

        train_loss = total_train_loss / total_train_samples
        train_acc = accuracy_score(all_train_targets, np.argmax(all_train_probs, axis=1))
        try:
            train_auc = roc_auc_score(
                torch.nn.functional.one_hot(torch.tensor(all_train_targets), num_classes=10).numpy(),
                np.array(all_train_probs),
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            train_auc = 0.0

        train_loss_curve.append(train_loss)
        train_acc_curve.append(train_acc)
        train_auc_curve.append(train_auc)

        # --- Validation ---
        model.eval()
        total_val_loss, total_val_samples = 0.0, 0
        all_val_targets, all_val_probs = [], []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                total_val_loss += loss.item() * y.size(0)
                total_val_samples += y.size(0)

                all_val_targets.extend(y.cpu().numpy())
                all_val_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())

        val_loss = total_val_loss / total_val_samples
        val_acc = accuracy_score(all_val_targets, np.argmax(all_val_probs, axis=1))
        try:
            val_auc = roc_auc_score(
                torch.nn.functional.one_hot(torch.tensor(all_val_targets), num_classes=10).numpy(),
                np.array(all_val_probs),
                multi_class="ovr",
                average="macro"
            )
        except Exception:
            val_auc = 0.0

        val_loss_curve.append(val_loss)
        val_acc_curve.append(val_acc)
        val_auc_curve.append(val_auc)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # --- TensorBoard Logging ---
        writer.add_scalar("train/Loss", train_loss, epoch)
        writer.add_scalar("val/Loss", val_loss, epoch)
        writer.add_scalar("train/Accuracy", train_acc, epoch)
        writer.add_scalar("val/Accuracy", val_acc, epoch)
        writer.add_scalar("train/AUC", train_auc, epoch)
        writer.add_scalar("val/AUC", val_auc, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        print(
            f"Epoch {epoch+1}/{epochs}: "
            f"TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}, "
            f"TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, "
            f"TrainAUC={train_auc:.4f}, ValAUC={val_auc:.4f}, LR={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    writer.close()

    # --- Save model + curves ---
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

    # --- Save JSON summary ---
    json_path = os.path.join(save_dir, "results_metrics.json")
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

    return float(best_val_loss)
