import os
import json
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.mlp import MLP10, MLP2, MLP4
from utils import get_cifar_loaders, get_device


def train_mlp(
    lr: float = 1e-3,
    num_layers: int = 2,
    hidden_dim: int = 256,
    weight_decay: float = 0.0,
    epochs: int = 10,
    batch_size: int = 128,
    trial_id: Optional[str] = None,
    log_dir: str = "./runs",
    save_dir: str = "./results"
) -> float:

    device = get_device()

    # Model selection
    model = MLP2(hidden_dim=hidden_dim) if num_layers == 2 else MLP4(hidden_dim=hidden_dim)
    model.to(device)

    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Loss_run_layers{num_layers}_lr{lr:.0e}_hd{hidden_dim}_wd{weight_decay}_{time_tag}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    val_curve = []
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss_total = 0.0
        total_samples = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
                val_loss_total += loss.item() * y.size(0)
                total_samples += y.size(0)

        val_loss = val_loss_total / total_samples
        val_curve.append(val_loss)
        writer.add_scalar("val/Loss", val_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs} Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

    writer.close()

    # Save curve to .pt file
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "hyperparameters": [lr, hidden_dim, weight_decay],
        "val_curve": val_curve
    }, os.path.join(save_dir, f"{num_layers}layers_lr{lr}_loss.pt"))

    # ✅ Save curve also into JSON for future plotting
    json_path = os.path.join(save_dir, "results_loss_curves.json")
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    key = f"layers{num_layers}_lr{lr}_hd{hidden_dim}_wd{weight_decay}"
    all_results[key] = {
        "lr": lr,
        "num_layers": num_layers,
        "hidden_dim": hidden_dim,
        "weight_decay": weight_decay,
        "val_curve": val_curve,
        "best_val_loss": best_val_loss
    }

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Stored loss curve in {json_path}")

    return float(best_val_loss)
