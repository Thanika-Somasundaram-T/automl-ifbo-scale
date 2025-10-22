import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from models.mlp import MLP2, MLP4
from utils import get_cifar_loaders, get_device, get_mnist_loaders, load_context_curves, normalize_hyperparameters, save_context_curves
from ifbo.surrogate import FTPFN
from ifbo import Curve
import os
from datetime import datetime
import json

# Global FT-PFN context (shared across runs)

def train(
    lr: float = 1e-3,
    num_layers: int = 2,
    hidden_dim: int = 256,
    weight_decay: float = 0.0,
    epochs: int = 10,
    batch_size: int = 128,
    use_context: bool = False,
    trial_id = None,
    log_dir: str = "./runs",
    save_dir: str = "./results"
):  
    device = get_device()

    # Select model
    model = MLP2(hidden_dim=hidden_dim) if num_layers == 2 else MLP4(hidden_dim=hidden_dim)
    model.to(device)

    # Load MNIST
    train_loader, test_loader = get_cifar_loaders(batch_size=batch_size)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # TensorBoard setup
    timestamp = datetime.now().strftime("%H%M%S")
    run_name = f"Full_run_layer_{num_layers}_lr_{lr:.0e}__full_lr_{lr}_hd_{hidden_dim}_Wd_{weight_decay}_{timestamp}"
    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    val_curve = []
    best_val_acc = 0.0

    # --- Train for a few epochs only if using larger layers ---

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        val_curve.append(val_acc)
        best_val_acc = max(best_val_acc, val_acc)
        writer.add_scalar("val/Accuracy", val_acc, epoch)
        writer.add_scalar("val/Accuracy_all", val_acc, epoch)
        print(f"Epoch {epoch+1}/{epochs}: Val Acc = {val_acc:.4f}")

    # --- Save trained curve ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{num_layers}layer_lr{lr}.pt")
    torch.save({
        "hyperparameters": [lr, hidden_dim, weight_decay],
        "val_curve": val_curve
    }, save_path)

    # --- Update FT-PFN context ---
    hp = normalize_hyperparameters(lr, hidden_dim, weight_decay)
    t = torch.linspace(0, 1, steps=len(val_curve))
    curve = Curve(hyperparameters=hp, t=t, y=torch.tensor(val_curve, dtype=torch.float32))
    # print("curve:::::::::: ", curve)

    # Load existing context if any
    existing_curves = load_context_curves("./context_layer.pt")
    # print("existing curve:::::::::", existing_curves)
    context_curves_to_save = existing_curves + [curve]  # append new curve
    save_context_curves(context_curves_to_save, path="./context_layer.pt")

    # --- Predict remaining epochs for higher layers ---
    if use_context:
        print("is it inside????????")
        context_curves = load_context_curves("./context_layer.pt")
        if len(context_curves) > 1:
            ft_pfn_model = FTPFN(version="0.0.1", device=device)
            query = [Curve(hyperparameters=normalize_hyperparameters(lr, hidden_dim, weight_decay),
                        t=torch.linspace(0, 1, steps=epochs))]
            prediction = ft_pfn_model.predict(context=context_curves, query=query)[0]
            predicted_val_acc = prediction.quantile(0.95).max().item()
            print("Predicted val_acc: ", predicted_val_acc)
            # Predicted continuation
            predicted_steps = 5
            predicted_values = torch.linspace(val_curve[-1], predicted_val_acc, steps=predicted_steps)
            for i, val in enumerate(predicted_values, start=len(val_curve)):
                writer.add_scalar("val/Accuracy_all", val, i)
                writer.add_scalar("val/Accuracy_Predicted", val, i)
            writer.close()
            best_val_acc = predicted_val_acc  # NEPS uses this as the objective

    writer.close()
    results_path = os.path.join(save_dir, "results_summary.json")
    key = f"{num_layers}_{use_context}"
    subkey = f"lr{lr}_hd{hidden_dim}_wd{weight_decay}"

    # Load existing results if file exists
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Ensure nested dicts exist
    if key not in all_results:
        all_results[key] = {}

    # Store value
    all_results[key][subkey] = round(best_val_acc, 4)

    # Write back to JSON
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Saved best val_acc = {best_val_acc:.4f} to {results_path}")
    return 1.0 - best_val_acc
