import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ifbo import Curve


def get_device():
    """
    Determine and return the best available computation device.

    Returns:
        torch.device
            Available device in priority order: MPS, CUDA, CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU.")
    return device

def normalize_hyperparameters(lr, hidden_dim, weight_decay):
    """
    Normalize hyperparameters to [0, 1] range for FT-PFN.

    Learning rate is normalized in log-scale.

    Args:
        lr : float
            Learning rate.
        num_layer : int
            Number of layers.
        hidden_dim : int
            Hidden layer dimension.
        weight_decay : float
            Weight decay value.

    Returns:
        torch.Tensor
            Normalized hyperparameter vector.
    """
    lr_min, lr_max = 1e-6, 3e-2
    hidden_min, hidden_max = 2, 256
    wd_min, wd_max = 0.0, 0.1
    # layer_min, layer_max = 2, 10

    lr_norm = (
        math.log10(lr) - math.log10(lr_min)
    ) / (math.log10(lr_max) - math.log10(lr_min) + 1e-8)

    hidden_norm = (
        math.log2(hidden_dim) - math.log2(hidden_min)
    ) / (math.log2(hidden_max) - math.log2(hidden_min) + 1e-8)

    weight_decay_norm = (weight_decay - wd_min) / (wd_max - wd_min + 1e-8)
    # layer_norm = (num_layer - layer_min) / (layer_max - layer_min)  # BUG FIX

    return torch.tensor(
        [lr_norm, hidden_norm, weight_decay_norm],
        dtype=torch.float32,
    )

def parse_key(key):
    """
    Parse a hyperparameter key string.

    Example:
        'lr3e-02_hd128_wd0.001'

    Args:
        key : str
            Encoded hyperparameter string.

    Returns:
        tuple
            (learning_rate, hidden_dim, weight_decay)
    """
    parts = key.split("_")
    lr = float(parts[0].replace("lr", ""))
    hd = float(parts[1].replace("hd", ""))
    wd = float(parts[2].replace("wd", ""))
    return lr, hd, wd

def get_fashion_mnist_loaders(batch_size=128, normalise=True):
    """
    Return train and test dataloaders for Fashion-MNIST.

    Args:
        batch_size : int, optional
            Batch size.
        normalise : bool, optional
            Whether to apply dataset normalization.

    Returns:
        train_loader, test_loader
            PyTorch DataLoader objects.
    """
    transform_list = [transforms.ToTensor()]

    if normalise:
        transform_list.append(
            transforms.Normalize((0.2860,), (0.3530,))
        )

    transform = transforms.Compose(transform_list)

    train_dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def min_max_normalize(y):
    """
    Normalize values using per-array min-max scaling.

    Args:
        y : list or torch.Tensor
            Input values to normalize.

    Returns:
        torch.Tensor
            Min-max normalized values.
    """
    if isinstance(y, list):
        y = torch.tensor(y, dtype=torch.float32)

    y_min = y.min()
    y_max = y.max()

    if y_max == y_min:
        return torch.zeros_like(y)

    return (y - y_min) / (y_max - y_min)

def fixed_range_normalize(y, y_min=0.0, y_max=1.0):
    """
    Normalize values to [0, 1] using a fixed range.

    Args:
        y : array-like
            Input values to normalize.
        y_min : float, optional
            Minimum value of the fixed range.
        y_max : float, optional
            Maximum value of the fixed range.

    Returns:
        np.ndarray
            Normalized values clipped to [0, 1].
    """
    y = np.array(y, dtype=float)
    y_clipped = np.clip(y, y_min, y_max)
    return (y_clipped - y_min) / (y_max - y_min)


import numpy as np



import re
import json
import numpy as np

def compute_min_max(hidden_dim_min, hidden_dim_max, buffer=0.05,
                     path="./results/results_metrics.json", min_value=1e-8):
    with open(path) as f:
        results = json.load(f)

    selected_losses = []
    for key, entry in results.items():
        match = re.search(r"_hd(\d+)_", key)
        if match is None:
            continue
        hd = int(match.group(1))
        if hidden_dim_min <= hd <= hidden_dim_max:
            curve = entry.get("val_loss_curve", [])
            if curve:
                selected_losses.append(np.asarray(curve))

    if not selected_losses:
        raise ValueError(
            f"No runs found with hidden_dim in [{hidden_dim_min}, {hidden_dim_max}]"
        )

    all_losses = np.concatenate(selected_losses)
    log_others = np.log(np.clip(all_losses, 1e-8, None))
    lo = float(log_others.min())
    hi = float(log_others.max())
    margin = (hi - lo) * buffer

    lo_adj = max(lo - margin, np.log(min_value))
    hi_adj = hi + margin
    if hi_adj <= lo_adj:
        hi_adj = lo_adj + 1e-6

    return lo_adj, hi_adj  # log-space, matches log_curve's units

def normalize_log_loss_curve(val_loss,eps=1e-8):
    """
    Converts validation loss into an IfBO-compatible score in [0,1]
    where higher = better (like accuracy).
    """

    # Convert to numpy
    val_loss = np.asarray(val_loss, dtype=np.float64)

    # Log-transform (stabilizes scale if losses vary a lot)
    log_curve = np.log(val_loss + eps)

    # Get reference range (from dataset / prior evaluations)
    log_min, log_max = compute_min_max(4, 128)

    print("log range:", log_min, log_max)

    # Handle degenerate case
    if abs(log_max - log_min) < 1e-12:
        return np.ones_like(log_curve)

    # Min-max normalize (loss space: 0=good, 1=bad)
    norm = (log_curve - log_min) / (log_max - log_min)

    # Clip to valid range
    norm = np.clip(norm, 0.0, 1.0)

    # Invert → convert to "accuracy-like" score
    score = 1.0 - norm

    return score

def fixed_normalize(val_loss,eps=1e-8):
    """
    Converts validation loss into an IfBO-compatible score in [0,1]
    where higher = better (like accuracy).
    """

    # Convert to numpy
    val_loss = np.asarray(val_loss, dtype=np.float64)

    # Log-transform (stabilizes scale if losses vary a lot)
    log_curve = np.log(val_loss + eps)

    # Get reference range (from dataset / prior evaluations)
    log_min, log_max = min(log_curve), max(log_curve)
    print("log range:", log_min, log_max)

    # Handle degenerate case
    if abs(log_max - log_min) < 1e-12:
        return np.ones_like(log_curve)

    # Min-max normalize (loss space: 0=good, 1=bad)
    norm = (log_curve - log_min) / (log_max - log_min)

    # Clip to valid range
    norm = np.clip(norm, 0.0, 1.0)

    # Invert → convert to "accuracy-like" score
    score = 1.0 - norm

    return score