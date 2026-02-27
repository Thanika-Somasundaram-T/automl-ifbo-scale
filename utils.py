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
    ) / (math.log10(lr_max) - math.log10(lr_min))

    hidden_norm = (
        math.log2(hidden_dim) - math.log2(hidden_min)
    ) / (math.log2(hidden_max) - math.log2(hidden_min))

    weight_decay_norm = (weight_decay - wd_min) / (wd_max - wd_min)
    # layer_norm = (num_layer - layer_min) / (layer_max - layer_min)  # BUG FIX

    return torch.tensor(
        [lr_norm, hidden_norm, weight_decay_norm],
        dtype=torch.float32,
    ).clamp(0.0, 1.0)

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
import math

def normalize_log_loss_curve(curve_values, loss_min: float= 0.01, loss_max: float = 2.0) -> np.ndarray:
    """
    Convert a validation loss curve into IFBO-compatible performance values.

    Steps:
    1) log-transform losses
    2) min-max normalize in log-space using global bounds
    3) invert so higher = better
    4) clip to [0,1]

    Args:
        curve_values : list or array-like
            Raw validation loss values (>0)
        loss_min : float
            Global minimum expected loss
        loss_max : float
            Global maximum expected loss

    Returns:
        np.ndarray
            Normalized performance values in [0,1]
    """
    curve_values = np.array(curve_values, dtype=float)

    if np.any(curve_values <= 0):
        raise ValueError("Loss values must be positive for log transform.")

    log_losses = np.log(curve_values)

    log_min = math.log(loss_min)
    log_max = math.log(loss_max)

    norm = (log_losses - log_min) / (log_max - log_min)

    y = 1.0 - norm
    return np.clip(y, 0.0, 1.0)


def unnormalize_pred(norm_curve, loss_min=1e-6, loss_max=3.0):
    """
    Convert normalized log-loss predictions back to actual NLL.
    """
    norm_curve = np.clip(norm_curve, 0.0, 1.0)
    log_min = math.log(loss_min)
    log_max = math.log(loss_max)
    log_losses = log_min + (1.0 - norm_curve) * (log_max - log_min)
    return np.exp(log_losses)
