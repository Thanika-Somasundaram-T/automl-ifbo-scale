import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from ifbo import Curve
import os
import math

def get_device():
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
    
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar_loaders(batch_size=128, normalise=True):
    """
    Return train and test dataloaders for CIFAR-10.
    
    Args:
        batch_size (int): batch size
        normalise (bool): whether to normalize
        
    Returns:
        train_loader, test_loader
    """
    transform_list = [transforms.ToTensor()]
    
    if normalise:
        # CIFAR-10 mean/std
        transform_list.append(transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        ))
    
    transform = transforms.Compose(transform_list)
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_mnist_loaders(batch_size=128, normalise=True):
    """
        Return train and test dataloaders for MNIST
        
        Args:
            batch_size (int):
            normalise (bool):
            
        Returns:
            train_loader, test_loader
    """
    transform_list = [transforms.ToTensor()]
    if normalise:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081, )))
        
    transform = transforms.Compose(transform_list)
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



def save_context_curves(curves, path="./context_curves.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save normally
    torch.save(curves, path)
    print(f"Saved {len(curves)} context curves to {path}")


def load_context_curves(path="./context_2layer.pt"):
    if not os.path.exists(path):
        print(f"No context curves found at {path}")
        return []

    # Allow ifbo.utils.Curve for unpickling
    with torch.serialization.safe_globals([Curve]):
        curves = torch.load(path, weights_only=False)
    print(f"Loaded {len(curves)} context curves from {path}")
    return curves

def normalize_hyperparameters(lr, hidden_dim, weight_decay):
    """
    Normalize hyperparameters to [0,1] range for FT-PFN.
    Learning rate is normalized in log-scale.
    """
    # Define min/max for each hyperparameter
    lr_min, lr_max = 1e-5, 1e-1
    hidden_min, hidden_max = 64, 512
    wd_min, wd_max = 0.0, 0.1

    # Normalize each hyperparameter
    lr_norm = (math.log10(lr) - math.log10(lr_min)) / (math.log10(lr_max) - math.log10(lr_min))
    hidden_norm = (hidden_dim - hidden_min) / (hidden_max - hidden_min)
    weight_decay_norm = (weight_decay - wd_min) / (wd_max - wd_min)

    # Clamp to [0,1] just in case
    return torch.tensor([lr_norm, hidden_norm, weight_decay_norm], dtype=torch.float32).clamp(0.0, 1.0)


def parse_key(key):
    """
    Parse keys like 'lr3e-02_hd128_wd0.001'
    into floats (lr, hd, wd)
    """
    parts = key.split("_")
    lr = float(parts[0].replace("lr", ""))
    hd = float(parts[1].replace("hd", ""))
    wd = float(parts[2].replace("wd", ""))
    return lr, hd, wd
