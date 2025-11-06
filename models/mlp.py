import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP2(nn.Module):
    """
    Two-layer MLP for image classification (e.g., MNIST or CIFAR10 flattened).
    Includes dropout for regularization.
    """

    def __init__(self, hidden_dim=256, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 10)

    def forward(self, x):
        # Flatten input (batch_size, 1, 28, 28) → (batch_size, 784)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # apply dropout only during training
        x = self.fc2(x)
        return x
