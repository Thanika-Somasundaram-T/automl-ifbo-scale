import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP2(nn.Module):
    """

    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class MLP4(nn.Module):
    """
    
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        
        x = self.fc4(x)
        return x
    
class MLP10(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.fc10 = nn.Linear(hidden_dim, 10)  # output layer

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = self.fc10(x)
        return x

    