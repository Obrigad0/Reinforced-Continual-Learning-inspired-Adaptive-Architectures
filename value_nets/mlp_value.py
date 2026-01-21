import torch
import torch.nn as nn


class MLPValueNet(nn.Module):
    """
    Value net minimale: nel paper lo 'state' è fisso per task durante il training del controller,
    quindi qui usiamo un input costante.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, device: torch.device) -> torch.Tensor:
        x = torch.ones(1, 1, device=device)
        h = torch.relu(self.fc1(x))
        v = self.fc2(h)
        return v.squeeze()
