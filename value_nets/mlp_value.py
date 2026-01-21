import torch
import torch.nn as nn

class MLPValueNet(nn.Module):
    def __init__(self, hiddensize: int):
        super().__init__()
        self.fc1 = nn.Linear(1, hiddensize)
        self.fc2 = nn.Linear(hiddensize, 1)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: [B,1] float, sul device corretto
        h = torch.relu(self.fc1(s))
        v = self.fc2(h)
        return v.squeeze(-1)
