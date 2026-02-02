# valuenet_tc.py
import torch
import torch.nn as nn

class ValueNetTC(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        h = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(2, h),
            nn.Tanh(),
            nn.Linear(h, h),
            nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.view(1, -1)
        y = self.net(x)      # (B,1)
        return y.view(-1)    # (B,)
