import torch
import torch.nn as nn

class RouterActorCritic(nn.Module):
    def __init__(self, num_tasks: int, input_dim: int = 784, hidden: int = 1024, emb: int = 256, dropout: float = 0.1):
        super().__init__()
        self.num_tasks = int(num_tasks)
        self.input_dim = int(input_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden, emb),
            nn.LayerNorm(emb),
            nn.GELU(),
        )

        self.actor = nn.Linear(emb, self.num_tasks)
        self.critic = nn.Linear(emb, 1)

    def forward(self, x: torch.Tensor):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        z = self.encoder(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def greedy_task(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)
