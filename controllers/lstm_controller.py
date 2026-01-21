from typing import List, Tuple

import torch
import torch.nn as nn


class LSTMController(nn.Module):
    """
    Controller autoregressivo: genera una azione discreta per ogni layer i.
    action_spec = [n1, n2, ..., nm] dove ni è la size della softmax al passo i.
    """
    def __init__(self, hidden_size: int, num_layers: int, action_spec: List[int]):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.action_spec = list(action_spec)
        self.num_steps = len(self.action_spec)

        self.max_actions = max(self.action_spec)
        self.action_emb = nn.Embedding(self.max_actions, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, n) for n in self.action_spec])
        self.start_token = nn.Parameter(torch.zeros(hidden_size))

    def sample(self, device: torch.device) -> Tuple[List[int], torch.Tensor]:
        actions = []
        logps = []
        hx = None
        inp = self.start_token.view(1, 1, -1).to(device)

        for i in range(self.num_steps):
            out, hx = self.lstm(inp, hx)
            h = out[:, -1, :]  # [1,H]
            logits = self.heads[i](h)  # [1, ni]
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()  # [1]
            actions.append(int(a.item()))
            logps.append(dist.log_prob(a))  # [1]
            inp = self.action_emb(a).view(1, 1, -1)

        logp_sum = torch.stack(logps).sum()
        return actions, logp_sum
