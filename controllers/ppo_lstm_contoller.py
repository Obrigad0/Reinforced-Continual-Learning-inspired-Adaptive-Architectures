from __future__ import annotations
from typing import List, Tuple, Dict
import torch
import torch.nn as nn


class PPOLSTMController(nn.Module):
    """
    Controller LSTM autoregressivo come lstm_controller.py, ma con API extra per PPO:
    - sample(device) -> (actions: List[int], logp_sum: Tensor)
    - log_prob(actions, device) -> logp_sum: Tensor
    - entropy(actions, device) -> ent_sum: Tensor
    """

    def __init__(self, hiddensize: int, numlayers: int, actionspec: List[int]):
        super().__init__()
        self.hiddensize = int(hiddensize)
        self.numlayers = int(numlayers)
        self.actionspec = list(actionspec)

        self.numsteps = len(self.actionspec)
        self.maxactions = max(self.actionspec)

        self.actionemb = nn.Embedding(self.maxactions, self.hiddensize)
        self.lstm = nn.LSTM(
            input_size=self.hiddensize,
            hidden_size=self.hiddensize,
            num_layers=self.numlayers,
            batch_first=True,
        )
        self.heads = nn.ModuleList([nn.Linear(self.hiddensize, n) for n in self.actionspec])
        self.starttoken = nn.Parameter(torch.zeros(self.hiddensize))

    @torch.no_grad()
    def sample(self, device: torch.device) -> Tuple[List[int], torch.Tensor]:
        actions: List[int] = []
        logps: List[torch.Tensor] = []

        hx = None
        inp = self.starttoken.view(1, 1, -1).to(device)

        for i in range(self.numsteps):
            out, hx = self.lstm(inp, hx)
            h = out[:, -1, :]                 # (1,H)
            logits = self.heads[i](h)         # (1,n_i)
            dist = torch.distributions.Categorical(logits=logits)

            a = dist.sample()                 # (1,)
            logp = dist.log_prob(a)           # (1,)

            actions.append(int(a.item()))
            logps.append(logp)

            inp = self.actionemb(a).view(1, 1, -1)

        return actions, torch.stack(logps).sum()

    def _teacher_force(self, actions: List[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(actions) != self.numsteps:
            raise ValueError(f"Expected {self.numsteps} actions, got {len(actions)}")

        hx = None
        inp = self.starttoken.view(1, 1, -1).to(device)

        logps: List[torch.Tensor] = []
        ents: List[torch.Tensor] = []

        for i, ai in enumerate(actions):
            out, hx = self.lstm(inp, hx)
            h = out[:, -1, :]
            logits = self.heads[i](h)
            dist = torch.distributions.Categorical(logits=logits)

            a = torch.tensor([int(ai)], device=device, dtype=torch.long)
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())

            inp = self.actionemb(a).view(1, 1, -1)

        return torch.stack(logps).sum(), torch.stack(ents).sum()

    def log_prob(self, actions: List[int], device: torch.device) -> torch.Tensor:
        lp, _ = self._teacher_force(actions, device)
        return lp

    def entropy(self, actions: List[int], device: torch.device) -> torch.Tensor:
        _, ent = self._teacher_force(actions, device)
        return ent
