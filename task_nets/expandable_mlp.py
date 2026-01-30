from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandableMLP(nn.Module):
    """
    MLP espandibile: input_dim -> h1 -> h2 -> num_classes
    + Task-aware inference: forward(x, task_id) usa slicing dei pesi.
    + Modularità: apply_freeze_policy(parent=...) e actions_complexity(actions).
    """

    def __init__(self, input_dim: int, hidden_sizes: List[int], num_classes: int, action_spec: List[int]):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.num_classes = int(num_classes)
        self._action_spec = list(action_spec)

        self.task_slices: List[Tuple[int, int]] = []
        self._build()

    def action_spec(self) -> List[int]:
        return list(self._action_spec)

    def _build(self):
        dims = [self.input_dim] + self.hidden_sizes + [self.num_classes]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def register_task_slice(self):
        if len(self.hidden_sizes) < 2:
            raise ValueError("Expected hidden_sizes like [h1, h2].")
        self.task_slices.append((int(self.hidden_sizes[0]), int(self.hidden_sizes[1])))

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if task_id is None:
            return self._forward_full(x)
        return self._forward_task(x, int(task_id))

    def _forward_full(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def _forward_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if task_id < 0 or task_id >= len(self.task_slices):
            raise ValueError(f"task_id={task_id} out of range (0..{len(self.task_slices)-1}).")

        h1_j, h2_j = self.task_slices[task_id]

        l0 = self.layers[0]
        w0 = l0.weight[:h1_j, :]
        b0 = l0.bias[:h1_j]
        h1 = F.relu(F.linear(x, w0, b0))

        l1 = self.layers[1]
        w1 = l1.weight[:h2_j, :h1_j]
        b1 = l1.bias[:h2_j]
        h2 = F.relu(F.linear(h1, w1, b1))

        l2 = self.layers[2]
        w2 = l2.weight[:, :h2_j]
        b2 = l2.bias
        out = F.linear(h2, w2, b2)
        return out

    def expanded_copy(self, actions: List[int]) -> "ExpandableMLP":
        if len(actions) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} actions, got {len(actions)}")

        # Qui actions sono già "delta" perché nel tuo setup MLP usavi action_spec grandi (30)
        # e trattavi l'indice come delta diretto.
        new_hidden = self.hidden_sizes.copy()
        new_hidden[0] += int(actions[0])
        if len(new_hidden) > 1:
            new_hidden[1] += int(actions[1]) + int(actions[2])

        child = ExpandableMLP(
            input_dim=self.input_dim,
            hidden_sizes=new_hidden,
            num_classes=self.num_classes,
            action_spec=self._action_spec,
        )

        child.task_slices = list(self.task_slices)

        with torch.no_grad():
            for old_layer, new_layer in zip(self.layers, child.layers):
                ow, ob = old_layer.weight, old_layer.bias
                nw, nb = new_layer.weight, new_layer.bias
                out_common = min(ow.shape[0], nw.shape[0])
                in_common = min(ow.shape[1], nw.shape[1])
                nw[:out_common, :in_common].copy_(ow[:out_common, :in_common])
                nb[:out_common].copy_(ob[:out_common])

        return child

    def actions_complexity(self, actions: List[int]) -> float:
        # Manteniamo il comportamento storico del tuo codice (proxy semplice).
        return float(sum(actions))

    def apply_freeze_policy(self, parent: "ExpandableMLP") -> bool:
        """
        Paper-style: allena solo i parametri nuovi. Implementazione uguale a quella
        che avevi in rcl.py ma spostata dentro al modello per modularità.
        """
        if not hasattr(self, "layers") or not hasattr(parent, "hidden_sizes"):
            return False
        if len(self.layers) != 3 or len(self.hidden_sizes) < 2:
            return False
        if parent.hidden_sizes is None or len(parent.hidden_sizes) < 2:
            return False

        old_h1, old_h2 = int(parent.hidden_sizes[0]), int(parent.hidden_sizes[1])
        new_h1, new_h2 = int(self.hidden_sizes[0]), int(self.hidden_sizes[1])

        # layer0
        l0 = self.layers[0]
        m_w0 = torch.zeros_like(l0.weight)
        m_b0 = torch.zeros_like(l0.bias)
        if new_h1 > old_h1:
            m_w0[old_h1:new_h1, :] = 1.0
            m_b0[old_h1:new_h1] = 1.0
        l0.weight.register_hook(lambda g, m=m_w0: g * m.to(g.device))
        l0.bias.register_hook(lambda g, m=m_b0: g * m.to(g.device))

        # layer1
        l1 = self.layers[1]
        m_w1 = torch.zeros_like(l1.weight)
        m_b1 = torch.zeros_like(l1.bias)
        if new_h2 > old_h2:
            m_w1[old_h2:new_h2, :] = 1.0
            m_b1[old_h2:new_h2] = 1.0
        if new_h1 > old_h1:
            m_w1[:, old_h1:new_h1] = 1.0
        l1.weight.register_hook(lambda g, m=m_w1: g * m.to(g.device))
        l1.bias.register_hook(lambda g, m=m_b1: g * m.to(g.device))

        # layer2
        l2 = self.layers[2]
        m_w2 = torch.zeros_like(l2.weight)
        m_b2 = torch.zeros_like(l2.bias)
        if new_h2 > old_h2:
            m_w2[:, old_h2:new_h2] = 1.0
        m_b2[:] = 0.0
        l2.weight.register_hook(lambda g, m=m_w2: g * m.to(g.device))
        l2.bias.register_hook(lambda g, m=m_b2: g * m.to(g.device))

        return True
    
    def task_costs(self, device=None) -> torch.Tensor:
        # costo per task = h1 + h2 (come il vecchio slice_cost)
        dev = device if device is not None else next(self.parameters()).device
        return torch.tensor([a + b for (a, b) in self.task_slices], dtype=torch.float32, device=dev)
