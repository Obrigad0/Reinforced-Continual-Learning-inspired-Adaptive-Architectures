from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandableMLP(nn.Module):
    """
    MLP espandibile: 784 -> h1 -> h2 -> 10.
    + Task-aware inference: forward(x, task_id) usa slicing dei pesi fino alle dimensioni del task.
    """

    def __init__(self, input_dim: int, hidden_sizes: List[int], num_classes: int, action_spec: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)  # [h1, h2]
        self.num_classes = num_classes
        self._action_spec = list(action_spec)

        # task_slices[j] = (h1_j, h2_j) dimensioni "timestamped" dopo aver finito il task j
        self.task_slices: List[Tuple[int, int]] = []

        self._build()

    def action_spec(self) -> List[int]:
        return list(self._action_spec)

    def _build(self):
        dims = [self.input_dim] + self.hidden_sizes + [self.num_classes]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def register_task_slice(self):
        """Chiama questo quando hai finito di addestrare il task corrente (dopo la promozione)."""
        if len(self.hidden_sizes) < 2:
            raise ValueError("Expected hidden_sizes like [h1, h2].")
        self.task_slices.append((int(self.hidden_sizes[0]), int(self.hidden_sizes[1])))

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
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

        # layer0: input -> h1
        l0 = self.layers[0]
        w0 = l0.weight[:h1_j, :]
        b0 = l0.bias[:h1_j]
        h1 = F.relu(F.linear(x, w0, b0))

        # layer1: h1 -> h2
        l1 = self.layers[1]
        w1 = l1.weight[:h2_j, :h1_j]
        b1 = l1.bias[:h2_j]
        h2 = F.relu(F.linear(h1, w1, b1))

        # layer2: h2 -> out
        l2 = self.layers[2]
        w2 = l2.weight[:, :h2_j]
        b2 = l2.bias  # bias condiviso (nel nostro freezing lo blocchiamo dopo task 0)
        out = F.linear(h2, w2, b2)
        return out

    def expanded_copy(self, actions: List[int]) -> "ExpandableMLP":
        if len(actions) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} actions, got {len(actions)}")

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

        # copia lo storico task_slices (timestamp) nel child
        child.task_slices = list(self.task_slices)

        # copia pesi nella parte comune
        with torch.no_grad():
            for old_layer, new_layer in zip(self.layers, child.layers):
                ow, ob = old_layer.weight, old_layer.bias
                nw, nb = new_layer.weight, new_layer.bias
                out_common = min(ow.shape[0], nw.shape[0])
                in_common = min(ow.shape[1], nw.shape[1])
                nw[:out_common, :in_common].copy_(ow[:out_common, :in_common])
                nb[:out_common].copy_(ob[:out_common])

        return child
