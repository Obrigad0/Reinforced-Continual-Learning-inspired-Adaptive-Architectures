from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpandableMLP(nn.Module):
    """
    MLP espandibile: 784 -> h1 -> h2 -> 10.
    Le azioni controllano la crescita di (h1, h2, input dell'output).
    Per replicare il paper: action_spec tipicamente [30,30,30] e azioni in 0..29. 
    """
    def __init__(self, input_dim: int, hidden_sizes: List[int], num_classes: int, action_spec: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.num_classes = num_classes
        self._action_spec = list(action_spec)
        self._build()

    def action_spec(self) -> List[int]:
        # Ritorna la size della softmax per step (n_i). Qui = [30,30,30] nel setup MNIST. 
        return list(self._action_spec)

    def _build(self):
        dims = [self.input_dim] + self.hidden_sizes + [self.num_classes]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return h

    def expanded_copy(self, actions: List[int]) -> "ExpandableMLP":
        """
        actions: [a0, a1, a2] (len = 3) per MLP con 3 layer lineari.
        - a0 cresce hidden1
        - a1 cresce hidden2
        - a2 (per semplicità) cresce hidden2 (equivalente a far crescere input dell'output)
        """
        if len(actions) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} actions, got {len(actions)}")

        # nuova dimensione hidden
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
