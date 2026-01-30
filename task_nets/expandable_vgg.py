from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VGGBlockSpec:
    num_convs: int
    out_channels: int


class ExpandableVGG(nn.Module):
    """
    VGG small espandibile (per canali) con task-aware inference.

    Convenzione azioni:
      - controller campiona actions[b] in [0..action_spec[b]-1]
      - delta_channels_b = actions[b] * delta_step
    """

    def __init__(
        self,
        in_channels: int,
        block_specs: List[List[int]] | List[Tuple[int, int]],
        num_classes: int,
        action_spec: List[int],
        delta_step: int = 8,
        use_gap: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.block_specs: List[VGGBlockSpec] = [VGGBlockSpec(int(nc), int(oc)) for nc, oc in block_specs]
        self.num_classes = int(num_classes)
        self._action_spec = list(action_spec)
        self.delta_step = int(delta_step)
        self.use_gap = bool(use_gap)
        self.dropout = float(dropout)

        # task_slices[j] = [C1_j, ..., CB_j]
        self.task_slices: List[List[int]] = []

        self._build()

    def action_spec(self) -> List[int]:
        return list(self._action_spec)

    def register_task_slice(self):
        self.task_slices.append([int(s.out_channels) for s in self.block_specs])

    def actions_to_deltas(self, actions: List[int]) -> List[int]:
        if len(actions) != len(self.block_specs):
            raise ValueError(f"Expected {len(self.block_specs)} actions, got {len(actions)}")
        return [int(a) * int(self.delta_step) for a in actions]

    def actions_complexity(self, actions: List[int]) -> float:
        # penalizziamo la somma dei delta canali (proxy semplice ma sensato)
        return float(sum(self.actions_to_deltas(actions)))

    def _build(self):
        self.blocks = nn.ModuleList()
        cin = self.in_channels

        for spec in self.block_specs:
            layers = []
            for _ in range(spec.num_convs):
                layers.append(nn.Conv2d(cin, spec.out_channels, kernel_size=3, padding=1, bias=True))
                layers.append(nn.ReLU(inplace=True))
                cin = spec.out_channels
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.blocks.append(nn.Sequential(*layers))

        self.feat_out_channels = cin

        if not self.use_gap:
            raise ValueError("use_gap=False non supportato qui (evita dipendere da H,W).")

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.feat_out_channels, self.num_classes)
        self.drop = nn.Dropout(self.dropout) if self.dropout > 0 else None

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        if task_id is None:
            return self._forward_full(x)
        return self._forward_task(x, int(task_id))

    def _forward_full(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.blocks:
            h = blk(h)
        h = self.gap(h).flatten(1)
        if self.drop is not None:
            h = self.drop(h)
        return self.classifier(h)

    def _forward_task(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        if task_id < 0 or task_id >= len(self.task_slices):
            raise ValueError(f"task_id={task_id} out of range (0..{len(self.task_slices)-1}).")

        chans = self.task_slices[task_id]  # [C1, ..., CB]
        h = x
        for b, blk in enumerate(self.blocks):
            h = self._forward_block_sliced(blk, h, chans[b])

        h = self.gap(h).flatten(1)
        if self.drop is not None:
            h = self.drop(h)

        c_last = chans[-1]
        w = self.classifier.weight[:, :c_last]
        b = self.classifier.bias
        return F.linear(h[:, :c_last], w, b)

    def _forward_block_sliced(self, blk: nn.Sequential, x: torch.Tensor, c_out: int) -> torch.Tensor:
        h = x
        for layer in blk:
            if isinstance(layer, nn.Conv2d):
                cin = h.shape[1]
                w = layer.weight[:c_out, :cin, :, :]
                b = layer.bias[:c_out] if layer.bias is not None else None
                h = F.conv2d(h, w, b, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
            elif isinstance(layer, nn.ReLU):
                h = F.relu(h, inplace=False)
                h = h[:, :c_out, :, :]
            else:
                h = layer(h)
        return h

    def expanded_copy(self, actions: List[int]) -> "ExpandableVGG":
        if len(actions) != len(self.block_specs):
            raise ValueError(f"Expected {len(self.block_specs)} actions, got {len(actions)}")

        deltas = self.actions_to_deltas(actions)
        new_specs = [(spec.num_convs, int(spec.out_channels) + int(d)) for spec, d in zip(self.block_specs, deltas)]

        child = ExpandableVGG(
            in_channels=self.in_channels,
            block_specs=new_specs,
            num_classes=self.num_classes,
            action_spec=self._action_spec,
            delta_step=self.delta_step,
            use_gap=self.use_gap,
            dropout=self.dropout,
        )
        child.task_slices = [list(s) for s in self.task_slices]

        with torch.no_grad():
            for old_blk, new_blk in zip(self.blocks, child.blocks):
                for old_layer, new_layer in zip(old_blk, new_blk):
                    if isinstance(old_layer, nn.Conv2d) and isinstance(new_layer, nn.Conv2d):
                        ow, ob = old_layer.weight, old_layer.bias
                        nw, nb = new_layer.weight, new_layer.bias
                        out_common = min(ow.shape[0], nw.shape[0])
                        in_common = min(ow.shape[1], nw.shape[1])
                        nw[:out_common, :in_common].copy_(ow[:out_common, :in_common])
                        if ob is not None and nb is not None:
                            nb[:out_common].copy_(ob[:out_common])

            ow, ob = self.classifier.weight, self.classifier.bias
            nw, nb = child.classifier.weight, child.classifier.bias
            out_common = min(ow.shape[0], nw.shape[0])
            in_common = min(ow.shape[1], nw.shape[1])
            nw[:out_common, :in_common].copy_(ow[:out_common, :in_common])
            nb[:out_common].copy_(ob[:out_common])

        return child

    def apply_freeze_policy(self, parent: "ExpandableVGG") -> bool:
        """
        Paper-style: aggiorna solo i canali NUOVI rispetto al parent.
        Implementazione via gradient mask (register_hook) come nel tuo MLP.

        Assunzione: stessi num_convs per blocco; cambia solo out_channels.
        """
        if not isinstance(parent, ExpandableVGG):
            return False
        if len(parent.block_specs) != len(self.block_specs):
            return False

        old_ch = [int(s.out_channels) for s in parent.block_specs]
        new_ch = [int(s.out_channels) for s in self.block_specs]

        # Maschera conv per conv, bloccando:
        # - tutti i filtri "vecchi" (out < old_out)
        # - e, nei layer successivi, le connessioni verso input channels vecchi (se vuoi più rigore)
        #
        # Qui facciamo: per ogni conv, permettiamo grad solo su:
        # - nuovi output channels (old_out:new_out)
        # - e su input channels nuovi (se esistono) quando il layer li vede.
        masked_any = False

        prev_old_out = self.in_channels
        prev_new_out = self.in_channels

        for b, (blk, o_out, n_out) in enumerate(zip(self.blocks, old_ch, new_ch)):
            # dentro un blocco ci sono N conv; per VGG, tutte producono n_out canali
            for layer in blk:
                if not isinstance(layer, nn.Conv2d):
                    continue

                ow = layer.weight
                ob = layer.bias

                m_w = torch.zeros_like(ow)
                m_b = torch.zeros_like(ob) if ob is not None else None

                # Permetti grad sui nuovi filtri in uscita
                if n_out > o_out:
                    m_w[o_out:n_out, :, :, :] = 1.0
                    if m_b is not None:
                        m_b[o_out:n_out] = 1.0
                    masked_any = True

                # Permetti grad anche sulle connessioni verso nuovi input channels (se presenti)
                # (questo serve perché se un blocco precedente è cresciuto, il primo conv del blocco
                # successivo deve potersi adattare ai nuovi canali in ingresso)
                cin = ow.shape[1]
                old_cin = min(int(prev_old_out), cin)
                new_cin = min(int(prev_new_out), cin)
                if new_cin > old_cin:
                    m_w[:, old_cin:new_cin, :, :] = 1.0
                    masked_any = True

                layer.weight.register_hook(lambda g, m=m_w: g * m.to(g.device))
                if layer.bias is not None and m_b is not None:
                    layer.bias.register_hook(lambda g, m=m_b: g * m.to(g.device))

                # dopo una conv, per VGG l'output channels diventa n_out (o_out per il parent)
                prev_old_out = o_out
                prev_new_out = n_out

        # classifier: permette grad solo sulle nuove features (canali finali aggiunti)
        ow = self.classifier.weight
        m_w = torch.zeros_like(ow)
        o_last = old_ch[-1]
        n_last = new_ch[-1]
        if n_last > o_last:
            m_w[:, o_last:n_last] = 1.0
            masked_any = True
        self.classifier.weight.register_hook(lambda g, m=m_w: g * m.to(g.device))

        # bias classifier: in genere lo terrei congelato (come nel tuo MLP layer2 bias mask=0)
        mb = torch.zeros_like(self.classifier.bias)
        self.classifier.bias.register_hook(lambda g, m=mb: g * m.to(g.device))

        return masked_any
    
    def task_costs(self, device=None) -> torch.Tensor:
        # costo per task = somma canali per blocco (proxy semplice)
        dev = device if device is not None else next(self.parameters()).device
        return torch.tensor([sum(chs) for chs in self.task_slices], dtype=torch.float32, device=dev)

