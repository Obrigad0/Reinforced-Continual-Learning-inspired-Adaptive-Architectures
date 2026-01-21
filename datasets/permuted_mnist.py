from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class PermutePixelsTransform:
    def __init__(self, perm: torch.Tensor):
        self.perm = perm

    def __call__(self, x):
        # x: Tensor [1,28,28]
        v = x.view(-1)          # 784
        v = v[self.perm]        # permutazione arbitraria degli indici
        return v                # ritorno flat: perfetto per MLP


@dataclass
class PermutedMNISTDataset:
    data_dir: str
    num_tasks: int
    permutation_seed: int
    train_val_split: float
    num_workers: int = 2

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.permutation_seed)
        self.perms = [torch.randperm(28 * 28, generator=g) for _ in range(self.num_tasks)]

    def _make_loaders(self, task_id: int, batch_size: int):
        perm = self.perms[task_id]
        tfm = transforms.Compose([transforms.ToTensor(), PermutePixelsTransform(perm)])

        train_full = datasets.MNIST(self.data_dir, train=True, download=True, transform=tfm)
        test = datasets.MNIST(self.data_dir, train=False, download=True, transform=tfm)

        n = len(train_full)
        n_train = int(n * self.train_val_split)
        n_val = n - n_train
        train_ds, val_ds = random_split(train_full, [n_train, n_val], generator=torch.Generator().manual_seed(task_id))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=self.num_workers, pin_memory=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True)

        return train_loader, val_loader, test_loader

    def get_task(self, task_id: int, batch_size: int) -> Dict[str, DataLoader]:
        tr, va, te = self._make_loaders(task_id, batch_size)
        return {"train": tr, "val": va, "test": te}
