"""Synthetic classification data: Gaussian clusters, one per class.

The clusters overlap on purpose (best possible accuracy is ~84% with the default
settings), so loss curves look like a real training run instead of dropping to 0.

It is only here so the template runs end-to-end without downloads. Replace it with
your project's dataset (keep the same contract: items are dicts of tensors).
"""

import torch
from torch.utils.data import Dataset


class SyntheticClassification(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        assert len(x) == len(y)
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[idx], "y": self.y[idx]}


def make_synthetic_splits(
    num_samples: int,
    in_dim: int,
    num_classes: int,
    noise: float = 1.0,
    separation: float = 0.4,
    val_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[SyntheticClassification, SyntheticClassification]:
    # Own generator: the data is identical on every process and every run.
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_classes, in_dim, generator=g) * separation
    y = torch.randint(0, num_classes, (num_samples,), generator=g)
    x = centers[y] + noise * torch.randn(num_samples, in_dim, generator=g)

    n_val = int(num_samples * val_fraction)
    perm = torch.randperm(num_samples, generator=g)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (
        SyntheticClassification(x[train_idx], y[train_idx]),
        SyntheticClassification(x[val_idx], y[val_idx]),
    )
