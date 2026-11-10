import torch
from torch.utils.data import DataLoader

from mltemplate.data.synthetic import make_synthetic_splits


def test_split_sizes_and_items():
    train, val = make_synthetic_splits(1000, in_dim=8, num_classes=3, val_fraction=0.2)
    assert len(train) == 800 and len(val) == 200
    item = train[0]
    assert item["x"].shape == (8,) and item["x"].dtype == torch.float32
    assert item["y"].dtype == torch.int64 and 0 <= item["y"] < 3


def test_same_seed_same_data():
    a, _ = make_synthetic_splits(100, 8, 3, seed=1)
    b, _ = make_synthetic_splits(100, 8, 3, seed=1)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)


def test_loader_batches():
    train, _ = make_synthetic_splits(256, in_dim=8, num_classes=3)
    batch = next(iter(DataLoader(train, batch_size=32)))
    assert batch["x"].shape == (32, 8)
    assert batch["y"].shape == (32,)
    assert torch.isfinite(batch["x"]).all()
