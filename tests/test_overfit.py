"""The most useful sanity test in ML: a correct model + loss must memorize one batch.

If this fails, there is a bug (wrong loss, detached graph, bad init, ...). Find it
before spending GPU hours.
"""

import torch

from mltemplate.models.mlp import MLP
from mltemplate.tasks.classification import classification_loss


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = MLP(in_dim=32, hidden_dim=128, num_layers=2, num_classes=4)
    # Random labels: the only way to reach ~0 loss is to memorize the batch.
    batch = {"x": torch.randn(16, 32), "y": torch.randint(0, 4, (16,))}
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(300):
        loss, metrics = classification_loss(model, batch)
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss.item() < 0.01
    assert metrics["accuracy"].item() == 1.0
