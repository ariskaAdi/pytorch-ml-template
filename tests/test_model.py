import torch

from mltemplate.models.mlp import MLP


def test_forward_shape():
    model = MLP(in_dim=32, hidden_dim=64, num_layers=2, num_classes=4)
    out = model(torch.randn(8, 32))
    assert out.shape == (8, 4)


def test_backward_reaches_all_params():
    model = MLP(in_dim=32, hidden_dim=64, num_layers=2, num_classes=4)
    model(torch.randn(8, 32)).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
