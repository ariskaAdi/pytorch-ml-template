"""Example model. Replace with your project's architecture."""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float = 0.0
    ):
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_dim) -> logits: (batch, num_classes)"""
        return self.net(x)
