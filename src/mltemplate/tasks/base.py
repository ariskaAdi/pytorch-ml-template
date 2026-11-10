from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

# loss_fn(model, batch) -> (loss, extra metrics). Metrics are averaged and logged.
LossFn = Callable[[nn.Module, Any], tuple[torch.Tensor, dict[str, torch.Tensor]]]


@dataclass
class Task:
    model: nn.Module
    train_dataset: Dataset
    val_dataset: Dataset
    loss_fn: LossFn
