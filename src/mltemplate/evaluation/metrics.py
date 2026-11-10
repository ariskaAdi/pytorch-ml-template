import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (logits.argmax(dim=-1) == targets).float().mean()
