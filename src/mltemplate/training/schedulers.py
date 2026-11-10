import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_factor(
    step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float
) -> float:
    """LR multiplier: linear warmup to 1.0, then cosine decay to min_lr_ratio."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_warmup_cosine(
    optimizer: Optimizer, warmup_steps: int, max_steps: int, min_lr_ratio: float
) -> LambdaLR:
    return LambdaLR(
        optimizer,
        lambda step: warmup_cosine_factor(step, warmup_steps, max_steps, min_lr_ratio),
    )
