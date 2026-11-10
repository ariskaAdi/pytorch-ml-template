import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

from mltemplate.data.synthetic import make_synthetic_splits
from mltemplate.evaluation.metrics import accuracy
from mltemplate.models.mlp import MLP
from mltemplate.tasks.base import Task


def build_model(model_cfg: DictConfig) -> nn.Module:
    if model_cfg.name != "mlp":
        raise ValueError(f"Unknown model: {model_cfg.name}")
    return MLP(
        in_dim=model_cfg.in_dim,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_classes=model_cfg.num_classes,
        dropout=model_cfg.dropout,
    )


def classification_loss(model: nn.Module, batch: dict):
    logits = model(batch["x"])
    loss = F.cross_entropy(logits.float(), batch["y"])
    return loss, {"accuracy": accuracy(logits, batch["y"])}


def build_classification_task(cfg: DictConfig) -> Task:
    if cfg.data.name != "synthetic":
        raise ValueError(f"Unknown dataset: {cfg.data.name}")
    train_ds, val_ds = make_synthetic_splits(
        num_samples=cfg.data.num_samples,
        in_dim=cfg.data.in_dim,
        num_classes=cfg.data.num_classes,
        noise=cfg.data.noise,
        separation=cfg.data.separation,
        val_fraction=cfg.data.val_fraction,
        seed=cfg.seed,
    )
    return Task(
        model=build_model(cfg.model),
        train_dataset=train_ds,
        val_dataset=val_ds,
        loss_fn=classification_loss,
    )
