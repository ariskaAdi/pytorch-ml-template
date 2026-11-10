"""Load a trained checkpoint and run predictions. Used by the API in app/serve.py."""

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from mltemplate.tasks.classification import build_model
from mltemplate.training.checkpoint import load_checkpoint


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> tuple[nn.Module, DictConfig]:
    state = load_checkpoint(checkpoint_path, map_location=device)
    cfg = OmegaConf.create(state["config"])
    model = build_model(cfg.model)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, cfg


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> dict[str, list]:
    probs = torch.softmax(model(x).float(), dim=-1)
    return {"labels": probs.argmax(dim=-1).tolist(), "probs": probs.tolist()}
