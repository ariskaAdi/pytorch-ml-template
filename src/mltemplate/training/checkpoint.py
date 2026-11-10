"""Checkpoint save/load, including RNG state so a resumed run continues exactly."""

import os
import random
from pathlib import Path

import numpy as np
import torch


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, state: dict) -> None:
    """Write atomically: a pod that stops mid-save never leaves a broken checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    # weights_only=False because the checkpoint also stores RNG and optimizer state.
    # Only load checkpoints you created yourself.
    return torch.load(path, map_location=map_location, weights_only=False)
