"""Save everything needed to reproduce a run: config, git commit and environment."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def git_info() -> dict:
    """Commit hash and whether there are uncommitted changes (None outside a git repo)."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def save_run_info(run_dir: Path, cfg: DictConfig, world_size: int = 1) -> dict:
    """Write config.yaml and run_info.json into run_dir. Call it on the main process only."""
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    info = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "world_size": world_size,
        "git": git_info(),
        "env": environment_info(),
    }
    if info["git"]["dirty"]:
        print("[repro] warning: uncommitted changes; this run is not fully reproducible from git")

    # Keep one entry per launch, so resumed runs keep their history.
    path = run_dir / "run_info.json"
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(info)
    path.write_text(json.dumps(history, indent=2))
    return info
