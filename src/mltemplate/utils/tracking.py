"""Experiment tracking: Weights & Biases, or a local console + metrics.jsonl fallback.

Metrics are always written to run_dir/metrics.jsonl, so a run can be inspected
even without W&B.
"""

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


class Tracker:
    def __init__(self, cfg: DictConfig, run_dir: Path, enabled: bool = True):
        # Only the main process logs; other ranks get a silent tracker.
        self.enabled = enabled
        self.run = None
        self.metrics_path = Path(run_dir) / "metrics.jsonl"
        if not enabled:
            return

        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.tracking.backend == "wandb":
            import wandb  # optional dependency: pip install -e ".[tracking]"

            self.run = wandb.init(
                project=cfg.tracking.project,
                entity=cfg.tracking.entity,
                name=cfg.experiment_name,
                id=cfg.experiment_name,  # same id + resume = one W&B run across restarts
                resume="allow",
                dir=str(run_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
        elif cfg.tracking.backend != "console":
            raise ValueError(f"Unknown tracking backend: {cfg.tracking.backend}")

    def log(self, metrics: dict, step: int) -> None:
        if not self.enabled:
            return
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if self.run is not None:
            self.run.log(metrics, step=step)
        text = " | ".join(
            f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items()
        )
        print(f"step {step:>6} | {text}", flush=True)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
