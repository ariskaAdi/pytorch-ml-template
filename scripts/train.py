"""Training entry point.

Single process:   python scripts/train.py
Override config:  python scripts/train.py train.lr=1.0e-3 experiment_name=lr_1e-3
Multi-GPU (DDP):  torchrun --nproc_per_node=4 scripts/train.py experiment_name=ddp4
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from mltemplate.tasks import build_task
from mltemplate.training.distributed import cleanup_distributed, setup_distributed
from mltemplate.training.trainer import Trainer
from mltemplate.utils.repro import save_run_info
from mltemplate.utils.seed import seed_everything
from mltemplate.utils.tracking import Tracker

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


def run(cfg: DictConfig) -> dict[str, float]:
    dist_info = setup_distributed()
    tracker = None
    try:
        # Same seed on every rank: identical model init (DDP also broadcasts rank 0's weights).
        seed_everything(cfg.seed, cfg.deterministic)
        run_dir = Path(cfg.run_dir)
        if dist_info.is_main:
            save_run_info(run_dir, cfg, dist_info.world_size)

        task = build_task(cfg)
        tracker = Tracker(cfg, run_dir, enabled=dist_info.is_main)
        trainer = Trainer(cfg, task, dist_info, tracker)
        return trainer.fit()
    finally:
        if tracker is not None:
            tracker.finish()
        cleanup_distributed()


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
