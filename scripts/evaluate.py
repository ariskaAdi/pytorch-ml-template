"""Evaluate a checkpoint on the validation set.

python scripts/evaluate.py --checkpoint outputs/debug/checkpoints/best.pt
"""

import argparse
import json

from omegaconf import OmegaConf

from mltemplate.tasks import build_task
from mltemplate.training.checkpoint import load_checkpoint
from mltemplate.training.distributed import setup_distributed
from mltemplate.training.trainer import Trainer
from mltemplate.utils.tracking import Tracker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()

    state = load_checkpoint(args.checkpoint)
    cfg = OmegaConf.create(state["config"])  # the exact config the model was trained with
    dist_info = setup_distributed()
    task = build_task(cfg)
    task.model.load_state_dict(state["model"])

    trainer = Trainer(cfg, task, dist_info, Tracker(cfg, cfg.run_dir, enabled=False))
    metrics = trainer.evaluate()
    print(json.dumps({"checkpoint": args.checkpoint, "step": state["step"], **metrics}, indent=2))


if __name__ == "__main__":
    main()
