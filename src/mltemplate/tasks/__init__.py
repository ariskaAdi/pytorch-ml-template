"""A task bundles what changes between projects: model, datasets and loss.

The Trainer only talks to a Task, so a new project (GPT, LoRA, detection, ...)
adds a new task here and reuses everything else.
"""

from omegaconf import DictConfig

from mltemplate.tasks.base import Task
from mltemplate.tasks.classification import build_classification_task

TASKS = {
    "classification": build_classification_task,
}


def build_task(cfg: DictConfig) -> Task:
    try:
        builder = TASKS[cfg.task]
    except KeyError:
        raise ValueError(f"Unknown task {cfg.task!r}. Available: {sorted(TASKS)}") from None
    return builder(cfg)


__all__ = ["Task", "build_task", "TASKS"]
