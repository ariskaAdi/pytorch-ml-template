"""Generic training loop.

Features: mixed precision (bf16/fp16), gradient accumulation, gradient clipping,
warmup + cosine LR, torch.compile, DDP, periodic eval, and exact checkpoint/resume.
The trainer knows nothing about the model or data: it receives a Task.
"""

import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from mltemplate.tasks.base import Task
from mltemplate.training.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from mltemplate.training.distributed import DistInfo, all_reduce_mean, barrier
from mltemplate.training.schedulers import build_warmup_cosine
from mltemplate.utils.tracking import Tracker

PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def move_to_device(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas) -> torch.optim.AdamW:
    """AdamW with weight decay only on matrices (not on biases / norm weights)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas))


class Trainer:
    def __init__(self, cfg: DictConfig, task: Task, dist_info: DistInfo, tracker: Tracker):
        self.cfg = cfg
        self.tc = cfg.train
        self.dist = dist_info
        self.device = dist_info.device
        self.tracker = tracker
        self.loss_fn = task.loss_fn
        self.run_dir = Path(cfg.run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints"

        if self.tc.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {list(PRECISIONS)}")
        self.amp_dtype = PRECISIONS[self.tc.precision]
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.tc.precision == "fp16")

        # Model: keep a handle on the raw module for saving; train through the wrapped one.
        self.raw_model = task.model.to(self.device)
        model: nn.Module = self.raw_model
        if self.dist.is_distributed:
            ids = [self.device.index] if self.device.type == "cuda" else None
            model = DDP(model, device_ids=ids)
        if self.tc.compile:
            model = torch.compile(model)
        self.model = model

        self.optimizer = build_optimizer(
            self.raw_model, self.tc.lr, self.tc.weight_decay, self.tc.betas
        )
        self.scheduler = build_warmup_cosine(
            self.optimizer, self.tc.warmup_steps, self.tc.max_steps, self.tc.min_lr_ratio
        )

        # DistributedSampler is used even for 1 process: it gives a seeded,
        # per-epoch shuffle that we can replay exactly on resume.
        self.train_sampler = DistributedSampler(
            task.train_dataset,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
            shuffle=True,
            seed=cfg.seed,
            drop_last=True,
        )
        self.train_loader = DataLoader(
            task.train_dataset,
            batch_size=self.tc.batch_size,
            sampler=self.train_sampler,
            num_workers=self.tc.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        if len(self.train_loader) == 0:
            raise ValueError("Train set is smaller than one batch per process")
        val_sampler = DistributedSampler(
            task.val_dataset,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
            shuffle=False,
        )
        self.val_loader = DataLoader(
            task.val_dataset,
            batch_size=self.tc.batch_size,
            sampler=val_sampler,
            num_workers=self.tc.num_workers,
        )

        # Progress counters (all saved in checkpoints).
        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.best_val_loss = math.inf
        self._pending_rng: dict | None = None

    # ------------------------------------------------------------------ helpers

    def autocast(self):
        if self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _batches(self):
        """Endless batch stream that tracks (epoch, batch_in_epoch) for exact resume."""
        while True:
            self.train_sampler.set_epoch(self.epoch)
            skip = self.batch_in_epoch
            it = iter(self.train_loader)  # creating an iterator draws from the torch RNG
            if self._pending_rng is not None:
                # Restore RNG *after* that draw, matching an uninterrupted run.
                restore_rng_state(self._pending_rng)
                self._pending_rng = None
            for i, batch in enumerate(it):
                if i < skip:  # replay position after a resume
                    continue
                self.batch_in_epoch = i + 1
                yield batch
            self.epoch += 1
            self.batch_in_epoch = 0

    # --------------------------------------------------------------- checkpoint

    def state_dict(self) -> dict:
        return {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "best_val_loss": self.best_val_loss,
            "rng": capture_rng_state(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }

    def save(self, name: str = "last.pt") -> None:
        if self.dist.is_main:
            save_checkpoint(self.ckpt_dir / name, self.state_dict())
        barrier()

    def load(self, path: Path) -> None:
        state = load_checkpoint(path, map_location=self.device)
        self.raw_model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.step = state["step"]
        self.epoch = state["epoch"]
        self.batch_in_epoch = state["batch_in_epoch"]
        self.best_val_loss = state["best_val_loss"]
        self._pending_rng = state["rng"]  # applied in _batches(), see comment there
        if self.dist.is_main:
            print(f"[trainer] resumed from {path} at step {self.step}", flush=True)

    # --------------------------------------------------------------------- eval

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), device=self.device))
        n = 0
        for batch in self.val_loader:
            batch = move_to_device(batch, self.device)
            with self.autocast():
                loss, metrics = self.loss_fn(self.model, batch)
            totals["loss"] += loss.detach().float()
            for k, v in metrics.items():
                totals[k] += v.detach().float()
            n += 1
        self.model.train()
        return {f"val/{k}": all_reduce_mean(v / max(n, 1)).item() for k, v in totals.items()}

    # -------------------------------------------------------------------- train

    def train_step(self, batches) -> dict[str, float]:
        """One optimizer step = grad_accum_steps micro-batches."""
        accum = self.tc.grad_accum_steps
        totals: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros((), device=self.device))

        for micro in range(accum):
            batch = move_to_device(next(batches), self.device)
            is_last = micro == accum - 1
            # Skip the DDP gradient all-reduce on all but the last micro-batch.
            sync_ctx = (
                self.model.no_sync()
                if self.dist.is_distributed and not is_last and hasattr(self.model, "no_sync")
                else nullcontext()
            )
            with sync_ctx:
                with self.autocast():
                    loss, metrics = self.loss_fn(self.model, batch)
                self.scaler.scale(loss / accum).backward()
            totals["loss"] += loss.detach().float() / accum
            for k, v in metrics.items():
                totals[k] += v.detach().float() / accum

        self.scaler.unscale_(self.optimizer)  # so clipping sees the real gradient norm
        max_norm = self.tc.grad_clip if self.tc.grad_clip > 0 else math.inf
        grad_norm = torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), max_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.step += 1

        out = {f"train/{k}": v.item() for k, v in totals.items()}
        out["train/grad_norm"] = grad_norm.item()
        return out

    def fit(self) -> dict[str, float]:
        last = self.ckpt_dir / "last.pt"
        if self.tc.resume and last.exists():
            self.load(last)

        batches = self._batches()
        self.model.train()
        samples_per_step = self.tc.batch_size * self.tc.grad_accum_steps * self.dist.world_size
        t0 = time.perf_counter()
        steps_since_log = 0
        val_metrics: dict[str, float] = {}

        while self.step < self.tc.max_steps:
            metrics = self.train_step(batches)
            steps_since_log += 1

            if self.step % self.tc.log_every == 0 or self.step == self.tc.max_steps:
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                    metrics["perf/peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                elapsed = time.perf_counter() - t0
                metrics["perf/samples_per_sec"] = samples_per_step * steps_since_log / elapsed
                metrics["train/lr"] = self.scheduler.get_last_lr()[0]
                metrics["train/epoch"] = self.epoch
                self.tracker.log(metrics, self.step)
                t0, steps_since_log = time.perf_counter(), 0

            if self.step % self.tc.eval_every == 0 or self.step == self.tc.max_steps:
                val_metrics = self.evaluate()
                self.tracker.log(val_metrics, self.step)
                if val_metrics["val/loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val/loss"]
                    self.save("best.pt")

            if self.step % self.tc.save_every == 0 or self.step == self.tc.max_steps:
                self.save("last.pt")

        return val_metrics
