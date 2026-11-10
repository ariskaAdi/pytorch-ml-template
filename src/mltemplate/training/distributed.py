"""DDP helpers. Works the same for a plain `python` launch (1 process) and `torchrun`."""

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


def setup_distributed() -> DistInfo:
    """Read the env vars set by torchrun and start the process group.

    Without torchrun (RANK/WORLD_SIZE unset) this returns a single-process setup.
    """
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    return DistInfo(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a tensor over all processes (no-op for a single process)."""
    if dist.is_initialized():
        t = t.clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t
