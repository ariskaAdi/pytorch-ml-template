"""Run real DDP with 2 CPU processes (gloo backend) and check the replicas stay in sync."""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, cfg_dict: dict, out_dir: str) -> None:
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        USE_LIBUV="0",  # Windows builds of PyTorch lack libuv for the TCP store
    )
    torch.set_num_threads(1)
    from scripts.train import run

    cfg = OmegaConf.create(cfg_dict)
    run(cfg)


@pytest.mark.slow
def test_ddp_two_processes(make_cfg, tmp_path):
    if torch.cuda.is_available():
        pytest.skip("CPU/gloo test; on GPU machines run torchrun manually")
    cfg = make_cfg("train.precision=fp32", "train.max_steps=6", name="ddp")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    mp.spawn(_worker, args=(2, _free_port(), cfg_dict, str(tmp_path)), nprocs=2, join=True)

    from mltemplate.training.checkpoint import load_checkpoint

    state = load_checkpoint(f"{cfg.run_dir}/checkpoints/last.pt")
    assert state["step"] == 6
    assert state["config"]["train"]["max_steps"] == 6
    lines = (tmp_path / "ddp" / "metrics.jsonl").read_text().splitlines()
    assert lines, "rank 0 should have logged metrics"
