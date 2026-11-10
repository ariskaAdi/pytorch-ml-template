"""Local multi-process DDP launcher that does not need torchrun.

Use it on Windows, where torchrun's rendezvous requires libuv and fails. On Linux
(and on RunPod) prefer torchrun.

python scripts/spawn_ddp.py --nproc 2 experiment_name=ddp_local train.max_steps=100
"""

import argparse
import os
import socket

import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from train import CONFIG_DIR, run  # scripts/ is on sys.path when run as a script


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, cfg_dict: dict) -> None:
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        USE_LIBUV="0",
    )
    run(OmegaConf.create(cfg_dict))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nproc", type=int, default=2)
    p.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. train.lr=1.0e-3")
    args = p.parse_args()

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train", overrides=args.overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    mp.spawn(_worker, args=(args.nproc, _free_port(), cfg_dict), nprocs=args.nproc, join=True)


if __name__ == "__main__":
    main()
