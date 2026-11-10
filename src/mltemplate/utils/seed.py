import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + all GPUs).

    deterministic=True also asks PyTorch for deterministic kernels. That is slower,
    so use it for debugging or for runs that must be bit-reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        # Required by cuBLAS for deterministic matmuls on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
