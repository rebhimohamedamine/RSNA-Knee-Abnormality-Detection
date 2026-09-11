"""Reproducibility helpers: seed every RNG the pipeline touches."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python's `random`, NumPy, and Torch (CPU + all CUDA devices).

    `deterministic=True` also asks cuDNN for deterministic algorithms, at a
    potential performance cost. Leave it True for reproducible experiments;
    a caller chasing raw training throughput can pass False.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """`worker_init_fn` for `torch.utils.data.DataLoader` so each dataloader
    worker process gets an independent-but-reproducible seed derived from
    Torch's own initial seed for that worker (set by DataLoader before this
    runs)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
