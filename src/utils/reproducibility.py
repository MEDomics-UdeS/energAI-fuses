"""
File:
    src/utils/reproducibility.py

Authors:
    - Simon Giard-Leroux
    - Shreyas Sunil Kulkarni

Description:
    Reproducibility functions to set seed and deterministic behavior

    Inspired by: https://pytorch.org/docs/stable/notes/randomness.html
"""

import torch
import numpy as np
import random


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_deterministic(deterministic: bool, seed: int) -> None:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)

    if deterministic:
        set_seed(seed)
