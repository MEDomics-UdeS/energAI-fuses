"""
File:
    src/utils/reproducibility.py

Authors:
    - Simon Giard-Leroux
    - Guillaume Cléroux
    - Shreyas Sunil Kulkarni

Description:
    Reproducibility functions to set seed and deterministic behavior

    Inspired by: https://pytorch.org/docs/stable/notes/randomness.html
"""

import torch
import numpy as np
import random


def seed_worker(worker_id: int) -> None:
    """Dataloader worker seed function to ensure results reproducibility

    Args:
        worker_id(int): worker id

    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_seed(seed: int) -> None:
    """Function to set a manual seed to ensure results reproducibility

    Args:
        seed(int): seed value

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_deterministic(deterministic: bool) -> None:
    """Function to set deterministic behavior to ensure results reproducibility

    Args:
        deterministic(bool): True to activate deterministic behavior

    """
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)
