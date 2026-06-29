"""Deterministic seeding across ``random``, ``numpy`` and ``torch``."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant RNGs for reproducible runs.

    Args:
        seed: Integer seed shared by all libraries.
        deterministic: If ``True`` force deterministic cuDNN behaviour
            (slower but reproducible). Torch is imported lazily so that the
            rest of the package can be used without a torch install.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch optional for some scripts
        pass
