"""Training orchestration: per-fold trainer and multi-seed CV runner."""

from .trainer import Trainer, TrainerConfig, gather_seg_labels
from .cv_runner import CVRunner, CVConfig

__all__ = ["Trainer", "TrainerConfig", "gather_seg_labels", "CVRunner", "CVConfig"]
