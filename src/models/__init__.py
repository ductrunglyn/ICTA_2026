"""Model components for TransVal-Dep."""

from .encoders import AttentionPool, ModalityEncoder, pool_segments_to_bags
from .grl import GradientReversal, grad_reverse
from .adversary import DomainAdversary
from .heads import DepressionHead, SelectiveHead
from .transval_net import TransValNet

__all__ = [
    "AttentionPool",
    "ModalityEncoder",
    "pool_segments_to_bags",
    "GradientReversal",
    "grad_reverse",
    "DomainAdversary",
    "DepressionHead",
    "SelectiveHead",
    "TransValNet",
]
