"""Loss functions: Group-DRO, IRM penalty, intermodal consistency."""

from .group_dro import GroupDROLoss
from .irm import irm_penalty
from .consistency import intermodal_consistency

__all__ = ["GroupDROLoss", "irm_penalty", "intermodal_consistency"]
