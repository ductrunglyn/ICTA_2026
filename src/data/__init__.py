"""Data layer: segmentation, feature extraction, confounds, datasets, splits."""

from .splitter import LeakageFreeSplitter, add_corpus_id
from .segmentation import Segment, segment_participant, segment_with_prompts, segment_with_vad
from .confounds import ConfoundExtractor, CONFOUND_COLUMNS
from .dataset import BagDataset, collate_bags, Bag
from .bag_builder import BagBuilder

__all__ = [
    "BagBuilder",
    "LeakageFreeSplitter",
    "add_corpus_id",
    "Segment",
    "segment_participant",
    "segment_with_prompts",
    "segment_with_vad",
    "ConfoundExtractor",
    "CONFOUND_COLUMNS",
    "BagDataset",
    "collate_bags",
    "Bag",
]
