"""multi-label tag 方案（详见 docs/apm 记忆与 RULE.md）。"""

from tags.dataset import TagsDataset, stratified_split_tags
from tags.features import FeatureDataset, precompute_features
from tags.filename_tags import parse_filename, parse_stem_tags
from tags.model import TagModel, load_wd_pretrained
from tags.preprocess import (
    CAMIE_IMG_SIZE,
    WD_IMG_SIZE,
    build_wd_preprocess,
    composite_onto_white,
    load_rgb_white_background,
    load_white_square_tensor,
)
from tags.train import (
    TrainResult,
    compute_pos_weight,
    compute_tag_metrics,
    load_v0_checkpoint,
    save_v0_checkpoint,
    train_head,
)
from tags.vocab import (
    ALL_TAGS,
    NEGATIVE_TAGS,
    PREFERENCE_SET,
    PREFERENCE_TAGS,
    V0_TRAIN_TAGS,
    canonical_tag,
)

__all__ = [
    "TagsDataset",
    "stratified_split_tags",
    "FeatureDataset",
    "precompute_features",
    "TrainResult",
    "compute_pos_weight",
    "compute_tag_metrics",
    "save_v0_checkpoint",
    "load_v0_checkpoint",
    "train_head",
    "TagModel",
    "load_wd_pretrained",
    "WD_IMG_SIZE",
    "CAMIE_IMG_SIZE",
    "build_wd_preprocess",
    "composite_onto_white",
    "load_rgb_white_background",
    "load_white_square_tensor",
    "parse_filename",
    "parse_stem_tags",
    "ALL_TAGS",
    "NEGATIVE_TAGS",
    "PREFERENCE_SET",
    "PREFERENCE_TAGS",
    "V0_TRAIN_TAGS",
    "canonical_tag",
]
