"""multi-label tag 方案（详见 docs/apm 记忆与 RULE.md）。"""

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
from tags.vocab import (
    ALL_TAGS,
    NEGATIVE_TAGS,
    PREFERENCE_SET,
    PREFERENCE_TAGS,
    V0_TRAIN_TAGS,
    canonical_tag,
)

__all__ = [
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
