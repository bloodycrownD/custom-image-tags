"""tag 词表 v1：分组、互斥语义与 v0 训练集。

与 TagSpaces 文件名标签一一对应（词表演进见 docs/apm 记忆）。
"""

from __future__ import annotations

# 喜好程度组：互斥，每张图恰好一个
PREFERENCE_TAGS: tuple[str, ...] = ("灵魂", "喜欢", "一般", "删除")

# 负面标签组：兼容多选
NEGATIVE_TAGS: tuple[str, ...] = (
    "官方图",
    "无背景",
    "大水印",
    "低像素",
    "小水印",
    "NSFW",
    "漫画图",
    "男角色",
)

ALL_TAGS: tuple[str, ...] = PREFERENCE_TAGS + NEGATIVE_TAGS

# v0 进入训练的 tag；官方图/大水印/男角色正样本 <30，暂不进损失
V0_TRAIN_TAGS: tuple[str, ...] = PREFERENCE_TAGS + (
    "无背景",
    "漫画图",
    "NSFW",
    "小水印",
    "低像素",
)

PREFERENCE_SET = frozenset(PREFERENCE_TAGS)


def normalize_tag(raw: str) -> str:
    """规范化标签：去首尾空白；ASCII 标签统一大写（nsfw → NSFW）。"""
    t = raw.strip()
    return t.upper() if t.isascii() else t


_TAG_KEY_TO_CANONICAL: dict[str, str] = {normalize_tag(t): t for t in ALL_TAGS}


def canonical_tag(raw: str) -> str | None:
    """把文件名里的标签映射为词表规范名；不在词表返回 None。"""
    key = normalize_tag(raw)
    if key in _TAG_KEY_TO_CANONICAL:
        return _TAG_KEY_TO_CANONICAL[key]
    # TagSpaces 层级标签（如 负面/水印）按末段匹配
    if "/" in key:
        return _TAG_KEY_TO_CANONICAL.get(normalize_tag(key.rsplit("/", 1)[-1]))
    return None
