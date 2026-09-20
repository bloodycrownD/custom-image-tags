"""解析 TagSpaces 文件名标签协议。

TagSpaces 将标签追加在文件名末尾的方括号组中，多标签的默认写盘格式为
单方括号组内空格分隔，逗号（半角/全角）亦接受：
    2398_126599139_p0[无背景 一般].png
    图 [灵魂] [NSFW].png
    [作者] 标题 (C98) [喜欢, 漫画图].jpg

动漫文件名本身常含方括号（[作者]/[C98]），因此只把**结尾**的方括号组
视为标签候选，且仅接受词表内的标签；未知的结尾方括号保留为文件名一部分
并单独上报，便于发现拼写错误或误判。
"""

from __future__ import annotations

import re
from pathlib import Path

from tags.vocab import PREFERENCE_SET, canonical_tag

# 结尾连续方括号组
_TRAILING_GROUPS_RE = re.compile(r"((?:\s*\[[^\[\]]+\])+)\s*$")
# 组内多标签分隔符：空格（TagSpaces 默认写盘）与半角/全角逗号。
# 词表标签均为无空白单 token，按空白拆分安全。
_SPLIT_RE = re.compile(r"[\s,，]+")


def parse_stem_tags(stem: str) -> tuple[str, list[str]]:
    """拆出 stem 结尾的方括号标签候选，返回 (去标签后的 stem, 候选列表)。"""
    m = _TRAILING_GROUPS_RE.search(stem)
    if not m:
        return stem, []
    base = stem[: m.start()].rstrip()
    candidates: list[str] = []
    for group in re.findall(r"\[([^\[\]]*)\]", m.group(0)):
        for part in _SPLIT_RE.split(group):
            part = part.strip()
            if part:
                candidates.append(part)
    return base, candidates


def parse_filename(path: Path | str) -> dict:
    """解析文件名标签。

    返回 {"name", "stem", "tags", "unknown_tags", "preference_count"}：
    - tags: 词表内规范标签；
    - unknown_tags: 结尾方括号中不在词表的字符串（可能为拼错的标签）；
    - preference_count: 喜好程度组标签数，训练要求恰好 1。
    """
    p = Path(path)
    base, candidates = parse_stem_tags(p.stem)
    tags: list[str] = []
    unknown: list[str] = []
    for c in candidates:
        canon = canonical_tag(c)
        if canon is None:
            unknown.append(c)
        else:
            tags.append(canon)
    return {
        "name": p.name,
        "stem": base,
        "tags": tags,
        "unknown_tags": unknown,
        "preference_count": sum(1 for t in tags if t in PREFERENCE_SET),
    }
