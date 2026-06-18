"""train_group 构建规则单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import touch_images
from tools.build_groups import build_train_group_map


def test_build_groups_rare(tmp_path: Path) -> None:
    """图片数 < 5 的作者应并入 __rare__。"""
    data_root = tmp_path / "data"
    rare_author = data_root / "author_sparse"
    rare_author.mkdir(parents=True)
    touch_images(rare_author, "img", 3)

    normal_author = data_root / "author_ok"
    normal_author.mkdir()
    touch_images(normal_author, "img", 6)

    group_map = build_train_group_map(data_root, min_images=5, max_per_group=80)

    assert "__rare__" in group_map["groups"]
    assert group_map["authors"]["author_sparse"] == group_map["groups"]["__rare__"]
    assert group_map["authors"]["author_ok"] == group_map["groups"]["author_ok"]


def test_build_groups_split(tmp_path: Path) -> None:
    """>80 张图片的作者应按 part 拆分，且 images 映射到正确 part。"""
    data_root = tmp_path / "data"
    big_author = data_root / "author_big"
    big_author.mkdir(parents=True)
    touch_images(big_author, "img", 85)

    group_map = build_train_group_map(data_root, min_images=5, max_per_group=80)

    part_names = [name for name in group_map["groups"] if name.startswith("author_big__part")]
    assert len(part_names) >= 2
    assert group_map["authors"]["author_big"] == group_map["groups"][part_names[0]]

    images = group_map.get("images", {})
    assert len(images) == 85
    part1_paths = [rel for rel, grp in images.items() if grp == "author_big__part1"]
    part2_paths = [rel for rel, grp in images.items() if grp == "author_big__part2"]
    assert len(part1_paths) > 0
    assert len(part2_paths) > 0
    assert len(part1_paths) + len(part2_paths) == 85
