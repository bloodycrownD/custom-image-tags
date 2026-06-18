"""PairsDataset 目标映射单元测试。"""

from __future__ import annotations

import pytest

from rank.dataset import prefer_to_target


@pytest.mark.parametrize(
    ("prefer", "expected"),
    [
        ("a", 1.0),
        ("b", 0.0),
        ("tie", 0.5),
    ],
)
def test_prefer_to_target(prefer: str, expected: float) -> None:
    """prefer 标签应映射为 a=1, b=0, tie=0.5。"""
    assert prefer_to_target(prefer) == expected


def test_prefer_to_target_invalid() -> None:
    """无效 prefer 应抛出 ValueError。"""
    with pytest.raises(ValueError, match="无效的 prefer"):
        prefer_to_target("invalid")
