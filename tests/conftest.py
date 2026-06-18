"""pytest 公共 fixture。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    """创建最小 data_root 目录结构（无真实图片）。"""
    root = tmp_path / "data"
    root.mkdir()
    return root


def touch_images(author_dir: Path, prefix: str, count: int) -> None:
    """在作者目录下创建占位图片文件。"""
    for label in ("good", "keep", "trash"):
        label_dir = author_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        label = ("good", "keep", "trash")[i % 3]
        (author_dir / label / f"{prefix}{i:03d}.jpg").write_bytes(b"\xff\xd8\xff")
