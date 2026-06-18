"""pytest 公共 fixture。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def tmp_data_root(tmp_path: Path) -> Path:
    """创建最小 data_root 目录结构（无真实图片）。"""
    root = tmp_path / "data"
    root.mkdir()
    return root


def make_rgb_jpeg(path: Path, size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (100, 150, 200)) -> None:
    """创建可被 PIL 读取的最小 JPEG 测试图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=color).save(path, "JPEG")


def touch_images(author_dir: Path, prefix: str, count: int) -> None:
    """在作者目录下创建占位图片文件。"""
    for label in ("good", "keep", "trash"):
        label_dir = author_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        label = ("good", "keep", "trash")[i % 3]
        make_rgb_jpeg(author_dir / label / f"{prefix}{i:03d}.jpg")
