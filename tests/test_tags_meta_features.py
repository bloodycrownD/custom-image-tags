"""元数据特征（分辨率/清晰度/压缩率）单测。"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from tags.features import META_DIM, compute_meta_features


def test_meta_features_basic(tmp_path: Path) -> None:
    """基本量纲：log像素/log短边/字节像素比可精确对拍，清晰度非负。"""
    p = tmp_path / "a.png"
    Image.new("RGB", (800, 400), (120, 130, 140)).save(p, "PNG")
    row = compute_meta_features(p)
    assert row is not None and len(row) == META_DIM
    assert abs(row[0] - math.log(800 * 400)) < 1e-6
    assert abs(row[1] - math.log(400)) < 1e-6
    assert row[2] >= 0.0  # log1p(var) ≥ 0
    assert abs(row[3] - p.stat().st_size / (800 * 400)) < 1e-9


def test_meta_features_bad_file_returns_none(tmp_path: Path) -> None:
    """坏图返回 None（调用方以 0 占位），不抛异常。"""
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image")
    assert compute_meta_features(p) is None


def test_meta_features_sharp_beats_smooth(tmp_path: Path) -> None:
    """清晰度指标分辨力：高频棋盘格 > 纯色平滑图（低像素判断的模糊代理）。"""
    arr = (np.indices((256, 256)).sum(0) % 2 * 255).astype("uint8")
    sharp = tmp_path / "sharp.png"
    Image.fromarray(arr).save(sharp)
    smooth = tmp_path / "smooth.png"
    Image.new("L", (256, 256), 128).save(smooth)
    sharp_row = compute_meta_features(sharp)
    smooth_row = compute_meta_features(smooth)
    assert sharp_row is not None and smooth_row is not None
    assert sharp_row[2] > smooth_row[2]
