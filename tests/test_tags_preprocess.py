"""tags.preprocess 单测：透明→白底、白填充正方形、归一化语义、坏图占位。"""

from __future__ import annotations

from PIL import Image
import numpy as np
import torch

from tags.preprocess import (
    WD_IMG_SIZE,
    build_wd_preprocess,
    load_rgb_white_background,
    load_white_square_tensor,
)


def _write_rgba_png(path, size=(60, 40)) -> None:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(20, 40):
        for y in range(10, 30):
            img.putpixel((x, y), (255, 0, 0, 255))
    img.save(path)


def test_alpha_composites_onto_white(tmp_path):
    p = tmp_path / "rgba.png"
    _write_rgba_png(p)
    rgb = load_rgb_white_background(p)
    assert rgb.mode == "RGB"
    arr = np.asarray(rgb)
    assert tuple(arr[0, 0]) == (255, 255, 255)  # 透明区 → 白
    assert tuple(arr[20, 20]) == (255, 0, 0)  # 不透明区保留


def test_opaque_non_rgb_mode(tmp_path):
    p = tmp_path / "p_mode.png"
    Image.new("P", (8, 8), 128).save(p)
    rgb = load_rgb_white_background(p)
    assert rgb.mode == "RGB"


def test_pad_white_centered_and_normalized():
    img = Image.new("RGB", (100, 40), (10, 20, 30))
    t = build_wd_preprocess(WD_IMG_SIZE)(img)
    assert t.shape == (3, WD_IMG_SIZE, WD_IMG_SIZE)
    # 顶部填充行：白 → 归一化后恰为 +1
    top = t[:, 0, :]
    assert torch.allclose(top, torch.ones_like(top))
    # 水平中线在内容区内：(10,20,30)/255 经 (x-0.5)/0.5 变换
    mid = t[:, WD_IMG_SIZE // 2, WD_IMG_SIZE // 2]
    expected = (torch.tensor([10.0, 20.0, 30.0]) / 255.0 - 0.5) / 0.5
    assert torch.allclose(mid, expected, atol=3 / 255)


def test_square_passthrough_no_distortion():
    img = Image.new("RGB", (64, 64), (200, 100, 50))
    t = build_wd_preprocess(64)(img)
    mid = t[:, 32, 32]
    expected = (torch.tensor([200.0, 100.0, 50.0]) / 255.0 - 0.5) / 0.5
    assert torch.allclose(mid, expected, atol=2 / 255)


def test_corrupt_image_white_fallback(tmp_path):
    p = tmp_path / "bad.png"
    p.write_bytes(b"not an image")
    t = load_white_square_tensor(p, 224)
    assert t.shape == (3, 224, 224)
    assert torch.allclose(t, torch.ones_like(t))
