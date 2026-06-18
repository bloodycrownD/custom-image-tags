"""图像预处理变换。"""

from __future__ import annotations

from PIL import Image
import torchvision.transforms as transforms


class KeepRatioResizePad:
    """等比缩放后居中填充为正方形，避免竖图拉伸变形。"""

    def __init__(self, target_size: int, fill: int = 0):
        """
        Args:
            target_size: 目标边长（像素）。
            fill: 填充色，默认 0（黑色），训练与推理须一致。
        """
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        old_w, old_h = img.size
        if old_w == 0 or old_h == 0:
            return Image.new("RGB", (self.target_size, self.target_size), color=(0, 0, 0))

        ratio = min(self.target_size / old_w, self.target_size / old_h)
        new_w = max(1, int(old_w * ratio))
        new_h = max(1, int(old_h * ratio))
        img = transforms.functional.resize(img, (new_h, new_w))
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - delta_w // 2,
            delta_h - delta_h // 2,
        )
        return transforms.functional.pad(img, padding, fill=self.fill)
