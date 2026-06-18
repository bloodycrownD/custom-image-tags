"""图像预处理变换。"""

from __future__ import annotations

from PIL import Image
import torch
import torchvision.transforms as transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def build_deterministic_transform(img_size: int) -> transforms.Compose:
    """构建确定性变换：等比缩放填充 → ToTensor → Normalize（训练/验证/推理一致）。"""
    return transforms.Compose(
        [
            KeepRatioResizePad(img_size, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_pil_augment() -> transforms.Compose:
    """构建 PIL 级随机增强（须在 ToTensor 之前应用）。"""
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    )


def pil_augment_then_tensor(
    pil_image: Image.Image,
    augment: transforms.Compose | None,
    deterministic: transforms.Compose,
) -> torch.Tensor:
    """先可选增强，再执行确定性变换流水线。"""
    if augment is not None:
        pil_image = augment(pil_image)
    return deterministic(pil_image)
