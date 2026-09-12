"""tag 预处理：与 WD tagger 预训练分布对齐。

对齐 SmilingWolf wd-tagger space 的官方 prepare_image 管线：
  1. alpha 透明通道合成到白底（直接 .convert("RGB") 会得到黑底，属分布外输入）；
  2. 原始分辨率下居中填充为白色正方形（官方顺序：先 pad 后 resize）；
  3. bicubic 缩放到目标边长；
  4. mean/std = 0.5 归一化（timm 版权重为 RGB 输入；ONNX 版的 BGR/0-255 路径不适用）。

rank 分支的黑色填充 + ImageNet 归一化服务于 ImageNet 初始化的
EfficientNet-B4，与本模块互不通用。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
import torchvision.transforms as transforms

WD_IMG_SIZE = 448  # wd-eva02-large-tagger-v3
CAMIE_IMG_SIZE = 512  # camie-tagger-v2，接入时以其官方代码复核
WHITE = (255, 255, 255)
WD_MEAN = (0.5, 0.5, 0.5)
WD_STD = (0.5, 0.5, 0.5)


def composite_onto_white(img: Image.Image) -> Image.Image:
    """将带透明通道的图片合成到白底，返回 RGB；无透明通道则直接转换。"""
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return img.convert("RGB")


def load_rgb_white_background(path: Path | str) -> Image.Image:
    """加载图片并合成透明通道到白底。"""
    with Image.open(path) as img:
        return composite_onto_white(img.copy())


class PadToSquareWhite:
    """原始分辨率下居中填充为白色正方形（WD 官方顺序：先 pad 后 resize）。"""

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        max_dim = max(w, h)
        canvas = Image.new("RGB", (max_dim, max_dim), WHITE)
        canvas.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
        return canvas


def build_wd_preprocess(img_size: int = WD_IMG_SIZE) -> transforms.Compose:
    """构建确定性预处理：pad 白色正方形 → bicubic 缩放 → 0.5 归一化。"""
    return transforms.Compose(
        [
            PadToSquareWhite(),
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(WD_MEAN, WD_STD),
        ]
    )


def load_white_square_tensor(path: Path | str, img_size: int = WD_IMG_SIZE) -> torch.Tensor:
    """端到端单图预处理（白底加载 + 全套变换）。

    加载失败时返回白色占位，与白色填充语义一致；rank 分支的黑图占位不适用。
    """
    try:
        pil = load_rgb_white_background(path)
    except Exception:
        pil = Image.new("RGB", (img_size, img_size), WHITE)
    return build_wd_preprocess(img_size)(pil)
