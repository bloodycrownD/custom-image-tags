"""验证 wd-eva02-large-tagger-v3 权重可载入 timm 骨干并完成前向。

用法：
    python tools/verify_wd_load.py [weights_path]

默认权重路径 data/pretrained/wd-eva02-large-tagger-v3/model.safetensors。
检查三件事：
  1. safetensors 键与 timm eva02_large_patch14_448 骨干完全对齐（仅丢弃原始分类头）；
  2. 白色 448 输入可完成前向；
  3. 骨干特征非退化（非常数/非零）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tags.model import WD_ARCH, TagModel, load_wd_pretrained  # noqa: E402
from tags.preprocess import WD_IMG_SIZE, build_wd_preprocess  # noqa: E402

DEFAULT_WEIGHTS = Path("data/pretrained/wd-eva02-large-tagger-v3/model.safetensors")


def main() -> int:
    weights = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WEIGHTS
    if not weights.exists():
        print(f"权重不存在: {weights}")
        return 2

    print(f"构建骨干 {WD_ARCH} ...")
    model = TagModel(WD_ARCH, num_tags=8, pretrained=False)

    print("加载 safetensors ...")
    dropped = load_wd_pretrained(model, weights)
    print(f"丢弃的原始头键（共 {len(dropped)} 个）: {sorted(dropped)[:6]}")

    model.eval()
    white = build_wd_preprocess(WD_IMG_SIZE)(
        __import__("PIL.Image", fromlist=["Image"]).new("RGB", (64, 64), (255, 255, 255))
    ).unsqueeze(0)
    with torch.no_grad():
        feats = model.backbone(white)
        logits = model(white)
    print(f"骨干特征: shape={tuple(feats.shape)}, mean={feats.mean():.4f}, std={feats.std():.4f}")
    print(f"tag 头 logits: shape={tuple(logits.shape)}")
    assert feats.std() > 1e-4, "特征退化"
    print("OK: 权重加载与前向验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
