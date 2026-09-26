"""tag 模型：预训练骨干 + 多标签 sigmoid 头。"""

from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file
import timm
import torch
import torch.nn as nn

WD_ARCH = "eva02_large_patch14_448"
DEFAULT_DROPOUT = 0.1


class TagModel(nn.Module):
    """timm 骨干（num_classes=0 取特征）+ dropout + 多标签线性头。

    输出为 raw logits，训练用 BCEWithLogitsLoss；推理经 sigmoid 得 tag 概率。
    eval 前向是确定性函数；mc_forward 提供认知不确定性估计。
    """

    def __init__(
        self,
        arch: str,
        num_tags: int,
        *,
        pretrained: bool = False,
        dropout: float = DEFAULT_DROPOUT,
        extra_features: int = 0,
    ):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0)
        # extra_features：头部附加的元数据特征维度（如分辨率/清晰度/压缩率——
        # 原图尺寸在 448 预处理中被销毁，这是模型"看见"源尺寸的唯一通道）
        self.extra_features = int(extra_features)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone.num_features + self.extra_features, num_tags),
        )

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None) -> torch.Tensor:
        """图像前向；extra_features>0 时必须显式传入元数据特征（无法从像素推出）。"""
        feat = self.backbone(x)
        if self.extra_features:
            if extra is None:
                raise ValueError(
                    "模型含元数据特征（extra_features>0），forward 需传入 extra 张量（元数据特征）"
                )
            feat = torch.cat([feat, extra], dim=-1)
        return self.head(feat)

    @torch.no_grad()
    def mc_forward(self, x: torch.Tensor, n_mc: int, extra: torch.Tensor | None = None) -> torch.Tensor:
        """MC dropout 采样：仅激活 Dropout 模块的 n_mc 次前向。

        不整体切 train()，避免 BatchNorm 骨干的运行统计被污染。
        返回 (n_mc, batch, num_tags) 的 sigmoid 概率。
        """
        was_training = self.training
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train(True)
        try:
            samples = torch.stack([torch.sigmoid(self(x, extra)) for _ in range(n_mc)])
        finally:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train(was_training)
        return samples


def load_wd_pretrained(model: TagModel, weights_path: Path | str) -> set[str]:
    """将 wd-*-tagger-v3 的 safetensors 载入骨干，返回被丢弃的原始头键名。

    原始 checkpoint 携带 10861 类分类头；骨干按 num_classes=0 构建，
    头部键（如 head.weight/bias）不在骨干参数表中，被过滤丢弃。
    其余键必须与骨干完全对齐，否则报错。
    """
    sd = load_file(str(weights_path))
    backbone_sd = model.backbone.state_dict()
    filtered = {k: v for k, v in sd.items() if k in backbone_sd}
    dropped = set(sd) - set(filtered)
    missing = set(backbone_sd) - set(filtered)
    if missing:
        preview = sorted(missing)[:5]
        raise RuntimeError(f"骨干有 {len(missing)} 个键在权重中缺失，如 {preview}")
    model.backbone.load_state_dict(filtered, strict=True)
    return dropped
