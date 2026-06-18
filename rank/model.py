"""PreferenceRanker：成对偏好排序模型。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class PreferenceRanker(nn.Module):
    """
    EfficientNet-B4 骨干 + 条件组 Embedding + 回归头。

    每张图使用各自的 group 索引计算偏好分；成对训练时 A/B 各自 embed。
    Embedding index 0 表示 unknown。
    """

    def __init__(self, num_groups: int, embed_dim: int = 64):
        """
        Args:
            num_groups: 已知组数量（不含 unknown；Embedding 大小为 num_groups + 1）。
            embed_dim: 条件 Embedding 维度。
        """
        super().__init__()
        self.num_groups = num_groups
        self.embed_dim = embed_dim

        self.backbone = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.DEFAULT
        )
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.group_embed = nn.Embedding(num_groups + 1, embed_dim)

        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features + embed_dim),
            nn.Linear(num_features + embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def _score(self, image: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        """单张图偏好分（logit）。"""
        image_feat = self.backbone(image)
        group_feat = self.group_embed(group_idx)
        combined = torch.cat((image_feat, group_feat), dim=1)
        return self.head(combined).squeeze(-1)

    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        group_a: torch.Tensor,
        group_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        成对前向：分别计算 A/B 偏好分。

        Returns:
            (score_a, score_b)，形状均为 (batch,)。
        """
        score_a = self._score(img_a, group_a)
        score_b = self._score(img_b, group_b)
        return score_a, score_b

    def predict(self, image: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
        """单图推理，返回偏好分 logit（eval 模式，关闭 Dropout）。"""
        was_training = self.training
        self.eval()
        try:
            return self._score(image, group_idx)
        finally:
            self.train(was_training)
