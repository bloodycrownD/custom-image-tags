"""MC Dropout 不确定性单元测试。"""

from __future__ import annotations

import torch

from rank.model import PreferenceRanker
from rank.predict import enable_mc_dropout


def test_uncertainty_mc_std_positive() -> None:
    """MC Dropout 多次采样时 uncertainty（std）应大于 0。"""
    model = PreferenceRanker(num_groups=2, embed_dim=8, pretrained=False)
    enable_mc_dropout(model)

    image = torch.randn(1, 3, 64, 64)
    group_idx = torch.tensor([1], dtype=torch.long)

    mc_scores: list[torch.Tensor] = []
    for _ in range(8):
        with torch.no_grad():
            mc_scores.append(model._score(image, group_idx))

    stacked = torch.stack(mc_scores, dim=0)
    uncertainty = stacked.std(dim=0).item()
    assert uncertainty > 0.0
