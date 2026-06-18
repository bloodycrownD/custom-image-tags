"""PreferenceRanker 前向传播单元测试。"""

from __future__ import annotations

import torch

from rank.model import PreferenceRanker


def test_model_cross_author() -> None:
    """不同 group 索引的成对前向应正常返回 score_a/score_b。"""
    model = PreferenceRanker(num_groups=4, embed_dim=16)
    model.eval()

    batch = 2
    img_a = torch.randn(batch, 3, 384, 384)
    img_b = torch.randn(batch, 3, 384, 384)
    group_a = torch.tensor([1, 2], dtype=torch.long)
    group_b = torch.tensor([3, 4], dtype=torch.long)

    score_a, score_b = model(img_a, img_b, group_a, group_b)

    assert score_a.shape == (batch,)
    assert score_b.shape == (batch,)
    assert torch.isfinite(score_a).all()
    assert torch.isfinite(score_b).all()
