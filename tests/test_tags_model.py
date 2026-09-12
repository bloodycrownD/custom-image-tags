"""tags.model 单测：小骨干验证换头、eval 确定性与 MC dropout 采样。"""

from __future__ import annotations

import torch

from tags.model import TagModel

ARCH = "resnet18"  # 仅测试用小骨干；生产为 eva02_large_patch14_448


def test_forward_shape_and_eval_determinism():
    torch.manual_seed(0)
    model = TagModel(ARCH, num_tags=6, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y1, y2 = model(x), model(x)
    assert y1.shape == (2, 6)
    assert torch.allclose(y1, y2)


def test_mc_forward_samples_differ_and_stay_probabilistic():
    torch.manual_seed(0)
    model = TagModel(ARCH, num_tags=6, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y_before = model(x)
    mc = model.mc_forward(x, n_mc=4)
    assert mc.shape == (4, 2, 6)
    assert not torch.allclose(mc[0], mc[1])  # dropout 采样导致前向不同
    assert (mc >= 0).all() and (mc <= 1).all()  # sigmoid 输出
    # mc_forward 不应改变 eval 确定性
    with torch.no_grad():
        y_after = model(x)
    assert torch.allclose(y_before, y_after)


def test_pretrained_flag_false_avoids_download():
    # 冒烟：pretrained=False 构建不应触发网络访问
    model = TagModel(ARCH, num_tags=3, pretrained=False)
    assert model.backbone.num_classes == 0
