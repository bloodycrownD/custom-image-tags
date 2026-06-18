"""checkpoint 增量加载结构单元测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from rank.checkpoint import load_checkpoint, save_checkpoint
from rank.groups import GroupMap
from rank.model import PreferenceRanker


def test_incremental_load_expands_embedding(tmp_path: Path) -> None:
    """resume 时 group 增加应扩展 Embedding 并保留旧权重。"""
    ckpt_path = tmp_path / "best.pth"
    old_map = GroupMap(groups={"author_a": 1, "author_b": 2}, authors={"author_a": 1, "author_b": 2})
    model = PreferenceRanker(num_groups=2, embed_dim=8, pretrained=False)
    old_weight = model.group_embed.weight.data.clone()

    save_checkpoint(ckpt_path, model, old_map, condition="train_group")

    new_map = GroupMap(
        groups={"author_a": 1, "author_b": 2, "author_c": 3},
        authors={"author_a": 1, "author_b": 2, "author_c": 3},
    )
    loaded, merged_map, meta = load_checkpoint(ckpt_path, "cpu", group_map=new_map, pretrained=False)

    assert loaded.num_groups >= 3
    assert "author_c" in merged_map.groups
    assert meta["condition"] == "train_group"

    new_weight = loaded.group_embed.weight.data
    assert new_weight.shape[0] == loaded.num_groups + 1
    assert torch.allclose(new_weight[: old_weight.shape[0]], old_weight)


def test_incremental_load_resume_training_loss_decreases(tmp_path: Path) -> None:
    """resume 后冻结骨干训练若干步，loss 应下降。"""
    from rank.train import _pairwise_loss, set_backbone_trainable

    ckpt_path = tmp_path / "best.pth"
    group_map = GroupMap(groups={"a": 1}, authors={"a": 1})
    model = PreferenceRanker(num_groups=1, embed_dim=8, pretrained=False)
    save_checkpoint(ckpt_path, model, group_map)

    loaded, _, _ = load_checkpoint(ckpt_path, "cpu", pretrained=False)
    set_backbone_trainable(loaded, False)
    for module in loaded.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    loaded.train()
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    torch.manual_seed(0)
    batch = {
        "img_a": torch.randn(4, 3, 64, 64),
        "img_b": torch.randn(4, 3, 64, 64),
        "group_a": torch.tensor([1, 1, 1, 1]),
        "group_b": torch.tensor([1, 1, 1, 1]),
        "target": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "weight": torch.tensor([1.0, 1.0, 1.0, 1.0]),
    }
    device = torch.device("cpu")
    loss_before = _pairwise_loss(loaded, batch, criterion, device).item()

    optimizer = torch.optim.Adam(
        (p for p in loaded.parameters() if p.requires_grad),
        lr=1e-2,
    )
    for _ in range(30):
        optimizer.zero_grad()
        loss = _pairwise_loss(loaded, batch, criterion, device)
        loss.backward()
        optimizer.step()

    loss_after = _pairwise_loss(loaded, batch, criterion, device).item()
    assert torch.isfinite(torch.tensor(loss_before))
    assert loss_after < loss_before
