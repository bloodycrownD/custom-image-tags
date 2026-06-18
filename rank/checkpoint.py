"""训练 checkpoint 保存与加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from rank.groups import ConditionMode, GroupMap
from rank.model import PreferenceRanker


def save_checkpoint(
    path: str | Path,
    model: PreferenceRanker,
    group_map: GroupMap,
    *,
    condition: ConditionMode = "train_group",
    percentiles: dict[str, float] | None = None,
    u_threshold: float | None = None,
    epoch: int | None = None,
    val_loss: float | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    保存训练 checkpoint，含模型权重与元数据。

    Args:
        path: 输出 .pth 路径。
        model: PreferenceRanker 实例。
        group_map: 训练时使用的组映射表。
        condition: 条件模式 train_group | author | none。
        percentiles: 分位数（如 p5、p95），用于 score_0_100 映射。
        u_threshold: 验证集 uncertainty 分位阈值，用于待复核区判定。
        epoch: 当前 epoch。
        val_loss: 验证集 loss。
        extra: 附加字段。
    """
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "train_group_map": group_map.to_dict(),
        "condition": condition,
        "embed_dim": model.embed_dim,
        "num_groups": model.num_groups,
    }
    if percentiles is not None:
        payload["percentiles"] = percentiles
    if u_threshold is not None:
        payload["u_threshold"] = u_threshold
    if epoch is not None:
        payload["epoch"] = epoch
    if val_loss is not None:
        payload["val_loss"] = val_loss
    if extra:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _resize_group_embedding(
    model: PreferenceRanker,
    old_state: dict[str, torch.Tensor],
    new_num_groups: int,
) -> dict[str, torch.Tensor]:
    """扩展 group Embedding：保留旧权重，新索引随机初始化。"""
    old_weight = old_state["group_embed.weight"]
    old_size = old_weight.shape[0]
    new_size = new_num_groups + 1

    if new_size <= old_size:
        return old_state

    new_weight = model.group_embed.weight.data.clone()
    new_weight[:old_size] = old_weight
    # index 0 (unknown) 与新增槽位保持 model 当前初始化
    if old_size > 0:
        new_weight[old_size:] = model.group_embed.weight.data[old_size:]

    state = dict(old_state)
    state["group_embed.weight"] = new_weight
    return state


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    group_map: GroupMap | None = None,
    pretrained: bool = True,
) -> tuple[PreferenceRanker, GroupMap, dict[str, Any]]:
    """
    加载 checkpoint 并构建模型。

    若提供 group_map 且组数增加，则扩展 Embedding 并保留已有权重。

    Args:
        pretrained: 构建模型时是否下载 ImageNet 预训练骨干（加载后会覆盖权重）。

    Returns:
        (model, group_map, metadata) — metadata 含 condition、percentiles、u_threshold 等。
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    loaded_map = GroupMap.from_dict(checkpoint.get("train_group_map", {}))
    if group_map is not None:
        merged_groups = dict(loaded_map.groups)
        merged_groups.update(group_map.groups)
        merged_authors = dict(loaded_map.authors)
        merged_authors.update(group_map.authors)
        merged_images = dict(loaded_map.images)
        merged_images.update(group_map.images)
        active_map = GroupMap(merged_groups, merged_authors, merged_images)
    else:
        active_map = loaded_map

    embed_dim = checkpoint.get("embed_dim", 64)
    num_groups = max(
        active_map.num_groups,
        checkpoint.get("num_groups", active_map.num_groups),
    )
    num_groups = max(num_groups, 0)

    model = PreferenceRanker(num_groups=num_groups, embed_dim=embed_dim, pretrained=pretrained)
    state_dict = checkpoint["state_dict"]

    old_num = checkpoint.get("num_groups", num_groups)
    if num_groups > old_num:
        state_dict = _resize_group_embedding(model, state_dict, num_groups)

    model.load_state_dict(state_dict)
    model.to(device)

    metadata = {
        "condition": checkpoint.get("condition", "train_group"),
        "percentiles": checkpoint.get("percentiles", {}),
        "u_threshold": checkpoint.get("u_threshold"),
        "epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
    }
    return model, active_map, metadata


def raw_to_score_0_100(
    score_raw: float,
    percentiles: dict[str, float],
) -> float:
    """将原始偏好分线性映射到 [0, 100] 并 clip。"""
    p5 = percentiles.get("p5", 0.0)
    p95 = percentiles.get("p95", 1.0)
    if p95 <= p5:
        return 50.0
    mapped = (score_raw - p5) / (p95 - p5) * 100.0
    return float(max(0.0, min(100.0, mapped)))
