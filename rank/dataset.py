"""成对比较数据集。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image
import torch
from torch.utils.data import Dataset

from rank.groups import ConditionMode, GroupMap

PreferLabel = Literal["a", "b", "tie"]

_LABEL_TO_TARGET = {"a": 1.0, "b": 0.0, "tie": 0.5}
_CATEGORY_ORDER = {"trash": 0, "keep": 1, "good": 2}


def prefer_to_target(prefer: PreferLabel | str) -> float:
    """将 prefer 标签转为 BCE 目标：a=1, b=0, tie=0.5。"""
    try:
        return _LABEL_TO_TARGET[prefer]  # type: ignore[index]
    except KeyError as exc:
        raise ValueError(f"无效的 prefer 值: {prefer!r}") from exc


def _infer_category(image_path: str) -> str | None:
    """从路径片段推断 good/keep/trash 类别（种子 pair 用）。"""
    parts = Path(image_path.replace("\\", "/")).parts
    for part in reversed(parts[:-1]):
        if part in _CATEGORY_ORDER:
            return part
    return None


def compute_pair_weight(
    pair: dict[str, Any],
    seed_pair_weight: float = 0.5,
) -> float:
    """
    计算样本权重。

    - annotator 来源默认 1.0
    - seed 来源：极端区边 1.0，keep↔keep tie 为 seed_pair_weight，其余 seed 为 seed_pair_weight
    """
    source = pair.get("source", "annotator")
    if source != "seed":
        return 1.0

    prefer = pair.get("prefer", "a")
    cat_a = pair.get("category_a") or _infer_category(pair.get("image_a", ""))
    cat_b = pair.get("category_b") or _infer_category(pair.get("image_b", ""))

    if cat_a is None or cat_b is None:
        return seed_pair_weight

    if prefer == "tie" and cat_a == "keep" and cat_b == "keep":
        return seed_pair_weight

    if cat_a != cat_b:
        return 1.0

    return seed_pair_weight


class PairsDataset(Dataset):
    """读取 pairs.json，返回成对训练样本。"""

    def __init__(
        self,
        pairs_path: str | Path,
        data_root: str | Path,
        group_map: GroupMap,
        transform=None,
        condition: ConditionMode = "train_group",
        seed_pair_weight: float = 0.5,
    ):
        """
        Args:
            pairs_path: pairs.json 路径。
            data_root: 图片根目录，与 pair 中相对路径拼接。
            group_map: 条件组映射表。
            transform: 应用于 A/B 两张图的 PIL 变换。
            condition: train_group | author | none。
            seed_pair_weight: 种子 pair 默认权重（keep↔keep tie 等同）。
        """
        self.data_root = Path(data_root)
        self.group_map = group_map
        self.transform = transform
        self.condition = condition
        self.seed_pair_weight = seed_pair_weight

        with open(pairs_path, encoding="utf-8") as f:
            data = json.load(f)
        self.pairs: list[dict[str, Any]] = data.get("pairs", [])

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, rel_path: str) -> Image.Image:
        path = self.data_root / rel_path
        try:
            return Image.open(path).convert("RGB")
        except OSError:
            return Image.new("RGB", (384, 384), color=(0, 0, 0))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        img_a = self._load_image(pair["image_a"])
        img_b = self._load_image(pair["image_b"])

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        group_a = self.group_map.resolve(
            pair.get("author_a", ""),
            pair.get("image_a"),
            self.condition,
        )
        group_b = self.group_map.resolve(
            pair.get("author_b", ""),
            pair.get("image_b"),
            self.condition,
        )

        target = prefer_to_target(pair.get("prefer", "a"))
        weight = compute_pair_weight(pair, self.seed_pair_weight)

        return {
            "img_a": img_a,
            "img_b": img_b,
            "group_a": group_a,
            "group_b": group_b,
            "target": target,
            "weight": weight,
        }


def pairs_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """DataLoader 批整理函数。"""
    return {
        "img_a": torch.stack([item["img_a"] for item in batch]),
        "img_b": torch.stack([item["img_b"] for item in batch]),
        "group_a": torch.tensor([item["group_a"] for item in batch], dtype=torch.long),
        "group_b": torch.tensor([item["group_b"] for item in batch], dtype=torch.long),
        "target": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
        "weight": torch.tensor([item["weight"] for item in batch], dtype=torch.float32),
    }
