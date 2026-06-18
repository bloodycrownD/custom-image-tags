"""成对比较数据集。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from rank.groups import ConditionMode, GroupMap
from rank.image_cache import ImageTensorStore, _load_rgb_image, apply_tensor_flip_augment
from rank.transforms import IMAGENET_MEAN, IMAGENET_STD, KeepRatioResizePad, build_pil_augment

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
        *,
        img_size: int = 384,
        augment: bool = False,
        tensor_store: ImageTensorStore | None = None,
        tensor_augment_flip: bool = True,
        resize_pad_cache: bool = False,
    ):
        """
        Args:
            pairs_path: pairs.json 路径。
            data_root: 图片根目录，与 pair 中相对路径拼接。
            group_map: 条件组映射表。
            transform: 已废弃；请使用 img_size + augment。若传入则覆盖内置流水线。
            condition: train_group | author | none。
            seed_pair_weight: 种子 pair 默认权重。
            img_size: 输入边长（transform 未指定时使用）。
            augment: 是否启用随机增强（训练集 True）。
            tensor_store: 预热后的张量缓存；命中时跳过磁盘解码。
            tensor_augment_flip: 缓存命中时是否在张量上做水平翻转（替代 PIL 增强）。
            resize_pad_cache: 未命中 tensor_store 时，是否缓存 resize_pad 后的 PIL（减轻重复解码）。
        """
        self.data_root = Path(data_root)
        self.group_map = group_map
        self.condition = condition
        self.seed_pair_weight = seed_pair_weight
        self.augment = augment
        self.tensor_store = tensor_store
        self.tensor_augment_flip = tensor_augment_flip
        self.resize_pad_cache = resize_pad_cache
        self._img_size = img_size

        if transform is not None:
            self._legacy_transform = transform
            self._resize_pad = None
            self._pil_augment = None
            self._tensor_norm = None
        else:
            self._legacy_transform = None
            self._resize_pad = KeepRatioResizePad(img_size, fill=0)
            self._pil_augment = build_pil_augment() if augment else None
            self._tensor_norm = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )

        self._resize_pad_cache: dict[str, Image.Image] = {}

        with open(pairs_path, encoding="utf-8") as f:
            data = json.load(f)
        self.pairs: list[dict[str, Any]] = data.get("pairs", [])

    def __len__(self) -> int:
        return len(self.pairs)

    def unique_image_paths(self) -> list[str]:
        """返回本数据集涉及的去重图片相对路径。"""
        seen: set[str] = set()
        paths: list[str] = []
        for pair in self.pairs:
            for key in ("image_a", "image_b"):
                rel = pair.get(key, "")
                if isinstance(rel, str) and rel and rel not in seen:
                    seen.add(rel)
                    paths.append(rel)
        return paths

    def _tensor_from_store(self, rel_path: str) -> torch.Tensor:
        """从共享张量缓存读取，并按配置做轻量增强。"""
        assert self.tensor_store is not None
        tensor = self.tensor_store.get(rel_path)
        if self.augment and self.tensor_augment_flip:
            tensor = apply_tensor_flip_augment(tensor)
        return tensor

    def _load_tensor_from_disk(self, rel_path: str) -> torch.Tensor:
        """从磁盘加载并变换为张量。"""
        if self._legacy_transform is not None:
            pil = _load_rgb_image(self.data_root, rel_path)
            out = self._legacy_transform(pil)
            return out if isinstance(out, torch.Tensor) else torch.as_tensor(out)

        if self.tensor_store is not None and self.tensor_store.contains(rel_path):
            return self._tensor_from_store(rel_path)

        pil = None
        if self.resize_pad_cache and rel_path in self._resize_pad_cache:
            pil = self._resize_pad_cache[rel_path].copy()
        else:
            pil = _load_rgb_image(self.data_root, rel_path)
            assert self._resize_pad is not None
            pil = self._resize_pad(pil)
            if self.resize_pad_cache:
                self._resize_pad_cache[rel_path] = pil.copy()

        if self._pil_augment is not None:
            pil = self._pil_augment(pil)
        assert self._tensor_norm is not None
        return self._tensor_norm(pil)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pair = self.pairs[idx]
        rel_a = pair["image_a"]
        rel_b = pair["image_b"]

        if self.tensor_store is not None and self.tensor_store.contains(rel_a):
            img_a = self._tensor_from_store(rel_a)
        else:
            img_a = self._load_tensor_from_disk(rel_a)

        if self.tensor_store is not None and self.tensor_store.contains(rel_b):
            img_b = self._tensor_from_store(rel_b)
        else:
            img_b = self._load_tensor_from_disk(rel_b)

        group_a = self.group_map.resolve(
            pair.get("author_a", ""),
            rel_a,
            self.condition,
        )
        group_b = self.group_map.resolve(
            pair.get("author_b", ""),
            rel_b,
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
