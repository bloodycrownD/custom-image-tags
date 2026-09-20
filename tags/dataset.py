"""多标签数据集：labels.json → multi-hot 目标与按喜好组分层划分。

labels.json 由 tools/scan_tagspaces.py 生成（对图库只读），schema：
  {"version": 1, "data_root": "...", "vocab": {...}, "stats": {...},
   "images": [{"path": "<posix 相对路径>", "tags": ["灵魂", ...]}]}

只对 tag_list（默认 V0_TRAIN_TAGS）做 multi-hot；词表外标签（官方图/大水印/
男角色等）不进入损失。喜好组互斥：缺失或多于 1 个喜好 tag 的图片跳过并计数。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tags.preprocess import WD_IMG_SIZE, load_white_square_tensor
from tags.vocab import PREFERENCE_SET, V0_TRAIN_TAGS
from tools import load_json


class TagsDataset(Dataset):
    """multi-label tags 数据集。

    每个样本返回 ``{"path": str, "img": Tensor, "target": Tensor}``：
      - img: (3, img_size, img_size) WD 白底预处理张量（供冻结骨干提特征）；
      - target: len(tag_list) 维 multi-hot float32 张量。

    喜好组互斥校验失败（0 个或 >1 个喜好 tag）的图片被跳过，
    计数记录在 ``skip_stats``；``tag_counts`` 统计保留样本的 per-tag 正样本数。

    Args:
        labels_path: labels.json 路径。
        data_root: 图片根目录；为 None 时回退 labels.json 内嵌的 data_root。
        img_size: 预处理边长（生产 448，与 WD 权重对齐）。
        tag_list: 进入损失的 tag 顺序（同时是 checkpoint 中 tag_list 的顺序）。
    """

    def __init__(
        self,
        labels_path: Path | str,
        data_root: Path | str | None = None,
        *,
        img_size: int = WD_IMG_SIZE,
        tag_list: tuple[str, ...] | list[str] = V0_TRAIN_TAGS,
    ):
        doc = load_json(Path(labels_path))
        images = doc.get("images")
        if not isinstance(images, list):
            raise ValueError(f"labels.json 缺少 images 列表: {labels_path}")

        root = Path(data_root) if data_root else Path(doc.get("data_root", ""))
        if not str(root).strip():
            raise ValueError("未提供 data_root，且 labels.json 内嵌 data_root 为空")
        self.data_root = root
        self.img_size = int(img_size)
        self.tag_list = tuple(tag_list)

        tag_index = {tag: i for i, tag in enumerate(self.tag_list)}
        self.entries: list[dict] = []
        self.skip_stats = {"no_preference": 0, "conflict_preference": 0}
        self.tag_counts: dict[str, int] = {tag: 0 for tag in self.tag_list}

        for item in images:
            path = item.get("path", "")
            tags = [t for t in item.get("tags", []) if t in tag_index]
            prefs = PREFERENCE_SET.intersection(tags)
            if len(prefs) == 0:
                self.skip_stats["no_preference"] += 1
                continue
            if len(prefs) > 1:
                self.skip_stats["conflict_preference"] += 1
                continue
            self.entries.append({"path": path, "tags": tags, "pref": next(iter(prefs))})
            for tag in tags:
                self.tag_counts[tag] += 1

        self.targets = self._build_targets()

    def _build_targets(self) -> torch.Tensor:
        """把每张图保留的 tags 编码为 (N, T) multi-hot float32 矩阵。"""
        tag_index = {tag: i for i, tag in enumerate(self.tag_list)}
        targets = torch.zeros(len(self.entries), len(self.tag_list), dtype=torch.float32)
        for row, entry in enumerate(self.entries):
            for tag in entry["tags"]:
                targets[row, tag_index[tag]] = 1.0
        return targets

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict:
        entry = self.entries[index]
        # 坏图由 load_white_square_tensor 静默替换为全白占位（预计算阶段另行统计）
        img = load_white_square_tensor(self.data_root / entry["path"], self.img_size)
        return {
            "path": entry["path"],
            "img": img,
            "target": self.targets[index],
        }

    @property
    def paths(self) -> list[str]:
        """保留样本的相对路径列表（与 targets 行序一致，特征缓存键）。"""
        return [entry["path"] for entry in self.entries]


def stratified_split_tags(
    dataset: TagsDataset,
    *,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """按喜好组分层抽样，返回排序后的 (train_indices, val_indices)。

    组内随机抽取 ``round(n * val_ratio)`` 张进验证集（至少 1 张、至多 n-1 张），
    保证稀有喜好（如 灵魂）在验证集中也有代表；单样本组不拆分（留在训练集）。
    使用独立 ``torch.Generator(seed)``，与全局随机态解耦、可复现。
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        groups[entry["pref"]].append(index)

    generator = torch.Generator().manual_seed(seed)
    val_indices: set[int] = set()
    for pref in sorted(groups):
        members = groups[pref]
        n = len(members)
        if n <= 1:
            continue
        k = max(1, int(round(n * val_ratio)))
        k = min(k, n - 1)
        perm = torch.randperm(n, generator=generator).tolist()
        val_indices.update(members[i] for i in perm[:k])

    train_indices = sorted(set(range(len(dataset))) - val_indices)
    return train_indices, sorted(val_indices)
