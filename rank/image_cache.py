"""图片张量缓存：训练前预热，多进程 DataLoader 共享同一份内存。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from PIL import Image
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def collect_unique_paths_from_pairs(pairs: list[dict[str, Any]]) -> list[str]:
    """从 pairs 列表收集去重后的相对图片路径。"""
    seen: set[str] = set()
    paths: list[str] = []
    for pair in pairs:
        for key in ("image_a", "image_b"):
            rel = pair.get(key, "")
            if isinstance(rel, str) and rel and rel not in seen:
                seen.add(rel)
                paths.append(rel)
    return paths


def _load_rgb_image(data_root: Path, rel_path: str) -> Image.Image:
    """从磁盘加载 RGB 图片；失败时返回黑图占位。"""
    try:
        return Image.open(data_root / rel_path).convert("RGB")
    except OSError:
        return Image.new("RGB", (384, 384), color=(0, 0, 0))


class ImageTensorStore:
    """
    将多张图片预处理为张量后存入连续 Tensor，供 DataLoader 多 worker 共享。

    典型用途：验证集或无增强训练集，避免重复 JPEG 解码与 resize。
    """

    def __init__(self) -> None:
        self._path_to_idx: dict[str, int] = {}
        self._storage: torch.Tensor | None = None

    def __len__(self) -> int:
        return len(self._path_to_idx)

    @property
    def nbytes(self) -> int:
        """缓存占用字节数（用于日志）。"""
        if self._storage is None:
            return 0
        return self._storage.nelement() * self._storage.element_size()

    def contains(self, rel_path: str) -> bool:
        return rel_path in self._path_to_idx

    def get(self, rel_path: str) -> torch.Tensor:
        """按相对路径取张量副本（避免原地增强污染缓存）。"""
        if self._storage is None or rel_path not in self._path_to_idx:
            raise KeyError(f"图片不在缓存中: {rel_path}")
        idx = self._path_to_idx[rel_path]
        return self._storage[idx].clone()

    def build(
        self,
        paths: Iterable[str],
        data_root: str | Path,
        transform: Callable[[Image.Image], torch.Tensor],
        *,
        desc: str = "预热图片缓存",
        show_progress: bool = True,
    ) -> int:
        """
        批量加载并变换图片，写入连续存储。

        Returns:
            成功缓存的图片数量。
        """
        data_root = Path(data_root)
        unique_paths = list(dict.fromkeys(paths))
        if not unique_paths:
            self._path_to_idx = {}
            self._storage = None
            return 0

        tensors: list[torch.Tensor] = []
        path_to_idx: dict[str, int] = {}

        iterator: Iterable[str] = unique_paths
        if show_progress:
            iterator = tqdm(unique_paths, desc=desc, unit="img")

        for rel_path in iterator:
            pil = _load_rgb_image(data_root, rel_path)
            tensor = transform(pil)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("transform 须返回 torch.Tensor")
            path_to_idx[rel_path] = len(tensors)
            tensors.append(tensor.contiguous())

        self._path_to_idx = path_to_idx
        self._storage = torch.stack(tensors, dim=0)
        return len(tensors)

    def share_memory(self) -> ImageTensorStore:
        """将底层 Tensor 放入共享内存，供 DataLoader worker 读取。"""
        if self._storage is not None:
            self._storage.share_memory_()
        return self

    def __getstate__(self) -> dict[str, Any]:
        return {"_path_to_idx": self._path_to_idx, "_storage": self._storage}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._path_to_idx = state["_path_to_idx"]
        self._storage = state["_storage"]


def apply_tensor_flip_augment(tensor: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """在张量上做随机水平翻转（配合缓存使用，替代 PIL RandomHorizontalFlip）。"""
    if torch.rand(()) < p:
        return torch.flip(tensor, dims=(-1,))
    return tensor
