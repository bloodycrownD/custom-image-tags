"""DataLoader 构建辅助。"""

from __future__ import annotations

import os
from typing import Any

from torch.utils.data import DataLoader, Dataset


def resolve_num_workers(requested: int | None) -> int:
    """
    解析 worker 数量。

    requested < 0 时按 CPU 核数自动估算（上限 8）。
    """
    if requested is None:
        return 0
    if requested < 0:
        cpu_count = os.cpu_count() or 4
        return max(1, min(8, cpu_count - 1))
    return requested


def build_pairs_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    collate_fn,
    dl_cfg: dict[str, Any] | None = None,
) -> DataLoader:
    """
    构建 PairsDataset 用的 DataLoader。

    默认启用 pin_memory（CUDA 可用时）、persistent_workers 与 prefetch。
    """
    dl_cfg = dl_cfg or {}
    num_workers = resolve_num_workers(int(dl_cfg.get("num_workers", 0)))
    pin_memory = bool(dl_cfg.get("pin_memory", True))
    if not _cuda_available():
        pin_memory = False

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(dl_cfg.get("persistent_workers", True))
        prefetch = int(dl_cfg.get("prefetch_factor", 2))
        if prefetch > 0:
            loader_kwargs["prefetch_factor"] = prefetch

    return DataLoader(dataset, **loader_kwargs)


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
