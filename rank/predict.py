"""批量推理与 MC Dropout 不确定性估计。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rank.checkpoint import raw_to_score_0_100
from rank.groups import GroupMap
from rank.model import PreferenceRanker


def enable_mc_dropout(model: nn.Module) -> None:
    """启用 Dropout 进行 MC 采样，BatchNorm 保持 eval 模式。"""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.eval()


@dataclass
class PredictItem:
    """单张图推理结果。"""

    path: str
    author: str
    group_id: int
    score_raw: float
    score_0_100: float
    uncertainty: float
    zone: str


class _ImageListDataset(Dataset):
    """待推理图片列表。"""

    def __init__(
        self,
        items: list[tuple[str, str]],
        data_root: Path,
        group_map: GroupMap,
        transform,
        condition: str,
    ):
        self.items = items
        self.data_root = data_root
        self.group_map = group_map
        self.transform = transform
        self.condition = condition

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rel_path, author = self.items[idx]
        try:
            image = Image.open(self.data_root / rel_path).convert("RGB")
        except OSError:
            image = Image.new("RGB", (384, 384), color=(0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        group_id = self.group_map.resolve(author, rel_path, self.condition)  # type: ignore[arg-type]

        return {
            "image": image,
            "group_id": group_id,
            "path": rel_path,
            "author": author,
        }


def _collate_predict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "group_id": torch.tensor([item["group_id"] for item in batch], dtype=torch.long),
        "path": [item["path"] for item in batch],
        "author": [item["author"] for item in batch],
    }


def _score_zone(score_0_100: float) -> str:
    if score_0_100 <= 10:
        return "extreme_low"
    if score_0_100 >= 90:
        return "extreme_high"
    return "mid"


@torch.no_grad()
def predict_batch(
    model: PreferenceRanker,
    items: Iterable[tuple[str, str]],
    data_root: str | Path,
    group_map: GroupMap,
    transform,
    device: torch.device,
    *,
    condition: str = "train_group",
    batch_size: int = 16,
    n_mc: int = 1,
    percentiles: dict[str, float] | None = None,
    num_workers: int = 0,
) -> list[PredictItem]:
    """
    批量推理，支持 MC Dropout 不确定性（n_mc > 1）。

    Args:
        items: (相对路径, author_id) 迭代器。
        n_mc: MC 采样次数；1 表示关闭不确定性（uncertainty=0）。
        percentiles: p5/p95，用于 score_0_100 映射。
    """
    item_list = list(items)
    if not item_list:
        return []

    percentiles = percentiles or {}
    dataset = _ImageListDataset(
        item_list,
        Path(data_root),
        group_map,
        transform,
        condition,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_predict,
    )

    if n_mc > 1:
        enable_mc_dropout(model)
    else:
        model.eval()

    results: list[PredictItem] = []

    for batch in tqdm(loader, desc="Predict"):
        images = batch["image"].to(device)
        group_ids = batch["group_id"].to(device)

        if n_mc > 1:
            mc_scores: list[torch.Tensor] = []
            for _ in range(n_mc):
                scores = model._score(images, group_ids)
                mc_scores.append(scores)
            stacked = torch.stack(mc_scores, dim=0)
            score_raw_t = stacked.mean(dim=0)
            uncertainty_t = stacked.std(dim=0)
        else:
            score_raw_t = model.predict(images, group_ids)
            uncertainty_t = torch.zeros_like(score_raw_t)

        for i, path in enumerate(batch["path"]):
            score_raw = score_raw_t[i].item()
            uncertainty = uncertainty_t[i].item()
            score_0_100 = raw_to_score_0_100(score_raw, percentiles)
            results.append(
                PredictItem(
                    path=path,
                    author=batch["author"][i],
                    group_id=int(batch["group_id"][i]),
                    score_raw=score_raw,
                    score_0_100=score_0_100,
                    uncertainty=uncertainty,
                    zone=_score_zone(score_0_100),
                )
            )

    return results
