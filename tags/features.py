"""冻结骨干特征预计算与缓存。

EVA02-L 骨干在 v0 阶段全程冻结：一次前向把 453 张图编码为 (N, 1024) fp16
特征矩阵（约 1MB）落盘缓存，此后线性头训练/评估只读缓存，不再触碰图片与骨干。
缓存键为 (arch, img_size, 路径列表顺序, 特征维度)，任一不匹配即重算覆盖。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tags.dataset import TagsDataset
from tags.model import TagModel
from tags.preprocess import load_rgb_white_background

FEATURES_CACHE_VERSION = 1


def count_bad_images(data_root: Path | str, paths: list[str]) -> list[str]:
    """显式探测无法解码的坏图，返回坏图相对路径列表。

    ``load_white_square_tensor`` 对坏图静默返回全白占位，训练流程不会失败；
    此处按同一加载路径（PIL open + 白底合成）逐张探测，供预计算阶段统计打印。
    """
    bad: list[str] = []
    root = Path(data_root)
    for rel in paths:
        try:
            load_rgb_white_background(root / rel)
        except Exception:
            bad.append(rel)
    return bad


def _load_feature_cache(
    cache_path: Path,
    dataset: TagsDataset,
    *,
    arch: str,
    img_size: int,
    num_features: int,
) -> torch.Tensor | None:
    """校验并加载特征缓存；键不匹配或文件损坏时返回 None（由调用方重算）。"""
    if not cache_path.is_file():
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except Exception:
        print(f"[警告] 特征缓存无法读取，将重新计算: {cache_path}")
        return None
    if not isinstance(payload, dict):
        return None
    checks = (
        # [tags/C-1] version 纳入缓存键校验：bump FEATURES_CACHE_VERSION 时旧缓存自动失效
        payload.get("version") == FEATURES_CACHE_VERSION,
        payload.get("arch") == arch,
        payload.get("img_size") == int(img_size),
        payload.get("num_features") == int(num_features),
        payload.get("paths") == dataset.paths,
    )
    if not all(checks):
        return None
    features = payload.get("features")
    if not isinstance(features, torch.Tensor):
        return None
    if features.shape != (len(dataset), num_features):
        return None
    return features


@torch.no_grad()
def precompute_features(
    model: TagModel,
    dataset: TagsDataset,
    *,
    cache_path: Path | str | None = None,
    batch_size: int = 16,
    device: torch.device | str | None = None,
    num_workers: int = 0,
    arch: str | None = None,
) -> torch.Tensor:
    """用冻结骨干批量提取特征，返回 (N, F) fp16 张量。

    坏图以全白占位提取（不中断流程），提取前显式统计个数并打印；
    cache_path 存在且缓存键匹配时直接复用并跳过重算。

    Args:
        model: TagModel 实例（骨干权重应已加载，本函数只读不更新）。
        dataset: TagsDataset（提供 img 与路径顺序）。
        cache_path: 特征缓存 .pt 路径；None 表示不落盘。
        arch: 缓存键中的骨干架构名；缺省时回退 torch 常量占位（仍可缓存，
            但换骨干不会自动失效，生产入口应显式传入）。

    Returns:
        (N, num_features) fp16 特征矩阵，行序与 dataset.paths 一致。
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    num_features = model.backbone.num_features
    cache_key_arch = arch if arch is not None else "unknown"

    if cache_path is not None:
        cached = _load_feature_cache(
            Path(cache_path), dataset, arch=cache_key_arch, img_size=dataset.img_size, num_features=num_features
        )
        if cached is not None:
            print(f"命中特征缓存: {cache_path}（{tuple(cached.shape)}, {cached.dtype}），跳过重算")
            return cached

    bad_paths = count_bad_images(dataset.data_root, dataset.paths)
    if bad_paths:
        print(f"[警告] 坏图 {len(bad_paths)} 张将以全白占位提取特征:")
        for rel in bad_paths[:10]:
            print(f"  - {rel}")

    model.eval()
    model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    chunks: list[torch.Tensor] = []
    for batch in tqdm(loader, desc="特征预计算"):
        features = model.backbone(batch["img"].to(device))
        chunks.append(features.detach().to("cpu", dtype=torch.float16))
    features_all = torch.cat(chunks, dim=0)
    assert features_all.shape == (len(dataset), num_features)

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "version": FEATURES_CACHE_VERSION,
            "arch": cache_key_arch,
            "img_size": dataset.img_size,
            "num_features": num_features,
            "paths": dataset.paths,
            "features": features_all,
        }
        torch.save(payload, cache_path)
        print(f"已写入特征缓存: {cache_path}（{tuple(features_all.shape)}, fp16）")
    return features_all


class FeatureDataset(Dataset):
    """预计算特征 + multi-hot 目标，供线性头训练与评估。

    Args:
        features: (N, F) 特征矩阵（fp16；__getitem__ 时转 float32）。
        targets: (N, T) multi-hot 矩阵，行序与 features 一致。
        indices: 本子集使用的样本索引（来自分层划分）。
    """

    def __init__(self, features: torch.Tensor, targets: torch.Tensor, indices: list[int]):
        assert features.shape[0] == targets.shape[0]
        self.features = features
        self.targets = targets
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.indices[index]
        return self.features[source].float(), self.targets[source]
