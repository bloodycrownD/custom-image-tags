"""冻结骨干特征预计算与缓存。

EVA02-L 骨干在 v0 阶段全程冻结：一次前向把 953 张图编码为 (N, 1024) fp16
特征矩阵（约 1MB）落盘缓存，此后线性头训练/评估只读缓存，不再触碰图片与骨干。
缓存键为 (version, arch, img_size, 路径列表顺序, 特征维度)，任一不匹配即重算覆盖。

v2（2026-09-25）新增 4 维元数据特征（log像素/log短边/log清晰度/压缩率）：
原图分辨率在 448 预处理中被销毁，模型对"低像素"类标签结构性失明，元数据
特征补上这一通道；附训练集均值/标准差用于推理侧同口径归一化。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tags.dataset import TagsDataset
from tags.model import TagModel
from tags.preprocess import load_rgb_white_background

FEATURES_CACHE_VERSION = 2
# 元数据特征维度：[log像素, log短边, log1p(清晰度), 字节/像素]
META_DIM = 4
# 清晰度测量的归一化短边上限（超过则缩放，使 lap_var 与源尺寸解耦）
_SHARPNESS_NORM_SIDE = 512
_LAPLACIAN_KERNEL = [0, 1, 0, 1, -4, 1, 0, 1, 0]


def compute_meta_features(path: Path | str) -> list[float] | None:
    """计算单图 4 维元数据特征：[log像素, log短边, log1p(清晰度), 字节/像素]。

    清晰度为归一化短边≤512 后的灰度 Laplacian 方差（模糊/放大的代理指标，
    2026-09-25 低像素标签分析：单指标 AUC 0.67~0.70，组合后逻辑回归 CV
    AUC 0.75）；字节/像素为压缩率代理。解码失败返回 None（调用方以 0 占位）。
    """
    try:
        p = Path(path)
        size_bytes = p.stat().st_size
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = _SHARPNESS_NORM_SIDE / min(w, h) if min(w, h) > _SHARPNESS_NORM_SIDE else 1.0
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
            gray = im.convert("L")
            lap = gray.filter(ImageFilter.Kernel((3, 3), _LAPLACIAN_KERNEL, scale=1, offset=0))
            lap_var = float(np.asarray(lap, dtype=np.float32).var())
        return [math.log(w * h), math.log(min(w, h)), math.log1p(lap_var), size_bytes / (w * h)]
    except Exception:
        return None


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """校验并加载特征缓存（含元数据特征与归一化统计）；不匹配返回 None。"""
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
    meta = payload.get("meta_features")
    meta_mean = payload.get("meta_mean")
    meta_std = payload.get("meta_std")
    if not all(isinstance(t, torch.Tensor) for t in (features, meta, meta_mean, meta_std)):
        return None
    if features.shape != (len(dataset), num_features):
        return None
    if meta.shape != (len(dataset), META_DIM):
        return None
    return features, meta, meta_mean, meta_std


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """用冻结骨干批量提取特征，返回 (特征, 元数据, 元数据均值, 元数据标准差)。

    特征为 (N, F) fp16；元数据为 (N, META_DIM) fp32 原始值，均值/标准差供
    训练与推理两侧同口径归一化（推理侧从 checkpoint meta 读取同值）。
    坏图以全白占位提取（不中断流程），提取前显式统计个数并打印；
    cache_path 存在且缓存键匹配时直接复用并跳过重算。

    Args:
        model: TagModel 实例（骨干权重应已加载，本函数只读不更新）。
        dataset: TagsDataset（提供 img 与路径顺序）。
        cache_path: 特征缓存 .pt 路径；None 表示不落盘。
        arch: 缓存键中的骨干架构名；缺省时回退 torch 常量占位（仍可缓存，
            但换骨干不会自动失效，生产入口应显式传入）。

    Returns:
        (features(N,F) fp16, meta(N,4) fp32, meta_mean(4,), meta_std(4,))，
        行序与 dataset.paths 一致。
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    num_features = model.backbone.num_features
    cache_key_arch = arch if arch is not None else "unknown"

    if cache_path is not None:
        cached = _load_feature_cache(
            Path(cache_path), dataset, arch=cache_key_arch, img_size=dataset.img_size, num_features=num_features
        )
        if cached is not None:
            features, meta, meta_mean, meta_std = cached
            print(f"命中特征缓存: {cache_path}（{tuple(features.shape)}, {features.dtype}），跳过重算")
            return features, meta, meta_mean, meta_std

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

    # 元数据特征（分辨率/清晰度/压缩率）：需原图，单独一遍解码
    meta_rows: list[list[float]] = []
    for rel in tqdm(dataset.paths, desc="元数据特征", unit="img"):
        row = compute_meta_features(dataset.data_root / rel)
        meta_rows.append(row if row is not None else [0.0] * META_DIM)
    meta = torch.tensor(meta_rows, dtype=torch.float32)
    meta_mean = meta.mean(dim=0)
    meta_std = meta.std(dim=0).clamp_min(1e-6)

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
            "meta_features": meta,
            "meta_mean": meta_mean,
            "meta_std": meta_std,
        }
        torch.save(payload, cache_path)
        print(f"已写入特征缓存: {cache_path}（{tuple(features_all.shape)}, fp16 + 元数据 {META_DIM} 维）")
    return features_all, meta, meta_mean, meta_std


class FeatureDataset(Dataset):
    """预计算特征 + 元数据特征 + multi-hot 目标，供线性头训练与评估。

    Args:
        features: (N, F) 特征矩阵（fp16；__getitem__ 时转 float32）。
        targets: (N, T) multi-hot 矩阵，行序与 features 一致。
        indices: 本子集使用的样本索引（来自分层划分）。
        meta: (N, META_DIM) 元数据原始值；None 表示不附加（向后兼容）。
        meta_mean/meta_std: 元数据归一化统计（训练集口径）。
    """

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        indices: list[int],
        *,
        meta: torch.Tensor | None = None,
        meta_mean: torch.Tensor | None = None,
        meta_std: torch.Tensor | None = None,
    ):
        assert features.shape[0] == targets.shape[0]
        self.features = features
        self.targets = targets
        self.indices = list(indices)
        self.meta = meta
        self.meta_mean = meta_mean
        self.meta_std = meta_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.indices[index]
        feat = self.features[source].float()
        if self.meta is not None:
            # 归一化用训练集统计（推理侧从 checkpoint meta 读同值，保证同口径）
            m = (self.meta[source] - self.meta_mean) / self.meta_std
            feat = torch.cat([feat, m])
        return feat, self.targets[source]
