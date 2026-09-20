"""冻结特征上的多标签线性头训练、评估与 v0 checkpoint。

镜像 rank/train.py 的训练骨架（AdamW + ReduceLROnPlateau + 早停 + tqdm +
TrainResult），但独立实现、不依赖 rank 包：本模块在预计算特征上只训练
TagModel.head（Dropout + Linear），损失为带 per-tag pos_weight 的
BCEWithLogitsLoss；评估以阈值无关的 per-tag Average Precision 为主。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from tags.model import TagModel

# 正样本低于该数的 tag 在报告中标注为"排序辅助档，不做阈值判定"
SMALL_TAG_SUPPORT = 30


@dataclass
class TrainResult:
    """头训练结果摘要。"""

    best_val_loss: float
    epochs_run: int
    stopped_early: bool
    history: list[dict[str, float]] = field(default_factory=list)
    # 最优 epoch 的头权重快照（已拷贝到 CPU），供外层评估与最终落盘
    best_head_state: dict[str, torch.Tensor] | None = None


def compute_pos_weight(targets: torch.Tensor) -> torch.Tensor:
    """按列统计正负样本比，返回 per-tag pos_weight (T,) float32。

    pos_weight = 负样本数 / 正样本数；正样本数为 0 的 tag 回退为 1 避免
    除零（该 tag 在本 split 内无监督信号，加权无意义）。当前词表最大
    不平衡约 10:1，按 v0 设计不做 cap。
    """
    total = targets.shape[0]
    pos = targets.sum(dim=0)
    neg = float(total) - pos
    pw = neg / pos.clamp(min=1.0)
    return torch.where(pos > 0, pw, torch.ones_like(pw)).float()


def _step(
    head: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """单批前向：特征与目标搬至设备并计算 BCE loss。"""
    logits = head(features.to(device))
    loss = criterion(logits, targets.to(device))
    return loss, features.size(0)


@torch.no_grad()
def evaluate_head(
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """在验证集上计算平均 BCE loss（样本均值）。"""
    head.eval()
    total_loss = 0.0
    total_samples = 0
    for features, targets in loader:
        loss, batch_size = _step(head, features, targets, criterion, device)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


@torch.no_grad()
def predict_probs(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """收集 loader 上的预测概率与目标，返回 (probs (N,T), targets (N,T))。"""
    head.eval()
    probs_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    for features, targets in loader:
        probs_list.append(torch.sigmoid(head(features.to(device))).cpu())
        targets_list.append(targets.cpu())
    return torch.cat(probs_list, dim=0), torch.cat(targets_list, dim=0)


def train_head(
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    pos_weight: torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-2,
    early_stop_patience: int = 10,
    checkpoint_path: Path | str | None = None,
    checkpoint_fields: dict[str, Any] | None = None,
) -> TrainResult:
    """在预计算特征上训练线性头。

    AdamW + ReduceLROnPlateau(factor=0.2, patience=3) + 早停（val loss）；
    val loss 刷新最优时快照头权重，并在提供 checkpoint_path 时覆盖写
    基础版 checkpoint（外层评估后会补写 per-tag metrics 完整版）。

    Args:
        head: 待训练的头模块（TagModel.head 或等价的 nn.Module）。
        pos_weight: per-tag 正负比 (T,)；None 表示不加权。
        checkpoint_fields: 写 checkpoint 所需的重建元信息（arch/tag_list 等）。
    """
    head.to(device)
    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        if pos_weight is not None
        else nn.BCEWithLogitsLoss()
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    stopped_early = False
    best_head_state: dict[str, torch.Tensor] | None = None

    for epoch in range(epochs):
        head.train()
        running_loss = 0.0
        total_samples = 0
        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)
            loss, batch_size = _step(head, features, targets, criterion, device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / max(total_samples, 1)
        val_loss = evaluate_head(head, val_loader, criterion, device)
        scheduler.step(val_loss)
        history.append(
            {"train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]}
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            if checkpoint_path is not None:
                save_v0_checkpoint(
                    checkpoint_path,
                    best_head_state,
                    epoch=epoch + 1,
                    val_loss=val_loss,
                    history=history,
                    **(checkpoint_fields or {}),
                )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            stopped_early = True
            break

    return TrainResult(
        best_val_loss=best_val_loss,
        epochs_run=len(history),
        stopped_early=stopped_early,
        history=history,
        best_head_state=best_head_state,
    )


def average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """阈值无关的 per-tag Average Precision（PR 曲线 step-wise 积分）。

    按置信度降序累计 TP，AP = 各正样本位置 precision 的平均；无并列分数时
    与 sklearn ``average_precision_score`` 结果一致（有并列分数时本实现
    不做并列组内插值，按 stable 排序的单一顺序累计，结果可能偏低）；
    无正样本时约定返回 0。
    """
    pos_total = int(labels.sum().item())
    if pos_total == 0 or scores.numel() == 0:
        return 0.0
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_labels = labels[order]
    tp_cum = torch.cumsum(sorted_labels, dim=0)
    precision = tp_cum / torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    return float((precision * sorted_labels).sum().item() / pos_total)


def compute_tag_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    tag_list: tuple[str, ...] | list[str],
    *,
    small_tag_support: int = SMALL_TAG_SUPPORT,
) -> dict[str, dict[str, Any]]:
    """逐 tag 计算 AP / F1@0.5 / 支持数。

    AP 为主指标（阈值无关）；F1@0.5 与 precision/recall 仅作参考。
    验证集正样本 < small_tag_support 的 tag 标注 ``ranking_only=True``
    （排序辅助档，不做阈值判定）。
    """
    metrics: dict[str, dict[str, Any]] = {}
    for j, tag in enumerate(tag_list):
        tag_probs = probs[:, j]
        tag_labels = targets[:, j]
        support = int(tag_labels.sum().item())

        pred = (tag_probs >= 0.5).float()
        tp = float((pred * tag_labels).sum().item())
        fp = float((pred * (1.0 - tag_labels)).sum().item())
        fn = float(((1.0 - pred) * tag_labels).sum().item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        metrics[tag] = {
            "ap": average_precision(tag_probs, tag_labels),
            "f1_at_050": f1,
            "precision": precision,
            "recall": recall,
            "support": support,
            "ranking_only": support < small_tag_support,
        }
    return metrics


def save_v0_checkpoint(
    path: Path | str,
    head_state_dict: dict[str, torch.Tensor],
    *,
    arch: str,
    num_features: int,
    tag_list: tuple[str, ...] | list[str],
    img_size: int,
    dropout: float,
    pos_weight: torch.Tensor | None = None,
    backbone_weights: str | None = None,
    epoch: int | None = None,
    val_loss: float | None = None,
    history: list[dict[str, float]] | None = None,
    per_tag_metrics: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """保存 v0 头 checkpoint（仅头权重 + 重建元信息，不含骨干权重）。

    payload 全部为 tensor / 基本类型 / 基本容器，可用
    ``torch.load(weights_only=True)`` 安全加载。
    """
    payload: dict[str, Any] = {
        "format": "tags_v0",
        "head_state_dict": head_state_dict,
        "arch": arch,
        "num_features": int(num_features),
        "tag_list": list(tag_list),
        "img_size": int(img_size),
        "dropout": float(dropout),
    }
    if pos_weight is not None:
        payload["pos_weight"] = pos_weight.detach().cpu()
    # 骨干权重路径恒写入（None 表示随机骨干），保证字段可预期
    payload["backbone_weights"] = str(backbone_weights) if backbone_weights is not None else None
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if val_loss is not None:
        payload["val_loss"] = float(val_loss)
    if history is not None:
        payload["history"] = history
    if per_tag_metrics is not None:
        payload["per_tag_metrics"] = per_tag_metrics

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_v0_checkpoint(
    path: Path | str,
    device: torch.device | str = "cpu",
) -> tuple[TagModel, dict[str, Any]]:
    """从 v0 checkpoint 重建 TagModel 并加载头权重。

    骨干按 payload 的 arch 随机初始化（不联网、不读权重文件）；
    生产推理需再调用 ``load_wd_pretrained`` 按需加载骨干权重。

    Returns:
        (model, payload)，payload 含 tag_list/pos_weight/per_tag_metrics 等元信息。
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "tags_v0":
        raise ValueError(f"非 tags_v0 checkpoint: {path}")
    tag_list = payload["tag_list"]
    model = TagModel(
        payload["arch"],
        len(tag_list),
        pretrained=False,
        dropout=float(payload.get("dropout", 0.1)),
    )
    model.head.load_state_dict(payload["head_state_dict"])
    model.to(torch.device(device))
    model.eval()
    return model, payload
