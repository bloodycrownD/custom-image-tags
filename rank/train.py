"""PreferenceRanker 训练循环。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from rank.checkpoint import save_checkpoint
from rank.groups import ConditionMode, GroupMap
from rank.model import PreferenceRanker


@dataclass
class TrainResult:
    """训练结果摘要。"""

    best_val_loss: float
    epochs_run: int
    stopped_early: bool
    history: list[dict[str, float]]


def _pairwise_loss(
    model: PreferenceRanker,
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    """计算加权 BCEWithLogitsLoss。"""
    img_a = batch["img_a"].to(device, non_blocking=non_blocking)
    img_b = batch["img_b"].to(device, non_blocking=non_blocking)
    group_a = batch["group_a"].to(device, non_blocking=non_blocking)
    group_b = batch["group_b"].to(device, non_blocking=non_blocking)
    target = batch["target"].to(device, non_blocking=non_blocking)
    weight = batch["weight"].to(device, non_blocking=non_blocking)

    score_a, score_b = model(img_a, img_b, group_a, group_b)
    diff = score_a - score_b
    loss = criterion(diff, target)
    return (loss * weight).mean()


@torch.no_grad()
def evaluate(
    model: PreferenceRanker,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    non_blocking: bool = False,
) -> float:
    """在验证集上计算平均 pair loss。"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        loss = _pairwise_loss(
            model, batch, criterion, device, non_blocking=non_blocking
        )
        batch_size = batch["target"].size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0
    return total_loss / total_samples


def train_ranker(
    model: PreferenceRanker,
    train_loader: DataLoader,
    val_loader: DataLoader,
    group_map: GroupMap,
    device: torch.device,
    *,
    epochs: int = 30,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    early_stop_patience: int = 7,
    condition: ConditionMode = "train_group",
    checkpoint_dir: str | None = None,
    use_amp: bool = True,
    pin_memory: bool = False,
) -> TrainResult:
    """
    训练 PreferenceRanker。

    使用 AdamW、ReduceLROnPlateau、AMP 混合精度与早停（基于 val pair loss）。
    """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    non_blocking = pin_memory and device.type == "cuda"

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history: list[dict[str, float]] = []
    stopped_early = False

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp and device.type == "cuda"):
                loss = _pairwise_loss(
                    model, batch, criterion, device, non_blocking=non_blocking
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = batch["target"].size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / max(total_samples, 1)
        val_loss = evaluate(
            model, val_loader, criterion, device, non_blocking=non_blocking
        )
        scheduler.step(val_loss)

        history.append({"train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if checkpoint_dir:
                save_checkpoint(
                    f"{checkpoint_dir}/best.pth",
                    model,
                    group_map,
                    condition=condition,
                    epoch=epoch + 1,
                    val_loss=val_loss,
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
    )


def set_backbone_trainable(model: PreferenceRanker, trainable: bool) -> None:
    """冻结或解冻 EfficientNet 骨干。"""
    for param in model.backbone.parameters():
        param.requires_grad = trainable
