"""tags v0 训练 CLI：labels → 分层划分 → 冻结骨干特征预计算 → 线性头训练 → 评估落盘。

用法：
    python tags_train.py --config configs/tags_v0.yaml [--labels ... --epochs ...]

CLI 参数采用 None 哨兵覆盖 config 字段（镜像 rank_train.py 风格）；
流程为函数式入口 ``run_tags_training``，测试可直接调用（不起子进程）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tags.dataset import TagsDataset, stratified_split_tags
from tags.features import META_DIM, FeatureDataset, precompute_features
from tags.model import WD_ARCH, TagModel, load_wd_pretrained
from tags.train import (
    TrainResult,
    calibrate_tag_thresholds,
    compute_pos_weight,
    compute_tag_metrics,
    predict_probs,
    save_v0_checkpoint,
    train_head,
)
from tools import load_json, save_json, utc_now_iso

DEFAULT_CHECKPOINT = Path("models/tags/v0_best.pth")
DEFAULT_REPORT = Path("data/classification/tags_v0_report.json")
# 摘要行 tag 列对齐宽度（显示宽度：中文字符记 2；当前词表 tag 最宽 6，取 8 留白）
TAG_COL_WIDTH = 8


def load_config(path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_device(spec: str | None) -> torch.device:
    """解析设备：auto / cpu / cuda（无 cuda 时 auto 回退 cpu）。"""
    if not spec or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _display_width(text: str) -> int:
    """粗略显示宽度：ASCII 记 1、其余（含 CJK）记 2，用于终端列对齐。"""
    return sum(1 if ch.isascii() else 2 for ch in text)


def _fmt_summary_line(tag: str, m: dict[str, Any]) -> str:
    """格式化单个 tag 的评估行；tag 列按东亚显示宽度对齐；排序辅助档追加标注。"""
    pad = " " * max(1, TAG_COL_WIDTH - _display_width(tag))
    flag = "  [排序辅助档，不做阈值判定]" if m["ranking_only"] else ""
    return f"  {tag}{pad} AP={m['ap']:.4f} F1@0.5={m['f1_at_050']:.4f} 支持={m['support']}{flag}"


def run_tags_training(
    config: dict,
    *,
    labels_path: Path | None,
    data_root: Path | None,
    checkpoint_path: Path,
    report_path: Path,
    arch: str | None = None,
    img_size: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    epochs: int | None = None,
    val_ratio: float | None = None,
    seed: int | None = None,
    feature_cache: Path | str | None = None,
    backbone_weights: Path | None = None,
    device: str | None = None,
    use_meta_features: bool | None = None,
) -> int:
    """执行 v0 完整训练流程，成功返回 0、失败返回 2。

    None 参数回退 config 字段（None 哨兵覆盖式 CLI）。流程：
    加载 labels → 喜好组分层划分 → 冻结骨干特征预计算（带缓存）→
    per-tag pos_weight 线性头训练（早停）→ best 头评估 per-tag 指标 →
    写 checkpoint（models/tags/v0_best.pth）与报告 json + stdout 摘要。
    """
    arch = arch or str(config.get("arch", WD_ARCH))
    img_size = int(img_size if img_size is not None else config.get("img_size", 448))
    batch_size = int(batch_size if batch_size is not None else config.get("batch_size", 16))
    lr = float(lr if lr is not None else config.get("lr", 1.0e-3))
    weight_decay = float(config.get("weight_decay", 1.0e-2))
    epochs = int(epochs if epochs is not None else config.get("epochs", 100))
    early_stop_patience = int(config.get("early_stop_patience", 10))
    val_ratio = float(val_ratio if val_ratio is not None else config.get("val_ratio", 0.15))
    seed = int(seed if seed is not None else config.get("seed", 42))
    dropout = float(config.get("dropout", 0.1))
    # 元数据特征开关（分辨率/清晰度/压缩率接入头部）；关闭时头输入退化为纯骨干特征。
    # 默认关闭：2026-09-25 受控实验（3 种子×开关 + OOF 诚实评估）证明该通道对
    # 现有标签（尤其低像素）无可测收益——低像素的模型输出折外 AUC≈0.5（等于瞎猜），
    # 可计算上限仅 AUC≈0.75；开关保留，待数据规模或标签需求变化时重评
    use_meta = (
        bool(config.get("use_meta_features", False)) if use_meta_features is None else use_meta_features
    )
    dl_cfg = config.get("dataloader", {})
    num_workers = int(dl_cfg.get("num_workers", 0))
    if feature_cache is None:
        feature_cache = config.get("feature_cache") or None
    # [tags/B-1] backbone_weights 与 feature_cache 对称：CLI 未传时回退 config 字段，
    # 避免默认入口静默以随机骨干训练
    if backbone_weights is None:
        backbone_weights = config.get("backbone_weights") or None

    if labels_path is None:
        labels_path = Path(config.get("labels_path", "data/classification/tags.labels.json"))
    labels_path = Path(labels_path)
    if not labels_path.is_file():
        print(f"ERROR: labels.json 不存在: {labels_path}", file=sys.stderr)
        return 2

    # 1) 加载 labels 并构建 multi-hot 数据集（data_root：参数 > config > labels 内嵌）
    root_override = data_root or (Path(config["data_root"]) if config.get("data_root") else None)
    try:
        dataset = TagsDataset(labels_path, root_override, img_size=img_size)
    except (OSError, ValueError) as exc:
        print(f"ERROR: 加载 labels.json 失败: {exc}", file=sys.stderr)
        return 2

    skip = dataset.skip_stats
    print(
        f"labels: {labels_path}（{len(dataset)} 张入集，"
        f"跳过 无喜好 {skip['no_preference']} / 喜好冲突 {skip['conflict_preference']}）"
    )
    if len(dataset) == 0:
        print("ERROR: 没有可用样本（全部被跳过或 labels 为空）", file=sys.stderr)
        return 2

    # 2) 按喜好组分层划分 train/val
    train_indices, val_indices = stratified_split_tags(dataset, val_ratio=val_ratio, seed=seed)
    if not train_indices or not val_indices:
        print(
            f"ERROR: 划分失败 train={len(train_indices)} val={len(val_indices)}（样本过少或组结构退化）",
            file=sys.stderr,
        )
        return 2
    print(f"分层划分: train {len(train_indices)} / val {len(val_indices)}（val_ratio={val_ratio}, seed={seed}）")

    dev = _resolve_device(device or str(config.get("device", "auto")))
    print(f"使用设备: {dev}")
    torch.manual_seed(seed)

    # 3) 构建冻结骨干（可选加载 WD 预训练权重）并预计算特征
    # 头输入 = 骨干特征 +（可选）META_DIM 维元数据特征（分辨率/清晰度/压缩率）
    meta_dim = META_DIM if use_meta else 0
    model = TagModel(arch, len(dataset.tag_list), pretrained=False, dropout=dropout, extra_features=meta_dim)
    if backbone_weights is not None:
        backbone_weights = Path(backbone_weights)
        if not backbone_weights.is_file():
            print(f"ERROR: 骨干权重不存在: {backbone_weights}", file=sys.stderr)
            return 2
        dropped = load_wd_pretrained(model, backbone_weights)
        print(f"已加载骨干权重: {backbone_weights}（丢弃原始头键 {len(dropped)} 个）")
    else:
        # [tags/B-1] 显式警告，避免随机骨干训练产物只能事后从 meta 识别
        print("警告: 未加载骨干权重，使用随机骨干")
    model.to(dev)

    features, meta, meta_mean, meta_std = precompute_features(
        model,
        dataset,
        cache_path=feature_cache,
        batch_size=batch_size,
        device=dev,
        num_workers=num_workers,
        arch=arch,
    )

    # 4) 特征子集（骨干特征 + 归一化元数据特征拼接）+ per-tag pos_weight（只按训练 split 统计）
    meta_kwargs = {"meta": meta, "meta_mean": meta_mean, "meta_std": meta_std} if use_meta else {}
    train_set = FeatureDataset(features, dataset.targets, train_indices, **meta_kwargs)
    val_set = FeatureDataset(features, dataset.targets, val_indices, **meta_kwargs)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    pos_weight = compute_pos_weight(dataset.targets[train_indices])

    backbone_weights_str = str(backbone_weights) if backbone_weights is not None else None
    checkpoint_fields = {
        "arch": arch,
        "num_features": model.backbone.num_features,
        "tag_list": dataset.tag_list,
        "img_size": img_size,
        "dropout": dropout,
        "pos_weight": pos_weight,
        "backbone_weights": backbone_weights_str,
        # 元数据特征协议：头输入 = 骨干特征 + 元数据（推理侧按 meta_norm 归一）
        "meta_dim": meta_dim,
        "meta_norm": {"mean": meta_mean.tolist(), "std": meta_std.tolist()} if use_meta else None,
    }

    # 5) 线性头训练（best 时覆盖写基础版 checkpoint）
    result: TrainResult = train_head(
        model.head,
        train_loader,
        val_loader,
        dev,
        pos_weight=pos_weight,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_stop_patience=early_stop_patience,
        checkpoint_path=checkpoint_path,
        checkpoint_fields=checkpoint_fields,
    )
    print(
        f"训练完成: best_val_loss={result.best_val_loss:.4f}, "
        f"epochs={result.epochs_run}/{epochs}, early_stop={result.stopped_early}"
    )

    # 6) 用 best 头在 val 上评估 per-tag 指标，补写完整 checkpoint
    assert result.best_head_state is not None
    model.head.load_state_dict(result.best_head_state)
    model.head.to(dev)
    probs, val_targets = predict_probs(model.head, val_loader, dev)
    val_metrics = compute_tag_metrics(probs, val_targets, dataset.tag_list)
    # 频次匹配校准：中和 pos_weight 通胀，阈值进 checkpoint meta 供预填决策
    tag_thresholds = calibrate_tag_thresholds(probs, val_targets, dataset.tag_list)
    supported = [m["ap"] for m in val_metrics.values() if m["support"] > 0]
    mean_ap = sum(supported) / len(supported) if supported else 0.0

    save_v0_checkpoint(
        checkpoint_path,
        result.best_head_state,
        per_tag_metrics=val_metrics,
        tag_thresholds=tag_thresholds,
        **checkpoint_fields,
        epoch=result.epochs_run,
        val_loss=result.best_val_loss,
        history=result.history,
    )

    # 7) 报告 json 与 stdout 摘要
    report = {
        "version": 1,
        "created_at": utc_now_iso(),
        "labels_path": str(labels_path),
        "data_root": str(dataset.data_root),
        "arch": arch,
        "img_size": img_size,
        "backbone_weights": backbone_weights_str,
        "num_images": len(dataset),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "skip_stats": skip,
        "tag_counts": dataset.tag_counts,
        "pos_weight": {tag: float(pos_weight[i]) for i, tag in enumerate(dataset.tag_list)},
        "tag_thresholds": tag_thresholds,
        "best_val_loss": result.best_val_loss,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "history": result.history,
        "val_metrics": val_metrics,
        "mean_ap": mean_ap,
        "checkpoint": str(checkpoint_path),
    }
    save_json(Path(report_path), report)

    print("== v0 训练摘要 ==")
    print(f"数据: {len(dataset)} 张, train {len(train_indices)} / val {len(val_indices)}")
    for tag in dataset.tag_list:
        print(_fmt_summary_line(tag, val_metrics[tag]))
    print(f"mAP(val, 有支持 tag): {mean_ap:.4f}")
    thr_str = "  ".join(f"{t}={tag_thresholds[t]:.2f}" for t in dataset.tag_list)
    print(f"校准阈值(频次匹配): {thr_str}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"报告: {report_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="训练 tags v0（冻结骨干 + 线性头多标签分类）")
    parser.add_argument("--config", type=Path, default=Path("configs/tags_v0.yaml"), help="YAML 配置文件路径")
    parser.add_argument("--labels", type=Path, default=None, help="labels.json 路径")
    parser.add_argument("--data-root", type=Path, default=None, help="图片根目录（缺省用 labels.json 内嵌值）")
    parser.add_argument("--arch", type=str, default=None, help="骨干架构（默认 eva02_large_patch14_448）")
    parser.add_argument("--backbone-weights", type=Path, default=None, help="骨干 safetensors 权重路径")
    parser.add_argument("--img-size", type=int, default=None, help="预处理边长")
    parser.add_argument("--batch-size", type=int, default=None, help="批大小")
    parser.add_argument("--lr", type=float, default=None, help="头学习率")
    parser.add_argument("--epochs", type=int, default=None, help="训练 epoch 数")
    parser.add_argument("--val-ratio", type=float, default=None, help="验证集比例")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--feature-cache", type=Path, default=None, help="特征缓存 .pt 路径")
    parser.add_argument("--checkpoint-out", type=Path, default=None, help="输出 checkpoint 路径（缺省回退 config 的 checkpoint_out）")
    parser.add_argument("--report-out", type=Path, default=None, help="输出报告 json 路径（缺省回退 config 的 report_out）")
    parser.add_argument("--device", type=str, default=None, help="设备：auto / cpu / cuda")
    parser.add_argument(
        "--no-meta-features", action="store_true",
        help="关闭元数据特征（头输入只用骨干特征，用于受控对比实验）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.config.is_file():
        print(f"ERROR: 配置文件不存在: {args.config}", file=sys.stderr)
        return 2
    config = load_config(args.config)

    # [tags/C-2] CLI default=None 哨兵：未传时回退 config 字段，兑现「CLI 覆盖 config」语义
    checkpoint_out = args.checkpoint_out or Path(config.get("checkpoint_out", DEFAULT_CHECKPOINT))
    report_out = args.report_out or Path(config.get("report_out", DEFAULT_REPORT))

    return run_tags_training(
        config,
        labels_path=args.labels,
        data_root=args.data_root,
        checkpoint_path=checkpoint_out.resolve(),
        report_path=report_out.resolve(),
        arch=args.arch,
        img_size=args.img_size,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        feature_cache=args.feature_cache,
        backbone_weights=args.backbone_weights,
        device=args.device,
        use_meta_features=(False if args.no_meta_features else None),
    )


if __name__ == "__main__":
    raise SystemExit(main())
