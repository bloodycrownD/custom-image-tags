"""PreferenceRanker 训练 CLI。"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, Subset, random_split

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rank.checkpoint import load_checkpoint, save_checkpoint
from rank.dataset import PairsDataset, pairs_collate_fn
from rank.groups import ConditionMode, GroupMap
from rank.model import PreferenceRanker
from rank.predict import predict_batch
from rank.train import set_backbone_trainable, train_ranker
from rank.transforms import KeepRatioResizePad
from tools import iter_images_under, load_json, rel_posix_path, save_json
from tools.build_groups import build_train_group_map, resolve_image_group
from tools.merge_pairs import merge_pairs_files
from tools.validate_pairs import validate_pairs_doc

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_CHECKPOINT = Path("models/rank/best.pth")


def load_config(path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_transform(img_size: int, *, augment: bool = False):
    """构建与 demo2 一致的图像变换流水线。"""
    steps: list = [KeepRatioResizePad(img_size)]
    if augment:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def build_group_map_from_data_root(
    data_root: Path,
    *,
    min_images: int,
    max_per_group: int,
) -> GroupMap:
    """调用 build_groups 规则并填充按图 train_group 映射。"""
    doc = build_train_group_map(
        data_root,
        min_images=min_images,
        max_per_group=max_per_group,
    )
    images: dict[str, str] = {}
    for author_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for img_path in iter_images_under(author_dir):
            rel = rel_posix_path(img_path, data_root)
            group_name = resolve_image_group(rel, doc, data_root)
            if group_name:
                images[rel] = group_name
    return GroupMap(doc["groups"], doc["authors"], images)


def merge_training_pairs(
    *,
    pairs_path: Path | None,
    pairs_new_path: Path | None,
    train_on_new_only: bool,
    out_path: Path,
) -> Path:
    """合并训练用 pairs 文件并写入临时或指定路径。"""
    inputs: list[Path] = []
    if train_on_new_only:
        if pairs_new_path is None:
            raise ValueError("train_on_new_only=true 时必须提供 --pairs-new")
        inputs = [pairs_new_path]
    else:
        if pairs_path is not None:
            inputs.append(pairs_path)
        if pairs_new_path is not None:
            inputs.append(pairs_new_path)

    if not inputs:
        raise ValueError("未指定 --pairs 或 --pairs-new，无法训练")

    merged, _, _ = merge_pairs_files(inputs)
    save_json(out_path, merged)
    return out_path


def split_train_val(
    dataset: PairsDataset,
    val_ratio: float,
    *,
    seed: int = 42,
) -> tuple[Subset, Subset]:
    """按 pair 数量划分训练集与验证集。"""
    total = len(dataset)
    if total == 0:
        raise ValueError("pairs 为空，无法训练")
    val_size = int(total * val_ratio)
    if total > 1:
        val_size = max(1, val_size)
    val_size = min(val_size, total - 1) if total > 1 else 0
    train_size = total - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def collect_unique_images(pairs_path: Path) -> list[tuple[str, str]]:
    """从 pairs.json 收集去重后的 (相对路径, author) 列表。"""
    dataset = PairsDataset(pairs_path, data_root=".", group_map=GroupMap({}, {}))
    seen: set[str] = set()
    items: list[tuple[str, str]] = []
    for pair in dataset.pairs:
        for key_path, key_author in (
            ("image_a", "author_a"),
            ("image_b", "author_b"),
        ):
            rel = pair.get(key_path, "")
            author = pair.get(key_author, "")
            if not isinstance(rel, str) or not rel or rel in seen:
                continue
            seen.add(rel)
            items.append((rel, str(author)))
    return items


def compute_percentiles(
    model: PreferenceRanker,
    pairs_path: Path,
    data_root: Path,
    group_map: GroupMap,
    transform,
    device: torch.device,
    *,
    condition: ConditionMode,
    batch_size: int,
) -> dict[str, float]:
    """在训练集图片上统计 score_raw 的 p5、p95。"""
    items = collect_unique_images(pairs_path)
    if not items:
        return {"p5": 0.0, "p95": 1.0}

    results = predict_batch(
        model,
        items,
        data_root,
        group_map,
        transform,
        device,
        condition=condition,
        batch_size=batch_size,
        n_mc=1,
    )
    scores = sorted(r.score_raw for r in results)
    n = len(scores)
    p5_idx = max(0, int(n * 0.05) - 1)
    p95_idx = min(n - 1, int(n * 0.95))
    return {"p5": scores[p5_idx], "p95": scores[p95_idx]}


def run_training(
    config: dict,
    *,
    data_root: Path,
    pairs_path: Path | None,
    pairs_new_path: Path | None,
    resume_path: Path | None,
    epochs: int | None,
    lr: float | None,
    freeze_backbone_epochs: int | None,
    build_groups: bool,
    checkpoint_path: Path,
) -> int:
    """执行完整训练流程。"""
    group_cfg = config.get("group", {})
    finetune_cfg = config.get("finetune", {})
    condition: ConditionMode = config.get("condition", "train_group")
    img_size = int(config.get("img_size", 384))
    batch_size = int(config.get("batch_size", 16))
    embed_dim = int(config.get("embed_dim", 64))
    weight_decay = float(config.get("weight_decay", 0.01))
    early_stop_patience = int(config.get("early_stop_patience", 7))
    val_ratio = float(config.get("val_ratio", 0.1))
    seed_pair_weight = float(config.get("seed_pair_weight", 0.5))
    train_on_new_only = bool(finetune_cfg.get("train_on_new_only", False))

    total_epochs = int(epochs if epochs is not None else config.get("epochs", 30))
    freeze_epochs = int(
        freeze_backbone_epochs
        if freeze_backbone_epochs is not None
        else finetune_cfg.get("freeze_backbone_epochs", 0)
    )
    is_resume = resume_path is not None

    if is_resume:
        learning_rate = float(
            lr if lr is not None else config.get("lr_finetune", config.get("lr", 5e-5))
        )
    else:
        learning_rate = float(lr if lr is not None else config.get("lr", 5e-5))

    if not data_root.is_dir():
        print(f"ERROR: data_root 不存在或不是目录: {data_root}", file=sys.stderr)
        return 2

    built_map: GroupMap | None = None
    if build_groups or not is_resume:
        built_map = build_group_map_from_data_root(
            data_root,
            min_images=int(group_cfg.get("min_images", 5)),
            max_per_group=int(group_cfg.get("max_per_group", 80)),
        )
        print(
            f"已构建 train_group: {len(built_map.groups)} 组, "
            f"{len(built_map.authors)} 位作者"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if is_resume:
        model, group_map, meta = load_checkpoint(resume_path, device, group_map=built_map)
        condition = meta.get("condition", condition)  # type: ignore[assignment]
        print(f"已加载 checkpoint: {resume_path}")
    else:
        if built_map is None:
            print("ERROR: 全量训练需要构建 train_group（默认已启用）", file=sys.stderr)
            return 2
        group_map = built_map
        model = PreferenceRanker(num_groups=group_map.num_groups, embed_dim=embed_dim)
        model.to(device)

    with tempfile.TemporaryDirectory(prefix="rank_train_") as tmp_dir:
        merged_pairs = Path(tmp_dir) / "pairs_merged.json"
        try:
            merge_training_pairs(
                pairs_path=pairs_path,
                pairs_new_path=pairs_new_path,
                train_on_new_only=train_on_new_only and is_resume,
                out_path=merged_pairs,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

        errors, warnings, stats = validate_pairs_doc(
            load_json(merged_pairs),
            data_root=data_root,
        )
        if warnings:
            print(f"pairs 校验警告 {len(warnings)} 条（继续训练）")
        if errors:
            for msg in errors[:10]:
                print(f"ERROR: {msg}", file=sys.stderr)
            return 2
        print(f"训练 pairs: {stats.get('pairs_total', 0)} 条")

        train_transform = build_transform(img_size, augment=True)
        val_transform = build_transform(img_size, augment=False)

        full_dataset = PairsDataset(
            merged_pairs,
            data_root,
            group_map,
            transform=train_transform,
            condition=condition,
            seed_pair_weight=seed_pair_weight,
        )
        train_subset, val_subset = split_train_val(full_dataset, val_ratio)

        val_dataset = PairsDataset(
            merged_pairs,
            data_root,
            group_map,
            transform=val_transform,
            condition=condition,
            seed_pair_weight=seed_pair_weight,
        )
        val_subset = Subset(val_dataset, val_subset.indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=pairs_collate_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=pairs_collate_fn,
        )

        checkpoint_dir = str(checkpoint_path.parent)
        remaining_epochs = total_epochs

        if freeze_epochs > 0 and is_resume:
            freeze_epochs = min(freeze_epochs, remaining_epochs)
            set_backbone_trainable(model, False)
            print(f"冻结骨干训练 {freeze_epochs} epoch, lr={learning_rate}")
            result = train_ranker(
                model,
                train_loader,
                val_loader,
                group_map,
                device,
                epochs=freeze_epochs,
                lr=learning_rate,
                weight_decay=weight_decay,
                early_stop_patience=early_stop_patience,
                condition=condition,
                checkpoint_dir=checkpoint_dir,
            )
            remaining_epochs -= result.epochs_run
            if remaining_epochs <= 0:
                print("训练结束（冻结阶段已耗尽 epoch 预算）")
            else:
                set_backbone_trainable(model, True)
                print(f"解冻骨干，继续训练 {remaining_epochs} epoch")

        if remaining_epochs > 0:
            result = train_ranker(
                model,
                train_loader,
                val_loader,
                group_map,
                device,
                epochs=remaining_epochs,
                lr=learning_rate,
                weight_decay=weight_decay,
                early_stop_patience=early_stop_patience,
                condition=condition,
                checkpoint_dir=checkpoint_dir,
            )
            print(
                f"训练完成: best_val_loss={result.best_val_loss:.4f}, "
                f"epochs={result.epochs_run}, early_stop={result.stopped_early}"
            )

        if not checkpoint_path.is_file():
            save_checkpoint(
                checkpoint_path,
                model,
                group_map,
                condition=condition,
            )

        model, group_map, meta = load_checkpoint(checkpoint_path, device)
        percentiles = compute_percentiles(
            model,
            merged_pairs,
            data_root,
            group_map,
            build_transform(img_size, augment=False),
            device,
            condition=condition,
            batch_size=batch_size,
        )
        save_checkpoint(
            checkpoint_path,
            model,
            group_map,
            condition=condition,
            percentiles=percentiles,
            extra={"percentile_source": "train_pairs"},
        )
        print(f"已写入 checkpoint（含 percentiles）: {checkpoint_path}")
        print(f"  p5={percentiles['p5']:.4f}, p95={percentiles['p95']:.4f}")

    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="训练 PreferenceRanker（成对偏好排序）")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rank_default.yaml"),
        help="YAML 配置文件路径",
    )
    parser.add_argument("--data-root", type=Path, default=None, help="图片根目录")
    parser.add_argument("--pairs", type=Path, default=None, help="pairs.json 路径")
    parser.add_argument("--pairs-new", type=Path, default=None, help="增量 pairs 路径（配合 --resume）")
    parser.add_argument("--resume", type=Path, default=None, help="增量训练 checkpoint 路径")
    parser.add_argument("--epochs", type=int, default=None, help="训练 epoch 数")
    parser.add_argument("--lr", type=float, default=None, help="学习率（resume 时默认 lr_finetune）")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=None,
        help="增量训练时冻结骨干的 epoch 数",
    )
    parser.add_argument(
        "--build-groups",
        action="store_true",
        help="强制从 data_root 重建 train_group（resume 时与 checkpoint 合并）",
    )
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="输出 checkpoint 路径（默认 models/rank/best.pth）",
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
    data_root = args.data_root or Path(config.get("data_root") or "")
    if not data_root or not str(data_root).strip():
        print("ERROR: 须通过 --data-root 或配置 data_root 指定图片根目录", file=sys.stderr)
        return 2
    data_root = data_root.resolve()

    pairs_path = args.pairs or (Path(config["pairs_path"]) if config.get("pairs_path") else None)
    if pairs_path is not None:
        pairs_path = pairs_path.resolve()

    build_groups = args.build_groups or args.resume is None

    return run_training(
        config,
        data_root=data_root,
        pairs_path=pairs_path,
        pairs_new_path=args.pairs_new.resolve() if args.pairs_new else None,
        resume_path=args.resume.resolve() if args.resume else None,
        epochs=args.epochs,
        lr=args.lr,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        build_groups=build_groups,
        checkpoint_path=args.checkpoint_out.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
