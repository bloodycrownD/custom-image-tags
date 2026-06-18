"""PreferenceRanker 训练 CLI。"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
import torchvision.transforms as transforms
import yaml
from torch.utils.data import Subset, random_split

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rank.checkpoint import load_checkpoint, save_checkpoint
from rank.dataloader_utils import build_pairs_dataloader
from rank.dataset import PairsDataset, pairs_collate_fn
from rank.groups import ConditionMode, GroupMap
from rank.image_cache import ImageTensorStore, collect_unique_paths_from_pairs
from rank.model import PreferenceRanker
from rank.predict import predict_batch
from rank.train import set_backbone_trainable, train_ranker
from rank.transforms import build_deterministic_transform
from rank.transforms import KeepRatioResizePad
from tools import load_json, save_json
from tools.build_groups import build_train_group_map
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
    """构建图像变换（兼容旧调用；新代码优先用 PairsDataset 的 img_size/augment）。"""
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


def maybe_build_tensor_store(
    pairs: list,
    data_root: Path,
    img_size: int,
    dl_cfg: dict,
) -> ImageTensorStore | None:
    """按配置预热张量缓存并放入共享内存。"""
    if not bool(dl_cfg.get("cache_tensors", True)):
        return None
    if not bool(dl_cfg.get("warmup_cache", True)):
        return None

    paths = collect_unique_paths_from_pairs(pairs)
    if not paths:
        return None

    transform = build_deterministic_transform(img_size)
    store = ImageTensorStore()
    count = store.build(paths, data_root, transform, desc="预热图片张量缓存")
    store.share_memory()
    nbytes_mb = store.nbytes / (1024 * 1024)
    print(f"已预热张量缓存: {count} 张, 约 {nbytes_mb:.1f} MB")
    return store


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
    return GroupMap(doc["groups"], doc["authors"], doc.get("images"))


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


def _percentile_value(values: list[float], pct: float) -> float:
    """计算分位数（线性插值）。"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def collect_val_images(val_subset: Subset, dataset: PairsDataset) -> list[tuple[str, str]]:
    """从验证集 subset 收集去重后的 (相对路径, author) 列表。"""
    seen: set[str] = set()
    items: list[tuple[str, str]] = []
    for idx in val_subset.indices:
        pair = dataset.pairs[idx]
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


def compute_u_threshold(
    model: PreferenceRanker,
    val_items: list[tuple[str, str]],
    data_root: Path,
    group_map: GroupMap,
    transform,
    device: torch.device,
    *,
    condition: ConditionMode,
    batch_size: int,
    n_mc: int,
    review_percentile: float,
) -> float:
    """在验证集图片上跑 MC Dropout，取 uncertainty 的指定分位作为 u_threshold。"""
    if not val_items:
        return 0.0

    results = predict_batch(
        model,
        val_items,
        data_root,
        group_map,
        transform,
        device,
        condition=condition,
        batch_size=batch_size,
        n_mc=max(n_mc, 2),
    )
    uncertainties = [r.uncertainty for r in results]
    return _percentile_value(uncertainties, review_percentile)


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
    pretrained: bool = True,
) -> int:
    """执行完整训练流程。"""
    group_cfg = config.get("group", {})
    finetune_cfg = config.get("finetune", {})
    uncertainty_cfg = config.get("uncertainty", {})
    condition: ConditionMode = config.get("condition", "train_group")
    img_size = int(config.get("img_size", 384))
    batch_size = int(config.get("batch_size", 16))
    embed_dim = int(config.get("embed_dim", 64))
    weight_decay = float(config.get("weight_decay", 0.01))
    early_stop_patience = int(config.get("early_stop_patience", 7))
    val_ratio = float(config.get("val_ratio", 0.1))
    seed_pair_weight = float(config.get("seed_pair_weight", 0.5))
    train_on_new_only = bool(finetune_cfg.get("train_on_new_only", False))
    n_mc = int(uncertainty_cfg.get("n_mc", 10))
    review_percentile = float(uncertainty_cfg.get("review_percentile", 75))
    dl_cfg = config.get("dataloader", {})

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
        model, group_map, meta = load_checkpoint(
            resume_path, device, group_map=built_map, pretrained=pretrained
        )
        condition = meta.get("condition", condition)  # type: ignore[assignment]
        print(f"已加载 checkpoint: {resume_path}")
    else:
        if built_map is None:
            print("ERROR: 全量训练需要构建 train_group（默认已启用）", file=sys.stderr)
            return 2
        group_map = built_map
        model = PreferenceRanker(
            num_groups=group_map.num_groups,
            embed_dim=embed_dim,
            pretrained=pretrained,
        )
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

        pairs_list = load_json(merged_pairs).get("pairs", [])
        tensor_store = maybe_build_tensor_store(pairs_list, data_root, img_size, dl_cfg)
        resize_pad_cache = bool(dl_cfg.get("resize_pad_cache", True))
        tensor_augment_flip = bool(dl_cfg.get("tensor_augment_flip", True))
        pin_memory = bool(dl_cfg.get("pin_memory", True)) and device.type == "cuda"

        num_workers = int(dl_cfg.get("num_workers", 4))
        if num_workers != 0:
            print(
                f"DataLoader: num_workers={num_workers}, pin_memory={pin_memory}, "
                f"cache_tensors={tensor_store is not None}"
            )

        full_dataset = PairsDataset(
            merged_pairs,
            data_root,
            group_map,
            condition=condition,
            seed_pair_weight=seed_pair_weight,
            img_size=img_size,
            augment=True,
            tensor_store=tensor_store,
            tensor_augment_flip=tensor_augment_flip,
            resize_pad_cache=resize_pad_cache and tensor_store is None,
        )
        train_subset, val_subset = split_train_val(full_dataset, val_ratio)

        val_dataset = PairsDataset(
            merged_pairs,
            data_root,
            group_map,
            condition=condition,
            seed_pair_weight=seed_pair_weight,
            img_size=img_size,
            augment=False,
            tensor_store=tensor_store,
            resize_pad_cache=False,
        )
        val_subset = Subset(val_dataset, val_subset.indices)

        train_loader = build_pairs_dataloader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=pairs_collate_fn,
            dl_cfg=dl_cfg,
        )
        val_loader = build_pairs_dataloader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pairs_collate_fn,
            dl_cfg=dl_cfg,
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
                pin_memory=pin_memory,
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
                pin_memory=pin_memory,
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

        model, group_map, meta = load_checkpoint(checkpoint_path, device, pretrained=pretrained)
        percentiles = compute_percentiles(
            model,
            merged_pairs,
            data_root,
            group_map,
            build_deterministic_transform(img_size),
            device,
            condition=condition,
            batch_size=batch_size,
        )
        val_items = collect_val_images(val_subset, val_dataset)
        u_threshold = compute_u_threshold(
            model,
            val_items,
            data_root,
            group_map,
            build_deterministic_transform(img_size),
            device,
            condition=condition,
            batch_size=batch_size,
            n_mc=n_mc,
            review_percentile=review_percentile,
        )
        save_checkpoint(
            checkpoint_path,
            model,
            group_map,
            condition=condition,
            percentiles=percentiles,
            u_threshold=u_threshold,
            extra={"percentile_source": "train_pairs", "u_threshold_source": "val_images"},
        )
        print(f"已写入 checkpoint（含 percentiles / u_threshold）: {checkpoint_path}")
        print(f"  p5={percentiles['p5']:.4f}, p95={percentiles['p95']:.4f}")
        print(f"  u_threshold={u_threshold:.4f} (P{review_percentile:.0f} on val)")

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
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="不加载 ImageNet 预训练骨干（测试用，避免下载权重）",
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
        pretrained=not args.no_pretrained,
    )


if __name__ == "__main__":
    raise SystemExit(main())
