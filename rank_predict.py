"""PreferenceRanker 批量推理 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision.transforms as transforms
import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rank.checkpoint import load_checkpoint
from rank.predict import predict_batch
from rank.transforms import KeepRatioResizePad
from tools import iter_images_under, rel_posix_path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_transform(img_size: int):
    """构建推理用图像变换（与训练验证一致，无增强）。"""
    return transforms.Compose(
        [
            KeepRatioResizePad(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def iter_data_root_images(data_root: Path):
    """
    遍历 data_root 下作者子目录中的全部图片。

    Yields:
        (相对路径, author_id) 元组。
    """
    for author_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        author = author_dir.name
        for img_path in iter_images_under(author_dir):
            yield rel_posix_path(img_path, data_root), author


def run_predict(
    *,
    checkpoint_path: Path,
    data_root: Path,
    out_path: Path,
    config: dict,
    n_mc: int | None,
    batch_size: int | None,
) -> int:
    """执行批量推理并写入 scores.json。"""
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint 不存在: {checkpoint_path}", file=sys.stderr)
        return 2
    if not data_root.is_dir():
        print(f"ERROR: data_root 不存在或不是目录: {data_root}", file=sys.stderr)
        return 2

    img_size = int(config.get("img_size", 384))
    batch_size = int(batch_size if batch_size is not None else config.get("batch_size", 16))
    uncertainty_cfg = config.get("uncertainty", {})
    n_mc = int(n_mc if n_mc is not None else uncertainty_cfg.get("n_mc", 10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model, group_map, meta = load_checkpoint(checkpoint_path, device)
    condition = meta.get("condition", config.get("condition", "train_group"))
    percentiles = meta.get("percentiles", {})

    items = list(iter_data_root_images(data_root))
    if not items:
        print(f"WARNING: {data_root} 下未找到图片", file=sys.stderr)

    print(f"待推理图片: {len(items)} 张, n_mc={n_mc}")
    transform = build_transform(img_size)

    results = predict_batch(
        model,
        items,
        data_root,
        group_map,
        transform,
        device,
        condition=condition,
        batch_size=batch_size,
        n_mc=n_mc,
        percentiles=percentiles,
    )

    payload = [
        {
            "path": item.path,
            "author": item.author,
            "group_id": item.group_id,
            "score_raw": item.score_raw,
            "score_0_100": item.score_0_100,
            "uncertainty": item.uncertainty,
            "zone": item.zone,
        }
        for item in results
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"已写入 {out_path}: {len(payload)} 条评分")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="PreferenceRanker 批量推理")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="模型 checkpoint 路径",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="图片根目录")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scores.json"),
        help="输出 scores.json 路径",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rank_default.yaml"),
        help="YAML 配置文件（img_size、batch_size、n_mc 默认值）",
    )
    parser.add_argument("--n-mc", type=int, default=None, help="MC Dropout 采样次数")
    parser.add_argument("--batch-size", type=int, default=None, help="推理批大小")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config: dict = {}
    if args.config.is_file():
        config = load_config(args.config)
    elif args.n_mc is None:
        print(f"WARNING: 配置文件不存在，使用内置默认值: {args.config}", file=sys.stderr)

    return run_predict(
        checkpoint_path=args.checkpoint.resolve(),
        data_root=args.data_root.resolve(),
        out_path=args.out.resolve(),
        config=config,
        n_mc=args.n_mc,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
