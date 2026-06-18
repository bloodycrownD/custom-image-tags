"""在 held-out pairs 上评估 PreferenceRanker 成对准确率。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Literal

import torch
import torchvision.transforms as transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rank.checkpoint import load_checkpoint
from rank.groups import GroupMap
from rank.model import PreferenceRanker
from rank.transforms import KeepRatioResizePad
from tools import load_json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ZoneBucket = Literal["extreme", "mid"]


def build_eval_transform(img_size: int = 384):
    """构建评估用图像变换。"""
    return transforms.Compose(
        [
            KeepRatioResizePad(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _predict_prefer(
    model: PreferenceRanker,
    pair: dict[str, Any],
    data_root: Path,
    group_map: GroupMap,
    transform,
    device: torch.device,
    *,
    condition: str,
    percentiles: dict[str, float],
) -> tuple[str, float, float, float, float]:
    """
    对单条 pair 推理 A/B 分数。

    Returns:
        (predicted_prefer, score_a, score_b, score_0_100_a, score_0_100_b)
    """
    from PIL import Image

    from rank.checkpoint import raw_to_score_0_100

    def load_tensor(rel_path: str) -> torch.Tensor:
        try:
            image = Image.open(data_root / rel_path).convert("RGB")
        except OSError:
            image = Image.new("RGB", (384, 384), color=(0, 0, 0))
        return transform(image)

    img_a = load_tensor(pair["image_a"]).unsqueeze(0).to(device)
    img_b = load_tensor(pair["image_b"]).unsqueeze(0).to(device)

    group_a = group_map.resolve(pair.get("author_a", ""), pair.get("image_a"), condition)  # type: ignore[arg-type]
    group_b = group_map.resolve(pair.get("author_b", ""), pair.get("image_b"), condition)  # type: ignore[arg-type]
    ga = torch.tensor([group_a], dtype=torch.long, device=device)
    gb = torch.tensor([group_b], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        score_a = model.predict(img_a, ga).item()
        score_b = model.predict(img_b, gb).item()

    s100_a = raw_to_score_0_100(score_a, percentiles)
    s100_b = raw_to_score_0_100(score_b, percentiles)

    diff = score_a - score_b
    tie_eps = 0.05
    if abs(diff) <= tie_eps:
        predicted = "tie"
    elif diff > 0:
        predicted = "a"
    else:
        predicted = "b"

    return predicted, score_a, score_b, s100_a, s100_b


def _pair_zone_bucket(score_0_100_a: float, score_0_100_b: float) -> ZoneBucket:
    """按 A/B 分数判断 pair 所属区域（极端 / 中间）。"""
    def is_extreme(score: float) -> bool:
        return score <= 10.0 or score >= 90.0

    if is_extreme(score_0_100_a) or is_extreme(score_0_100_b):
        return "extreme"
    return "mid"


def _is_correct(predicted: str, prefer: str) -> bool:
    """判断预测 prefer 是否与标注一致。"""
    return predicted == prefer


def evaluate_pairs(
    model: PreferenceRanker,
    pairs: list[dict[str, Any]],
    *,
    data_root: Path,
    group_map: GroupMap,
    transform,
    device: torch.device,
    condition: str = "train_group",
    percentiles: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    计算 held-out pairs 的成对准确率，并按极端/中间分区统计。

    Returns:
        含 overall / extreme / mid 准确率与样本数的字典。
    """
    percentiles = percentiles or {}
    stats = {
        "overall": {"correct": 0, "total": 0},
        "extreme": {"correct": 0, "total": 0},
        "mid": {"correct": 0, "total": 0},
    }

    for pair in pairs:
        prefer = pair.get("prefer", "a")
        predicted, _, _, s100_a, s100_b = _predict_prefer(
            model,
            pair,
            data_root,
            group_map,
            transform,
            device,
            condition=condition,
            percentiles=percentiles,
        )
        bucket = _pair_zone_bucket(s100_a, s100_b)
        correct = _is_correct(predicted, prefer)

        for key in ("overall", bucket):
            stats[key]["total"] += 1
            if correct:
                stats[key]["correct"] += 1

    def _accuracy_block(key: str) -> dict[str, Any]:
        bucket = stats[key]
        total = bucket["total"]
        acc = bucket["correct"] / total if total else 0.0
        return {
            "correct": bucket["correct"],
            "total": total,
            "accuracy": round(acc, 4),
        }

    return {
        "pairs_total": len(pairs),
        "overall": _accuracy_block("overall"),
        "by_zone": {
            "extreme": _accuracy_block("extreme"),
            "mid": _accuracy_block("mid"),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="评估 PreferenceRanker 成对准确率")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型 checkpoint")
    parser.add_argument("--pairs", type=Path, required=True, help="held-out pairs.json")
    parser.add_argument("--data-root", type=Path, required=True, help="图片根目录")
    parser.add_argument("--img-size", type=int, default=384, help="输入尺寸")
    args = parser.parse_args(argv)

    for path, label in (
        (args.checkpoint, "checkpoint"),
        (args.pairs, "pairs"),
        (args.data_root, "data_root"),
    ):
        if label == "data_root":
            if not path.is_dir():
                print(f"ERROR: {label} 不存在或不是目录: {path}", file=sys.stderr)
                return 2
        elif not path.is_file():
            print(f"ERROR: {label} 不存在: {path}", file=sys.stderr)
            return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, group_map, meta = load_checkpoint(args.checkpoint, device)
    condition = meta.get("condition", "train_group")
    percentiles = meta.get("percentiles", {})

    pairs_doc = load_json(args.pairs)
    pairs = pairs_doc.get("pairs", [])

    transform = build_eval_transform(args.img_size)
    metrics = evaluate_pairs(
        model,
        pairs,
        data_root=args.data_root.resolve(),
        group_map=group_map,
        transform=transform,
        device=device,
        condition=condition,
        percentiles=percentiles,
    )

    overall = metrics["overall"]
    extreme = metrics["by_zone"]["extreme"]
    mid = metrics["by_zone"]["mid"]
    print(
        f"成对准确率: {overall['accuracy']:.2%} "
        f"({overall['correct']}/{overall['total']})"
    )
    print(
        f"  极端区: {extreme['accuracy']:.2%} ({extreme['correct']}/{extreme['total']})"
    )
    print(f"  中间区: {mid['accuracy']:.2%} ({mid['correct']}/{mid['total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
