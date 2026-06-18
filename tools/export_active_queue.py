"""从 scores.json 采样主动学习队列，输出 active_queue.json。"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import load_json, parse_scores_doc, pair_identity, save_json


def load_scores(path: Path) -> list[dict[str, Any]]:
    """读取 scores.json 中的 scores 列表（兼容旧版纯列表格式）。"""
    data = load_json(path)
    scores, _meta = parse_scores_doc(data)
    return scores


def load_existing_pair_keys(pairs_path: Path | None) -> set[tuple[str, str]]:
    """加载 pairs.json 中已有边的无序键集合。"""
    if pairs_path is None or not pairs_path.is_file():
        return set()
    doc = load_json(pairs_path)
    keys: set[tuple[str, str]] = set()
    for pair in doc.get("pairs", []):
        image_a = pair.get("image_a", "")
        image_b = pair.get("image_b", "")
        if image_a and image_b:
            keys.add(pair_identity(image_a, image_b))
    return keys


def _percentile(values: list[float], pct: float) -> float:
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


def _score_index(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按 path 建立索引。"""
    return {item["path"]: item for item in scores if item.get("path")}


def _priority_normalize(value: float, lo: float, hi: float) -> float:
    """将 value 线性归一化到 [0, 1]。"""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _make_candidate(
    image_a: str,
    image_b: str,
    reason: str,
    priority: float,
) -> dict[str, Any]:
    """构造单条队列候选。"""
    return {
        "image_a": image_a,
        "image_b": image_b,
        "reason": reason,
        "priority": round(float(priority), 4),
    }


def generate_active_queue(
    scores: list[dict[str, Any]],
    *,
    existing_pairs: set[tuple[str, str]] | None = None,
    budget: int = 200,
    score_close_threshold: float = 5.0,
) -> dict[str, Any]:
    """
    按 spec 三策略采样主动学习队列。

    策略顺序：
    1. 双高不确定性且分数接近（难分辨对）
    2. 极端区候选 + 高不确定性（边界确认）
    3. 跨 train_group 且不确定性之和最高（补稀疏边）

    去重：跳过已有边；同对只保留一条；按 priority 降序截断至 budget。
    """
    if budget < 1:
        raise ValueError("budget 必须 >= 1")

    existing_pairs = existing_pairs or set()
    by_path = _score_index(scores)
    paths = list(by_path.keys())
    if len(paths) < 2:
        return {"version": 1, "budget": budget, "pairs": []}

    uncertainties = [float(item.get("uncertainty", 0.0)) for item in scores]
    score_0_100_list = [float(item.get("score_0_100", 50.0)) for item in scores]
    u_p75 = _percentile(uncertainties, 75.0)
    score_p10 = _percentile(score_0_100_list, 10.0)
    score_p90 = _percentile(score_0_100_list, 90.0)

    u_min, u_max = min(uncertainties), max(uncertainties)
    sum_min = u_min * 2
    sum_max = u_max * 2

    seen_pairs: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []

    def is_existing(image_a: str, image_b: str) -> bool:
        key = pair_identity(image_a, image_b)
        return key in existing_pairs or key in seen_pairs

    def add_candidate(image_a: str, image_b: str, reason: str, priority: float) -> None:
        if image_a == image_b:
            return
        key = pair_identity(image_a, image_b)
        if key in existing_pairs or key in seen_pairs:
            return
        seen_pairs.add(key)
        candidates.append(_make_candidate(image_a, image_b, reason, priority))

    # 策略 1：双高不确定性 + 分数接近
    for path_a, path_b in itertools.combinations(paths, 2):
        item_a = by_path[path_a]
        item_b = by_path[path_b]
        u_a = float(item_a.get("uncertainty", 0.0))
        u_b = float(item_b.get("uncertainty", 0.0))
        if u_a <= u_p75 or u_b <= u_p75:
            continue
        s_a = float(item_a.get("score_0_100", 50.0))
        s_b = float(item_b.get("score_0_100", 50.0))
        if abs(s_a - s_b) >= score_close_threshold:
            continue
        priority = (
            _priority_normalize(min(u_a, u_b), u_p75, u_max)
            + _priority_normalize(score_close_threshold - abs(s_a - s_b), 0.0, score_close_threshold)
        ) / 2.0
        add_candidate(path_a, path_b, "high_uncertainty_pair", priority)

    # 策略 2：极端区 + 高不确定性
    for path_a, path_b in itertools.combinations(paths, 2):
        item_a = by_path[path_a]
        item_b = by_path[path_b]
        s_a = float(item_a.get("score_0_100", 50.0))
        s_b = float(item_b.get("score_0_100", 50.0))
        u_a = float(item_a.get("uncertainty", 0.0))
        u_b = float(item_b.get("uncertainty", 0.0))

        extreme_a = s_a <= score_p10 or s_a >= score_p90
        extreme_b = s_b <= score_p10 or s_b >= score_p90
        if not extreme_a and not extreme_b:
            continue

        if extreme_a and u_b > u_p75:
            extreme_strength = abs(s_a - 50.0) / 50.0
            priority = (
                _priority_normalize(u_b, u_p75, u_max) + _priority_normalize(extreme_strength, 0.0, 1.0)
            ) / 2.0
            add_candidate(path_a, path_b, "extreme_boundary_confirm", priority)
        if extreme_b and u_a > u_p75:
            extreme_strength = abs(s_b - 50.0) / 50.0
            priority = (
                _priority_normalize(u_a, u_p75, u_max) + _priority_normalize(extreme_strength, 0.0, 1.0)
            ) / 2.0
            add_candidate(path_b, path_a, "extreme_boundary_confirm", priority)

    # 策略 3：跨 train_group，按不确定性之和排序
    cross_group: list[tuple[float, str, str]] = []
    for path_a, path_b in itertools.combinations(paths, 2):
        item_a = by_path[path_a]
        item_b = by_path[path_b]
        group_a = item_a.get("group_id")
        group_b = item_b.get("group_id")
        if group_a is None or group_b is None or group_a == group_b:
            continue
        u_sum = float(item_a.get("uncertainty", 0.0)) + float(item_b.get("uncertainty", 0.0))
        cross_group.append((u_sum, path_a, path_b))

    cross_group.sort(key=lambda x: x[0], reverse=True)
    for u_sum, path_a, path_b in cross_group:
        priority = _priority_normalize(u_sum, sum_min, sum_max)
        add_candidate(path_a, path_b, "cross_group_sparse", priority)

    candidates.sort(key=lambda x: x["priority"], reverse=True)
    selected = candidates[:budget]

    return {
        "version": 1,
        "budget": budget,
        "pairs": selected,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="从 scores.json 导出主动学习队列")
    parser.add_argument("--scores", type=Path, required=True, help="scores.json 路径")
    parser.add_argument("--pairs", type=Path, default=None, help="已有 pairs.json（跳过已有边）")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("active_queue.json"),
        help="输出 active_queue.json 路径",
    )
    parser.add_argument("--budget", type=int, default=200, help="采样上限（默认 200）")
    parser.add_argument(
        "--score-close-threshold",
        type=float,
        default=5.0,
        help="策略 1 分数接近阈值（默认 5.0）",
    )
    args = parser.parse_args(argv)

    if not args.scores.is_file():
        print(f"ERROR: scores 文件不存在: {args.scores}", file=sys.stderr)
        return 2

    try:
        scores = load_scores(args.scores)
        existing = load_existing_pair_keys(args.pairs)
        doc = generate_active_queue(
            scores,
            existing_pairs=existing,
            budget=args.budget,
            score_close_threshold=args.score_close_threshold,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    save_json(args.out, doc)
    print(f"已写入 {args.out}: {len(doc['pairs'])} 条候选（budget={doc['budget']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
