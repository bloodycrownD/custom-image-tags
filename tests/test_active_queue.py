"""主动学习队列导出单元测试。"""

from __future__ import annotations

from tools import pair_identity
from tools.export_active_queue import generate_active_queue


def _make_score(
    path: str,
    *,
    score: float,
    uncertainty: float,
    group_id: int,
) -> dict:
    """构造单条 score 记录。"""
    zone = "mid"
    if score <= 10:
        zone = "extreme_low"
    elif score >= 90:
        zone = "extreme_high"
    return {
        "path": path,
        "author": path.split("/")[0],
        "group_id": group_id,
        "score_raw": score / 100.0,
        "score_0_100": score,
        "uncertainty": uncertainty,
        "zone": zone,
    }


def test_active_queue_skips_existing_pairs() -> None:
    """应跳过 pairs.json 中已有边。"""
    scores = [
        _make_score("a/g1.jpg", score=50, uncertainty=0.9, group_id=1),
        _make_score("a/g2.jpg", score=51, uncertainty=0.85, group_id=1),
        _make_score("b/g1.jpg", score=5, uncertainty=0.8, group_id=2),
        _make_score("b/g2.jpg", score=95, uncertainty=0.75, group_id=2),
    ]
    existing = {pair_identity("a/g1.jpg", "a/g2.jpg")}

    doc = generate_active_queue(scores, existing_pairs=existing, budget=50)

    for pair in doc["pairs"]:
        key = pair_identity(pair["image_a"], pair["image_b"])
        assert key not in existing


def test_active_queue_respects_budget() -> None:
    """输出条数不应超过 budget。"""
    scores = [
        _make_score(f"g{i // 2}/img{i}.jpg", score=50 + (i % 3), uncertainty=0.5 + i * 0.01, group_id=i % 3 + 1)
        for i in range(20)
    ]

    budget = 5
    doc = generate_active_queue(scores, budget=budget)

    assert doc["budget"] == budget
    assert len(doc["pairs"]) <= budget
    assert doc["version"] == 1
    for pair in doc["pairs"]:
        assert "reason" in pair
        assert "priority" in pair
