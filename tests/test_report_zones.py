"""报告分区互斥性单元测试。"""

from __future__ import annotations

from tools.report_results import partition_report_zones


def _score(path: str, score: float, uncertainty: float) -> dict:
    """构造测试用 score 条目。"""
    zone = "mid"
    if score <= 10:
        zone = "extreme_low"
    elif score >= 90:
        zone = "extreme_high"
    return {
        "path": path,
        "score_0_100": score,
        "uncertainty": uncertainty,
        "zone": zone,
    }


def test_report_zones_mutually_exclusive_and_covering() -> None:
    """极端低/极端高/待复核三集合应互斥，且与全量条目一致划分。"""
    scores = [
        _score("a/low.jpg", 5, 0.1),
        _score("b/high.jpg", 95, 0.2),
        _score("c/review.jpg", 50, 0.9),
        _score("d/mid.jpg", 50, 0.1),
        _score("g/mid2.jpg", 55, 0.05),
        _score("f/review2.jpg", 65, 0.95),
    ]

    parts = partition_report_zones(scores, review_percentile=75.0)
    low_paths = {item["path"] for item in parts["extreme_low"]}
    high_paths = {item["path"] for item in parts["extreme_high"]}
    review_paths = {item["path"] for item in parts["needs_review"]}

    assert low_paths == {"a/low.jpg"}
    assert high_paths == {"b/high.jpg"}
    assert review_paths == {"c/review.jpg", "f/review2.jpg"}

    overlap = (low_paths & high_paths) | (low_paths & review_paths) | (high_paths & review_paths)
    assert not overlap

    classified = low_paths | high_paths | review_paths
    all_paths = {item["path"] for item in scores}
    unclassified = all_paths - classified
    assert unclassified == {"d/mid.jpg", "g/mid2.jpg"}
