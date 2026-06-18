"""从 scores.json 生成 F8 排序报告（JSON + HTML）。"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import load_json, parse_scores_doc, save_json


def load_scores(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """读取 scores.json，返回 (scores 列表, meta 字典)。"""
    data = load_json(path)
    return parse_scores_doc(data)


def _percentile(values: list[float], pct: float) -> float:
    """计算分位数。"""
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


def partition_report_zones(
    scores: list[dict[str, Any]],
    *,
    review_percentile: float = 75.0,
    u_threshold: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    将 scores 划分为报告用集合。

    - extreme_low: score_0_100 <= 10
    - extreme_high: score_0_100 >= 90
    - needs_review: 30 <= score_0_100 <= 70 且 uncertainty > u_threshold

    u_threshold 优先使用传入值（来自 scores.json meta）；未提供时回退为全库 uncertainty 分位。

    三集合互斥；其余条目归入 sorted 但不进入 needs_review。
    """
    if u_threshold is None:
        uncertainties = [float(item.get("uncertainty", 0.0)) for item in scores]
        u_threshold = _percentile(uncertainties, review_percentile)

    extreme_low: list[dict[str, Any]] = []
    extreme_high: list[dict[str, Any]] = []
    needs_review: list[dict[str, Any]] = []

    for item in scores:
        score = float(item.get("score_0_100", 50.0))
        uncertainty = float(item.get("uncertainty", 0.0))

        if score <= 10.0:
            extreme_low.append(item)
        elif score >= 90.0:
            extreme_high.append(item)
        elif 30.0 <= score <= 70.0 and uncertainty > u_threshold:
            needs_review.append(item)

    return {
        "extreme_low": extreme_low,
        "extreme_high": extreme_high,
        "needs_review": needs_review,
        "u_threshold": u_threshold,
    }


def _render_table(title: str, items: list[dict[str, Any]], data_root: Path | None) -> str:
    """渲染单列表格 HTML 片段。"""
    rows: list[str] = []
    for item in items:
        path = str(item.get("path", ""))
        if data_root is not None:
            thumb_href = (data_root / path).resolve().as_uri()
        else:
            thumb_href = path
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(thumb_href)}\">{html.escape(path)}</a></td>"
            f"<td>{item.get('score_0_100', '')}</td>"
            f"<td>{item.get('uncertainty', '')}</td>"
            f"<td>{html.escape(str(item.get('zone', '')))}</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan=\"4\">（无）</td></tr>"
    return (
        f"<section><h2>{html.escape(title)} ({len(items)})</h2>"
        "<table><thead><tr>"
        "<th>路径</th><th>score_0_100</th><th>uncertainty</th><th>zone</th>"
        "</tr></thead><tbody>"
        f"{body}</tbody></table></section>"
    )


def build_index_html(
    partitions: dict[str, list[dict[str, Any]]],
    *,
    data_root: Path | None = None,
) -> str:
    """生成 index.html 内容。"""
    sections = [
        _render_table("极端低分", partitions["extreme_low"], data_root),
        _render_table("极端高分", partitions["extreme_high"], data_root),
        _render_table("待复核", partitions["needs_review"], data_root),
    ]
    return (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head>"
        "<meta charset=\"utf-8\"/>"
        "<title>审美评分报告</title>"
        "<style>"
        "body{font-family:sans-serif;margin:1.5rem;}"
        "table{border-collapse:collapse;width:100%;margin-bottom:2rem;}"
        "th,td{border:1px solid #ccc;padding:0.4rem;text-align:left;}"
        "h2{margin-top:0;}"
        "</style></head><body>"
        "<h1>审美评分报告</h1>"
        + "".join(sections)
        + "</body></html>"
    )


def write_report(
    scores: list[dict[str, Any]],
    out_dir: Path,
    *,
    review_percentile: float = 75.0,
    u_threshold: float | None = None,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """
    写入 sorted / extreme / needs_review JSON 与 index.html。

    Returns:
        分区摘要（含 u_threshold 与各集合数量）。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    sorted_scores = sorted(
        scores,
        key=lambda x: float(x.get("score_0_100", 0.0)),
        reverse=True,
    )
    partitions = partition_report_zones(
        scores,
        review_percentile=review_percentile,
        u_threshold=u_threshold,
    )

    save_json(out_dir / "sorted.json", sorted_scores)
    save_json(out_dir / "extreme_low.json", partitions["extreme_low"])
    save_json(out_dir / "extreme_high.json", partitions["extreme_high"])
    save_json(out_dir / "needs_review.json", partitions["needs_review"])

    html_content = build_index_html(partitions, data_root=data_root)
    (out_dir / "index.html").write_text(html_content, encoding="utf-8")

    return {
        "total": len(scores),
        "sorted": len(sorted_scores),
        "extreme_low": len(partitions["extreme_low"]),
        "extreme_high": len(partitions["extreme_high"]),
        "needs_review": len(partitions["needs_review"]),
        "u_threshold": partitions["u_threshold"],
    }


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="生成审美评分报告")
    parser.add_argument("--scores", type=Path, required=True, help="scores.json 路径")
    parser.add_argument("--out", type=Path, required=True, help="报告输出目录")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="图片根目录（HTML 缩略图链接用，可选）",
    )
    parser.add_argument(
        "--review-percentile",
        type=float,
        default=75.0,
        help="待复核 uncertainty 分位阈值（默认 75）",
    )
    args = parser.parse_args(argv)

    if not args.scores.is_file():
        print(f"ERROR: scores 文件不存在: {args.scores}", file=sys.stderr)
        return 2

    try:
        scores, meta = load_scores(args.scores)
        u_threshold = meta.get("u_threshold")
        if u_threshold is not None:
            u_threshold = float(u_threshold)
        summary = write_report(
            scores,
            args.out.resolve(),
            review_percentile=args.review_percentile,
            u_threshold=u_threshold,
            data_root=args.data_root.resolve() if args.data_root else None,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"已写入 {args.out}: sorted={summary['sorted']}, "
        f"extreme_low={summary['extreme_low']}, "
        f"extreme_high={summary['extreme_high']}, "
        f"needs_review={summary['needs_review']}, "
        f"u_threshold={summary['u_threshold']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
