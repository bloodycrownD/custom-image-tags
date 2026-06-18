"""校验 pairs.json schema、路径存在性并输出统计报告。"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import (
    PAIRS_VERSION,
    VALID_PREFER,
    VALID_SOURCES,
    load_json,
    pair_identity,
)


REQUIRED_PAIR_FIELDS = ("image_a", "image_b", "author_a", "author_b", "prefer", "source")


def validate_pairs_doc(
    doc: dict,
    data_root: Path | None = None,
) -> tuple[list[str], list[str], Counter]:
    """
    校验 pairs 文档。

    返回 (errors, warnings, stats_counter)。
    """
    errors: list[str] = []
    warnings: list[str] = []
    stats: Counter = Counter()

    if not isinstance(doc, dict):
        errors.append("根对象必须是 JSON object")
        return errors, warnings, stats

    version = doc.get("version")
    if version != PAIRS_VERSION:
        errors.append(f"version 必须为 {PAIRS_VERSION}，当前为 {version!r}")

    pairs = doc.get("pairs")
    if pairs is None:
        errors.append("缺少 pairs 字段")
        return errors, warnings, stats
    if not isinstance(pairs, list):
        errors.append("pairs 必须是数组")
        return errors, warnings, stats

    seen_keys: set[tuple[str, str]] = set()

    for i, pair in enumerate(pairs):
        prefix = f"pairs[{i}]"
        if not isinstance(pair, dict):
            errors.append(f"{prefix}: 必须是 object")
            continue

        for field in REQUIRED_PAIR_FIELDS:
            if field not in pair:
                errors.append(f"{prefix}: 缺少必填字段 {field}")

        image_a = pair.get("image_a")
        image_b = pair.get("image_b")
        author_a = pair.get("author_a")
        author_b = pair.get("author_b")
        prefer = pair.get("prefer")
        source = pair.get("source")

        if not isinstance(image_a, str) or not image_a.strip():
            errors.append(f"{prefix}: image_a 必须为非空字符串")
        if not isinstance(image_b, str) or not image_b.strip():
            errors.append(f"{prefix}: image_b 必须为非空字符串")
        if isinstance(image_a, str) and isinstance(image_b, str) and image_a == image_b:
            errors.append(f"{prefix}: image_a 与 image_b 不能相同")

        if prefer not in VALID_PREFER:
            errors.append(f"{prefix}: prefer 必须为 a/b/tie，当前为 {prefer!r}")
        if source not in VALID_SOURCES:
            errors.append(f"{prefix}: source 必须为 seed/annotator，当前为 {source!r}")

        if isinstance(image_a, str) and isinstance(author_a, str):
            expected = Path(image_a).parts[0] if Path(image_a).parts else None
            if expected and expected != author_a:
                warnings.append(
                    f"{prefix}: author_a={author_a!r} 与 image_a 路径作者 {expected!r} 不一致"
                )
        if isinstance(image_b, str) and isinstance(author_b, str):
            expected = Path(image_b).parts[0] if Path(image_b).parts else None
            if expected and expected != author_b:
                warnings.append(
                    f"{prefix}: author_b={author_b!r} 与 image_b 路径作者 {expected!r} 不一致"
                )

        if data_root is not None and isinstance(image_a, str):
            if not (data_root / image_a).is_file():
                errors.append(f"{prefix}: 图片不存在: {data_root / image_a}")
        if data_root is not None and isinstance(image_b, str):
            if not (data_root / image_b).is_file():
                errors.append(f"{prefix}: 图片不存在: {data_root / image_b}")

        if isinstance(image_a, str) and isinstance(image_b, str):
            key = pair_identity(image_a, image_b)
            if key in seen_keys:
                warnings.append(f"{prefix}: 与先前条目重复的无序 pair {key}")
            else:
                seen_keys.add(key)

        if isinstance(prefer, str) and prefer in VALID_PREFER:
            stats[f"prefer:{prefer}"] += 1
        if isinstance(source, str) and source in VALID_SOURCES:
            stats[f"source:{source}"] += 1

    stats["pairs_total"] = len(pairs)
    return errors, warnings, stats


def print_report(errors: list[str], warnings: list[str], stats: Counter) -> None:
    """打印校验报告。"""
    print("=== pairs.json 校验报告 ===")
    print(f"总 pair 数: {stats.get('pairs_total', 0)}")
    for key in sorted(stats):
        if key == "pairs_total":
            continue
        print(f"  {key}: {stats[key]}")

    if warnings:
        print(f"\n警告 ({len(warnings)}):")
        for msg in warnings:
            print(f"  WARN: {msg}")

    if errors:
        print(f"\n错误 ({len(errors)}):")
        for msg in errors:
            print(f"  ERROR: {msg}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """
    CLI 入口。

    退出码：0=通过，1=仅有警告，2=存在错误。
    """
    parser = argparse.ArgumentParser(description="校验 pairs.json")
    parser.add_argument("--pairs", type=Path, required=True, help="pairs.json 路径")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="可选，校验 image_a/image_b 文件是否存在",
    )
    args = parser.parse_args(argv)

    pairs_path = args.pairs
    if not pairs_path.is_file():
        print(f"ERROR: 文件不存在: {pairs_path}", file=sys.stderr)
        return 2

    try:
        doc = load_json(pairs_path)
    except (OSError, ValueError) as exc:
        print(f"ERROR: 无法解析 JSON: {exc}", file=sys.stderr)
        return 2

    data_root = args.data_root.resolve() if args.data_root else None
    if data_root is not None and not data_root.is_dir():
        print(f"ERROR: data_root 不存在或不是目录: {data_root}", file=sys.stderr)
        return 2

    errors, warnings, stats = validate_pairs_doc(doc, data_root=data_root)
    print_report(errors, warnings, stats)

    if errors:
        return 2
    if warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
