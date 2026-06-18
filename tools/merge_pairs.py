"""合并多个 pairs.json，去重；annotator 优先于 seed。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import (
    PAIRS_VERSION,
    SOURCE_PRIORITY,
    empty_pairs_doc,
    load_json,
    pair_identity,
    save_json,
)


def merge_pairs_files(inputs: list[Path]) -> tuple[dict, int, int]:
    """
    按顺序合并多个 pairs 文件。

    同一无序 pair 冲突时，source 优先级 annotator > seed；
    同优先级时后出现的文件覆盖先前条目。

    返回 (merged_doc, input_pair_count, deduped_count)。
    """
    merged: dict[tuple[str, str], dict] = {}
    input_total = 0

    for path in inputs:
        doc = load_json(path)
        pairs = doc.get("pairs", [])
        if not isinstance(pairs, list):
            raise ValueError(f"{path}: pairs 必须是数组")

        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            image_a = pair.get("image_a")
            image_b = pair.get("image_b")
            if not isinstance(image_a, str) or not isinstance(image_b, str):
                continue
            input_total += 1
            key = pair_identity(image_a, image_b)
            source = pair.get("source", "seed")
            priority = SOURCE_PRIORITY.get(source, -1)

            existing = merged.get(key)
            if existing is None:
                merged[key] = dict(pair)
                continue

            existing_source = existing.get("source", "seed")
            existing_priority = SOURCE_PRIORITY.get(existing_source, -1)
            if priority > existing_priority:
                merged[key] = dict(pair)
            elif priority == existing_priority:
                merged[key] = dict(pair)

    out_doc = empty_pairs_doc()
    out_doc["version"] = PAIRS_VERSION
    out_doc["pairs"] = list(merged.values())
    deduped = input_total - len(out_doc["pairs"])
    return out_doc, input_total, deduped


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="合并多个 pairs.json")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="待合并的 pairs.json（后者在同优先级时覆盖前者）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pairs.json"),
        help="合并结果输出路径",
    )
    args = parser.parse_args(argv)

    for path in args.inputs:
        if not path.is_file():
            print(f"ERROR: 文件不存在: {path}", file=sys.stderr)
            return 2

    try:
        merged, input_total, deduped = merge_pairs_files(args.inputs)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    save_json(args.out, merged)
    print(
        f"已写入 {args.out}: 输入 {input_total} 条, "
        f"去重 {deduped} 条, 输出 {len(merged['pairs'])} 条"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
