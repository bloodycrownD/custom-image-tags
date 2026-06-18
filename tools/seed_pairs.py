"""从 author/{good,keep,trash} 目录结构生成种子 pairs.json。"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import (
    LABEL_DIRS,
    LABEL_RANK,
    PAIRS_VERSION,
    empty_pairs_doc,
    is_image_file,
    rel_posix_path,
    save_json,
    utc_now_iso,
)


def _collect_label_images(author_dir: Path, data_root: Path) -> dict[str, list[str]]:
    """收集作者各标签下的图片相对路径（已排序）。"""
    by_label: dict[str, list[str]] = {label: [] for label in LABEL_DIRS}
    for label in LABEL_DIRS:
        label_dir = author_dir / label
        if not label_dir.is_dir():
            continue
        for img_path in sorted(label_dir.iterdir(), key=lambda p: p.name.lower()):
            if is_image_file(img_path):
                by_label[label].append(rel_posix_path(img_path, data_root))
    return by_label


def _make_pair(
    path_a: str,
    path_b: str,
    author_id: str,
    prefer: str,
    created_at: str,
) -> dict:
    """构造单条 pair 记录。"""
    return {
        "image_a": path_a,
        "image_b": path_b,
        "author_a": author_id,
        "author_b": author_id,
        "prefer": prefer,
        "source": "seed",
        "created_at": created_at,
    }


def _prefer_for_labels(label_a: str, label_b: str, path_a: str, path_b: str) -> str | None:
    """
    根据标签等级推导 prefer。

    good > keep > trash；同档 keep-keep 为 tie；同档 good/trash 跳过。
    """
    rank_a = LABEL_RANK[label_a]
    rank_b = LABEL_RANK[label_b]
    if rank_a == rank_b:
        if label_a == "keep":
            return "tie"
        return None
    if rank_a > rank_b:
        return "a"
    return "b"


def generate_seed_pairs(data_root: Path) -> list[dict]:
    """
    扫描 data_root，为每位作者生成跨标签种子 pair。

    规则（PRD / spec）：
    - good 优于 keep / trash
    - keep 优于 trash
    - 同档 keep 之间为 tie
    """
    data_root = data_root.resolve()
    created_at = utc_now_iso()
    pairs: list[dict] = []

    author_dirs = sorted(
        (p for p in data_root.iterdir() if p.is_dir()),
        key=lambda p: p.name.lower(),
    )

    for author_dir in author_dirs:
        author_id = author_dir.name
        by_label = _collect_label_images(author_dir, data_root)
        label_items = [(label, paths) for label, paths in by_label.items() if paths]

        for (label_a, paths_a), (label_b, paths_b) in itertools.combinations(label_items, 2):
            for path_a in paths_a:
                for path_b in paths_b:
                    prefer = _prefer_for_labels(label_a, label_b, path_a, path_b)
                    if prefer is None:
                        continue
                    # 保证 image_a/image_b 与 prefer 一致：必要时交换
                    if prefer == "b":
                        path_a, path_b = path_b, path_a
                        prefer = "a"
                    pairs.append(
                        _make_pair(path_a, path_b, author_id, prefer, created_at)
                    )

        # keep-keep：同标签内两两 tie
        keep_paths = by_label["keep"]
        for i, path_a in enumerate(keep_paths):
            for path_b in keep_paths[i + 1 :]:
                pairs.append(_make_pair(path_a, path_b, author_id, "tie", created_at))

    return pairs


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="从 good/keep/trash 生成种子 pairs.json")
    parser.add_argument("--data-root", type=Path, required=True, help="图片根目录")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pairs_seed.json"),
        help="输出 pairs.json 路径",
    )
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data_root 不存在或不是目录: {data_root}", file=sys.stderr)
        return 2

    pairs = generate_seed_pairs(data_root)
    doc = empty_pairs_doc()
    doc["version"] = PAIRS_VERSION
    doc["pairs"] = pairs
    save_json(args.out, doc)
    print(f"已写入 {args.out}: {len(pairs)} 条种子 pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
