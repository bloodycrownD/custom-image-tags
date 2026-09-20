"""扫描 TagSpaces 文件名标签，生成训练用 labels.json（对图库只读）。

用法：
    python tools/scan_tagspaces.py <data_root> [-o labels.json] [--all]

默认仅收录至少含一个词表标签的图片；--all 连未标注图一起收录（供推理池）。
产物结构：
    {"version": 1, "vocab": {...}, "stats": {...}, "images": [{"path", "tags"}]}
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tags.filename_tags import parse_filename  # noqa: E402
from tags.vocab import (  # noqa: E402
    ALL_TAGS,
    NEGATIVE_TAGS,
    PREFERENCE_SET,
    PREFERENCE_TAGS,
    V0_TRAIN_TAGS,
)
from tools import IMAGE_EXTENSIONS, save_json  # noqa: E402

# [tools/C-5] 复用共享扩展名常量消除双轨；推理池需收录动图，rank 链不收
IMAGE_EXTS = IMAGE_EXTENSIONS | {".gif"}
LABELS_VERSION = 1


def scan(data_root: Path, include_unlabeled: bool = False) -> dict:
    images: list[dict] = []
    tag_counts: collections.Counter[str] = collections.Counter()
    unknown_counter: collections.Counter[str] = collections.Counter()
    conflicts: list[str] = []
    total_images = 0

    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        total_images += 1
        rel = path.relative_to(data_root).as_posix()
        parsed = parse_filename(path)
        tags = parsed["tags"]
        unknown_counter.update(parsed["unknown_tags"])
        if parsed["preference_count"] > 1:
            conflicts.append(rel)
        if not tags and not include_unlabeled:
            continue
        tag_counts.update(tags)
        images.append({"path": rel, "tags": tags})

    pref_counts = {t: tag_counts.get(t, 0) for t in PREFERENCE_TAGS}
    stats = {
        "total_images": total_images,
        "labeled_images": sum(1 for img in images if img["tags"]),
        "preference_distribution": pref_counts,
        "preference_missing": sum(1 for img in images if not img["tags"] or not (PREFERENCE_SET & set(img["tags"]))),
        "preference_conflicts": conflicts,
        "tag_counts": {t: tag_counts.get(t, 0) for t in ALL_TAGS},
        "unknown_trailing_tags": dict(unknown_counter),
    }
    return {
        "version": LABELS_VERSION,
        "data_root": str(data_root),
        "vocab": {
            "preference": list(PREFERENCE_TAGS),
            "negative": list(NEGATIVE_TAGS),
            "v0_train": list(V0_TRAIN_TAGS),
        },
        "stats": stats,
        "images": images,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扫描 TagSpaces 文件名标签生成 labels.json")
    parser.add_argument("data_root", type=Path, help="图库根目录（只读）")
    parser.add_argument("-o", "--output", type=Path, default=Path("data/classification/tags.labels.json"))
    parser.add_argument("--all", action="store_true", help="连同未标注图收录")
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"数据目录不存在: {data_root}")
        return 2

    doc = scan(data_root, include_unlabeled=args.all)

    stats = doc["stats"]
    print(f"扫描图片总数: {stats['total_images']}")
    print(f"已标注图片: {stats['labeled_images']}")
    print("喜好程度分布:", stats["preference_distribution"])
    print("各标签计数:", stats["tag_counts"])
    if stats["preference_conflicts"]:
        print(f"[警告] 喜好程度互斥冲突 {len(stats['preference_conflicts'])} 张:")
        for rel in stats["preference_conflicts"][:10]:
            print(f"  - {rel}")
    if stats["unknown_trailing_tags"]:
        print("[警告] 未知结尾方括号（可能是拼错标签）:", stats["unknown_trailing_tags"])

    # [tools/C-5] 统一走共享 save_json 落盘（UTF-8、缩进 2、尾换行），不再手写序列化
    save_json(args.output, doc)
    print(f"已写入 {args.output}（{len(doc['images'])} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
