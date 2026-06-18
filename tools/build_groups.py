"""扫描 data_root，按作者图片数生成 train_group_map.json。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools import iter_images_under, rel_posix_path, save_json


def build_train_group_map(
    data_root: Path,
    min_images: int = 5,
    max_per_group: int = 80,
) -> dict:
    """
    根据 spec 规则构建 train_group 映射表。

    - 图片数 < min_images：并入 ``__rare__``
    - min_images ≤ 数 ≤ max_per_group：单组，组名为作者目录名
    - 数 > max_per_group：按文件名排序均分为 ``{author}__part{k}``
    """
    if min_images < 1:
        raise ValueError("min_images 必须 >= 1")
    if max_per_group < min_images:
        raise ValueError("max_per_group 必须 >= min_images")

    author_dirs = sorted(
        (p for p in data_root.iterdir() if p.is_dir()),
        key=lambda p: p.name.lower(),
    )

    groups: dict[str, int] = {}
    authors: dict[str, int] = {}
    next_idx = 1  # 0 保留给模型 unknown

    def ensure_group(name: str) -> int:
        nonlocal next_idx
        if name not in groups:
            groups[name] = next_idx
            next_idx += 1
        return groups[name]

    # 先确保 __rare__ 存在（若有冷门作者会用到）
    rare_authors: list[tuple[str, Path]] = []
    normal_authors: list[tuple[str, Path, list[Path]]] = []

    for author_dir in author_dirs:
        images = iter_images_under(author_dir)
        if not images:
            continue
        author_id = author_dir.name
        count = len(images)
        if count < min_images:
            rare_authors.append((author_id, author_dir))
        else:
            normal_authors.append((author_id, author_dir, images))

    if rare_authors:
        rare_idx = ensure_group("__rare__")
        for author_id, _ in rare_authors:
            authors[author_id] = rare_idx

    for author_id, author_dir, images in normal_authors:
        count = len(images)
        if count <= max_per_group:
            group_name = author_id
            idx = ensure_group(group_name)
            authors[author_id] = idx
        else:
            # 按文件名排序均分
            part_count = (count + max_per_group - 1) // max_per_group
            chunk_size = (count + part_count - 1) // part_count
            first_idx = None
            for part_no in range(part_count):
                start = part_no * chunk_size
                end = min(start + chunk_size, count)
                if start >= count:
                    break
                group_name = f"{author_id}__part{part_no + 1}"
                idx = ensure_group(group_name)
                if first_idx is None:
                    first_idx = idx
            authors[author_id] = first_idx if first_idx is not None else ensure_group(author_id)

    return {
        "version": 1,
        "params": {
            "min_images": min_images,
            "max_per_group": max_per_group,
            "data_root": str(data_root.resolve()),
        },
        "groups": groups,
        "authors": authors,
    }


def resolve_image_group(
    image_rel: str,
    group_map: dict,
    data_root: Path | None = None,
) -> str | None:
    """
    根据 train_group_map 与图片相对路径解析 train_group 名称。

    对拆分作者，按与 build 相同的排序与分块规则定位 part。
    """
    groups = group_map.get("groups", {})
    authors = group_map.get("authors", {})
    params = group_map.get("params", {})
    max_per_group = int(params.get("max_per_group", 80))

    parts = Path(image_rel).parts
    if not parts:
        return None
    author_id = parts[0]
    if author_id not in authors:
        return None

    author_idx = authors[author_id]
    author_group_names = [name for name, idx in groups.items() if idx == author_idx]
    if not author_group_names:
        return None
    base_group = author_group_names[0]
    if "__part" not in base_group:
        return base_group

    if data_root is None:
        data_root_str = params.get("data_root")
        if not data_root_str:
            return base_group
        data_root = Path(data_root_str)

    author_dir = data_root / author_id
    images = iter_images_under(author_dir)
    rel_paths = [rel_posix_path(p, data_root) for p in images]
    if image_rel not in rel_paths:
        return base_group

    count = len(rel_paths)
    part_count = (count + max_per_group - 1) // max_per_group
    chunk_size = (count + part_count - 1) // part_count
    pos = rel_paths.index(image_rel)
    part_no = pos // chunk_size + 1
    part_name = f"{author_id}__part{part_no}"
    return part_name if part_name in groups else base_group


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="扫描 data_root 生成 train_group_map.json")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="图片根目录（作者/标签/图片）",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("train_group_map.json"),
        help="输出 JSON 路径（默认 train_group_map.json）",
    )
    parser.add_argument("--min-images", type=int, default=5, help="最少图片数阈值（默认 5）")
    parser.add_argument("--max-per-group", type=int, default=80, help="每组最多图片数（默认 80）")
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data_root 不存在或不是目录: {data_root}", file=sys.stderr)
        return 2

    try:
        group_map = build_train_group_map(
            data_root,
            min_images=args.min_images,
            max_per_group=args.max_per_group,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    save_json(args.out, group_map)
    print(
        f"已写入 {args.out}: "
        f"{len(group_map['groups'])} 个 train_group, "
        f"{len(group_map['authors'])} 位作者"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
