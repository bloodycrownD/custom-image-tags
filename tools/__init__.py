"""共享工具层：JSON 读写、路径与图片枚举等无业务依赖的基础工具。

分层意图（CR 待拍板 #11）：本模块是 rank 与 tags 两条链共用的底层工具层，
保持零业务依赖——禁止 import tags / rank（tools/ 下的业务脚本可以反向依赖
本层与 tags/rank，但本层不依赖任何业务包，避免模块级环）。

现役工具概览：
- pairs 链（seed/merge/validate/active_queue 等）：已冻结留存，不再演进；
- train_group 构建：tools/build_groups.py；
- TagSpaces 扫描：tools/scan_tagspaces.py（labels.json 生成）；
- WD 权重验证：tools/verify_wd_load.py；
- 评估报告：tools/report_results.py（rank 冻结链）。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 支持的图片扩展名（rank 与 tags 两条链共用的图片枚举口径）
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# pairs.json 契约常量
PAIRS_VERSION = 1
VALID_PREFER = frozenset({"a", "b", "tie"})
VALID_SOURCES = frozenset({"seed", "annotator"})
SOURCE_PRIORITY = {"seed": 0, "annotator": 1}

# 三分类标签目录名
LABEL_DIRS = ("good", "keep", "trash")
LABEL_RANK = {"trash": 0, "keep": 1, "good": 2}


def utc_now_iso() -> str:
    """返回 UTC 时间的 ISO 8601 字符串（秒精度）。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_image_file(path: Path) -> bool:
    """判断路径是否为支持的图片文件。"""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_images_under(root: Path) -> list[Path]:
    """递归收集目录下所有图片文件（已排序）。"""
    if not root.is_dir():
        return []
    images = [p for p in root.rglob("*") if is_image_file(p)]
    images.sort(key=lambda p: p.as_posix().lower())
    return images


def count_author_images(author_dir: Path) -> int:
    """统计作者目录下的图片总数。"""
    return len(iter_images_under(author_dir))


def rel_posix_path(path: Path, root: Path) -> str:
    """将路径转为相对 data_root 的正斜杠字符串。"""
    return path.relative_to(root).as_posix()


def load_json(path: Path) -> Any:
    """读取 JSON 文件（兼容 UTF-8 BOM）。"""
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    """写入 JSON 文件（UTF-8、缩进 2）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def pair_identity(image_a: str, image_b: str) -> tuple[str, str]:
    """生成无序 pair 键，用于去重。"""
    if image_a <= image_b:
        return (image_a, image_b)
    return (image_b, image_a)


def empty_pairs_doc() -> dict[str, Any]:
    """构造空的 pairs.json 文档。"""
    return {"version": PAIRS_VERSION, "pairs": []}


def parse_scores_doc(data: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    解析 scores.json 文档。

    支持旧版纯列表格式与新版 ``{version, meta, scores}`` 格式。

    Returns:
        (scores 列表, meta 字典)。
    """
    if isinstance(data, list):
        return data, {}
    if isinstance(data, dict):
        scores = data.get("scores")
        if scores is None:
            raise ValueError("scores.json 缺少 scores 字段")
        if not isinstance(scores, list):
            raise ValueError("scores.json 中 scores 字段应为列表")
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError("scores.json 中 meta 字段应为对象")
        return scores, meta
    raise ValueError("scores.json 格式无效")
