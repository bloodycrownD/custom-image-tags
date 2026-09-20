"""tools/scan_tagspaces 单测：文件名标签解析、冲突与未知标签统计、--all 行为。"""

from __future__ import annotations

from pathlib import Path

from tags.vocab import NEGATIVE_TAGS, PREFERENCE_TAGS, V0_TRAIN_TAGS
from tools import IMAGE_EXTENSIONS
from tools.scan_tagspaces import IMAGE_EXTS, scan


def _touch(root: Path, rel: str) -> Path:
    """在 data_root 下创建占位文件（scan 只解析文件名，不读图片内容）。"""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


def _build_root(tmp_path: Path) -> Path:
    """构造含多标签/冲突/未知/未标注样本的最小图库。"""
    root = tmp_path / "library"
    root.mkdir()
    _touch(root, "作者A/图1 [无背景 一般].jpg")  # 空格分隔多标签
    _touch(root, "作者A/图2[喜欢, 漫画图].png")  # 逗号分隔多标签
    _touch(root, "作者B/图3 [nsfw].webp")  # 小写 nsfw 规范化为 NSFW
    _touch(root, "作者B/图4 [灵魂 删除].bmp")  # 双喜好 → conflicts（仍收录）
    _touch(root, "作者C/图5 [拼错标签x].jpg")  # 未知尾标签，无词表标签 → 默认不收录
    _touch(root, "作者C/plain.jpg")  # 未标注 → 默认不收录
    _touch(root, "作者C/notes.txt")  # 非图片扩展名不参与
    return root


def test_scan_parses_space_and_comma_multilabel(tmp_path: Path) -> None:
    """空格与逗号分隔的多标签均解析入 images；未标注图默认不收录。"""
    doc = scan(_build_root(tmp_path))
    by_path = {img["path"]: img["tags"] for img in doc["images"]}
    assert by_path["作者A/图1 [无背景 一般].jpg"] == ["无背景", "一般"]
    assert by_path["作者A/图2[喜欢, 漫画图].png"] == ["喜欢", "漫画图"]
    # 小写 ASCII 标签规范化（nsfw → NSFW）
    assert by_path["作者B/图3 [nsfw].webp"] == ["NSFW"]
    assert "作者C/plain.jpg" not in by_path


def test_scan_all_includes_unlabeled(tmp_path: Path) -> None:
    """--all 连未标注图（tags 为空列表）一起收录。"""
    doc = scan(_build_root(tmp_path), include_unlabeled=True)
    by_path = {img["path"]: img["tags"] for img in doc["images"]}
    assert by_path["作者C/plain.jpg"] == []
    assert by_path["作者C/图5 [拼错标签x].jpg"] == []


def test_scan_conflicts_and_unknown_trailing_tags(tmp_path: Path) -> None:
    """双喜好图计入 conflicts 但仍收录；未知尾标签单独上报。"""
    doc = scan(_build_root(tmp_path))
    stats = doc["stats"]
    assert stats["preference_conflicts"] == ["作者B/图4 [灵魂 删除].bmp"]
    paths = {img["path"] for img in doc["images"]}
    assert "作者B/图4 [灵魂 删除].bmp" in paths
    assert stats["unknown_trailing_tags"] == {"拼错标签x": 1}


def test_scan_stats_and_vocab_fields(tmp_path: Path) -> None:
    """stats 计数与 vocab 字段与词表逐项一致。"""
    doc = scan(_build_root(tmp_path))
    assert doc["version"] == 1
    assert doc["data_root"].endswith("library")

    stats = doc["stats"]
    # txt 不计：jpg/png/webp/bmp 共 6 个图片文件
    assert stats["total_images"] == 6
    # 默认收录的 4 张均有词表标签（图5 未知标签不算标注）
    assert stats["labeled_images"] == 4
    assert stats["preference_distribution"] == {"灵魂": 1, "喜欢": 1, "一般": 1, "删除": 1}
    # 收录图中仅图3（nsfw）缺喜好
    assert stats["preference_missing"] == 1
    assert stats["tag_counts"]["无背景"] == 1
    assert stats["tag_counts"]["NSFW"] == 1
    assert stats["tag_counts"]["漫画图"] == 1

    assert doc["vocab"]["preference"] == list(PREFERENCE_TAGS)
    assert doc["vocab"]["negative"] == list(NEGATIVE_TAGS)
    assert doc["vocab"]["v0_train"] == list(V0_TRAIN_TAGS)


def test_image_exts_superset_of_shared_constant() -> None:
    """[tools/C-5] 推理池扩展名 = 共享常量 + .gif；.gif 不反向污染共享常量。"""
    assert IMAGE_EXTS == IMAGE_EXTENSIONS | {".gif"}
    assert ".gif" not in IMAGE_EXTENSIONS
