"""tools JSON 工具单测：parse_scores_doc 分支、save/load round-trip 与 BOM 兼容。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import load_json, parse_scores_doc, save_json


def test_parse_scores_doc_legacy_list_format() -> None:
    """旧版纯列表格式 → (scores, 空 meta)。"""
    scores = [{"image": "a.jpg", "score": 1.0}]
    assert parse_scores_doc(scores) == (scores, {})


def test_parse_scores_doc_dict_format() -> None:
    """新版 dict 格式 → (scores, meta)；meta 缺省为空 dict。"""
    scores = [{"image": "a.jpg", "score": 2.0}]
    doc = {"version": 1, "scores": scores, "meta": {"k": "v"}}
    assert parse_scores_doc(doc) == (scores, {"k": "v"})
    assert parse_scores_doc({"scores": scores}) == (scores, {})


def test_parse_scores_doc_missing_scores_raises() -> None:
    """dict 缺 scores 字段 → ValueError。"""
    with pytest.raises(ValueError, match="缺少 scores"):
        parse_scores_doc({"version": 1})


def test_parse_scores_doc_wrong_types_raise() -> None:
    """scores 非列表 / meta 非对象 / 顶层类型不识别三分支均报错。"""
    with pytest.raises(ValueError, match="应为列表"):
        parse_scores_doc({"scores": {"a": 1}})
    with pytest.raises(ValueError, match="应为对象"):
        parse_scores_doc({"scores": [], "meta": [1]})
    with pytest.raises(ValueError, match="格式无效"):
        parse_scores_doc("not a doc")


def test_save_load_json_roundtrip(tmp_path: Path) -> None:
    """save_json → load_json 往返：中文、嵌套结构、父目录自动创建、尾换行。"""
    data = {"中文键": [1, 2.5, {"嵌": None}], "flag": True}
    path = tmp_path / "sub" / "doc.json"  # 父目录不存在，save_json 应自动创建
    save_json(path, data)
    assert path.read_text(encoding="utf-8").endswith("\n")  # 尾换行约定
    assert load_json(path) == data


def test_load_json_tolerates_bom(tmp_path: Path) -> None:
    """带 UTF-8 BOM 的 JSON 文件可正常读取。"""
    path = tmp_path / "bom.json"
    body = json.dumps({"k": "中文"}, ensure_ascii=False).encode("utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + body)
    assert load_json(path) == {"k": "中文"}
