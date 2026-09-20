"""TagSpaces 文件名标签解析单测。"""

from __future__ import annotations

from tags.filename_tags import parse_filename, parse_stem_tags
from tags.vocab import V0_TRAIN_TAGS, canonical_tag


def test_basic_multiple_groups():
    stem, cands = parse_stem_tags("图 [灵魂] [NSFW]")
    assert stem == "图"
    assert cands == ["灵魂", "NSFW"]


def test_comma_separated_inside_group():
    stem, cands = parse_stem_tags("x [喜欢, 漫画图]")
    assert stem == "x"
    assert cands == ["喜欢", "漫画图"]


def test_space_separated_inside_group():
    """TagSpaces 默认写盘格式：单方括号组内空格分隔多标签。"""
    stem, cands = parse_stem_tags("2398_126599139_p0[无背景 一般]")
    assert stem == "2398_126599139_p0"
    assert cands == ["无背景", "一般"]


def test_space_separated_three_tags_real_world_shape():
    r = parse_filename("x_118880212_p0[无背景 一般 小水印].jpg")
    assert r["tags"] == ["无背景", "一般", "小水印"]
    assert r["unknown_tags"] == []
    assert r["preference_count"] == 1


def test_mixed_separators_inside_group():
    _, cands = parse_stem_tags("x [低像素 一般, NSFW]")
    assert cands == ["低像素", "一般", "NSFW"]


def test_fullwidth_comma():
    _, cands = parse_stem_tags("x [小水印，低像素]")
    assert cands == ["小水印", "低像素"]


def test_original_brackets_not_trailing_are_kept():
    stem, cands = parse_stem_tags("[作者] 标题 (C98) [喜欢]")
    assert stem == "[作者] 标题 (C98)"
    assert cands == ["喜欢"]


def test_no_tags():
    stem, cands = parse_stem_tags("[作者] 标题 (C98)")
    assert stem == "[作者] 标题 (C98)"
    assert cands == []


def test_unknown_trailing_bracket_reported():
    r = parse_filename("图 [随便什么].png")
    assert r["tags"] == []
    assert r["unknown_tags"] == ["随便什么"]
    assert r["stem"] == "图"


def test_case_insensitive_ascii():
    r = parse_filename("图 [nsfw] [漫画图].png")
    assert r["tags"] == ["NSFW", "漫画图"]


def test_hierarchical_tag_last_segment():
    assert canonical_tag("负面/水印") is None  # 词表无"水印"，末段不匹配 → None
    assert canonical_tag("喜好/灵魂") == "灵魂"


def test_preference_count_and_conflict():
    r = parse_filename("图 [删除] [喜欢].png")
    assert r["preference_count"] == 2
    r2 = parse_filename("图 [删除].png")
    assert r2["preference_count"] == 1


def test_v0_vocab_subset():
    assert set(V0_TRAIN_TAGS) <= set(
        ("灵魂", "喜欢", "一般", "删除", "官方图", "无背景", "大水印", "低像素", "小水印", "NSFW", "漫画图", "男角色")
    )
