"""tags.dataset 单测：multi-hot 编码、无效图跳过与喜好组分层划分。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tags.dataset import TagsDataset, stratified_split_tags
from tests.conftest import make_rgb_jpeg


def _write_labels(path: Path, images: list[dict], data_root: str = "") -> Path:
    """写最小合法 labels.json（schema 与 tools/scan_tagspaces.py 一致）。"""
    doc = {"version": 1, "data_root": data_root, "vocab": {}, "stats": {}, "images": images}
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def test_missing_data_root_raises(tmp_path: Path) -> None:
    """[tags/B-2] 未传 data_root 且 labels.json 内嵌值为空时显式抛 ValueError。"""
    labels = _write_labels(
        tmp_path / "labels.json",
        [{"path": "a/1.jpg", "tags": ["灵魂"]}],
        data_root="",
    )
    with pytest.raises(ValueError, match="data_root"):
        TagsDataset(labels)


def test_multihot_encoding_and_vocab_filter(tmp_path: Path) -> None:
    """V0 顺序 multi-hot；词表外标签（大水印/男角色/未知）不进入目标与计数。

    2026-09-21 官方图并入 V0 词表（正样本 45 ≥ 30），目标向量 9 → 10 维。
    """
    labels = _write_labels(
        tmp_path / "labels.json",
        [
            {"path": "a/1.jpg", "tags": ["灵魂", "NSFW", "官方图"]},
            {"path": "a/2.jpg", "tags": ["删除", "小水印", "大水印", "未知标签"]},
        ],
        data_root=str(tmp_path),  # [tags/B-2] 显式传根，本用例不触盘
    )
    ds = TagsDataset(labels)
    assert len(ds) == 2
    # V0 顺序: 灵魂 喜欢 一般 删除 无背景 漫画图 NSFW 小水印 低像素 官方图
    assert torch.equal(ds.targets[0], torch.tensor([1, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype=torch.float32))
    assert torch.equal(ds.targets[1], torch.tensor([0, 0, 0, 1, 0, 0, 0, 1, 0, 0], dtype=torch.float32))
    assert ds.tag_counts["灵魂"] == 1
    assert ds.tag_counts["官方图"] == 1  # 已入 V0 词表，正常计数
    assert "大水印" not in ds.tag_counts  # 词表外不计数


def test_skip_missing_and_conflicting_preference(tmp_path: Path) -> None:
    """喜好缺失（含无标签）与喜好冲突的图跳过并计数。"""
    labels = _write_labels(
        tmp_path / "labels.json",
        [
            {"path": "a/ok.jpg", "tags": ["一般", "无背景"]},
            {"path": "a/no_pref.jpg", "tags": ["无背景", "NSFW"]},
            {"path": "a/unlabeled.jpg", "tags": []},
            {"path": "a/conflict.jpg", "tags": ["灵魂", "删除", "NSFW"]},
        ],
        data_root=str(tmp_path),  # [tags/B-2] 显式传根，本用例不触盘
    )
    ds = TagsDataset(labels)
    assert len(ds) == 1
    assert ds.skip_stats == {"no_preference": 2, "conflict_preference": 1}
    assert ds.paths == ["a/ok.jpg"]


def test_getitem_returns_preprocessed_image_and_target(tmp_path: Path) -> None:
    """__getitem__ 返回白底预处理图与目标；构造参数 data_root 覆盖内嵌值。"""
    root = tmp_path / "data"
    make_rgb_jpeg(root / "a" / "1.jpg", size=(40, 24))
    # 内嵌 data_root 指向不存在的路径，验证参数优先
    labels = _write_labels(
        tmp_path / "labels.json",
        [{"path": "a/1.jpg", "tags": ["喜欢"]}],
        data_root="F:/nonexistent",
    )
    ds = TagsDataset(labels, root, img_size=64)
    item = ds[0]
    assert item["path"] == "a/1.jpg"
    assert item["img"].shape == (3, 64, 64)  # 长方形已 pad 成正方形
    assert torch.equal(item["target"], ds.targets[0])
    assert item["target"][1] == 1.0  # 喜欢


def test_getitem_corrupt_image_returns_white_placeholder(tmp_path: Path) -> None:
    """坏图静默替换为全白占位（不抛异常）。"""
    root = tmp_path / "data"
    root.mkdir()
    (root / "bad.jpg").write_bytes(b"not an image")
    labels = _write_labels(
        tmp_path / "labels.json", [{"path": "bad.jpg", "tags": ["一般"]}], data_root=str(root)
    )
    ds = TagsDataset(labels, img_size=32)
    item = ds[0]
    assert item["img"].shape == (3, 32, 32)
    assert torch.allclose(item["img"], torch.ones_like(item["img"]))


def test_stratified_split_group_representation_and_reproducibility(tmp_path: Path) -> None:
    """每个喜好组在 val 有代表、索引守恒不重不漏、同 seed 可复现。"""
    images = []
    for i in range(12):
        images.append({"path": f"g/normal_{i}.jpg", "tags": ["一般"]})
    for i in range(8):
        images.append({"path": f"g/like_{i}.jpg", "tags": ["喜欢"]})
    for i in range(3):
        images.append({"path": f"g/soul_{i}.jpg", "tags": ["灵魂"]})
    for i in range(5):
        images.append({"path": f"g/del_{i}.jpg", "tags": ["删除"]})
    ds = TagsDataset(_write_labels(tmp_path / "labels.json", images, data_root=str(tmp_path)))

    train_idx, val_idx = stratified_split_tags(ds, val_ratio=0.25, seed=42)
    assert sorted(train_idx + val_idx) == list(range(len(ds)))  # 守恒且不重不漏
    assert not set(train_idx) & set(val_idx)
    # 每个组（大小 >=2）在 val 至少 1 张（稀有档 灵魂 也有代表）
    for pref in ("一般", "喜欢", "灵魂", "删除"):
        assert any(ds.entries[i]["pref"] == pref for i in val_idx)
        assert any(ds.entries[i]["pref"] == pref for i in train_idx)
    assert len(val_idx) == 7  # round(12*.25)+round(8*.25)+round(3*.25)+round(5*.25)=3+2+1+1
    # 同 seed 完全复现
    again = stratified_split_tags(ds, val_ratio=0.25, seed=42)
    assert again == (train_idx, val_idx)


def test_stratified_split_single_member_group_stays_in_train(tmp_path: Path) -> None:
    """单样本组不拆分，留在训练集（避免稀有组唯一样本进 val）。"""
    images = [{"path": f"m/{i}.jpg", "tags": ["一般"]} for i in range(10)]
    images.append({"path": "m/solo.jpg", "tags": ["灵魂"]})
    ds = TagsDataset(_write_labels(tmp_path / "labels.json", images, data_root=str(tmp_path)))
    train_idx, val_idx = stratified_split_tags(ds, val_ratio=0.5, seed=7)
    solo = next(i for i, entry in enumerate(ds.entries) if entry["pref"] == "灵魂")
    assert solo in train_idx
    assert solo not in val_idx
