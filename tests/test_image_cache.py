"""ImageTensorStore 与 DataLoader 配置测试。"""

from __future__ import annotations

from pathlib import Path

import torch

from rank.dataloader_utils import build_pairs_dataloader, resolve_num_workers
from rank.dataset import PairsDataset, pairs_collate_fn
from rank.groups import GroupMap
from rank.image_cache import ImageTensorStore, apply_tensor_flip_augment, collect_unique_paths_from_pairs
from rank.transforms import build_deterministic_transform
from tests.conftest import make_rgb_jpeg
from tools import empty_pairs_doc, save_json


def test_image_tensor_store_build_and_get(tmp_path: Path) -> None:
    """预热缓存后应能按路径取张量。"""
    data_root = tmp_path / "data"
    rel = "author_a/good/img.jpg"
    make_rgb_jpeg(data_root / rel)

    store = ImageTensorStore()
    transform = build_deterministic_transform(64)
    count = store.build([rel], data_root, transform, show_progress=False)
    assert count == 1
    tensor = store.get(rel)
    assert tensor.shape == (3, 64, 64)


def test_tensor_flip_augment_changes_tensor() -> None:
    """随机翻转应可能改变张量（多次采样至少一次不同）。"""
    base = torch.zeros(3, 8, 8)
    base[:, :, :4] = 1.0
    variants = {torch.equal(base, apply_tensor_flip_augment(base.clone(), p=1.0))}
    flipped = apply_tensor_flip_augment(base.clone(), p=1.0)
    assert not torch.equal(base, flipped)


def test_pairs_dataset_uses_tensor_store(tmp_path: Path) -> None:
    """PairsDataset 命中 tensor_store 时不应重复读盘。"""
    data_root = tmp_path / "data"
    rel_a = "author_a/good/a.jpg"
    rel_b = "author_b/good/b.jpg"
    make_rgb_jpeg(data_root / rel_a)
    make_rgb_jpeg(data_root / rel_b)

    pairs_doc = empty_pairs_doc()
    pairs_doc["pairs"] = [
        {
            "image_a": rel_a,
            "image_b": rel_b,
            "author_a": "author_a",
            "author_b": "author_b",
            "prefer": "a",
            "source": "annotator",
        }
    ]
    pairs_path = tmp_path / "pairs.json"
    save_json(pairs_path, pairs_doc)

    store = ImageTensorStore()
    transform = build_deterministic_transform(64)
    paths = collect_unique_paths_from_pairs(pairs_doc["pairs"])
    store.build(paths, data_root, transform, show_progress=False)

    group_map = GroupMap({"author_a": 1, "author_b": 2}, {"author_a": 1, "author_b": 2})
    dataset = PairsDataset(
        pairs_path,
        data_root,
        group_map,
        img_size=64,
        augment=False,
        tensor_store=store,
    )
    sample = dataset[0]
    assert sample["img_a"].shape == (3, 64, 64)
    assert sample["img_b"].shape == (3, 64, 64)


def test_resolve_num_workers_auto() -> None:
    """负数 worker 应解析为合理正整数。"""
    assert resolve_num_workers(-1) >= 1
    assert resolve_num_workers(0) == 0


def test_build_pairs_dataloader_num_workers_zero(tmp_path: Path) -> None:
    """num_workers=0 时应能构建可迭代的 DataLoader。"""
    data_root = tmp_path / "data"
    rel_a = "author_a/good/a.jpg"
    rel_b = "author_b/good/b.jpg"
    make_rgb_jpeg(data_root / rel_a)
    make_rgb_jpeg(data_root / rel_b)

    pairs_doc = empty_pairs_doc()
    pairs_doc["pairs"] = [
        {
            "image_a": rel_a,
            "image_b": rel_b,
            "author_a": "author_a",
            "author_b": "author_b",
            "prefer": "a",
            "source": "annotator",
        }
    ]
    pairs_path = tmp_path / "pairs.json"
    save_json(pairs_path, pairs_doc)

    group_map = GroupMap({"author_a": 1, "author_b": 2}, {"author_a": 1, "author_b": 2})
    dataset = PairsDataset(pairs_path, data_root, group_map, img_size=64, augment=False)
    loader = build_pairs_dataloader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=pairs_collate_fn,
        dl_cfg={"num_workers": 0, "pin_memory": False},
    )
    batch = next(iter(loader))
    assert batch["img_a"].shape[0] == 1
