"""端到端冒烟测试：小数据集全流程（步骤 8）。"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from rank.checkpoint import load_checkpoint
from rank.groups import GroupMap
from rank_train import run_training
from rank_predict import run_predict
from tests.conftest import make_rgb_jpeg, touch_images
from tools import empty_pairs_doc, load_json, save_json, utc_now_iso
from tools.build_groups import build_train_group_map
from tools.export_active_queue import generate_active_queue
from tools.report_results import load_scores, write_report
from tools.seed_pairs import generate_seed_pairs
from tools.validate_pairs import validate_pairs_doc


def _setup_data_root(data_root: Path) -> None:
    """创建两位作者、各 6 张图的最小数据集。"""
    for author in ("author_a", "author_b"):
        author_dir = data_root / author
        author_dir.mkdir(parents=True)
        touch_images(author_dir, "img", 6)


def _write_test_config(config_path: Path, data_root: Path) -> dict:
    """写入加速用测试配置。"""
    config = {
        "data_root": str(data_root),
        "img_size": 64,
        "batch_size": 4,
        "embed_dim": 16,
        "lr": 5.0e-4,
        "lr_finetune": 1.0e-4,
        "weight_decay": 0.01,
        "epochs": 1,
        "early_stop_patience": 100,
        "val_ratio": 0.15,
        "group": {"min_images": 5, "max_per_group": 80},
        "condition": "train_group",
        "seed_pair_weight": 0.5,
        "uncertainty": {"n_mc": 3, "review_percentile": 75},
        "active_learning": {"budget": 10, "score_close_threshold": 5.0},
        "finetune": {"freeze_backbone_epochs": 0, "train_on_new_only": False},
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config


def test_e2e_smoke_pipeline(tmp_path: Path) -> None:
    """冷启动 → 训练 → 预测 → 报告 → 队列 → 增量训练 全流程。"""
    data_root = tmp_path / "data"
    _setup_data_root(data_root)

    group_map_path = tmp_path / "train_group_map.json"
    group_doc = build_train_group_map(data_root, min_images=5, max_per_group=80)
    save_json(group_map_path, group_doc)
    assert "images" in group_doc
    assert len(group_doc["images"]) == 12

    pairs_path = tmp_path / "pairs.json"
    seed_pairs = generate_seed_pairs(data_root)
    assert len(seed_pairs) > 0
    pairs_doc = empty_pairs_doc()
    pairs_doc["pairs"] = seed_pairs
    save_json(pairs_path, pairs_doc)

    errors, _warnings, stats = validate_pairs_doc(pairs_doc, data_root=data_root)
    assert not errors, errors
    assert stats.get("pairs_total", 0) > 0

    config_path = tmp_path / "rank_test.yaml"
    config = _write_test_config(config_path, data_root)
    checkpoint_path = tmp_path / "models" / "best.pth"

    assert (
        run_training(
            config,
            data_root=data_root,
            pairs_path=pairs_path,
            pairs_new_path=None,
            resume_path=None,
            epochs=1,
            lr=None,
            freeze_backbone_epochs=0,
            build_groups=True,
            checkpoint_path=checkpoint_path,
            pretrained=False,
        )
        == 0
    )
    assert checkpoint_path.is_file()

    _model, _group_map, ckpt_meta = load_checkpoint(checkpoint_path, "cpu", pretrained=False)
    assert ckpt_meta.get("u_threshold") is not None
    assert ckpt_meta.get("percentiles")

    scores_path = tmp_path / "scores.json"
    assert (
        run_predict(
            checkpoint_path=checkpoint_path,
            data_root=data_root,
            out_path=scores_path,
            config=config,
            n_mc=3,
            batch_size=4,
            pretrained=False,
        )
        == 0
    )

    scores, meta = load_scores(scores_path)
    assert len(scores) == 12
    assert meta.get("u_threshold") == ckpt_meta["u_threshold"]

    report_dir = tmp_path / "report"
    summary = write_report(
        scores,
        report_dir,
        u_threshold=float(meta["u_threshold"]),
        data_root=data_root,
    )
    assert summary["u_threshold"] == meta["u_threshold"]
    assert (report_dir / "sorted.json").is_file()
    assert (report_dir / "index.html").is_file()

    queue_doc = generate_active_queue(scores, budget=10)
    assert queue_doc["version"] == 1
    assert len(queue_doc["pairs"]) <= 10

    pairs_new_path = tmp_path / "pairs_new.json"
    new_pair = {
        "image_a": "author_a/good/img000.jpg",
        "image_b": "author_b/good/img000.jpg",
        "author_a": "author_a",
        "author_b": "author_b",
        "prefer": "a",
        "source": "annotator",
        "created_at": utc_now_iso(),
    }
    save_json(pairs_new_path, {"version": 1, "pairs": [new_pair]})

    resume_checkpoint = tmp_path / "models" / "best_resume.pth"
    resume_checkpoint.write_bytes(checkpoint_path.read_bytes())

    assert (
        run_training(
            config,
            data_root=data_root,
            pairs_path=pairs_path,
            pairs_new_path=pairs_new_path,
            resume_path=resume_checkpoint,
            epochs=1,
            lr=None,
            freeze_backbone_epochs=0,
            build_groups=False,
            checkpoint_path=checkpoint_path,
            pretrained=False,
        )
        == 0
    )

    _model2, _gm2, ckpt_meta2 = load_checkpoint(checkpoint_path, "cpu", pretrained=False)
    assert ckpt_meta2.get("u_threshold") is not None
    assert torch.isfinite(torch.tensor(float(ckpt_meta2["u_threshold"])))
