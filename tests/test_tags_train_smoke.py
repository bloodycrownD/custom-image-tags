"""tags v0 训练端到端冒烟：resnet18 小骨干 + 真图假 labels，函数级调用入口。

不依赖真实 EVA02-L 权重与真实图库：小 img_size、CPU、秒级完成，
验证 dataset → 特征预计算 → 头训练 → checkpoint 与报告产出全链路。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from tags.features import FEATURES_CACHE_VERSION, count_bad_images
from tags.train import (
    average_precision,
    compute_pos_weight,
    compute_tag_metrics,
    load_v0_checkpoint,
)
from tags.vocab import V0_TRAIN_TAGS
from tags_train import main, run_tags_training
from tests.conftest import make_rgb_jpeg
from tools import load_json

ARCH = "resnet18"  # 仅测试用小骨干；生产为 eva02_large_patch14_448
IMG_SIZE = 64


def _build_dataset(data_root: Path) -> list[dict]:
    """构造 4 个喜好组共 24 张真 JPEG + 1 坏图 + 2 张待跳过图，返回 images 列表。"""
    images: list[dict] = []
    for pref, count in [("灵魂", 3), ("喜欢", 5), ("一般", 10), ("删除", 6)]:
        for i in range(count):
            rel = f"author/{pref}_{i}.jpg"
            tags = [pref]
            if i % 2 == 0:
                tags.append("无背景")
            if i % 3 == 0:
                tags.append("NSFW")
            make_rgb_jpeg(data_root / rel, color=(i * 8 % 256, 100, 150))
            images.append({"path": rel, "tags": tags})
    # 坏图：喜好正常 → 入集，特征阶段以全白占位提取
    (data_root / "author" / "bad.jpg").write_bytes(b"not an image")
    images.append({"path": "author/bad.jpg", "tags": ["一般", "小水印"]})
    # 喜好缺失 → 跳过
    make_rgb_jpeg(data_root / "author" / "nopref.jpg")
    images.append({"path": "author/nopref.jpg", "tags": ["无背景"]})
    # 喜好冲突 → 跳过
    make_rgb_jpeg(data_root / "author" / "conflict.jpg")
    images.append({"path": "author/conflict.jpg", "tags": ["灵魂", "删除"]})
    return images


def _write_labels(path: Path, images: list[dict], data_root: Path) -> Path:
    doc = {"version": 1, "data_root": str(data_root), "vocab": {}, "stats": {}, "images": images}
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path


def _make_config() -> dict:
    """最小训练配置（覆盖式 CLI 的 config 部分）。"""
    return {
        "arch": ARCH,
        "img_size": IMG_SIZE,
        "batch_size": 8,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-2,
        "epochs": 3,
        "early_stop_patience": 100,  # 冒烟不触发早停
        "val_ratio": 0.25,
        "seed": 0,
        "dropout": 0.1,
        "device": "cpu",
        "dataloader": {"num_workers": 0},
    }


def _run(
    tmp_path: Path, *, feature_cache: Path, epochs: int, img_size: int = IMG_SIZE
) -> tuple[int, Path, Path]:
    """搭建数据并跑一次完整入口，返回 (返回码, checkpoint, report)。"""
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)  # 缓存复用测试会第二次进入同一目录
    images = _build_dataset(data_root)
    labels_path = _write_labels(tmp_path / "labels.json", images, data_root)
    checkpoint_path = tmp_path / "models" / "tags_v0_best.pth"
    report_path = tmp_path / "tags_v0_report.json"
    rc = run_tags_training(
        _make_config(),
        labels_path=labels_path,
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        arch=ARCH,
        img_size=img_size,
        epochs=epochs,
        feature_cache=feature_cache,
        backbone_weights=None,  # 无真实 WD 权重，随机骨干即可验证流程
        device="cpu",
    )
    return rc, checkpoint_path, report_path


def test_v0_pipeline_smoke(tmp_path: Path) -> None:
    """全流程：入口返回 0，产出 checkpoint / 报告，且可重建模型。"""
    rc, checkpoint_path, report_path = _run(tmp_path, feature_cache=tmp_path / "features.pt", epochs=3)
    assert rc == 0
    assert checkpoint_path.is_file()
    assert report_path.is_file()

    # checkpoint 重建：resnet18 骨干 + 头权重，前向输出 9 维 logits
    model, payload = load_v0_checkpoint(checkpoint_path, "cpu")
    assert payload["format"] == "tags_v0"
    assert payload["arch"] == ARCH
    assert payload["img_size"] == IMG_SIZE
    assert payload["tag_list"] == list(V0_TRAIN_TAGS)
    assert "per_tag_metrics" in payload and "pos_weight" in payload
    assert payload["backbone_weights"] is None
    with torch.no_grad():
        logits = model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
    assert logits.shape == (2, len(V0_TRAIN_TAGS))

    # 报告内容：25 张入集（24 正常 + 1 坏图），2 张跳过
    report = load_json(report_path)
    assert report["num_images"] == 25
    assert report["skip_stats"] == {"no_preference": 1, "conflict_preference": 1}
    assert report["train_size"] + report["val_size"] == 25
    assert report["train_size"] > 0 and report["val_size"] > 0
    assert set(report["val_metrics"]) == set(V0_TRAIN_TAGS)
    assert report["epochs_run"] == 3
    assert report["stopped_early"] is False
    assert len(report["history"]) == 3
    assert report["checkpoint"] == str(checkpoint_path)
    # 每 tag 指标字段齐全
    for tag_metrics in report["val_metrics"].values():
        assert set(tag_metrics) >= {"ap", "f1_at_050", "support", "ranking_only"}


def test_feature_cache_reused_on_second_run(tmp_path: Path, capsys) -> None:
    """第二次运行命中特征缓存：不重算、缓存文件不被覆盖写。"""
    cache = tmp_path / "features.pt"
    rc1, checkpoint_path, _report = _run(tmp_path, feature_cache=cache, epochs=1)
    assert rc1 == 0
    assert cache.is_file()
    assert "已写入特征缓存" in capsys.readouterr().out
    mtime = cache.stat().st_mtime_ns

    rc2, checkpoint_path2, _ = _run(tmp_path, feature_cache=cache, epochs=1)
    assert rc2 == 0
    assert checkpoint_path2.is_file()
    out2 = capsys.readouterr().out
    assert "命中特征缓存" in out2
    assert "已写入特征缓存" not in out2
    assert cache.stat().st_mtime_ns == mtime  # 未重算


def test_count_bad_images_detects_corrupt_file(tmp_path: Path) -> None:
    """坏图显式探测：与白占位静默语义互补。"""
    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "bad.jpg").write_bytes(b"not an image")
    make_rgb_jpeg(data_root / "good.jpg")
    assert count_bad_images(data_root, ["bad.jpg", "good.jpg"]) == ["bad.jpg"]


def test_compute_pos_weight_ratios_and_fallback() -> None:
    """per-tag 正负比正确；无正样本 tag 回退 1。"""
    targets = torch.tensor([[1, 0], [0, 0], [0, 1], [0, 0]], dtype=torch.float32)
    assert torch.allclose(compute_pos_weight(targets), torch.tensor([3.0, 3.0]))
    assert torch.allclose(compute_pos_weight(torch.zeros(4, 1)), torch.tensor([1.0]))


def test_average_precision_known_values() -> None:
    """AP 手写实现与 step-wise 积分定义一致。"""
    # 降序位置 [1,0,1]：precision 1/1, 2/3 → AP = (1 + 2/3) / 2
    scores = torch.tensor([0.9, 0.8, 0.1])
    labels = torch.tensor([1.0, 0.0, 1.0])
    assert abs(average_precision(scores, labels) - (1 + 2 / 3) / 2) < 1e-6
    # 完美分离 AP = 1
    assert average_precision(torch.tensor([0.9, 0.8, 0.2]), torch.tensor([1.0, 1.0, 0.0])) == 1.0
    # 无正样本约定 0
    assert average_precision(torch.tensor([0.9, 0.1]), torch.tensor([0.0, 0.0])) == 0.0


def test_compute_tag_metrics_ranking_only_flag() -> None:
    """小支持数 tag 标注排序辅助档；F1@0.5 可计算。"""
    probs = torch.tensor([[0.9, 0.8], [0.2, 0.7], [0.1, 0.6]])
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    metrics = compute_tag_metrics(probs, targets, ["tag_a", "tag_b"], small_tag_support=30)
    assert metrics["tag_a"]["support"] == 1
    assert metrics["tag_a"]["ranking_only"] is True
    assert abs(metrics["tag_a"]["ap"] - 1.0) < 1e-6
    assert metrics["tag_a"]["f1_at_050"] == 1.0
    # tag_b 概率全在 0.5 之上：pred 全正 → precision=1/3, recall=1 → F1=0.5
    assert metrics["tag_b"]["f1_at_050"] == 0.5


def test_backbone_weights_fallback_to_config(monkeypatch, tmp_path: Path, capsys) -> None:
    """[tags/B-1] CLI 未传 backbone_weights 时回退 config 字段并加载；报告记录路径。"""
    import tags_train

    calls: list[Path] = []

    def _fake_load(model, weights_path):
        calls.append(Path(weights_path))
        return {"head.weight", "head.bias"}

    monkeypatch.setattr(tags_train, "load_wd_pretrained", _fake_load)
    weights = tmp_path / "fake.safetensors"
    weights.write_bytes(b"")  # 只需存在，加载逻辑已 mock

    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    images = _build_dataset(data_root)
    labels_path = _write_labels(tmp_path / "labels.json", images, data_root)
    config = _make_config()
    config["backbone_weights"] = str(weights)  # config 指定权重，CLI 传 None

    rc = run_tags_training(
        config,
        labels_path=labels_path,
        data_root=data_root,
        checkpoint_path=tmp_path / "m" / "best.pth",
        report_path=tmp_path / "report.json",
        arch=ARCH,
        img_size=IMG_SIZE,
        epochs=1,
        feature_cache=tmp_path / "features.pt",
        backbone_weights=None,
        device="cpu",
    )
    assert rc == 0
    assert calls == [weights]  # 权重确实被加载（而非静默随机骨干）
    out = capsys.readouterr().out
    assert "已加载骨干权重" in out and str(weights) in out
    assert "随机骨干" not in out
    report = load_json(tmp_path / "report.json")
    assert report["backbone_weights"] == str(weights)


def test_random_backbone_warning_when_no_weights(tmp_path: Path, capsys) -> None:
    """[tags/B-1] config 与 CLI 均未提供权重时，显式打印随机骨干警告。"""
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    images = _build_dataset(data_root)
    labels_path = _write_labels(tmp_path / "labels.json", images, data_root)
    rc = run_tags_training(
        _make_config(),  # config 不含 backbone_weights
        labels_path=labels_path,
        data_root=data_root,
        checkpoint_path=tmp_path / "m" / "best.pth",
        report_path=tmp_path / "report.json",
        arch=ARCH,
        img_size=IMG_SIZE,
        epochs=1,
        feature_cache=tmp_path / "features.pt",
        backbone_weights=None,
        device="cpu",
    )
    assert rc == 0
    assert "警告: 未加载骨干权重，使用随机骨干" in capsys.readouterr().out


def test_feature_cache_invalidated_on_img_size_change(tmp_path: Path, capsys) -> None:
    """[tags/G-1] 缓存键 img_size 变更后不命中旧缓存，重算并覆盖写。"""
    cache = tmp_path / "features.pt"
    rc1, _, _ = _run(tmp_path, feature_cache=cache, epochs=1)
    assert rc1 == 0
    assert "已写入特征缓存" in capsys.readouterr().out

    rc2, _, _ = _run(tmp_path, feature_cache=cache, epochs=1, img_size=IMG_SIZE + 32)
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "命中特征缓存" not in out2  # 旧缓存未命中
    assert "已写入特征缓存" in out2  # 重新提特征并落盘
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    assert payload["img_size"] == IMG_SIZE + 32  # 缓存已按新键覆盖


def test_feature_cache_invalidated_on_version_mismatch(tmp_path: Path, capsys) -> None:
    """[tags/C-1] 缓存 version 与 FEATURES_CACHE_VERSION 不符时失效重算。"""
    cache = tmp_path / "features.pt"
    rc1, _, _ = _run(tmp_path, feature_cache=cache, epochs=1)
    assert rc1 == 0
    capsys.readouterr()

    # 篡改缓存 version 模拟旧版本残留
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    payload["version"] = FEATURES_CACHE_VERSION - 1
    torch.save(payload, cache)

    rc2, _, _ = _run(tmp_path, feature_cache=cache, epochs=1)
    assert rc2 == 0
    out2 = capsys.readouterr().out
    assert "命中特征缓存" not in out2
    assert "已写入特征缓存" in out2
    rewritten = torch.load(cache, map_location="cpu", weights_only=True)
    assert rewritten["version"] == FEATURES_CACHE_VERSION


def test_run_labels_missing_returns_2(tmp_path: Path) -> None:
    """[tags/G-2] labels.json 不存在：打印 ERROR 并返回 2。"""
    rc = run_tags_training(
        _make_config(),
        labels_path=tmp_path / "nope.json",
        data_root=tmp_path,
        checkpoint_path=tmp_path / "m" / "best.pth",
        report_path=tmp_path / "report.json",
        device="cpu",
    )
    assert rc == 2


def test_run_all_samples_skipped_returns_2(tmp_path: Path) -> None:
    """[tags/G-2] labels 内图片全部喜好缺失：无可用样本，返回 2。"""
    data_root = tmp_path / "data"
    data_root.mkdir()
    labels = _write_labels(
        tmp_path / "labels.json",
        [{"path": "a/x.jpg", "tags": ["无背景"]}, {"path": "a/y.jpg", "tags": []}],
        data_root,
    )
    rc = run_tags_training(
        _make_config(),
        labels_path=labels,
        data_root=data_root,
        checkpoint_path=tmp_path / "m" / "best.pth",
        report_path=tmp_path / "report.json",
        device="cpu",
    )
    assert rc == 2


def test_load_v0_checkpoint_rejects_foreign_format(tmp_path: Path) -> None:
    """[tags/G-2] 非 tags_v0 格式的 .pth 文件：显式 ValueError 拒绝。"""
    fake = tmp_path / "foreign.pth"
    torch.save({"format": "rank_v1", "state_dict": {}}, fake)
    with pytest.raises(ValueError, match="非 tags_v0"):
        load_v0_checkpoint(fake)


def test_main_output_paths_fall_back_to_config(tmp_path: Path) -> None:
    """[tags/C-2] CLI 未传输出路径时，checkpoint/report 写到 config 指定位置。"""
    data_root = tmp_path / "data"
    data_root.mkdir()
    images = _build_dataset(data_root)
    labels_path = _write_labels(tmp_path / "labels.json", images, data_root)
    config = _make_config()
    config.update(
        {
            "labels_path": str(labels_path),
            "feature_cache": str(tmp_path / "features.pt"),
            "checkpoint_out": str(tmp_path / "out" / "cfg_best.pth"),
            "report_out": str(tmp_path / "out" / "cfg_report.json"),
        }
    )
    config_path = tmp_path / "tags_v0.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    rc = main(["--config", str(config_path)])  # CLI 不传任何输出路径
    assert rc == 0
    checkpoint = tmp_path / "out" / "cfg_best.pth"
    report_path = tmp_path / "out" / "cfg_report.json"
    assert checkpoint.is_file() and report_path.is_file()
    report = load_json(report_path)
    assert report["checkpoint"] == str(checkpoint)
