"""tags_predict 打标入口离线测试：resnet18 小骨干 + 假 WD 权重 + tmp_path 树。

覆盖：dry-run 不改名且报告正确、--apply 改名后可 parse_filename 读回、
pref_source=model 平铺目录、同图同种子逐位复现、权重缺失硬报错 rc=2、
已打标/坏图跳过。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from tags import PREFERENCE_TAGS, V0_TRAIN_TAGS, TagModel, parse_filename, save_v0_checkpoint
from tags_predict import main
from tests.conftest import make_rgb_jpeg
from tools import iter_images_under, load_json

ARCH = "resnet18"  # 仅测试用小骨干；生产为 eva02_large_patch14_448
IMG_SIZE = 64


def _make_checkpoint(path: Path) -> None:
    """用 resnet18 小骨干造 v0 checkpoint（头随机初始化即可验证推理链路）。"""
    model = TagModel(ARCH, len(V0_TRAIN_TAGS), pretrained=False, dropout=0.1)
    save_v0_checkpoint(
        path,
        model.head.state_dict(),
        arch=ARCH,
        num_features=model.backbone.num_features,
        tag_list=V0_TRAIN_TAGS,
        img_size=IMG_SIZE,
        dropout=0.1,
    )


def _build_tree(root: Path) -> Path:
    """构造 good/keep/trash + 平铺 + 已打标 + 坏图的测试树（共 11 个图片文件）。"""
    for dir_name, count in (("good", 3), ("keep", 2), ("trash", 2)):
        for i in range(count):
            make_rgb_jpeg(root / dir_name / f"{dir_name}_{i}.png", color=(i * 40 % 256, 100, 200))
    make_rgb_jpeg(root / "plain_0.jpg")
    make_rgb_jpeg(root / "labeled[喜欢].png")  # 结尾已有词表标签
    make_rgb_jpeg(root / "unknown_tag[贴错].png")  # 结尾未知标签同样视为已打标
    (root / "bad.jpg").write_bytes(b"not an image")  # 坏图字节文件
    return root


def _run_main(
    tmp_path: Path, data_root: Path, monkeypatch, *, extra: list[str] | None = None, report_name: str = "report.json"
) -> tuple[int, Path]:
    """mock WD 权重加载并调用 CLI 入口，返回 (返回码, 报告路径)。"""
    import tags_predict

    def _fake_load(model, weights_path):
        # 确定性覆写骨干参数：多次调用得到相同的"伪权重"，
        # 否则 load_v0_checkpoint 每次随机重建骨干，复现性测试不可比
        torch.manual_seed(123)
        for p in model.backbone.parameters():
            p.data.copy_(torch.randn_like(p) * 0.01)
        return set()

    monkeypatch.setattr(tags_predict, "load_wd_pretrained", _fake_load)
    checkpoint = tmp_path / "ckpt" / "v0_best.pth"
    if not checkpoint.exists():  # 同一 tmp_path 内多次调用复用同一模型（复现性测试前提）
        _make_checkpoint(checkpoint)
    weights = tmp_path / "wd.safetensors"
    weights.write_bytes(b"")  # 只需存在，加载逻辑已 mock
    config = tmp_path / "cfg.yaml"
    config.write_text("", encoding="utf-8")  # 空 config：全走内置默认
    report = tmp_path / report_name
    argv = [
        "--config", str(config),
        "--data-root", str(data_root),
        "--checkpoint", str(checkpoint),
        "--backbone-weights", str(weights),
        "--report-out", str(report),
        "--device", "cpu",
        *(extra or []),
    ]
    return main(argv), report


def test_dry_run_keeps_names_and_report(tmp_path: Path, monkeypatch) -> None:
    """dry-run：全部文件名不变；报告 action 分布、目录映射（显式 dir）、probs 键齐全。"""
    root = _build_tree(tmp_path / "data")
    before = {p.relative_to(root).as_posix() for p in iter_images_under(root)}
    rc, report_path = _run_main(
        tmp_path, root, monkeypatch, extra=["--n-mc", "5", "--pref-source", "dir"]
    )
    assert rc == 0
    after = {p.relative_to(root).as_posix() for p in iter_images_under(root)}
    assert after == before  # dry-run 不改名

    report = load_json(report_path)
    assert report["meta"]["mode"] == "dry-run"
    assert report["meta"]["pref_source"] == "dir"  # 显式 dir：目录映射生效
    assert report["meta"]["n_mc"] == 5
    assert report["meta"]["tag_list"] == list(V0_TRAIN_TAGS)

    actions = Counter(im["action"] for im in report["images"])
    assert actions == {"prefilled": 8, "skipped_existing": 2, "skipped_bad": 1}

    by_rel = {im["rel_path"]: im for im in report["images"]}
    # 目录映射：good→喜欢 / keep→一般 / trash→删除；平铺图无目录
    assert by_rel["good/good_0.png"]["dir_pref"] == "喜欢"
    assert by_rel["keep/keep_0.png"]["dir_pref"] == "一般"
    assert by_rel["trash/trash_0.png"]["dir_pref"] == "删除"
    assert by_rel["plain_0.jpg"]["dir_pref"] is None
    assert by_rel["good/good_0.png"]["chosen_tags"][0] == "喜欢"

    for im in report["images"]:
        if im["action"] == "prefilled":
            assert set(im["probs"]) == set(V0_TRAIN_TAGS)
            assert set(im["mc_std"]) == set(V0_TRAIN_TAGS)
            assert all(0.0 <= p <= 1.0 for p in im["probs"].values())
            assert im["chosen_tags"][0] in PREFERENCE_TAGS
            assert im["renamed_to"] and "[" in im["renamed_to"] and "]" in im["renamed_to"]
            # 方括号前无空格：原 stem 后紧跟 [
            assert f"{Path(im['rel_path']).stem}[" in im["renamed_to"]
        else:
            assert im["probs"] is None and im["mc_std"] is None
            assert im["chosen_tags"] == [] and im["renamed_to"] is None

    # agreement 仅 dir 模式给出：7 张目录图（平铺图不参与）
    assert report["agreement"]["total"] == 7
    assert 0.0 <= report["agreement"]["match_rate"] <= 1.0
    s = report["summary"]
    assert (s["total"], s["prefilled"], s["skipped_existing"], s["skipped_bad"], s["failed"]) == (11, 8, 2, 1, 0)
    assert sum(s["tag_counts"].values()) >= 8  # 每张预填图至少 1 个喜好 tag


def test_apply_renames_and_parse_back(tmp_path: Path, monkeypatch) -> None:
    """--apply：改名后 parse_filename 读回标签、扩展名保留、good 目录映射正确。"""
    root = _build_tree(tmp_path / "data")
    rc, report_path = _run_main(tmp_path, root, monkeypatch, extra=["--apply", "--n-mc", "3"])
    assert rc == 0
    report = load_json(report_path)
    assert report["meta"]["mode"] == "apply"
    assert report["failures"] == []

    for im in report["images"]:
        if im["action"] == "prefilled":
            new_path = root / im["renamed_to"]
            assert new_path.is_file()
            parsed = parse_filename(new_path)
            assert parsed["tags"] == im["chosen_tags"]  # 标签可读回且一致
            assert parsed["stem"] == Path(im["rel_path"]).stem  # 原 stem 保留
            assert new_path.suffix == Path(im["rel_path"]).suffix  # 扩展名保留
        else:
            assert (root / im["rel_path"]).is_file()  # 已打标/坏图原名未动

    good_files = sorted((root / "good").glob("*.png"))
    assert len(good_files) == 3
    for f in good_files:
        assert parse_filename(f)["tags"][0] == "喜欢"  # good → 喜欢


def test_pref_source_model_flat_dir(tmp_path: Path, monkeypatch) -> None:
    """pref_source=model 在无 good/keep/trash 的平铺目录生效；auto 亦回退 model。"""
    flat = tmp_path / "flat"
    for i in range(3):
        make_rgb_jpeg(flat / f"img_{i}.png", color=(i * 60 % 256, 80, 160))
    rc, report_path = _run_main(tmp_path, flat, monkeypatch, extra=["--pref-source", "model"])
    assert rc == 0
    report = load_json(report_path)
    assert report["meta"]["pref_source"] == "model"
    assert "agreement" not in report  # 仅 dir 模式给一致率
    assert report["soul_candidates"] == []
    for im in report["images"]:
        assert im["action"] == "prefilled"
        assert im["dir_pref"] is None
        assert im["model_pref"] in PREFERENCE_TAGS
        assert im["chosen_tags"][0] == im["model_pref"]

    # 默认 auto：平铺目录无三档子目录 → 回退 model
    rc2, report2_path = _run_main(
        tmp_path, flat, monkeypatch, extra=["--pref-source", "auto"], report_name="report2.json"
    )
    assert rc2 == 0
    assert load_json(report2_path)["meta"]["pref_source"] == "model"


def test_default_pref_source_is_auto_dir(tmp_path: Path, monkeypatch) -> None:
    """默认（不传 --pref-source）即 auto：三档目录齐备时走目录映射。

    2026-09-21 二次拍板：模型喜好预测偏差远大于目录映射的语义偏差
    （114299 修正实证），目录做预填先验、人工修正兜底。
    """
    root = _build_tree(tmp_path / "data")
    rc, report_path = _run_main(tmp_path, root, monkeypatch, extra=["--n-mc", "5"])
    assert rc == 0
    report = load_json(report_path)
    assert report["meta"]["pref_source"] == "dir"
    by_rel = {im["rel_path"]: im for im in report["images"]}
    assert by_rel["good/good_0.png"]["chosen_tags"][0] == "喜欢"
    assert by_rel["keep/keep_0.png"]["chosen_tags"][0] == "一般"
    assert by_rel["trash/trash_0.png"]["chosen_tags"][0] == "删除"
    # 目录未识别的平铺图回退 model argmax
    assert by_rel["plain_0.jpg"]["chosen_tags"][0] == by_rel["plain_0.jpg"]["model_pref"]


def test_negative_subsumption_rule() -> None:
    """大小负面吞并：任一大负面（官方图）过阈时小负面全部让位。"""
    from tags_predict import select_negative_tags

    tags = list(V0_TRAIN_TAGS)  # 含官方图（2026-09-21 并入，10 tag）
    assert "官方图" in tags
    probs = {"官方图": 0.9, "无背景": 0.8, "NSFW": 0.7, "小水印": 0.75, "低像素": 0.8}
    # 大负面入选 → 只留官方图
    assert select_negative_tags(probs, tags, 0.5, 0.7) == ["官方图"]
    # 无大负面 → 小负面按档位阈值组合
    probs2 = {"官方图": 0.3, "无背景": 0.8, "NSFW": 0.7, "小水印": 0.75, "低像素": 0.6}
    assert select_negative_tags(probs2, tags, 0.5, 0.7) == ["无背景", "NSFW", "小水印"]


def test_same_seed_reproducible_probs(tmp_path: Path, monkeypatch) -> None:
    """同图同种子：同参数两次运行 probs/mc_std 逐位一致（n_mc=5）。"""
    root = tmp_path / "data"
    for i in range(4):
        make_rgb_jpeg(root / f"img_{i}.png", color=(i * 50 % 256, 80, 160))
    common = ["--n-mc", "5", "--seed", "42"]
    rc1, r1 = _run_main(tmp_path, root, monkeypatch, extra=common)
    rc2, r2 = _run_main(tmp_path, root, monkeypatch, extra=common, report_name="report2.json")
    assert rc1 == 0 and rc2 == 0
    report1, report2 = load_json(r1), load_json(r2)
    assert [im["probs"] for im in report1["images"]] == [im["probs"] for im in report2["images"]]
    assert [im["mc_std"] for im in report1["images"]] == [im["mc_std"] for im in report2["images"]]


def test_missing_backbone_weights_returns_2(tmp_path: Path, monkeypatch) -> None:
    """骨干权重缺失：硬报错退出 rc=2（不做随机骨干推理）。"""
    root = tmp_path / "data"
    make_rgb_jpeg(root / "a.png")
    rc, _ = _run_main(
        tmp_path, root, monkeypatch, extra=["--backbone-weights", str(tmp_path / "nope.safetensors")]
    )
    assert rc == 2
