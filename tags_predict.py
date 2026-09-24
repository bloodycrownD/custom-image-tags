"""tags 预测打标 CLI：批量推理 9 tag 概率并生成 TagSpaces 文件名标签。

用法：
    python tags_predict.py --data-root <目标目录> [--apply] [--pref-source auto|dir|model]

流程：加载 v0 checkpoint + WD 基模权重 → 目标目录批量推理 →
按策略生成文件名标签 → 写报告 json；默认 dry-run 只出报告，
``--apply`` 才真正重命名文件。

打标策略：
- 喜好 4 tag 互斥单选：pref-source=auto（默认）在 good/keep/trash 三档目录
  齐备时用目录映射（good→喜欢/keep→一般/trash→删除），否则回退 model argmax。
  教训记录：目录是作者内相对排序、与全局喜好有语义偏差（953690 实证 38/38），
  但模型喜好预测偏差更大（114299 修正实证远差于目录）——两害相权目录是更好
  的预填先验，最终以人工修正为准；dir/model 为显式选项；
  dir 模式下目录未识别的图回退 model argmax，「灵魂」无目录对应——
  模型倾向灵魂的 good 图仅列入 soul_candidates 供人工补标；
- 负面 5 tag 多选：mean prob ≥ 阈值入选；强档（无背景/漫画图/NSFW/官方图）
  默认 0.5，弱档（小水印/低像素，v0 排序辅助档）保守取 max(阈值, 0.7)；
  **大小负面吞并**：官方图/大水印属大负面（独占），任一大负面入选时
  小负面全部让位（2026-09-21 用户修正 114299 后确立的标注约定）；
- 文件名格式 ``{原stem}[{tag1} {tag2}]{原扩展名}``（方括号前无空格，
  喜好 tag 在前、负面按 V0_TRAIN_TAGS 顺序）。

MC dropout 可复现协议（骨干批量前向 1 次，MC 只循环 head）：
- Dropout 仅存在于 head（骨干无 dropout），故先在 eval/no_grad 下批量
  提骨干特征，MC 循环只跑 head，成本近零；
- 每图独立种子：第 i 张图（全枚举顺序索引）MC 循环前
  ``torch.manual_seed(seed * 1_000_003 + i)``，严格同图同种子、与
  batch 划分无关；只把 head 内 Dropout 切 train（try/finally 恢复）；
- n_mc=1 走确定性 eval 前向（std=0）；运行开始设 cudnn.deterministic。

跳过规则：结尾已有方括号标签（含未知标签）不覆盖；坏图跳过；
重命名目标已存在或被占用计为失败并继续。骨干权重缺失直接报错退出
（exit 2）——随机骨干打出的标签会写进用户文件名，不可接受。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tags import (
    PREFERENCE_SET,
    PREFERENCE_TAGS,
    V0_TRAIN_TAGS,
    build_wd_preprocess,
    load_rgb_white_background,
    load_v0_checkpoint,
    load_wd_pretrained,
    parse_filename,
)
from tags.vocab import BIG_NEGATIVE_TAGS
from tools import iter_images_under, rel_posix_path, save_json, utc_now_iso

DEFAULT_CHECKPOINT = Path("models/tags/v0_best.pth")
DEFAULT_BACKBONE_WEIGHTS = Path("data/pretrained/wd-eva02-large-tagger-v3/model.safetensors")
DEFAULT_REPORT = Path("data/classification/tags_prefill_report.json")
# MC 协议默认值（记忆约定：n_mc 20~50；cudnn.deterministic）
DEFAULT_N_MC = 20
DEFAULT_SEED = 42
# 喜好来源默认 auto：目录映射做预填先验（好于模型喜好预测，114299 实证），
# 三档目录齐备走 dir、否则回退 model；语义偏差由人工修正兜底
DEFAULT_PREF_SOURCE = "auto"
# 每图 MC 种子 = seed * 1_000_003 + 全枚举顺序索引（大素数乘子拉开相邻图种子）
MC_SEED_STRIDE = 1_000_003
# good/keep/trash 三分类目录（rank 链约定）→ 喜好 tag 映射
DIR_TO_PREF = {"good": "喜欢", "keep": "一般", "trash": "删除"}
# 负面 tag 阈值分档：强档直接判定；弱档为 v0 排序辅助档（AP 0.30/0.18），阈值下限 0.7
NEG_STRONG_THRESHOLD = 0.5
NEG_WEAK_TAGS = frozenset({"小水印", "低像素"})
NEG_WEAK_FLOOR = 0.7


def select_negative_tags(
    probs: dict[str, float], tag_list: list[str], strong_th: float, weak_th: float
) -> list[str]:
    """按阈值选负面 tag，并应用大小负面吞并。

    任一大负面（官方图/大水印，见 tags/vocab.py BIG_NEGATIVE_TAGS）入选时，
    小负面全部让位——用户 2026-09-21 确立的标注约定：大负面独占。
    """
    negs = [
        tag
        for tag in tag_list
        if tag not in PREFERENCE_SET
        and probs.get(tag, 0.0) >= (weak_th if tag in NEG_WEAK_TAGS else strong_th)
    ]
    if any(tag in BIG_NEGATIVE_TAGS for tag in negs):
        negs = [tag for tag in negs if tag in BIG_NEGATIVE_TAGS]
    return negs


def load_config(path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_device(spec: str | None) -> torch.device:
    """解析设备：auto / cpu / cuda（无 cuda 时 auto 回退 cpu）。"""
    if not spec or spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


@torch.no_grad()
def _mc_head_probs(
    head: nn.Module, feat: torch.Tensor, n_mc: int, seed_i: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """单图 MC dropout 采样，返回 (mean, std) 概率张量 (T,)。

    协议：先 ``torch.manual_seed(seed_i)`` 再循环 head 前向 n_mc 次，
    保证同图同种子可复现；只把 head 内 Dropout 切 train（镜像
    ``TagModel.mc_forward`` 的只切 Dropout 语义，try/finally 恢复），
    不整体切 train()。n_mc <= 1 走确定性 eval 前向（std 恒 0）。
    """
    if n_mc <= 1:
        head.eval()
        prob = torch.sigmoid(head(feat))
        return prob, torch.zeros_like(prob)
    torch.manual_seed(seed_i)
    dropouts = [m for m in head.modules() if isinstance(m, nn.Dropout)]
    for m in dropouts:
        m.train(True)
    try:
        samples = torch.stack([torch.sigmoid(head(feat)) for _ in range(n_mc)])
    finally:
        for m in dropouts:
            m.train(False)  # 推理上下文恒为 eval
    return samples.mean(dim=0), samples.std(dim=0)


def run_tags_predict(
    config: dict,
    *,
    data_root: Path,
    checkpoint_path: Path | None,
    backbone_weights: Path | None,
    report_path: Path,
    apply: bool = False,
    pref_source: str | None = None,
    neg_threshold: float | None = None,
    n_mc: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> int:
    """执行批量打标，成功返回 0、硬错误返回 2。

    None 参数回退 config 字段（None 哨兵覆盖式 CLI）。流程：
    加载 v0 checkpoint（meta 自检）→ 加载 WD 基模权重（缺失硬报错）→
    枚举图片（已打标/坏图跳过）→ 骨干批量提特征 → 每图独立种子 MC →
    按策略生成标签 → dry-run 出报告 / apply 重命名 → 报告 + stdout 摘要。
    """
    # ---- 参数解析（None 哨兵回退 config）----
    if checkpoint_path is None:
        checkpoint_path = config.get("checkpoint") or config.get("checkpoint_out") or DEFAULT_CHECKPOINT
    checkpoint_path = Path(checkpoint_path)
    if backbone_weights is None:
        backbone_weights = config.get("backbone_weights") or DEFAULT_BACKBONE_WEIGHTS
    backbone_weights = Path(backbone_weights)
    n_mc = int(n_mc if n_mc is not None else config.get("n_mc", DEFAULT_N_MC))
    seed = int(seed if seed is not None else config.get("seed", DEFAULT_SEED))
    batch_size = int(batch_size if batch_size is not None else config.get("batch_size", 16))
    strong_th = float(
        neg_threshold if neg_threshold is not None else config.get("neg_threshold", NEG_STRONG_THRESHOLD)
    )
    weak_th = max(strong_th, NEG_WEAK_FLOOR)
    requested_source = str(pref_source or config.get("pref_source", DEFAULT_PREF_SOURCE))

    # ---- 输入校验（骨干权重缺失硬报错：随机骨干打出的标签不可接受）----
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint 不存在: {checkpoint_path}", file=sys.stderr)
        return 2
    if not backbone_weights.is_file():
        print(
            f"ERROR: 骨干权重不存在: {backbone_weights}（推理要求 WD 基模权重，随机骨干标签不可信）",
            file=sys.stderr,
        )
        return 2
    data_root = Path(data_root)
    if not data_root.is_dir():
        print(f"ERROR: data_root 不存在或不是目录: {data_root}", file=sys.stderr)
        return 2

    # ---- 模型重建与 meta 自检 ----
    try:
        model, payload = load_v0_checkpoint(checkpoint_path, "cpu")
    except Exception as exc:
        print(f"ERROR: 加载 v0 checkpoint 失败: {exc}", file=sys.stderr)
        return 2
    tag_list = list(payload["tag_list"])
    if tag_list != list(V0_TRAIN_TAGS):
        print(f"ERROR: checkpoint tag_list 与 V0_TRAIN_TAGS 不一致: {tag_list}", file=sys.stderr)
        return 2
    if int(payload.get("num_features", -1)) != int(model.backbone.num_features):
        print(
            f"ERROR: num_features 自检失败: payload={payload.get('num_features')} "
            f"vs 模型={model.backbone.num_features}",
            file=sys.stderr,
        )
        return 2
    img_size = int(payload["img_size"])  # 预处理边长以 checkpoint 为准

    dev = _resolve_device(device or str(config.get("device", "auto")))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model.to(dev)
    try:
        dropped = load_wd_pretrained(model, backbone_weights)
    except Exception as exc:
        print(f"ERROR: 骨干权重加载失败: {exc}", file=sys.stderr)
        return 2
    print(f"已加载骨干权重: {backbone_weights}（丢弃原始头键 {len(dropped)} 个）")
    print(f"使用设备: {dev}, img_size={img_size}, tag 数 {len(tag_list)}")

    # ---- 枚举与记录（全枚举顺序索引即 MC 种子索引，与 batch 划分无关）----
    images = iter_images_under(data_root)
    print(f"待打标图片: {len(images)} 张, n_mc={n_mc}, 模式: {'apply' if apply else 'dry-run'}")
    preprocess = build_wd_preprocess(img_size)
    records: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for idx, path in enumerate(images):
        rel = rel_posix_path(path, data_root)
        rec: dict[str, Any] = {
            "rel_path": rel,
            "path": path,
            "index": idx,
            "dir_pref": DIR_TO_PREF.get(rel.split("/", 1)[0]),
            "model_pref": None,
            "probs": None,
            "mc_std": None,
            "chosen_tags": [],
            "action": "pending",
            "renamed_to": None,
        }
        parsed = parse_filename(path)
        if parsed["tags"] or parsed["unknown_tags"]:
            rec["action"] = "skipped_existing"  # 结尾已有标签（含未知标签）不覆盖
            records.append(rec)
            continue
        try:
            tensor = preprocess(load_rgb_white_background(path))
        except Exception:
            rec["action"] = "skipped_bad"  # 坏图（探测/预处理异常）跳过
            records.append(rec)
            continue
        rec["tensor"] = tensor
        records.append(rec)
        pending.append(rec)

    # ---- 喜好来源判定（auto：good/keep/trash 三档目录都识别到才整体走 dir）----
    dir_ready = all(
        any(rec["dir_pref"] == pref for rec in records) for pref in DIR_TO_PREF.values()
    )
    if requested_source == "dir":
        effective_source = "dir"
    elif requested_source == "auto" and dir_ready:
        effective_source = "dir"
    else:
        effective_source = "model"

    # ---- 骨干批量前向 1 次（eval + no_grad；Dropout 仅在 head，MC 无需重复骨干）----
    # cuda 下用 fp16 autocast：EVA02-L@448 fp32 贴 8GB 边缘易触发 WDDM 共享内存
    # 溢出（极慢+OOM），fp16 稳定且更快；特征转回 fp32 供 head 使用。
    # 必须 no_grad：否则 autograd 保留整网激活图（实测 ~20GB，必 OOM）。
    use_amp = dev.type == "cuda"
    with torch.no_grad():
        for start in tqdm(range(0, len(pending), batch_size), desc="提取骨干特征", unit="batch"):
            chunk = pending[start : start + batch_size]
            batch = torch.stack([rec.pop("tensor") for rec in chunk]).to(dev)
            with torch.autocast(device_type=dev.type, enabled=use_amp):
                feats = model.backbone(batch)
            feats = feats.float().cpu()
            for rec, feat in zip(chunk, feats.unbind(0)):
                rec["feat"] = feat

    # ---- MC dropout：每图独立种子，只循环 head ----
    pref_indices = [tag_list.index(t) for t in PREFERENCE_TAGS]
    for rec in tqdm(pending, desc="MC dropout 采样", unit="img"):
        mean, std = _mc_head_probs(
            model.head, rec.pop("feat").to(dev), n_mc, seed * MC_SEED_STRIDE + rec["index"]
        )
        rec["probs"] = {tag: float(mean[j]) for j, tag in enumerate(tag_list)}
        rec["mc_std"] = {tag: float(std[j]) for j, tag in enumerate(tag_list)}
        rec["model_pref"] = PREFERENCE_TAGS[int(mean[pref_indices].argmax())]

    # ---- 打标决策与重命名 ----
    soul_candidates: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for rec in pending:
        probs = rec["probs"]
        assert probs is not None
        chosen_pref = rec["dir_pref"] if effective_source == "dir" and rec["dir_pref"] else rec["model_pref"]
        assert chosen_pref is not None
        rec["chosen_tags"] = [chosen_pref] + select_negative_tags(probs, tag_list, strong_th, weak_th)
        # 灵魂候选：dir 模式下 good 无「灵魂」对应，模型倾向灵魂的交人工补标
        if effective_source == "dir" and rec["dir_pref"] == "喜欢" and (
            rec["model_pref"] == "灵魂" or probs["灵魂"] >= 0.5
        ):
            soul_candidates.append(
                {
                    "rel_path": rec["rel_path"],
                    "model_pref": rec["model_pref"],
                    "soul_prob": probs["灵魂"],
                }
            )
        new_path = rec["path"].with_name(
            f"{rec['path'].stem}[{' '.join(rec['chosen_tags'])}]{rec['path'].suffix}"
        )
        # dry-run 下 renamed_to 表示预期新相对路径
        rec["renamed_to"] = rel_posix_path(new_path, data_root)
        if not apply:
            rec["action"] = "prefilled"
            continue
        try:
            if new_path.exists():
                raise FileExistsError(f"目标文件已存在: {new_path.name}")
            rec["path"].rename(new_path)
            rec["action"] = "prefilled"
        except (FileExistsError, PermissionError, OSError) as exc:
            rec["action"] = "failed"
            rec["renamed_to"] = None
            failures.append({"rel_path": rec["rel_path"], "error": f"{type(exc).__name__}: {exc}"})

    # ---- dir vs model 喜好一致率（仅 dir 模式；含所有完成推理的目录图）----
    agreement: dict[str, Any] | None = None
    if effective_source == "dir":
        pairs = [
            (rec["dir_pref"], rec["model_pref"])
            for rec in records
            if rec["dir_pref"] is not None and rec["model_pref"] is not None
        ]
        matched = sum(1 for d, m in pairs if d == m)
        confusion = Counter(f"{d}|{m}" for d, m in pairs)
        agreement = {
            "total": len(pairs),
            "matched": matched,
            "match_rate": (matched / len(pairs)) if pairs else 0.0,
            "confusion": dict(sorted(confusion.items())),
        }

    # ---- 汇总与报告 ----
    tag_counter: Counter = Counter()
    for rec in records:
        if rec["action"] == "prefilled":
            tag_counter.update(rec["chosen_tags"])
    summary = {
        "total": len(records),
        "prefilled": sum(1 for r in records if r["action"] == "prefilled"),
        "skipped_existing": sum(1 for r in records if r["action"] == "skipped_existing"),
        "skipped_bad": sum(1 for r in records if r["action"] == "skipped_bad"),
        "failed": sum(1 for r in records if r["action"] == "failed"),
        "tag_counts": dict(tag_counter),
    }
    report: dict[str, Any] = {
        "version": 1,
        "meta": {
            "mode": "apply" if apply else "dry-run",
            "checkpoint": str(checkpoint_path),
            "backbone_weights": str(backbone_weights),
            "arch": str(payload["arch"]),
            "tag_list": tag_list,
            "img_size": img_size,
            "n_mc": n_mc,
            "seed": seed,
            "pref_source": effective_source,
            "neg_thresholds": {"strong": strong_th, "weak": weak_th},
            "created": utc_now_iso(),
        },
    }
    if agreement is not None:
        report["agreement"] = agreement
    report["summary"] = summary
    report["soul_candidates"] = soul_candidates
    report["failures"] = failures
    report["images"] = [
        {
            k: rec[k]
            for k in (
                "rel_path",
                "dir_pref",
                "model_pref",
                "probs",
                "mc_std",
                "chosen_tags",
                "action",
                "renamed_to",
            )
        }
        for rec in records
    ]
    save_json(Path(report_path), report)

    # ---- stdout 摘要 ----
    mode_label = "apply" if apply else "dry-run"
    print(f"== tags 预填摘要（{mode_label}）==")
    print(
        f"总数 {summary['total']}, 预填 {summary['prefilled']}, "
        f"跳过(已有标签) {summary['skipped_existing']}, "
        f"跳过(坏图) {summary['skipped_bad']}, 失败 {summary['failed']}"
    )
    if requested_source == "auto":
        note = "三档目录齐备" if dir_ready else "目录不齐备，回退模型 argmax"
        print(f"喜好来源: auto → {effective_source}（{note}）")
    else:
        print(f"喜好来源: {effective_source}")
    if agreement is not None:
        print(
            f"dir vs model 喜好一致率: {agreement['match_rate']:.1%}"
            f"（{agreement['matched']}/{agreement['total']}）"
        )
    if soul_candidates:
        print(f"灵魂候选（good 中模型倾向 灵魂，供人工补标）: {len(soul_candidates)} 张")
    freq = ", ".join(f"{tag}×{n}" for tag, n in sorted(tag_counter.items(), key=lambda kv: -kv[1]))
    print(f"tag 频次: {freq or '（无）'}")
    print(f"报告: {report_path}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="tags 预测打标：批量推理 tag 概率并生成 TagSpaces 文件名标签（默认 dry-run）"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/tags_v0.yaml"),
        help="YAML 配置文件路径（可选，缺省回退内置默认值）",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="待打标图片根目录")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="v0 checkpoint 路径（缺省回退 config 的 checkpoint/checkpoint_out）",
    )
    parser.add_argument(
        "--backbone-weights", type=Path, default=None,
        help="WD 基模 safetensors 路径（缺失即报错退出，不做随机骨干推理）",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="真正执行重命名（缺省 dry-run 只出报告）",
    )
    parser.add_argument(
        "--pref-source", choices=("auto", "dir", "model"), default=None,
        help="喜好 tag 来源（默认 auto：三档目录齐备走 dir 目录映射，否则 model argmax；"
             "目录预填先验好于模型喜好预测，最终以人工修正为准）",
    )
    parser.add_argument(
        "--neg-threshold", type=float, default=None,
        help="负面强档阈值（默认 0.5；弱档取 max(该值, 0.7)）",
    )
    parser.add_argument(
        "--n-mc", type=int, default=None,
        help="MC dropout 采样次数（默认 20；1 为确定性前向）",
    )
    parser.add_argument("--seed", type=int, default=None, help="MC 随机种子（默认 42）")
    parser.add_argument("--batch-size", type=int, default=None, help="骨干特征提取批大小")
    parser.add_argument("--report-out", type=Path, default=None, help="报告 json 输出路径")
    parser.add_argument("--device", type=str, default=None, help="设备：auto / cpu / cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 入口。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config: dict = {}
    if args.config.is_file():
        config = load_config(args.config)
    else:
        print(f"WARNING: 配置文件不存在，使用内置默认值: {args.config}", file=sys.stderr)

    report_out = args.report_out or Path(config.get("prefill_report_out", DEFAULT_REPORT))
    return run_tags_predict(
        config,
        data_root=args.data_root.resolve(),
        checkpoint_path=args.checkpoint,
        backbone_weights=args.backbone_weights,
        report_path=report_out.resolve(),
        apply=args.apply,
        pref_source=args.pref_source,
        neg_threshold=args.neg_threshold,
        n_mc=args.n_mc,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
