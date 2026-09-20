---
date: 2026-09-20
agile_trace: true
---

# v0-tags-training 实现规格（SPEC）

## 根因 / 方案摘要

解析缺陷根因：`tags/filename_tags.py` 的 `_SPLIT_RE` 仅按半角/全角逗号拆组内多标签，
而 TagSpaces 默认写盘为空格分隔（词表 12 tag 均为无空白单 token，按空白拆安全）。

v0 方案（沿用记忆拍板）：冻结 wd-eva02-large-tagger-v3 骨干 + 预计算 1024 维特征
（453×1024 fp16 ≈ 1MB，缓存文件命中即跳过）+ 线性头 BCEWithLogitsLoss
（per-tag pos_weight = 训练 split 正负比，无 cap）+ 按喜好组分层划分 +
阈值无关的 per-tag Average Precision 评估。8GB 笔记本全程可跑。

## 变更点清单

| 提交 | 内容 |
|------|------|
| `8f3a9ea` | fix(tags): 解析器支持空格分隔多标签（+3 单测；RULE.md 协议描述同步） |
| `29435fc` | feat(tags): TagsDataset 多标签数据集与冻结骨干特征预计算 |
| `7e76a05` | feat(tags): 线性头 BCE 训练循环、v0 checkpoint 与训练 CLI 入口 |
| `2c65ad5` | test(tags): v0 数据集单测与训练端到端冒烟测试 |
| `48ca984` | docs(apm): 敏捷过程记忆（过程留痕，非代码） |

新文件：`tags/dataset.py`、`tags/features.py`、`tags/train.py`、`tags_train.py`、
`configs/tags_v0.yaml`、`tests/test_tags_dataset.py`、`tests/test_tags_train_smoke.py`。
修改：`tags/filename_tags.py`、`tests/test_tags_filename.py`、`tags/__init__.py`、
`docs/apm/RULE.md`。

## 详细改动说明

- **解析器**：`_SPLIT_RE = re.compile(r"[\s,，]+")`，docstring 更新为真实写盘格式。
- **TagsDataset**：labels.json → `V0_TRAIN_TAGS` 9 维 multi-hot；喜好缺失/冲突跳过并计数
  （`skip_stats`）；坏图经 `load_white_square_tensor` 静默白占位（预计算阶段另行显式统计）。
- **分层划分**：按喜好组分层抽样，组内至少 1 张进 val、单样本组留 train，
  独立 `torch.Generator(seed)` 可复现。
- **特征预计算**：冻结骨干 `model.backbone` 批量前向（fp16），缓存键 =
  arch/img_size/路径序/维度，匹配即跳过；坏图显式探测打印；损坏缓存安全降级重算。
- **头训练**：AdamW(lr=1e-3, wd=0.01) + ReduceLROnPlateau(0.2/3) + val loss 早停(10)，
  镜像 rank/train.py 骨架但独立实现（不 import rank）。
- **评估**：手写 step-wise Average Precision（无新依赖）；F1@0.5 与支持数为参考；
  val 正样本 <30 的 tag 标注"排序辅助档，不做阈值判定"。
- **checkpoint**：payload = 头 state_dict + arch/num_features/tag_list/pos_weight/
  img_size/dropout/backbone_weights 路径 + epoch/val_loss/history/per-tag metrics；
  `load_v0_checkpoint` 以 `weights_only=True` 加载并重建 TagModel（骨干随机初始化，
  生产打分需另行 `load_wd_pretrained`）。
- **入口**：`tags_train.py` CLI 镜像 rank_train.py 的 `--config` + None 哨兵覆盖风格；
  流程 = 加载 labels → 分层划分 → 特征预计算 → pos_weight → 训头 → 评估 → 写
  `models/tags/v0_best.pth` + `data/classification/tags_v0_report.json`。

## 测试策略

### 测试用例

- 定向：`python -m pytest tests/test_tags_dataset.py tests/test_tags_train_smoke.py tests/test_tags_filename.py -q` → 25 passed
- 全量回归：`python -m pytest tests/ -q` → 52 passed
- 真实数据验收（2026-09-20，4070Ti Laptop 8GB）：
  - 扫描：453/453 全解析、零冲突、零未知标签，计数与基准逐项一致
  - 训练：train 385 / val 68；特征预计算 4 分 35 秒（GPU，29 批）；44/100 轮早停，
    best_val_loss=0.6828
  - val per-tag AP：无背景 0.951 / 一般 0.943 / 漫画图 0.938 / NSFW 0.892 / 删除 0.888 /
    喜欢 0.429 / 小水印 0.301 / 灵魂 0.186（支持 4）/ 低像素 0.183（支持 2）；mAP=0.634
  - checkpoint 复核：`load_v0_checkpoint` 元数据齐全，头前向 (2,9) 有限值

## 风险与回滚方案

- 风险：①坏图静默白占位混入训练（已显式统计，当前 0 张）；②val 支持数小的 tag
  （灵魂4/低像素2/小水印4）指标噪声大，AP 低属预期，v0 仅作排序辅助；
  ③timm 无版本 pin（当前 1.0.29），大版本升级需回归冒烟；④MC dropout 打分的
  "同图同种子"协议 rank 侧即缺失，本期未实现，打分入口落地时必须补。
- 回滚：分支整体 revert（新增文件删除 + `8f3a9ea` revert 即恢复逗号解析）；
  训练产物（checkpoint/报告/特征缓存/labels.json）均在 gitignore 目录，删除无痕。
