---
date: 2026-06-17
confirmed: 2026-06-17
---

# 二次元图片个性化审美评分系统 技术规格（SPEC）

> **状态：用户已确认（2026-06-17）**，可进入实现。  
> 确认项：全 PRD 能力（F1–F8 classify 侧）；`train_group` 默认 `min=5/max=80`；条件默认 `train_group`；`demo*.py` 移至 `history/`。

## 设计目标

在 `d:\Dev\Python\classify` 落地 **完整 pairwise 训练与迭代管线**，覆盖 [prd.md](prd.md) F1–F8（classify 侧）；F2 采集 UI 由 [pairwise-annotator](../pairwise-annotator/prd.md) / `annotator.exe` 实现，本仓库提供数据契约与消费端。

| PRD | 能力 | SPEC 落点 |
|-----|------|-----------|
| F1 | 图片 + 条件组 | `PreferenceRanker` + `train_group_map`；`condition: train_group \| author \| none` |
| F2 | 成对比较持久化 | `pairs.json` 契约；`merge_pairs`；annotator 另仓 |
| F3 | 种子 pair | `tools/seed_pairs.py` |
| F4 | 偏好分 + 不确定性 | `rank_predict.py`：score + MC Dropout uncertainty |
| F5 | 极端区优先 | 种子边加权 + `eval_rank.py` 分区域指标 |
| F6 | 主动学习采样 | `tools/export_active_queue.py` → `active_queue.json` |
| F7 | 增量更新 | `rank_train.py --resume --pairs-new` |
| F8 | 排序 / 极端 / 待复核 | `tools/report_results.py`（HTML + JSON） |
| F9 | 竖图不变形 | `KeepRatioResizePad(384, fill=0)` |

**代码基线：**

- `history/demo2.py`（自根目录迁入）：EfficientNet-B4、作者 Embedding、AMP、早停；逻辑迁入 `rank/`，不再使用三分类头。
- `author_classify/rank/main.py`：pairwise 原型；**缺陷**为 A/B 共用单一 `author_id`——本方案改为双 `train_group`。

**硬约束：**

- 预处理：`KeepRatioResizePad(384)`，`fill=0`，训练/推理一致。
- 条件维默认 **`train_group_id`**（`min_images=5`，`max_per_group=80`）。
- 稀疏 `pairs.json` 即可训练；不依赖全局全序。

---

## 总体方案

### 架构

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│ good/keep/trash │────▶│ tools/seed_pairs │────▶│                     │
└─────────────────┘     └──────────────────┘     │   pairs.json        │
┌─────────────────┐     ┌──────────────────┐     │   (+ annotator)     │
│ annotator.exe   │────▶│ merge_pairs      │────▶│                     │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                              │
         ┌────────────────────────────────────────────────────┘
         ▼
┌─────────────────┐   ┌─────────────────┐   ┌──────────────────────────┐
│ validate_pairs  │   │ build_groups    │   │ rank_train.py            │
└─────────────────┘   │ → group_map     │   │ (全量 / --resume 增量)    │
                      └─────────────────┘   └────────────┬─────────────┘
                                                         ▼
                      ┌──────────────────────────────────────────────┐
                      │ rank_predict.py  →  scores.json               │
                      │   (score, uncertainty, score_0_100, zones)   │
                      └────────────┬─────────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
   ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
   │ export_active_   │ │ report_results   │ │ eval_rank        │
   │ queue.py         │ │ (F8 展示报告)     │ │ (分区域验收)      │
   │ → active_queue   │ └──────────────────┘ └──────────────────┘
   └──────────────────┘
```

### 模型：`PreferenceRanker`

每张图使用各自的 `train_group_id`：

```python
score_a = head(backbone(img_a), embed(group_a))
score_b = head(backbone(img_b), embed(group_b))
diff = score_a - score_b
loss = BCEWithLogits(diff, target) * sample_weight
# target: a=1, b=0, tie=0.5
```

- 骨干：EfficientNet-B4（ImageNet 预训练），`classifier=Identity`。
- 条件：`nn.Embedding(num_groups + 1, embed_dim)`，index **0 = unknown**。
- 头：`BatchNorm → Linear → ReLU → Dropout(0.5) → Linear(1)`。
- **极端区损失加权**：`source=seed` 且涉及 good↔trash / good↔keep / keep↔trash 的边，`weight=1.0`；keep↔keep tie `weight=0.5`；`source=annotator` 默认 `1.0`（可配置）。

### 不确定性（F4 / F5）

推理时 **MC Dropout**（`rank/predict.py`）：

- `model.train()` 仅启用 Dropout 层；`n_mc` 次前向（默认 10）。
- `score_raw = mean(scores)`；`uncertainty = std(scores)`。
- `score_0_100`：checkpoint 存 `p5`, `p95`，线性映射并 clip。
- **待复核区**：`30 ≤ score_0_100 ≤ 70` 且 `uncertainty > u_threshold`（默认 `u_threshold` 取验证集 uncertainty 的 75 分位）。

### `train_group_id`（`tools/build_groups.py`）

| 条件 | 动作 |
|------|------|
| 图片数 `< 5` | 并入 `__rare__` |
| `5 ≤ 数 ≤ 80` | `author_{id}` |
| 数 `> 80` | `author_{id}__part{k}`，按文件名排序均分 |

输出 `train_group_map.json`（`groups` 名→idx，`authors` author→idx）。

### `pairs.json` 契约

```json
{
  "version": 1,
  "pairs": [{
    "image_a": "author_1/foo.jpg",
    "image_b": "author_2/bar.jpg",
    "author_a": "author_1",
    "author_b": "author_2",
    "prefer": "a",
    "group_id": "shard-007",
    "source": "annotator",
    "created_at": "2026-06-17T12:00:00"
  }]
}
```

### 主动学习队列（F6）

`tools/export_active_queue.py` 读 `scores.json`，输出 `active_queue.json`：

```json
{
  "version": 1,
  "budget": 200,
  "pairs": [
    { "image_a": "...", "image_b": "...", "reason": "high_uncertainty_pair", "priority": 0.92 }
  ]
}
```

**采样策略（按 priority 降序，默认 budget=200）：**

1. 两张图 `uncertainty` 均高于 P75，且 `|score_a - score_b| < 5`（难分辨对）。
2. 一张在极端区候选（`score < p10` 或 `> p90`）、另一张 uncertainty 高（边界确认）。
3. 跨 `train_group` 且 uncertainty 之和最高的未标注候选对（补稀疏边）。
4. 去重：跳过 `pairs.json` 已有边；同对只保留一条。

annotator 可将 `active_queue.json` 作为可选导入（pairwise-annotator spec 对齐）。

### 增量训练（F7）

```bash
python rank_train.py --config configs/rank_default.yaml \
  --resume models/rank/best.pth \
  --pairs-new data/pairs_new.json \
  --epochs 10 --lr 1e-5 --freeze-backbone-epochs 2
```

- 合并 `pairs` + `pairs-new`（或仅新边，可配置 `train_on_new_only`）。
- 加载 checkpoint 的 `state_dict`、`train_group_map`；group 表变更时扩展 Embedding（新 idx 随机初始化，旧权重保留）。
- 早停基于 val pair loss；全量重训仍可用无 `--resume` 路径。

### 结果报告（F8）

`tools/report_results.py --scores scores.json --out report/`：

- `report/sorted.json`：全库按 `score_0_100` 排序。
- `report/extreme_low.json` / `extreme_high.json`：`score_0_100 ≤ 10` / `≥ 90`。
- `report/needs_review.json`：中间区 + 高 uncertainty。
- `report/index.html`：三列表格 + 缩略图路径链接（本地 file:// 或相对路径）。

---

## 最终项目结构

```text
classify/
├── history/
│   ├── demo0.py                 # 归档，只读参考
│   ├── demo1.py
│   └── demo2.py
├── requirements.txt
├── configs/
│   └── rank_default.yaml
├── tools/
│   ├── seed_pairs.py
│   ├── validate_pairs.py
│   ├── build_groups.py
│   ├── merge_pairs.py
│   ├── export_active_queue.py
│   ├── eval_rank.py
│   └── report_results.py
├── rank/
│   ├── __init__.py
│   ├── transforms.py            # KeepRatioResizePad(fill=0)
│   ├── groups.py
│   ├── dataset.py
│   ├── model.py                 # PreferenceRanker
│   ├── train.py
│   ├── predict.py               # 含 MC Dropout
│   └── checkpoint.py
├── rank_train.py
├── rank_predict.py
├── tests/
│   ├── test_build_groups.py
│   ├── test_dataset_targets.py
│   ├── test_model_forward.py
│   ├── test_uncertainty.py
│   ├── test_active_queue.py
│   └── test_incremental_load.py
└── models/
    └── rank/
```

**另仓：** `simple_viewer` 分支 `pairwise-annotator`（F2 采集 UI）。

---

## 变更点清单

| 路径 | 操作 | 说明 |
|------|------|------|
| `demo0.py` … `demo2.py` | **移至 `history/`** | 用户确认；根目录不再保留 |
| `rank/*` | 新增 | 核心包 |
| `tools/*` | 新增 | 数据 + 主动学习 + 报告 + 评估 |
| `rank_train.py` / `rank_predict.py` | 新增 | CLI |
| `configs/rank_default.yaml` | 新增 | 含 uncertainty / active / finetune 段 |
| `.gitignore` | 更新 | `models/rank/`，本地 `pairs.json` 可选 |

---

## 详细实现步骤

### 步骤 0：仓库整理

1. 创建 `history/`，移动 `demo0.py`、`demo1.py`、`demo2.py`。
2. 添加 `requirements.txt`、`.gitignore`。

### 步骤 1：配置

`configs/rank_default.yaml` 含：

```yaml
data_root: ""
img_size: 384
batch_size: 16
embed_dim: 64
lr: 5.0e-5
lr_finetune: 1.0e-5
weight_decay: 0.01
epochs: 30
early_stop_patience: 7
val_ratio: 0.1
group:
  min_images: 5
  max_per_group: 80
condition: train_group
seed_pair_weight: 0.5
uncertainty:
  n_mc: 10
  review_percentile: 75
active_learning:
  budget: 200
  score_close_threshold: 5.0
finetune:
  freeze_backbone_epochs: 2
  train_on_new_only: false
```

### 步骤 2：数据工具链

`build_groups` → `seed_pairs` → `validate_pairs` → `merge_pairs`。

### 步骤 3：rank 核心

`transforms`、`groups`、`dataset`、`model`、`train`、`predict`（含 MC）、`checkpoint`。

### 步骤 4：训练 CLI

全量训练 + `--resume` / `--pairs-new` 增量路径。

### 步骤 5：推理 CLI

`scores.json` 字段：`path`, `author`, `group_id`, `score_raw`, `score_0_100`, `uncertainty`, `zone`（`extreme_low|mid|extreme_high`）。

### 步骤 6：评估与报告

`eval_rank.py`：pairwise accuracy、分区域（极端/中间）一致性。  
`report_results.py`：F8 三列表 + HTML。

### 步骤 7：主动学习

`export_active_queue.py`；与 annotator 联调导入格式。

### 步骤 8：端到端验收

冷启动（种子 + 手写 pairs）→ 训练 → 预测 → 报告 → 导出队列 → 模拟增量一轮。

---

## 测试策略

### 单元测试

| 用例 | 断言 |
|------|------|
| `test_build_groups_rare` | `<5` 张 → `__rare__` |
| `test_build_groups_split` | `>80` 张 → 多 part |
| `test_prefer_to_target` | a/b/tie → 1/0/0.5 |
| `test_model_cross_author` | 不同 group forward OK |
| `test_uncertainty` | MC 时 `std > 0` |
| `test_active_queue` | 跳过已有边；budget 上限 |
| `test_incremental_load` | resume 后 loss 可降 |
| `test_report_zones` | 极端/待复核集合互斥且覆盖 |

### 集成测试

1. 小数据集全流程（步骤 8）。
2. 真实 `pairs.json`：`validate_pairs` 零 ERROR。

---

## 风险与回滚方案

| 风险 | 缓解 | 回滚 |
|------|------|------|
| MC Dropout 过慢 | 可配置 `n_mc`；推理批量化 | `n_mc=1` 关闭不确定性 |
| 主动学习队列质量差 | 多策略混合 + 人工抽检 reason | 退回随机分片（annotator） |
| 增量遗忘 | 混合旧 pairs 训练；`train_on_new_only=false` | 全量重训 |
| group 规则不适配 | yaml 调参；`condition=author` 对照 | 切配置 |
| annotator 未就绪 | 种子 + `active_queue` 手写 pairs | 不阻塞 classify |

**回滚：** 删除 `rank/`、`tools/`、CLI；自 `history/` 恢复 demo 脚本。

---

## 与 pairwise-annotator 的接口

| 字段 | annotator | classify |
|------|-----------|----------|
| `pairs.json` | 写入 | 训练读取 |
| `active_queue.json` | 可选导入 | `export_active_queue` 产出 |
| `author_a/b` | 写入 | → `train_group` |

---

## 实现顺序

1. 步骤 0–1：整理 + 配置  
2. 步骤 2–3：工具链 + rank 核心 + 单测  
3. 步骤 4–5：训练 / 推理 CLI  
4. 步骤 6–7：评估、报告、主动学习  
5. 步骤 8：端到端验收；并行 pairwise-annotator spec/实现
