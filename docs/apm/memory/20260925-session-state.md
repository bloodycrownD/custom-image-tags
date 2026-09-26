---
date: 2026-09-25 04:05
title: 会话状态快照（压缩恢复锚点）：数据 1054 张、模型待重训、93583 待预填
keywords: 状态快照, 恢复锚点, 待办, 1054, 93583, v0.3
abstract: 压缩前快照——数据：finish=1054 张（953690/114299/132450 三作者已并入并人工修正，labels.json 同步；喜好 灵魂65/喜欢198/一般404/删除387；负面 无背景349/低像素75/官方图75/男角色33/大水印7）；模型：models/tags/v0_best.pth = 10 tag + 频次匹配校准阈值，但训练集仍是旧 755 张 → 待重训（特征全量重提约 12 分钟）；new/ 里 93583（480 张）待预填。关键决策：喜好纯模型预估、负面大小互斥（大负面独占）、低像素手标、元数据特征默认关闭。恢复会话读本文件 + RULE.md 即可接力。
---

user:
更新记忆吧，我要压缩了。

assistant:
（状态快照——恢复会话读本文件 + docs/apm/RULE.md 即可接力）

**数据**
- finish = **1054 张**，labels.json 已同步（data/classification/tags.labels.json）。
- 已并入并修正的作者：953690（57）、114299（245）、132450（299）；基础 453。
- 喜好分布：灵魂 65 / 喜欢 198 / 一般 404 / 删除 387。
- 负面计数：无背景 349、小水印 97、漫画图 93、NSFW 77、低像素 75、官方图 75、男角色 33、大水印 7。
- classification 已被取走 3 位作者（93583/132450/114299）；**new/ 里还剩 93583（480 张，未预填）**。

**模型**
- 生产 checkpoint `models/tags/v0_best.pth`：10 tag（含官方图）、meta_dim=0、tag_thresholds 频次匹配校准齐全；**训练集还是 755 张（132450 并入前）→ 待重训吃 1054**。
- 测试基线：`python -m pytest tests/ -q` → 84 passed。
- 词表候选：男角色 33 样本已达 ≥30 门槛（下次重训评估并入）；大水印 7 仍不够。

**待办（按顺序）**
1. 重训：`python tags_train.py --config configs/tags_v0.yaml`（缓存键含路径清单必失效 → 特征全量重提 1054 张 ≈12 分钟 + 元数据计算 ~1 分钟）。
2. 预填 93583：`python tags_predict.py --data-root "F:\Pictures\Storage\new\93583" --apply --report-out data\classification\prefill_93583.json`。
3. 用户修正 → 并入 finish → 重扫 → 下一轮重训（循环）。

**关键决策索引**（细节见对应记忆文件与 RULE.md）
- 喜好预填 = 纯模型预估（三轮拍板定稿）→ 20260925-pref-calibration.md
- 负面大小互斥：大负面（官方图/大水印）独占 → RULE.md
- 频次匹配校准阈值入 checkpoint meta → 20260925-pref-calibration.md
- 元数据特征默认关闭；低像素维持手标（模型 OOF≈瞎猜、上限 AUC 0.75）→ 20260925-meta-features-negative-result.md
- CR（两轮全量）fix-spec 已执行闭环 → 20260920-cr-full-main.md + docs/iterations/multi-label-tags/cr-fix-spec.md

**环境/坑**
- PowerShell 对含方括号文件名必须 `-LiteralPath`，批量移动/改名优先用 Python。
- 改标签后必须先重扫 labels.json 再重训（标签事实来源是文件名）。
- 特征缓存 v2（含元数据字段；关闭时训练不使用但缓存仍带）。gitignore：data/、models/ 不入库。
