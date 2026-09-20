---
date: 2026-09-20
dependency: []
---

# v0-tags-training Feature PRD

## 背景与变更动机

multi-label tags 范式（2026-06-18 拍板转向）已具备全部前置条件：453 张人工标注图落在
`F:\Pictures\Storage\new\finish`（TagSpaces 文件名标签），词表 v1 定稿（喜好 4 互斥 + 负面 8，
v0 训练 9 tag），wd-eva02-large-tagger-v3 基模选型完成。记忆中已判定"453 张可开训 v0"，
目标是**辅助打标的基础模型**——不要求非常准，只要预填准确率能降低后续人工打标成本。

同时实扫发现一个阻断缺陷：TagSpaces 实际写盘为单方括号组内**空格**分隔
（`2398_..._p0[无背景 一般].png`），原解析器只按逗号拆组，351/453（77.5%）文件解析失败。

## 范围说明（相对原需求）

- 修复 TagSpaces 空格分隔解析缺陷（内含 bug 修复，trivial 豁免由主代理直接实现）
- 新建 v0 训练管线：TagsDataset + 冻结骨干特征预计算 + 线性头 BCE 训练（per-tag pos_weight）
- 产出 checkpoint（含重建元信息）与评估报告
- **不含**：全库打分/推理入口（tools 预填循环）、MC dropout 打分种子协议、阈值校准
  （小样本档不做阈值判定）、增量重训——留待后续敏捷项

## 影响模块与接口

- `tags/filename_tags.py`：`_SPLIT_RE` 分隔符扩展（`,` `，` → 空格+逗号），输出协议不变
- `tags/dataset.py`（新）：`TagsDataset`、`stratified_split_tags`
- `tags/features.py`（新）：`precompute_features`（缓存+坏图统计）、`FeatureDataset`
- `tags/train.py`（新）：`train_head`、`compute_pos_weight`、`average_precision`、
  `compute_tag_metrics`、`save_v0_checkpoint` / `load_v0_checkpoint`
- `tags_train.py`（新，根入口 CLI）+ `configs/tags_v0.yaml`（新）
- `tags/__init__.py`：导出新 API
- rank/ 包、tags/vocab.py、tags/preprocess.py、tags/model.py：不动

## 验收标准

1. 解析失败归零：453/453 全解析，计数与人工标注基准逐项一致
   （灵魂26/喜欢51/一般262/删除114；无背景199/漫画图78/NSFW76/小水印29/低像素23）
2. 真实数据端到端训练跑通：产出 checkpoint 与报告，早停生效
3. 全部测试离线通过（不下载权重、不依赖真实数据盘）
4. checkpoint 可独立加载重建（`weights_only=True`），meta 含词表/pos_weight/协议信息

## 测试用例

- 解析器：空格分隔 2/3 标签、混合分隔符、真实文件名形态（3 个新用例）
- TagsDataset：multi-hot 编码、喜好缺失/冲突跳过、词表外标签不入向量、分层划分守恒与可复现
- 训练冒烟：resnet18 小骨干 + 假 labels.json + 函数级入口调用，断言
  checkpoint/报告/重建前向/特征缓存命中/坏图探测/pos_weight 与 AP 手算对拍
