# 项目规则（classify）

## 项目定位
二次元图库审美评估系统。数据：作者/{good,keep,trash} 目录组织的图库（Windows 侧 F:/Pictures/Storage/classification，只读，1433 作者 / 26284 图）。

## 范式决策
- 2026-06：先弃三分类 → pairwise 偏好排序（feature/model-training-rank，EfficientNet-B4 + train_group 条件 Embedding，已实现 F1-F8）。
- 2026-06-18 晚：用户提出**转向 multi-label tag 范式**（分类转 tag，算法评估图片可能有的 tags）。理由：pairwise 对样本要求太高；tags 标注成本低（点式勾选）、无互斥约束、更准确。二次元领域先例：WD14/DeepDanbooru。
- 转向后：骨干/transforms/image_cache/训练循环/checkpoint/MC dropout 复用；pairs 全家桶（seed/merge/validate/active_queue 配对、PairsDataset）淘汰；新增 TagsDataset、per-tag pos_weight、per-tag 阈值与评估。pairwise 的 rank 代码保留在仓库中不删，但不再演进。

## 约定
- 数据目录只读，所有产物写入 classify 仓库 data/ 下。
- 打分输出协议以 checkpoint meta 为准（percentiles / u_threshold 等），工具读取而非重算。
