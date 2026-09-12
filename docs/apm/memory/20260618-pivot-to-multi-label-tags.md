---
date: 2026-06-18 23:45
title: 范式转向：pairwise 排序 → multi-label tag 分类
keywords: 范式转向, multi-label, tags, pairwise, WD14, 标注
abstract: 用户拍板换思路：pairwise 对样本要求太高，改用 multi-label tag（评估图片可能有的 tags）。分析了迁移成本与新核心工作（tag 词表设计），下一步是起草 tag 词表。
---

user:
我想换思路了，感觉（pairwise）这种算法对于样本要求太高了。我们之前处理过三分类、五分类，我们把分类转换为 tag，我理解算法评估一个图片可能有的 tag，感觉更准确，最主要的是样本好处理，人工好打标。

assistant:
确认转向 multi-label tags 范式。要点：
- 先例：WD14/DeepDanbooru 证明 multi-label 在二次元图上非常有效。
- 之前三/五分类"不准"的根因是互斥约束（上色好+构图差只能塞一个桶=噪声），不是分类范式本身；multi-label 解除互斥。
- 标注从成对比较降为点式勾选，每图提供 T bit 监督信号，样本要求大幅下降。
- 代码迁移：骨干/transforms/image_cache/dataloader/checkpoint/训练循环/MC dropout 全部复用；pairs 全家桶淘汰；新增 TagsDataset、per-tag pos_weight（类别不平衡）、per-tag 阈值校准、per-tag F1/AUPRC/mAP 评估；可选 WD14 预训练权重替代 ImageNet 初始化。
- 新核心工作=tag 词表设计（20~50 个质量维度 tag，每个一句话定义防标注漂移）；现有 good/keep/trash 目录可映射粗 tag 冷启动。
- 若仍需全局分数：tag 概率加权聚合；但 tags 本身即可作为筛图产品。
- 下一步：起草质量维度 tag 词表供用户审，拍板后再动代码。
