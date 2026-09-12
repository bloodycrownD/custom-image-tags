---
date: 2026-09-12 11:00
title: tag 模型骨干选型：WD14 v3 初始化为主路线，EfficientNet-B4 留作 baseline
keywords: 骨干选型, WD14, DeepDanbooru, EfficientNet-B4, 迁移学习, 微调
abstract: 拍板 tag 模型骨干路线：不用 DeepDanbooru 本体（2020 年老模型，已被 WD14 系取代）；主路线用 WD14 v3 tagger 预训练权重做初始化、换自建质量 tag 头端到端微调；EfficientNet-B4（ImageNet 初始化）作为 baseline 对照，同验证集比 per-tag mAP/F1。注意 SmilingWolf 权重多为 ONNX 发布，可能需 onnx2torch 转换。
---

user:
当前如果要训练标签分类模型，你建议用什么算法模型呢？还是用现在的 EfficientNet，还是那个 D 开头的模型（DeepDanbooru）？

assistant:
建议：主路线用 WD14 v3 tagger 预训练权重初始化 + 自建质量 tag 头端到端微调；EfficientNet-B4（ImageNet 初始化）保留作 baseline。理由与要点：
- 我们的 tag 是质量维度（上色/构图/线条），不是 Danbooru 内容 tag，所以任何预训练 tagger 都不能直接用头，迁移的是"二次元域适配的特征提取器"。26k 小数据集下，域适配初值的价值大。
- DeepDanbooru 是 2020 年自定义 VGG 风格网络，早已停止演进；WD14 系（SmilingWolf，Danbooru2021 数据，v3 持续更新，ConvNeXt/SwinV2/EVA02 多骨干、448px 输入）是其现代继任者，选 WD14。
- 质量类 tag 受益于高分辨率输入（构图/线条要看细节），WD14 的 448px 正合适。
- 实施：~10k tag 的头换成 T 个质量 tag 新头（随机初始化），骨干低 LR 微调或先冻结几轮；训练循环复用 BCE + per-tag pos_weight + masked loss + MC dropout；pos_weight/阈值按自建标注重算进 checkpoint meta。
- 坑：SmilingWolf 仓库主要以 ONNX 发布权重，PyTorch 版不一定有，可能需 onnx2torch 转换或找社区 port，属一次性成本。
- 验证方式：两条路线同验证集比 per-tag mAP/F1，用数据拍板；预期 WD14-init 在小样本 tag 上优势明显。
