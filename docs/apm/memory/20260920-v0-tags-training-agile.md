---
date: 2026-09-20 23:40
title: 敏捷迭代 v0-tags-training：解析器修复完成，训练管线实现中
keywords: v0, 训练管线, TagsDataset, 敏捷开发, 解析器, labels.json, 冻结骨干
abstract: 用户拍板"做吧，走敏捷迭代"后启动 feature/v0-tags-training 分支（迭代 multi-label-tags）。已完成：环境准备（timm 1.0.29、本机 HF 直连可用无需 clash、wd-eva02 权重后台下载）、双探索子代理报告（tags 侧训练件全新建 + rank 侧可复用清单）、trivial 豁免修复解析器空格分隔（主代理直接实现，351/453 失败→0 失败），labels.json 实扫计数与记忆基准逐项一致。进行中：子代理实现 TagsDataset + v0 训练脚本。关键协议缺口：rank 的 MC dropout 未实现同图同种子。
---

user:
做吧，可以走敏捷迭代流程。

assistant:
敏捷执行记录（feature/v0-tags-training，迭代 multi-label-tags）：
- Step1 环境：timm 1.0.29 装好（清华镜像）；本机 4070Ti Laptop HF 直连 302 可用（与服务器不同，无需 clash）；wd-eva02-large 1.26GB 后台下载中。
- Step2 探索（双子代理）：tags 侧 preprocess/model/vocab 齐但训练件全新建；rank 侧可复用训练循环骨架（AdamW+Plateau+早停+best.pth）、checkpoint payload 协议、enable_mc_dropout、dataloader_utils、tools IO；绕开 pairs 全家桶/GroupMap/黑填充/score_0_100；发现 rank 的 MC dropout 未实现"同图同种子"协议（v0 若用需补）。
- Step3a trivial 豁免（主代理直接实现）：filename_tags._SPLIT_RE 由 `[，,]` 改为 `[\s,，]+`，docstring 与 RULE.md 协议描述同步为真实写盘格式（单组空格分隔）。21 项单测过（含 3 个新用例）。
- 实扫校验：453/453 全解析，计数与记忆基准逐项一致（灵魂26/喜欢51/一般262/删除114；无背景199/漫画图78/NSFW76/小水印29/低像素23），零冲突零未知。labels.json 落 data/classification/tags.labels.json。
- Step3b 进行中：子代理实现 tags/dataset.py（V0_TRAIN_TAGS 9 维 multi-hot）+ 特征预计算 + 头训练（BCE per-tag pos_weight，无头校准用阈值无关 AP）+ tags_train.py 入口 + configs/tags_v0.yaml + 离线冒烟测试。
