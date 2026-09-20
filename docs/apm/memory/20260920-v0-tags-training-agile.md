---
date: 2026-09-20 23:55
title: 敏捷迭代 v0-tags-training 完成：v0 训练管线落地并真实数据验收通过
keywords: v0, 训练管线, TagsDataset, 敏捷开发, 解析器, labels.json, 冻结骨干, mAP
abstract: v0 训练管线已完成并合并 main（6b132a0，fast-forward，分支已删）：解析器空格分隔修复 + TagsDataset/特征预计算/头训练/checkpoint/CLI。真实数据验收：453 张全解析计数对基准；端到端训练 val mAP=0.634（强 tag 0.89~0.95，小样本档低属预期）。prd/spec 落 docs/iterations/multi-label-tags/features/v0-tags-training/。遗留到下个敏捷项：全库打分/预填入口（需补 MC dropout 同图同种子）、阈值校准、增量重训、伪标签铺负面组。
---

user:
做吧，可以走敏捷迭代流程。

assistant:
敏捷执行记录（feature/v0-tags-training，迭代 multi-label-tags）：
- Step1 环境：timm 1.0.29 装好（清华镜像）；本机 4070Ti Laptop HF 直连可用（与服务器不同，无需 clash）；wd-eva02-large 1.26GB 下载完成并校验（10862 张量、头键齐全）。
- Step2 探索（双子代理）：tags 侧训练件全新建；rank 侧可复用训练循环骨架/checkpoint 协议/enable_mc_dropout/dataloader_utils；绕开 pairs 全家桶/黑填充/score_0_100；发现 rank 的 MC dropout 未实现"同图同种子"协议。
- Step3a trivial 豁免（主代理直接实现，8f3a9ea）：_SPLIT_RE 改 `[\s,，]+`。实扫 453/453 全解析，计数与记忆基准逐项一致（灵魂26/喜欢51/一般262/删除114；无背景199/漫画图78/NSFW76/小水印29/低像素23），零冲突零未知。
- Step3b 实现子代理（29435fc/7e76a05/2c65ad5）：tags/dataset.py（multi-hot+按喜好分层划分）、tags/features.py（预计算+缓存+坏图统计）、tags/train.py（AdamW+Plateau+早停、手写 AP、checkpoint 存取 weights_only 兼容）、tags_train.py CLI、configs/tags_v0.yaml、tests/test_tags_dataset.py + test_tags_train_smoke.py（resnet18 离线）。
- Step4 验收（主代理复核）：定向 25 + 全量 52 测试全绿；真实训练端到端通过——train 385/val 68（分层 seed42），EVA02-L 特征提取 4 分 35 秒（缓存 (453,1024) fp16），头训练 44/100 轮早停 best_val_loss=0.6828，val mAP=0.634（无背景 0.951/一般 0.943/漫画图 0.938/NSFW 0.892/删除 0.888；喜欢 0.429、小水印 0.301、灵魂 0.186（支持4）、低像素 0.183（支持2）为排序辅助档，低属预期）；checkpoint 可加载重建，头前向 (2,9) 有限值。
- Step5 留痕：docs/iterations/multi-label-tags/features/v0-tags-training/{prd,spec}.md（含真实提交清单与训练数字）。
- Step6 用户确认后已 fast-forward 合并 main（6b132a0）并删除 feature 分支（本地+远端），敏捷项关闭。产物（gitignored）：models/tags/v0_best.pth、data/classification/tags_v0_report.json、tags_v0_features.pt、tags.labels.json。
- 遗留到下个敏捷项：全库打分/预填入口（需补 MC dropout 同图同种子协议）、阈值校准（≥50 正样本后）、增量重训、伪标签铺负面组候选。
