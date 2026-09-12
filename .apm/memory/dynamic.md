---
createdAt: '2026-06-18 20:54:28'
updatedAt: '2026-06-18 23:10:00'
---
【环境切换·22GB 2080 Ti 台式机】

训练已在 4070 Laptop 手动停止（Epoch 1/30 约 84%）。
数据：F:/Pictures/Storage/classification（只读，1433 作者 / 26284 图）
产物：data/classification/pairs.seed.json（20000 条）、train_group_map.json、train.log
配置：configs/classification_storage.yaml（当前 batch 8；22GB 机建议 batch 32 + lr 8e-5）

下一步：2080 Ti 机同步 classify 仓库后重新 rank_train.py。
