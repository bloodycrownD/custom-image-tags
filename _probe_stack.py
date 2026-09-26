"""决定性测试：低像素的三种信号源 5 折 CV AUC 对比（全池 54 正样本）。

- meta4：4 个元数据特征
- model：当前 checkpoint 的 低像素 概率（从 v2 特征缓存直接算，秒级）
- meta4+model：堆叠
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from tags.train import load_v0_checkpoint

doc = json.load(open(r"data\classification\tags.labels.json", encoding="utf-8"))
y = torch.tensor([1.0 if "低像素" in im["tags"] else 0.0 for im in doc["images"]])

cache = torch.load("data/classification/tags_v0_features.pt", map_location="cpu", weights_only=True)
feat, meta = cache["features"], cache["meta_features"]
mean, std = cache["meta_mean"], cache["meta_std"]

model, payload = load_v0_checkpoint("models/tags/v0_best.pth", "cpu")
tag_list = list(payload["tag_list"])
j = tag_list.index("低像素")
with torch.no_grad():
    meta_n = (meta - mean) / std
    # 注意：缓存模型 meta_norm 与当前一致
    x = torch.cat([feat.float(), meta_n], dim=1)
    probs = torch.sigmoid(model.head(x))[:, j]
print(f"样本 {len(y)}，低像素正例 {int(y.sum())}；模型 prob 分位: "
      f"p50={probs.median():.3f} p90={probs.quantile(0.9):.3f}")

meta_norm = (meta - mean) / std


def cv_auc(X: torch.Tensor, y: torch.Tensor, folds: int = 5, seed: int = 42) -> float:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]
    pos_w = torch.tensor([float((len(y) - y.sum()) / y.sum().clamp_min(1))])
    chunks = torch.chunk(torch.arange(len(X)), folds)
    aucs = []
    for i, val_idx in enumerate(chunks):
        tr = torch.cat([c for k, c in enumerate(chunks) if k != i])
        m = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(m.parameters(), lr=0.05, weight_decay=1e-3)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for _ in range(400):
            opt.zero_grad()
            lossf(m(X[tr]).squeeze(1), y[tr]).backward()
            opt.step()
        with torch.no_grad():
            s = m(X[val_idx]).squeeze(1)
        yv = y[val_idx]
        order = torch.argsort(s)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32)
        n_pos, n_neg = float(yv.sum()), len(yv) - float(yv.sum())
        if n_pos and n_neg:
            aucs.append(float((ranks[yv.bool()].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)))
    return sum(aucs) / len(aucs)


print(f"meta4       CV AUC: {cv_auc(meta_norm, y):.3f}")
print(f"model prob  CV AUC: {cv_auc(probs.unsqueeze(1), y):.3f}")
print(f"meta4+model CV AUC: {cv_auc(torch.cat([meta_norm, probs.unsqueeze(1)], 1), y):.3f}")
