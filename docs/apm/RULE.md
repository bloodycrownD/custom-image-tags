# 项目规则（classify）

## 项目定位
二次元图库审美评估系统。数据：作者/{good,keep,trash} 目录组织的图库（Windows 侧 F:/Pictures/Storage/classification，只读，1433 作者 / 26284 图）。

## 范式决策
- 2026-06：先弃三分类 → pairwise 偏好排序（feature/model-training-rank，EfficientNet-B4 + train_group 条件 Embedding，已实现 F1-F8）。
- 2026-06-18 晚：用户提出**转向 multi-label tag 范式**（分类转 tag，算法评估图片可能有的 tags）。理由：pairwise 对样本要求太高；tags 标注成本低（点式勾选）、无互斥约束、更准确。二次元领域先例：WD14/DeepDanbooru。
- 转向后：骨干/transforms/image_cache/训练循环/checkpoint/MC dropout 复用；pairs 全家桶（seed/merge/validate/active_queue 配对、PairsDataset）淘汰；新增 TagsDataset、per-tag pos_weight、per-tag 阈值与评估。pairwise 的 rank 代码保留在仓库中不删，但不再演进。
- 2026-09-12：tag 方案在 feature/multi-label-tags 分支开发，代码落 tags/ 包；v1 词表不含作者组（作者 base rate 走两段式先验融合，作者预测/画风相似留待 v2）。

## 约定
- 远程仓库：origin = git@github.com:bloodycrownD/custom-image-tags.git（SSH 关联，跨机开发；本机 SSH 22 端口直连可用，无需代理）。
- 数据目录只读，所有产物写入 classify 仓库 data/ 下。
- 打分输出协议以 checkpoint meta 为准（percentiles / u_threshold 等），工具读取而非重算。
- tag 预处理对齐基模官方管线（WD：alpha 白底合成、白色 pad-to-square、bicubic、448px、mean/std 0.5）；rank 的黑填充 + ImageNet 归一化仅服务于 EfficientNet-B4，两套互不通用。
- 训练环境：~/miniconda3/envs/classify（torch cu13 + timm）。网络：PyPI 走清华镜像直连，HuggingFace 走 clash 代理 127.0.0.1:7890（直连被墙）。
- 预训练基模权重放 data/pretrained/（gitignore），不进版本库。
