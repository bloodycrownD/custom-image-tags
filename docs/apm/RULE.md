# 项目规则（classify）

## 项目定位
二次元图库审美评估系统。数据：作者/{good,keep,trash} 目录组织的图库（Windows 侧 F:/Pictures/Storage/classification，只读，1433 作者 / 26284 图）。

## 范式决策
- 2026-06：先弃三分类 → pairwise 偏好排序（EfficientNet-B4 + train_group 条件 Embedding，已实现 F1-F8；rank 代码冻结留存于 main 历史）。
- 2026-06-18 晚：用户提出**转向 multi-label tag 范式**（分类转 tag，算法评估图片可能有的 tags）。理由：pairwise 对样本要求太高；tags 标注成本低（点式勾选）、无互斥约束、更准确。二次元领域先例：WD14/DeepDanbooru。
- 转向后：骨干/transforms/image_cache/训练循环/checkpoint/MC dropout 复用；pairs 全家桶（seed/merge/validate/active_queue 配对、PairsDataset）淘汰；新增 TagsDataset、per-tag pos_weight、per-tag 阈值与评估。pairwise 的 rank 代码保留在仓库中不删，但不再演进。
- 2026-09-12：tag 方案在 feature/multi-label-tags 分支开发，代码落 tags/ 包；v1 词表不含作者组（作者 base rate 走两段式先验融合，作者预测/画风相似留待 v2）。

## 约定
- 分支结构（2026-09-20 起）：main = 唯一长期分支与 GitHub 默认分支（rank 冻结代码 + tags 全部交付都在其历史中）。小步改动直接提交 main；阶段性工作可开短命 feature 分支，完成后合回并删除，历史靠 main 保留。
- 远程仓库：origin = git@github.com:bloodycrownD/custom-image-tags.git（跨机开发；笔记本侧 SSH 22 直连可用，台式机侧走 https，均指向同一仓库）。
- 数据目录只读，所有产物写入 classify 仓库 data/ 下；例外：经用户指令把整个作者文件夹从 classification 图库移入 new/（主动学习循环的取材入口，2026-09-21 起沿用）。
- 人工标注用 TagSpaces：标签写在文件名结尾的方括号组，多标签默认为单组内空格分隔（`2398_..._p0[无背景 一般].png`，逗号亦接受），无 sidecar 文件；训练数据用 tools/scan_tagspaces.py 扫描生成 labels.json，词表见 tags/vocab.py。
- 负面标签分大小（2026-09-21 确立）：大负面（官方图/大水印）**独占**——出现时不再标任何小负面；小负面（无背景/漫画图/NSFW/小水印/低像素/男角色）之间可组合。权威定义见 tags/vocab.py 的 BIG_NEGATIVE_TAGS。
- 打分输出协议以 checkpoint meta 为准（percentiles / u_threshold 等），工具读取而非重算。
- tag 预处理对齐基模官方管线（WD：alpha 白底合成、白色 pad-to-square、bicubic、448px、mean/std 0.5）；rank 的黑填充 + ImageNet 归一化仅服务于 EfficientNet-B4，两套互不通用。
- 训练环境：~/miniconda3/envs/classify（torch cu13 + timm）。网络：PyPI 走清华镜像直连，HuggingFace 走 clash 代理 127.0.0.1:7890（直连被墙）。
- 预训练基模权重放 data/pretrained/（gitignore），不进版本库。
- v0 训练入口：`python tags_train.py --config configs/tags_v0.yaml`（冻结骨干特征预计算 + 线性头 BCE）；checkpoint models/tags/v0_best.pth 的 meta（tag_list/pos_weight/img_size 等）是打分协议的事实来源，工具读取而非重算。
