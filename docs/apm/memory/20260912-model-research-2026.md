---
date: 2026-09-12 16:00
title: 2026 基模调研定论与分支/环境事实
keywords: 基模调研, wd-eva02-large-tagger-v3, camie-tagger-v2, GPL-3.0, DINOv3, clashctl, 分支, RTX 2080 Ti
abstract: 2026-09 调研结论：SmilingWolf 无公开 v4（最新为 2024-07 wd-eva02-large-tagger-v3，Apache-2.0，已有 timm 兼容 safetensors，ONNX 顾虑消除）；新秀 Camie Tagger v2（2026-01，ViT 143M，512px，67.3% micro-F1/70k tag，danbooru-2024 数据，GPL-3.0）基准更强且更适配 2080 Ti 22GB（无 bf16）。GPL-3.0 私用无义务、不分发不触发，唯一红线是将来公开 camie 衍生权重需随 GPL；WD Apache-2.0 无任何顾虑。两个基模已挂后台经 clash 代理（127.0.0.1:7890，直连被墙）下载到 data/pretrained/。分支：feature/model-training-rank 保留（3 提交），tag 方案在新分支 feature/multi-label-tags。样本硬盘暂不可用。
---

user:
当前分支保留，新建分支实现 tag 方案；clashctl 可帮助下载基模；样本硬盘暂时不在；2026 年了，检查下 HuggingFace 有没有更新更好的模型。GPL-3.0 对我们没影响吧？反正自己使用不对外。

assistant:
执行与调研结论：
- 分支：feature/model-training-rank 上未提交改动分三个提交保留（perf fp16缓存/下采样工具、docs APM接入、chore .apm残留），git 身份沿用 BloodyCrown <3170989534@qq.com>（repo级）。新分支 feature/multi-label-tags 基于其创建（复用 image_cache/训练循环/checkpoint/MC dropout）。
- 网络：clash(mihomo) 常驻，代理端口 127.0.0.1:7890，HF 直连被墙，curl 走代理 200。下载基模均需 export https_proxy=http://127.0.0.1:7890。
- 基模调研（2026-09 实查 HF API）：
  1. SmilingWolf 公开仓库共 12 个模型，无 v4（搜索结果提到的 wdtagger-v4-seed 不在公开列表，401）；最新为 wd-eva02-large-tagger-v3（2024-07-28，EVA02-L，448px）。重大更新：v3 large 系列已提供 model.safetensors 且标 timm 兼容、Apache-2.0——此前"只有 ONNX 需转换"的顾虑消除。
  2. Camie Tagger v2（Camais03/camie-tagger-v2，2026-01-01）：ViT 骨干，143M 参数（v1 的 EfficientNetV2-L 424M 缩 66%），多分辨率训练 384→512px，67.3% micro-F1 / 50.6% macro-F1（70,527 tag，danbooru-2024 数据，IRFS 稀有tag采样）；artist 识别 70% micro-F1（对将来作者组有利）；提供 safetensors+ONNX；GPL-3.0。基准显著优于 WD v3（不同词表协议不可直接比，但社区公认更强）。
  3. 对照项：DINOv3（Meta，LVD-1689M，自然图像域）通用特征强、小数据 linear probing 可比微调，但非二次元域适配，作备选不作主线。
  4. GPU 约束：RTX 2080 Ti 22GB（Turing，无 bf16，用 fp16 AMP）。EVA02-L@448 全微调偏重（batch 4-8+梯度累积+grad checkpointing 可行）；Camie v2 143M@512 更从容。
- GPL-3.0 结论：纯私用（训练/推理/管图库）零义务——GPL 义务仅在"对外分发/convey"时触发，且 GPL-3.0（非 AGPL）连对外提供网络服务都不触发源码义务。唯一红线：将来若公开 camie 衍生的微调权重/模型，需按 GPL-3.0 开源发布；届时选 WD（Apache-2.0）线发布即可。本项目权重均在 gitignore（*.pth/*.safetensors/*.onnx/data/pretrained/），仓库不会误分发。
- 已完成下载（走代理，已校验文件头为合法 safetensors）：wd-eva02-large-tagger-v3（model.safetensors 1.26GB + selected_tags.csv + config，config 确认 architecture=eva02_large_patch14_448、10861 类、mean/std=0.5，可 timm 直载）与 camie-tagger-v2（safetensors 572MB + metadata + config，架构代码需从其 app/utils 取）至 data/pretrained/。
- 环境已解决：新建 conda env `classify`（python 3.10，torch 2.14.0+cu130 CUDA 可用于 2080 Ti，timm 1.0.29）。踩坑记录：PyPI 官方源走代理仅 ~100KB/s，改清华镜像直连 ~10MB/s；HF 才需要走 clash 代理。
- 首批代码已落地（commit 3c2ad9c，feature/multi-label-tags）：tags/preprocess.py（alpha 白底合成/白 pad-to-square/bicubic/0.5 归一化/坏图白占位）、tags/model.py（timm 骨干 num_classes=0 + dropout + 线性头；mc_forward 只切 Dropout 不切 BN；load_wd_pretrained 过滤原始头键并 strict 校验）、tools/verify_wd_load.py（真实权重验证：仅丢弃 head.weight/bias 两键，骨干 1024 维特征 std=4.1 非退化，CPU 前向 8.5s）。单测 8 项全过（tests/test_tags_preprocess.py、test_tags_model.py，用 resnet18 小骨干避免下载）。RULE.md 已更新持久约定。下一步：tag 词表 v1 起草供用户审 → TagsDataset + 目录标签冷启动 → 训练循环（BCE+pos_weight+两阶段）。
- 候选策略微调：原计划"EVA02-L 主线"更新为"两候选并测"——camie v2（更强、更省显存、但 GPL 与自定义架构加载）vs wd-eva02-large（timm 直载、Apache、生态最成熟）；同验证集比 per-tag mAP/F1 拍板。样本硬盘接上后即可冷启动。
- v4-seed 确认为**数据集**而非模型（SmilingWolf/wdtagger-v4-seed，2025-04-24 公开，parquet train/val/test/rejected 划分），v4 权重未发布。
- 预处理对比（读 WD 官方 space app.py vs 本仓库 rank/transforms.py）：
  - WD 官方管线：①alpha_composite 到白底（Image.new RGBA (255,255,255)）→ 透明区域变白；②pad-to-square 用白色(255,255,255)居中；③bicubic 缩放到 448；④timm 版 mean/std=0.5、RGB（ONNX 版是 0-255 BGR，归一化烘焙进图）。无裁剪（crop_pct=1.0）。
  - rank 现状：KeepRatioResizePad 等比缩放+**黑色填充**(fill=0)、bilinear、ImageNet mean/std、.convert("RGB") 使透明 PNG 变**黑底**、384px。
  - 结论：tag 分支必须改用 WD 官方预处理（白底合成 alpha、白填充、bicubic、448、mean/std 0.5）——预处理与预训练分布一致是迁移学习纪律；黑边对 WD-init 是分布外输入，且 anime 图库透明 PNG 常见，透明→黑 vs 透明→白是实打实的域差异。KeepRatioResizePad 骨架可复用，参数化 fill=255+alpha 白底合成+改 bicubic 即可；rank 分支保持原样不演进（已冻结）。Camie v2 用 512px，预处理同样以其官方代码为准（接入时核对）。

user:
（2026-09-12 晚）样本盘仍未挂载；过节回家，改用 4070Ti Laptop 8GB 显存训练，服务器不可用。

assistant:
8GB 约束下的训练方案排序（不推翻架构，只调整实施顺序）：
- v0 首选"冻结骨干 + 预计算特征"：EVA02-L 前向一遍提特征（453 张，fp16 batch 4-8 轻松放进 8GB，CPU 也行），头在 1024 维特征上训练，秒级/epoch——与既定"冻结骨干先训头"完全一致，8GB 无压力。
- 若需轻量微调：解冻最后 1-2 个 block，fp16 AMP + batch 2-4 + 梯度累积 + 梯度检查点，8GB 可行但紧；EVA02-L 全参微调（AdamW fp32 状态约 4.8GB）留回服务器做。
- Camie v2（143M@512）在 8GB 上微调更从容，可作为假期机器上的骨干候选。
- 4070Ti Laptop 为 Ada 架构，torch cu13x 直接支持；笔记本需按 RULE.md 建 classify 环境（清华镜像装依赖）+ 从 GitHub 克隆 + data/pretrained 权重需重新下载（gitignore 不随仓库走）。
- 同日交付：TagSpaces 文件名标签解析落地（tags/vocab.py 词表、tags/filename_tags.py 解析器、tools/scan_tagspaces.py 扫描器，10 项单测过）——只认结尾方括号组+词表内标签，防动漫文件名原生方括号误判；未知结尾标签上报为疑似拼写错误。
