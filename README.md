# custom-image-tags

二次元图库的 multi-label 标签分类系统：扫描 TagSpaces 文件名标注生成训练集，用冻结的 WD 预训练骨干 + 线性头做 per-tag 多标签训练与评估。

## 环境安装
```bash
pip install -r requirements.txt
```
预训练骨干权重（gitignore，不入库）：从 HuggingFace `SmilingWolf/wd-eva02-large-tagger-v3` 下载 `model.safetensors`，放到 `data/pretrained/wd-eva02-large-tagger-v3/` 下。

## 入口命令
```bash
python tools/scan_tagspaces.py <图库根目录> -o data/classification/tags.labels.json  # 扫描 TagSpaces 标注
python tags_train.py --config configs/tags_v0.yaml  # tags v0 训练（现行主线）
python rank_train.py  # rank 排序训练（已冻结，仅留存历史，不再演进）
```

## 测试
```bash
python -m pytest tests/ -q
```

## 更多文档
- 项目规则与数据约定：`docs/apm/RULE.md`
- 迭代过程留痕（PRD/SPEC/CR）：`docs/iterations/`
