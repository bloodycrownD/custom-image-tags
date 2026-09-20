# CR Fix Spec: multi-label-tags v0 tags-training 首轮修复说明书

## 元信息

- repo: `D:\Dev\Python\custom-image-tags`
- base_sha: a4982e0
- head_sha: b51b71e
- prd_path: `docs/iterations/multi-label-tags/features/v0-tags-training/prd.md`
- spec_path: `docs/iterations/multi-label-tags/features/v0-tags-training/spec.md`
- review_round: 2
- dag_version: 3
- 状态: draft
- 说明: 全量 main 分支 CR（代码首次评审）。rank 区已冻结，其发现不入本迭代 must-fix，统一进「待拍板」附录（见 Open questions #2）。

## Must-fix（按 P0 → P1 → P2）

本批无 P0。P1×4 + P2×13，共 17 条。

### [P1] tags/B-1 config 的 backbone_weights 字段被忽略，默认入口静默用随机骨干训练
- 维度: B + A
- 文件: `tags_train.py:141-147,258`、`configs/tags_v0.yaml:12`
- 问题: `run_tags_training` 的 `backbone_weights` 无 `config.get` 回退（与 `feature_cache` 的处理不对称）；按 RULE 记载入口 `python tags_train.py --config configs/tags_v0.yaml` 不带 CLI 参数时，将以随机 EVA02-L 骨干训练且无任何警告，产物只能事后从 meta 识别。
- 改法: 参数解析段补 `if backbone_weights is None: backbone_weights = config.get("backbone_weights") or None`；未加载权重分支显式打印警告「警告： 未加载骨干权重，使用随机骨干」。
- 验收/测试: 单测构造含 `backbone_weights` 的 config + CLI None，断言权重被加载（tmp 权重文件或 monkeypatch 计数）。
- 来源: review-scope-tags / round 1

### [P1] tags/B-2 TagsDataset 的 data_root 空值防御永不生效（Path("")→"."）
- 维度: B
- 文件: `tags/dataset.py:54-56`
- 问题: `str(Path(""))` 恒为 "."，ValueError 分支不可达；labels.json 缺 `data_root` 且未传参时静默以 CWD 为根 → 全部图片失败 → 全白占位仍"训练成功"。
- 改法: 先判原始字符串再 Path 化：`root_raw = data_root if data_root else doc.get("data_root", "")`；为空则 `raise ValueError("未提供 data_root，且 labels.json 内嵌 data_root 为空")`；再 `root = Path(root_raw)`。
- 验收/测试: 新增单测 `pytest.raises(ValueError)` 覆盖「无 data_root 且未传参」；现有测试显式传 data_root。
- 来源: review-scope-tags / round 1

### [P1] tools/C-1 temp.py 无引用临时死代码
- 维度: C
- 文件: `temp.py`（36 行）
- 问题: 一次性计数脚本，硬编码 Linux 路径，零引用，功能被 `tools/__init__` 与 `build_groups` 覆盖。
- 改法: `git rm temp.py`。
- 验收/测试: 全量 pytest 通过数不变（52）。
- 来源: review-scope-tools-misc / round 1

### [P1] tools/C-2 history/ 三份演示死代码（含错误映射）
- 维度: C
- 文件: `history/demo0.py`、`history/demo1.py`、`history/demo2.py`；关联 `tools/__init__.py:10` 注释
- 问题: 零引用；`demo0.py:21/26` 类别 'trash' 误写为 'demo1'；依赖未登记的 matplotlib。RULE「rank 保留不删」仅指 rank/ 包，不覆盖 history/。
- 改法: 默认 `git rm -r history/`；同步把 `tools/__init__.py:10` 注释改为自述式（不再引用 history/demo2）。备选 `git mv history docs/archive/history`（二选一，见待拍板 #1，默认删除）。
- 验收/测试: `git grep -n "history/" -- "*.py"` 为 0 处；pytest 通过数不变。
- 来源: review-scope-tools-misc / round 1

### [P2] tags/C-1 特征缓存 version 字段写入但从不校验
- 维度: C
- 文件: `tags/features.py:21,58-64,137`
- 问题: payload 写入 version 但 checks 不比对，bump 版本时旧缓存不失效。
- 改法: checks 元组增加 `payload.get("version") == FEATURES_CACHE_VERSION`。
- 验收/测试: 配合 tags/G-1 的失效重算单测覆盖。
- 来源: review-scope-tags / round 1

### [P2] tags/C-2 checkpoint_out/report_out 为死配置
- 维度: C + A
- 文件: `tags_train.py:266-267`、`configs/tags_v0.yaml:31-32`
- 问题: config 写了但 `main()` 不读，CLI default 非 None，「CLI 覆盖 config」语义未兑现（当前默认值碰巧相同）。
- 改法: CLI 两参数 `default=None`；`checkpoint_out = args.checkpoint_out or Path(config.get("checkpoint_out", DEFAULT_CHECKPOINT))`（report_out 同理）。
- 验收/测试: 单测改 config 值断言输出路径随之变化。
- 来源: review-scope-tags / round 1

### [P2] tags/C-3 TagsDataset 重复构造 tag_index
- 维度: C
- 文件: `tags/dataset.py:61,84`
- 问题: `_build_targets` 每次调用都重新构造 tag_index，存在重复计算。
- 改法: `__init__` 构造 `self._tag_index` 供 `_build_targets` 复用。
- 验收/测试: 现有 dataset 单测全过。
- 来源: review-scope-tags / round 1

### [P2] tags/G-1 特征缓存"键变更→重算"路径无测试
- 维度: G
- 文件: `tags/features.py`；对照 `tests/test_tags_train_smoke.py:138-153`
- 问题: 缓存键（img_size/version/arch）变更后触发重算的路径无测试覆盖。
- 改法: 新增单测：首跑写缓存后改 img_size（或篡改缓存 version/arch）再跑，断言重新写入缓存且 shape 变化。
- 验收/测试: 新增用例离线通过。
- 来源: review-scope-tags / round 1

### [P2] tags/G-2 失败路径无测试
- 维度: G
- 文件: `tags_train.py:104-106,121-132`、`tags/train.py:304-305`
- 问题: labels 缺失 / checkpoint 格式不符等失败路径无测试覆盖。
- 改法: 补 2-3 个轻量单测：labels 不存在断言 `rc == 2`；format 不符 .pth 断言 `pytest.raises(ValueError)`。
- 验收/测试: 新增用例离线通过。
- 来源: review-scope-tags / round 1

### [P2] tools/C-3 tools/__init__ docstring 停留 pairs 时代
- 维度: C
- 文件: `tools/__init__.py:1`
- 问题: 模块 docstring 描述与现役工具集不符。
- 改法: 更新为覆盖全部现役工具（pairs 冻结、train_group、TagSpaces 扫描、WD 验证、评估报告；JSON/路径工具被两侧共用）。
- 验收/测试: 文档检查。
- 来源: review-scope-tools-misc / round 1

### [P2] tools/C-4 conftest 死 fixture tmp_data_root
- 维度: C
- 文件: `tests/conftest.py:16-21`
- 问题: 零引用的 fixture 死代码。
- 改法: 删除（9 行）。
- 验收/测试: pytest 收集与通过数不变。
- 来源: review-scope-tools-misc / round 1

### [P2] tools/C-5 scan_tagspaces 双轨：扩展名集合分叉 + 绕开 save_json
- 维度: C + C-orch
- 文件: `tools/scan_tagspaces.py:30,105-108`
- 问题: 自建 `IMAGE_EXTS` 与共享 `IMAGE_EXTENSIONS` 分叉，且手写落盘绕开 `save_json`，双轨无 parity 保障。
- 改法: `from tools import IMAGE_EXTENSIONS, save_json`；`IMAGE_EXTS = IMAGE_EXTENSIONS | {".gif"}` + 中文注释「推理池需收录动图，rank 链不收」（不得反向把 .gif 并入共享常量）；落盘改 `save_json(doc 路径)`。
- 验收/测试: 同目录扫描 labels.json 与旧实现一致（除尾换行）；真实 453 条计数不变（见合并后 QA）。
- 来源: review-scope-tools-misc / round 1

### [P2] tools/C-6 verify_wd_load num_tags=8 过时 + 晦涩动态导入
- 维度: C
- 文件: `tools/verify_wd_load.py:35,42-43`
- 问题: `num_tags=8` 与现行 9 类词表不符；`__import__` 动态导入写法晦涩。
- 改法: `from tags.vocab import V0_TRAIN_TAGS`，`num_tags = len(V0_TRAIN_TAGS)`；顶部 `from PIL import Image` 替换 `__import__` 写法。
- 验收/测试: 有权重环境跑一次输出 logits (1,9)；离线无新增测试要求。
- 来源: review-scope-tools-misc / round 1

### [P2] tools/C-7 tests/ 缺 __init__.py，conftest 双实例导入
- 维度: C
- 文件: `tests/conftest.py:11-13`
- 问题: tests 目录无 `__init__.py`，rootdir 推断可能导致 conftest 以双实例导入。
- 改法: 新增空 `tests/__init__.py`；conftest 的 sys.path 兜底保留。
- 验收/测试: 全量 pytest 通过数不变；单文件可独立运行。
- 来源: review-scope-tools-misc / round 1

### [P2] tools/G-1 scan_tagspaces 无离线单测
- 维度: G
- 文件: `tools/scan_tagspaces.py:34-76`
- 问题: 解析逻辑（多标签、--all、conflicts、unknown_trailing_tags）零测试覆盖。
- 改法: 新增 `tests/test_scan_tagspaces.py`：空格/逗号多标签入 images、`--all` 收未标注、双喜好进 conflicts、未知尾标签进 unknown_trailing_tags、stats/vocab 字段断言。
- 验收/测试: 离线通过。
- 来源: review-scope-tools-misc / round 1

### [P2] tools/G-2 parse_scores_doc/load_json 分支无测试
- 维度: G
- 文件: `tools/__init__.py:53-64,79-100`
- 问题: list/dict/缺字段/类型错等分支与 BOM 兼容无测试覆盖。
- 改法: 新增 `tests/test_tools_json.py`：list/dict/缺字段/类型错四分支；save_json→load_json round-trip；BOM 文件可读。
- 验收/测试: 离线通过。
- 来源: review-scope-tools-misc / round 1

### [P2] tools/C-8 requirements 声明 pandas 零引用
- 维度: C
- 文件: `requirements.txt:8`
- 问题: 代码库零引用 pandas，依赖声明与实际不符。
- 改法: 删除 `pandas>=2.0`（若用户保留分析用途则改注释注明；默认删除，见待拍板 #4）。
- 验收/测试: `git grep pandas` 为空；测试不依赖。
- 来源: review-scope-tools-misc / round 1

## Spec deviations

- **open** ×1：「None 哨兵覆盖 config」部分未兑现——`backbone_weights` / `checkpoint_out` / `report_out` 三字段 config 写入但 `main()` 不读取。
  - 处置：已由 must-fix **tags/B-1**、**tags/C-2** 承接，执行后闭合；下轮 review-full 验证。

## Open questions / 待拍板

以下不阻塞本 fix-spec 的 draft 状态，需用户拍板：

1. **history/ 处置二选一**：删除（默认，对应 tools/C-2 主方案）vs `git mv history docs/archive/history` —— 需用户拍板。
2. **rank-frozen 条目 17 项**（rank 区已冻结，用户拍板整体豁免或单独立项；若立项优先 rank/B-1、rank/C-1、rank/G-1 三条 P1）：
   - [P1] rank/B-1 resume + 显式 `--build-groups` 时 train_group 索引静默错位（`rank/checkpoint.py:107-115`、`tools/build_groups.py:43-72`；改法：merge 以旧索引为准、新组从 max+1 分配、冲突告警 + 一致性单测）
   - [P1] rank/C-1 预处理管线 4 处手写重复无 parity 测试（`rank/transforms.py:45-53` 为正主；`rank_train.py:45-61`、`rank_predict.py:32-40`、`tools/eval_rank.py:29-37` 重复；IMAGENET 常量重复 5 处；改法：统一调 `build_deterministic_transform` + parity 测试）
   - [P1] rank/C-2 死代码四处（`tools/build_groups.py:114-162` resolve_image_group、`rank_train.py:45-61` build_transform、`rank/transforms.py:66-74` pil_augment_then_tensor、`tools/__init__.py:43-45` count_author_images）
   - [P1] rank/G-1 测试缺口清单（compute_pair_weight 全分支、raw_to_score_0_100 边界、merge_pairs 冲突优先级、seed_pairs._prefer_for_labels、eval_rank 零覆盖、split_train_val/_percentile_value/compute_percentiles）
   - [P2] rank/B-2 seed_pairs 交换分支污染循环变量（`tools/seed_pairs.py:106-108`，当前不可达）
   - [P2] rank/B-3 merge_pairs 静默丢弃非法条目（`tools/merge_pairs.py:41-47`，补 skipped_invalid 计数）
   - [P2] rank/B-4 eval_rank tie_eps=0.05 硬编码未标定（`tools/eval_rank.py:85-91`，提为 `--tie-eps` 并记录 meta）
   - [P2] rank/B-5 GroupMap.num_groups author 模式低估（`rank/groups.py:55-64`）
   - [P2] rank/C-3 `_resize_group_embedding` 冗余 no-op 赋值（`rank/checkpoint.py:78-80`）
   - [P2] rank/C-4 merge_pairs 优先级两分支同体（`tools/merge_pairs.py:60-63`，合并为 `>=`）
   - [P2] rank/C-5 load_scores 双实现（`tools/report_results.py:18-21` vs `export_active_queue.py:18-22`）
   - [P2] rank/C-6 rank_train 重复解析 JSON 与绕路取数（`rank_train.py:368-380,217-233`）
   - [P2] rank/C-7 num_workers 打印值与生效值可能不一致（`rank_train.py:386-391`）
   - [P2] rank/C-8 私有方法访问 + 黑图占位三处重复（`rank/predict.py:163,65-68`、`tools/eval_rank.py:61-66`）
   - [P2] rank/C-9 重载自身 checkpoint 仍传 pretrained=True（`rank_train.py:491`，改固定 False）
   - [P2] rank/A-1 partition_report_zones u_threshold 回退重算与 RULE 17 张力（`tools/report_results.py:56-58`，回退路径加 WARNING）
   - [P2] rank/G-2 测试内死变量（`tests/test_image_cache.py:36`）
3. **AP ties 措辞**：average_precision docstring 与 sklearn 并列分数语义差异，建议改措辞，两可。
4. **pandas 去留**（tools/C-8 默认删除，若保留交互分析用途改注释）。
5. **labels.json version 无消费方校验**（v0 可接受，v2 演进时拍板）。
6. **labels.json 内嵌 data_root 为本机绝对路径**（跨机需 `--data-root` 覆盖，可接受待拍板）。
7. **两套图片遍历排序键不一致**（scan 码点序 vs iter_images_under 的 lower()，未认定）。
8. **仓库根无 README**（观察项）。
9. **CLI 覆盖面不对称**（weight_decay/dropout 无 CLI 参数，两可）。
10. **[E] export_active_queue 三次 O(N²) 全对遍历**（round 2 终审新增；`tools/export_active_queue.py:134,152,180`，26k 图 ≈ 10 亿次迭代，策略 2 候选无截断有 OOM 风险；属 rank 冻结工具链，随 #2 一并拍板。改法：策略 1 预过滤子集后组合、策略 2 极端区×高不确定子集笛卡尔、策略 3 堆取 top，语义不变）。
11. **[C-orch 观察] tags↔tools 包级环**（`tags/dataset.py:21` import tools 与 `tools/scan_tagspaces.py:21` import tags 形成模块级环，当前无运行时问题；执行 tools/C-3 时在 docstring 写明分层意图：tools/__init__ 为无业务依赖共享层、禁止 import tags/rank，不改结构）。
12. **[E 观察] count_bad_images 缓存未命中时全量二次解码**（探测一遍 + DataLoader 一遍；453 张 4'35'' 可接受，图量级增长时合并为单次解码）。

## 已豁免（用户确认不修）

- 暂无。待拍板项（含 rank-frozen 17 条）经用户决策后如有不修者，移入本节并附用户原话。

## 合并后 QA（manual_user）

- tools/C-5 执行后真实重扫一次 labels.json，对比 453 条计数不变。
- tags/B-1 修复后用 `configs/tags_v0.yaml` 默认入口重跑一次，确认加载权重与警告逻辑。

## K 节建议（下游执行时闭合）

- 执行 must-fix 时同步跑 `python -m pytest tests/ -q` 全量回归。
- tags/B-1、tags/C-2 涉及 CLI 行为变化，执行后更新 `docs/iterations/multi-label-tags/features/v0-tags-training/spec.md` 入口描述（如与实现不符）。
- 中文对齐样式（`_fmt_summary_line` `{tag:<6}`）可选优化，随 P2 批次顺手处理。
