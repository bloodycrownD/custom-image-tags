---
date: 2026-09-20 23:59
title: 全量 CR 完成：17 条 must-fix 入 fix-spec，待用户确认承接与拍板
keywords: code-review-loop, CR, fix-spec, must-fix, rank-frozen, 待拍板
abstract: 用户要求对整个 main 分支（代码从未 CR）走 code-review-loop full 模式。两轮完成：round1 三 scope 并行（tags 7 条 / tools 10 条 / rank scope-ready 且 17 项按冻结约定转待拍板），round2 review-full 终审建议 fix-spec-ready=yes。产物 docs/iterations/multi-label-tags/cr-fix-spec.md（17 条 must-fix：P0=0/P1×4/P2×13，每条含文件+改法+验收）。最重要发现 tags/B-1：config 的 backbone_weights 被入口忽略，默认入口会静默随机骨干训练。状态=待用户确认：①spec_deviation 承接方案（tags/B-1+C-2 修复后闭合）②12 项待拍板（history 处置、rank-frozen 17 条豁免或立项、export_active_queue O(N²) 等）。确认前不改代码。
---

user:
对整个 main 分支继续 CR，因为代码本身就没 CR 过。

assistant:
code-review-loop full 模式执行记录（base a4982e0 → head b51b71e，全量文件评审）：
- Round 1 三 scope 并行：review-scope-tags（not-ready，7 must-fix）、review-scope-rank（scope-ready，must-fix 通道空——全仓 import 链核实 rank 不污染 tags；17 项发现全按"rank 冻结"约定转待拍板）、review-scope-tools-misc（not-ready，10 must-fix）。
- Round 2 review-full 终审：17/17 落盘无遗漏无篡改、四要素齐备；B-K 补查无新 must-fix（D 安全基本 N/A、F 中文注释无整段英文遗漏、无调试残留）；新增 3 条待拍板（export_active_queue O(N²)×3 有 OOM 风险、tags↔tools 模块级环观察、count_bad_images 双解码观察，已由主代理 trivial 豁免补入 fix-spec 附录 #10-12）。
- 关键发现（P1×4）：tags/B-1 backbone_weights 被 config 忽略→默认入口静默随机骨干训练（RULE 记载的标准入口命令即踩中）；tags/B-2 data_root 空值防御不可达（Path("")→"."，错根时全白占位仍"训练成功"）；tools/C-1 temp.py 死代码；tools/C-2 history/ 死代码（demo0 类别映射还有错）。
- fix-spec：docs/iterations/multi-label-tags/cr-fix-spec.md，元信息 review_round=2/dag_version=3。
- 状态：待用户确认（spec_deviation 承接方案确认后即可宣布 fix-spec-ready；12 项待拍板并行等决策）。确认后另开实现任务执行，本 skill 不改代码。
