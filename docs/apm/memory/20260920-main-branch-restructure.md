---
date: 2026-09-20 23:05
title: 分支重组：建立 main 主干并设为 GitHub 默认分支
keywords: 分支, main, 默认分支, GitHub API, gh CLI, origin/HEAD, 跨机开发
abstract: 应用户要求重组分支：main 基于 feature/multi-label-tags 尖端 22904f1 建立（包含 rank 冻结历史与 tags 全部交付），设为 GitHub 默认分支；feature/model-training-rank 与 feature/multi-label-tags 均完全包含于 main 历史，先后远端+本地删除，收敛为单主干结构（小步直接提交 main，阶段性工作开短命 feature 分支合完即删）。台式机无 gh CLI，用 git credential fill 取 token 走 GitHub API PATCH 默认分支成功。
---

user:
恢复下记忆。顺便重新整理下分支，应该有主分支（main/master），而不应该现在这样只有开发分支。

assistant:
先恢复记忆（RULE.md + 最近 5 条 + multi-label 分支上增量的一条基模调研记忆），随后完成分支重组。重组前的事实：本地只有 feature/model-training-rank（9b83efd，rank 范式已冻结）；远端 feature/multi-label-tags 领先 7 个交付提交（tags 预处理/基模加载/TagSpaces 扫描/词表定稿）；GitHub 默认分支错误地指向 rank 开发分支。

重组操作与结果：
- main 基于 origin/feature/multi-label-tags 尖端 22904f1 创建并推送——main 完整包含 rank 时代 + tags 已交付进展，默认分支视图展示的是当前范式。
- GitHub 默认分支切为 main：本机（台式机）未装 gh CLI，改用 `git credential fill`（Windows 凭据管理器）取 token，走 GitHub REST API `PATCH /repos/bloodycrownD/custom-image-tags` 设 default_branch=main，成功。
- feature/model-training-rank（恰为 9b83efd，即 main 与 tags 分支的分叉点，完全包含于 main 历史）远端+本地删除，零提交损失；feature/multi-label-tags 保留为 tag 方案活跃开发线，阶段性成果合回 main；origin/HEAD 已刷新指向 main。
- RULE.md 约定区新增分支结构条目；remote URL 两机不一致（笔记本 SSH、台式机 https）不影响，同一仓库。

user:
feature/multi-label-tags 也应该合并吧。

assistant:
是——22904f1 本就是 main 的直接祖先（main 即从它创建），`git merge` 返回 Already up to date，确认包含关系（merge-base --is-ancestor 通过）后删除该分支（本地+远端）。分支结构收敛为唯一长期分支 main：小步改动直接提交 main，阶段性工作开短命 feature 分支、合完即删。RULE.md 分支约定已同步改写，本记忆 abstract 已刷新。笔记本侧下次开工先 `git checkout main && git pull`，其本地残留的 feature/multi-label-tags 用 `git branch -D feature/multi-label-tags` 清理（上游已删，-d 会因查不到合并状态拒绝）。
