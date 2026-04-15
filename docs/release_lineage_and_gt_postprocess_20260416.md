# Release Lineage And GT Postprocess — 2026-04-16

这份文档只回答两类问题：

1. 当前 full-release 数据集是怎么一步步来的
2. `future GT / support` 去重到底对哪些任务做过

---

## 1. 当前关键 release 谱系

### 1.1 `benchmark_full_curated_polished`

路径：
- `data/releases/benchmark_full_curated_polished/`

已知状态：
- `task_count = 422`
- `dropped_task_count = 111`

这一步做的主要是：
- 从更早的 `benchmark_full_curated` 继续清理
- 删除一批不稳定的 strategic agenda tasks
- 做 venue repair
- 做 language polish

注意：
- `422` 已经不是原始 full release
- 它本身已经经过一轮 curated 清理

---

### 1.2 从 `422` 按 strict 规则筛出 `251`

这一步对应的是：
- 从 `benchmark_full_curated_polished` 中筛出 family 关键 GT 结构非空的任务

当时的 strict 规则是：
- `bottleneck_opportunity_discovery` 要求 `ground_truth.future_descendants` 非空
- `direction_forecasting` 要求 `ground_truth.emergent_descendants` 非空
- `strategic_research_planning` 要求 `ground_truth.direction_records` 非空
- `venue_aware_research_positioning` 要求 `ground_truth.direction_records` 非空

相关脚本：
- `scripts/build_release_subset_by_task_ids.py`

所以：
- `251` 的本质是“从整理后的 `422` 里，再把 family 关键 GT 为空的任务筛掉”
- 如果把你口头里的“空 GT / 空 future”理解成这些关键未来 GT 结构为空，那么 `422 -> 251` 这个理解是对的

但要注意：
- 这一步不是 `future novelty` 语义去重
- 它只是“关键 GT 结构非空筛选”

---

### 1.3 `benchmark_full_curated_recovered21`

路径：
- `data/releases/benchmark_full_curated_recovered21/`

已知状态：
- `task_count = 443`
- `recovered_task_count = 21`
- `remaining_dropped_task_count = 90`

这一步做的主要是：
- 在 `benchmark_full_curated_polished` 基础上，回捞 `21` 个原先被删除、但其实仍保留 `candidate_directions` / `direction_records` 的 q1 strategic tasks

对应 strict 数量：
- 从 `251` 增长到 `272`

---

### 1.4 `benchmark_full_curated_recovered21_bottleneck18`

路径：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18/`

已知状态：
- `task_count = 443`

这一步没有新增 task，而是：
- 修了 `18` 个 bottleneck task 的 `future_descendants`
- 同步回填了：
  - `realized_opportunity_directions`
  - `public_metadata.future_themes`

相关 README 已明确写了：
- 只修“empty `future_descendants` but non-empty `historical_future_work_cluster`” 的 bottleneck tasks

---

### 1.5 `benchmark_full_curated_recovered21_bottleneck18_expanded75`

路径：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75/`

已知状态：
- `task_count = 518`
- `strict_task_count = 365`
- `strict_quarter_task_count = 105`
- `added_task_count = 75`

这一步做的主要是：
- 在 `recovered21_bottleneck18` 基础上
- 从 q1 / cluster judged pools 加入 `75` 个 strict-ready 候选

---

### 1.6 `benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover`

路径：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover/`

已知状态：
- `task_count = 528`
- `strict_task_count = 375`
- `added_task_count = 10`

这一步做的主要是：
- 在 `expanded75` 基础上
- 针对 venue bucket coverage 再补 `10` 个 venue tasks

这就是当前默认使用的 full-release 版本。

---

## 2. `future GT / support` 去重到底做过什么

### 2.1 做过的处理是什么

相关脚本：
- `scripts/postprocess_gt_future_novelty.py`

这个脚本做的不是“空 GT 筛选”，而是：
- 把 `future` 侧那些只是重述 `history / support` 的未来 idea 从 semantic GT 里删掉
- 同时过滤一部分与历史太像的 future validation papers

它拿来作为历史锚点的字段包括：
- `public_metadata.topic`
- `public_metadata.topic_title`
- `support_context.top_limitations`
- `support_context.top_future_work`
- `support_context.candidate_directions`
- `support_context.history_chain`
- `ground_truth.historical_limitation_signals`
- `ground_truth.historical_future_work_signals`

它拿来作为 future labels 的字段包括：
- `ground_truth.future_descendants`
- `ground_truth.emergent_descendants`
- `ground_truth.realized_opportunity_directions`
- `ground_truth.direction_records`
- `public_metadata.future_themes`

它会真正改写：
- `public_metadata.future_themes`
- `ground_truth.future_descendants`
- `ground_truth.emergent_descendants`
- `ground_truth.realized_opportunity_directions`
- `ground_truth.direction_records`
- `ground_truth.reference_papers.future_q4`
- `ground_truth.reference_papers.future_q1`
- `support_context.future_validation_set`

---

### 2.2 这个处理明确做到了哪些任务上

从当前磁盘上的 manifest 看，明确带有 `future_novelty_postprocess` 记录的 release 只有一个：

- `data/releases/benchmark_core100_future_novelty_v1/`

它的 source 是：
- `data/releases/benchmark_core100/`

也就是说：
- 这一步明确只对一个 `100` 题子集做过
- 不是对整个 `422`
- 也不是对整个 `251`
- 更不是对当前 `375` full release

---

### 2.3 这个处理在这 100 题上作用了多少 task

根据：
- `data/releases/benchmark_core100_future_novelty_v1/manifest.json`
- `data/releases/benchmark_core100_future_novelty_v1/future_novelty_dedup_report.json`

明确数字是：
- `task_count = 100`
- `tasks_touched = 35`
- `removed_future_labels = 81`
- `tasks_with_future_paper_prune = 24`
- `fallback_tasks = 19`

所以如果问题是：

“你这个对 future GT 的去重，到底对多少 task 做过？”

当前能被 artifact 明确支持的答案只有：

- 它明确跑过的版本是一个 `100` 题 release
- 在这 `100` 题里，真正被改动到的 task 是 `35` 个

---

## 3. 哪些说法是对的，哪些不严谨

### 可以说的

- `422 -> 251` 主要是把 family 关键 GT 结构为空的任务筛掉
- 我们确实做过一版 `future GT / support` 去重
- 那版 `future GT / support` 去重明确跑在一个 `100` 题子集上

### 不能直接说的

- “整个 `251` 都已经做过 future novelty 去重”
- “当前 `375` full release 已经做过 future novelty 去重”

从当前 manifest / report 证据看，这两句都不能成立。

---

## 4. 当前最稳妥的口径

如果之后要统一口径，建议固定用下面三层概念，不要混说：

- `curated422`
  - 指 `benchmark_full_curated_polished`
- `strict251`
  - 指从 `curated422` 按 family 关键 GT 非空规则筛出来的子集
- `future_novelty_processed`
  - 只指真正跑过 `postprocess_gt_future_novelty.py` 的 release

当前明确属于第三类的只有：
- `benchmark_core100_future_novelty_v1`

当前默认 full release：
- `benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover`

它现在是：
- `task_count = 528`
- `strict_task_count = 375`
- 但**还没有** manifest 级证据表明它做过 `future novelty` 后处理
