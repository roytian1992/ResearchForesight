# Benchmark Rebuild Log — 2026-04-16

这份日志只记录 benchmark/release 处理，不记录方法实验结果。

---

## Step 0. Current Base Before Correction

基线 release：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover/`

基线状态：
- `task_count = 528`
- `strict_task_count = 375`

说明：
- 这个版本已经做过 venue coverage 补题
- 但在这一步之前，还没有对整套 release 做 `future GT / support` novelty cleanup

---

## Step 1. Run Future GT / Support Novelty Cleanup

脚本：
- `scripts/postprocess_gt_future_novelty.py`

输入 release：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover/`

输出 release：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover_future_novelty_v1/`

结果摘要：
- `task_count = 528`
- `tasks_touched = 237`
- `removed_future_labels = 392`
- `tasks_with_future_paper_prune = 146`
- `fallback_tasks = 187`

对应 artifact：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover_future_novelty_v1/manifest.json`
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover_future_novelty_v1/future_novelty_dedup_report.json`
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover_future_novelty_v1/future_novelty_dedup_rows.json`

---

## Step 2. Recompute Strict After Novelty Cleanup

按 metrics 对应 GT 字段重新筛 strict：
- `bottleneck_opportunity_discovery` -> `ground_truth.future_descendants`
- `direction_forecasting` -> `ground_truth.emergent_descendants`
- `strategic_research_planning` -> `ground_truth.direction_records`
- `venue_aware_research_positioning` -> `ground_truth.direction_records`

输出文件：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover_future_novelty_v1/strict_task_ids.txt`
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover_future_novelty_v1/strict_summary_after_future_novelty.json`

重算结果：
- novelty cleanup 前：`strict = 375`
- novelty cleanup 后：`strict = 373`

family 分布：
- bottleneck: `89`
- forecasting: `94`
- strategic: `129`
- venue: `61`

horizon 分布：
- half_year: `265`
- quarter: `108`

---

## Step 3. Strict Loss From Novelty Cleanup

novelty cleanup 后从 strict 中掉出的 task 只有 `2` 个：

1. `RTLv3-0115`
   - family: `strategic_research_planning`
   - title: `Prioritizing Near-Term Research Agendas in Multimodal Instruction Tuning`

2. `RTLv3-EXP-VENUE-1142`
   - family: `venue_aware_research_positioning`
   - title: `Strategic Prioritization of Multimodal Instruction Tuning for AAAI Venue Alignment`

结论：
- 这次按正确顺序修完后，并没有把 benchmark 打崩
- 当前 `strict = 373`，已经高于“至少 350”的底线
- 如果后面目标改成 “400+ strict”，那才需要继续补题

---

## Step 4. Operational Decision

当前建议：
- 先把 `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover_future_novelty_v1/` 视为新的正确顺序版本
- 后续所有 strict / benchmark 数量都以这个 release 为准
- 如果继续补题，必须从这个 novelty-cleaned release 往上补，不要再回到旧的 `375` 版继续加

---

## Step 5. Process Fix

为避免再次出现顺序错乱，已新增 Codex skill：
- `~/.codex/skills/researchforesight-experiment-hygiene/`

作用：
- 固定 benchmark 修复顺序
- 强制区分 `task_count` / `strict_task_count` / `future-novelty-processed`
- 强制把 release 变更和实验运行记录写入 markdown
