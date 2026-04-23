# Benchmark Refine Log 20260423 Round 8

## Scope

- This round finishes the residual weak-evidence cleanup in `benchmark_release/task_refined.jsonl`.
- Focus: the last 23 tasks that had still been marked `provisional_low_evidence`.
- Main actions:
  - replaced placeholder or weakly grounded gold answers
  - rewrote a few structurally broken prompts into answerable forms
  - supplemented or corrected supporting paper sets where the prior evidence pack was off-topic
  - aligned audit status with the new evidence quality instead of keeping a residual provisional bucket

## Source And Output

- Source release path: `benchmark_release/tasks.jsonl`
- Refined output path: `benchmark_release/task_refined.jsonl`
- Script used: `scripts/optimize_remaining_tasks_round8_20260423.py`

## Status Counts

- Before round 8:
  - `validated_with_refinement`: 310
  - `substantially_corrected`: 89
  - `provisional_low_evidence`: 23
- After round 8:
  - `validated_with_refinement`: 322
  - `substantially_corrected`: 100
  - `provisional_low_evidence`: 0

## Task Groups Updated

- Diffusion / visual generation bottlenecks:
  - `RTLv3-0388`
  - `RTLv3-0396`
  - `RTLv3-0397`
  - `RTLv3-0403`
  - `RTLv3-0404`
  - `RTLv3-0405`
  - `RTLv3-0406`
  - `RTLv3-0407`
  - `RTLv3-0541`
- Video / agent / retrieval bottlenecks:
  - `RTLv3-0452`
  - `RTLv3-0508`
- IE / post-training / biomedical / graph / dialogue bottlenecks:
  - `RTLv3-0478`
  - `RTLv3-0480`
  - `RTLv3-0483`
  - `RTLv3-0504`
  - `RTLv3-0505`
  - `RTLv3-0511`
  - `RTLv3-0514`
  - `RTLv3-0515`
  - `RTLv3-0516`
- Venue-aware tasks:
  - `RTLv3-EXP-VENUE-1113`
  - `RTLv3-EXP-VENUE-1158`
  - `RTLv3-EXP-VENUE-1159`

## Notable Repairs

- `RTLv3-0403`: replaced an implausible compute-cost answer with a metric-validity bottleneck for CLIP-based evaluation, and swapped in metric-specific historical citations.
- `RTLv3-0452`: replaced the weak “limited internal knowledge” answer with a much stronger dynamic evidence-selection bottleneck for long-video reasoning.
- `RTLv3-0508`: replaced a placeholder gold answer and corrected the future evidence pack toward `FastV-RAG` and `R4`.
- `RTLv3-EXP-VENUE-1113`: inherited the stronger zero-shot planning evidence chain from adjacent tasks and upgraded the direction forecast.
- `RTLv3-EXP-VENUE-1158` and `RTLv3-EXP-VENUE-1159`: fixed a structural prompt defect where the question said “rank only the listed options” even though no options were actually listed.

## Files Updated

- `benchmark_release/task_refined.jsonl`
- `scripts/optimize_remaining_tasks_round8_20260423.py`
- `docs/benchmark_refine_log_20260423_round8.md`

## Caveats

- Zero provisional tasks remain, but some items still intentionally stay at `substantially_corrected` rather than `validated_with_refinement` because their near-term future-realization evidence is only moderate or weak.
- The remaining weaker items are now weak in a transparent way: the question and gold answer have been rewritten conservatively instead of being left as overconfident forecasts.
