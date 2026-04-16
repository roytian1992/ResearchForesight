# Added87 Future Audit — 2026-04-16

## Goal

Run a lightweight LLM future-novelty audit on the `87` evidence-chain-derived tasks only, without modifying the current expanded full release.

## Source Release

- expanded full release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- added-task subset:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002_evidenceexp_v1_added87_subset`
- subset `task_count`:
  - `87`

## Audit Target

- families covered:
  - `venue_aware_research_positioning`
  - `strategic_research_planning`
- subtypes covered:
  - `venue_aware_direction_forecast`
  - `venue_targeted_planning`
  - `comparative_opportunity_prioritization`

## Script And LLM Setup

- script:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/postprocess_gt_future_novelty_llm.py`
- LLM configs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8002.local.yaml`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8002.local.yaml`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8002.local.yaml`
- workers:
  - `6`

## Runtime

- tmux session:
  - `rf_added87_future_audit_20260416`
- log:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/logs/added87_future_audit_20260416.log`
- output release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002_evidenceexp_v1_added87_subset_futureaudit_v1`

## Why This Audit Is Isolated

- The current expanded full release remains unchanged.
- The audit runs on an extracted `87`-task subset only.
- If the audit shows negligible removals, we keep the expanded release as-is.
- If the audit shows real future/history novelty issues, we can patch only the derived additions and then rebuild the expanded release cleanly.

## Status

- first launch on `futureaudit_v1` was stopped early
- stop reason:
  - comparative tasks used synthetic `A vs B` topic titles in `public_metadata`
  - the audit script was incorrectly treating those synthetic comparative titles as historical labels
  - this caused false duplicate flags on the paired future labels
- fix applied:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/postprocess_gt_future_novelty_llm.py`
  - synthetic comparative topic titles are now excluded from `extract_history_labels()`
- relaunched:
  - tmux session: `rf_added87_future_audit_20260416`
  - output release:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002_evidenceexp_v1_added87_subset_futureaudit_v2`
  - log:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/logs/added87_future_audit_v2_20260416.log`
- early spot check after the fix:
  - comparative tasks no longer show the previous systematic false removals
- final status:
  - completed
- final output release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002_evidenceexp_v1_added87_subset_futureaudit_v2`
- final summary:
  - `task_count = 87`
  - `tasks_touched = 1`
  - `removed_future_labels = 1`
  - `tasks_cleared_to_empty = 0`
  - `tasks_using_llm = 54`
  - `heuristic_candidate_labels = 71`
  - `llm_reviewed_label_count = 71`
  - `auto_kept_label_count = 72`
- only removal observed:
  - task: `RTLv3-ECS-0029`
  - removed label: `Text to video generation`
  - matched kept label: `Video generation and manipulation`
  - judge code: `remove_too_generic`
- manual follow-up:
  - user chose to keep `Text to video generation`
  - manual-override release:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002_evidenceexp_v1_added87_subset_futureaudit_v3_manualkeep`
  - override scope:
    - only `RTLv3-ECS-0029`
    - restored future label: `Text to video generation`
  - effective result after manual override:
    - `effective_removed_future_labels_after_manual_overrides = 0`
    - `effective_tasks_touched_after_manual_overrides = 0`
- practical conclusion:
  - the `87` derived additions do not show a systematic future-novelty leakage or over-pruning problem after the comparative-topic fix
