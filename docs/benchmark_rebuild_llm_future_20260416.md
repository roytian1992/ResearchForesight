# Benchmark Rebuild With LLM Future Cleanup — 2026-04-16

## Goal

Rebuild the experiment subset pipeline from the original curated `422` baseline, replace heuristic future novelty cleanup with LLM-based judgment, recompute strict after cleanup, and then build a fresh `100`-task experiment subset.

## Source Lineage

- base release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated_polished`
  - `task_count = 422`
- supplement release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover`
  - `task_count = 528`
- candidate pool policy:
  - keep all rows from `422` exactly as-is
  - append only task IDs absent from `422`, taking those rows from the supplement release
  - expected candidate pool size: `528 = 422 + 106`

## New Cleanup Policy

- old heuristic:
  - token-overlap / reduced-token duplicate removal against history
- new policy:
  - LLM-based per-label novelty judgment
  - model cascade: `mimo -> qwen@8002 -> qwen@8001`
  - JSON parsing uses existing repair + retry utility in `src/researchworld/llm.py`
  - no forced fallback label retention; tasks may become structurally empty and be removed by strict recomputation later

## Scripts

- candidate pool build:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/build_candidate_pool_from_base_plus_supplements.py`
- LLM future cleanup:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/postprocess_gt_future_novelty_llm.py`
- balanced subset build:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/build_core100_subset.py`
- orchestration:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_rebuild_from_422_llm_future.sh`

## Planned Outputs

- candidate pool:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_v1`
- cleaned full candidate pool:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_v1`
- cleaned full candidate pool, candidate-review + majority-vote revision:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v2`
- experiment subset:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_v1`

## Status

- candidate pool merge script added
- LLM future cleanup script added
- local `qwen@8001` config added
- subset builders updated to regenerate `tasks_hidden_eval_v3*`
- 2-task smoke test completed on the LLM cleanup script
- full rebuild launch started, but the first full run stopped at `20/528`
- failure cause identified:
  - one LLM response was parseable JSON but violated the required schema
  - old script treated `missing label_decisions list` as fatal and terminated the thread-pool run
- 2026-04-16 fix applied:
  - schema-invalid JSON now triggers retry with an explicit schema-correction prompt
  - after local schema retries are exhausted, the task escalates to the next model in the cascade
  - orchestration script now uses `--resume` so completed decisions are reused on restart
- 2026-04-16 prompt / policy revision:
  - `candidate_directions` is no longer treated as historical context
  - old rule-based duplicate-like matching is now used only to surface candidate labels for LLM review
  - candidate review prompt is now per-label and concise instead of a long task-level JSON payload
  - the final decision is based on independent per-model votes with majority rule, rather than a single task-level LLM judgment
  - heuristic candidate recall is intentionally fuzzy rather than exact, so high-similarity pairs can still be sent to LLM adjudication
- 2026-04-16 full rerun launched with revised policy:
  - tmux session: `rf_llmfuture_vote_v2_20260416`
  - log: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/logs/rebuild_llm_future_vote_v2_20260416.log`
  - first confirmed write:
    - `1/528` judged
    - task: `RTLv3-0005`

## New 100-Task Experiment Subset

- source strict release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002`
  - `strict_task_count = 290`
- subset release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002`
  - `task_count = 100`
- family counts:
  - `bottleneck_opportunity_discovery = 25`
  - `direction_forecasting = 25`
  - `strategic_research_planning = 25`
  - `venue_aware_research_positioning = 25`
- domain counts:
  - `LLM agents = 25`
  - `LLM fine-tuning and post-training = 25`
  - `RAG and retrieval structuring = 25`
  - `Visual generative modeling and diffusion = 25`
- subtype note:
  - the current strict source only exposes enough eligible items for:
    - bottleneck: `pageindex_grounded_bottleneck`
    - forecasting: `chain_terminal_forecast`
    - strategic: `agenda_priority_selection`
    - venue: `venue_targeted_planning`
  - so the new 100-task subset is balanced by family and domain, but not currently diversified across strategic / venue subtype variants

## Spot Check Of The First 20 Judged Tasks

- output file:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_v1/future_novelty_llm_decisions.jsonl`
- current completed rows:
  - `20`
- current slice characteristics:
  - all `20` are `bottleneck_opportunity_discovery`
  - total judged future labels: `59`
  - kept labels: `58`
  - removed labels: `1`
  - no task in the first `20` was cleared to empty
- interpretation:
  - this early slice does **not** show evidence of the old over-aggressive future deletion problem
  - the single removal so far is a plausible future-vs-future redundancy case rather than history leakage pruning
