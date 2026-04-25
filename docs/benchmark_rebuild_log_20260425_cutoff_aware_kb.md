# Cutoff-Aware KB And Embedded Future Eval Log

Date: 2026-04-25

## Scope

- Release: `data/releases/researchforesight_refined_422`
- Task file: `data/releases/researchforesight_refined_422/task_refined.jsonl`
- Task count: 422
- Motivation: the refined release has mixed `time_cutoff` values (`2025-08-31` and `2025-11-30`), so the offline KB must cover the maximum cutoff while agents filter retrieval by each task's own cutoff.

## Changes

- Rebuilt `data/releases/researchforesight_refined_422/kb` as a cutoff-aware history pool with `max_history_cutoff = 2025-11-30`.
- Preserved the previous `2025-08-31` KB locally as `data/local_backups/researchforesight_refined_422_kb_20250831_backup`.
- Updated `scripts/export_offline_kb.py` so omitted `--history-cutoff` infers the maximum `time_cutoff` from `task_refined.jsonl`.
- Updated v3/v3.1/final metric evaluation so `--future-kb-dir` is optional. If omitted, future alignment and future-scoped fact evidence use each task's embedded `trace.future_evidence` from `task_refined.jsonl` instead of reusing the public history KB.
- Updated the final metric evaluator to fail on unknown or duplicate result `task_id` values instead of silently skipping them.

## KB Counts

- `llm_agent`: 32,998 papers, date range `2023-01-01` to `2025-11-30`, with 6,566 papers after `2025-08-31`.
- `llm_finetuning_post_training`: 23,072 papers, date range `2023-01-01` to `2025-11-30`, with 3,831 papers after `2025-08-31`.
- `rag_and_retrieval_structuring`: 5,524 papers, date range `2023-01-01` to `2025-11-30`, with 719 papers after `2025-08-31`.
- `visual_generative_modeling_and_diffusion`: 8,494 papers, date range `2023-01-01` to `2025-11-30`, with 948 papers after `2025-08-31`.

## Validation

- `python -m py_compile` passed for the changed export/eval modules.
- Runtime cutoff check passed on the rebuilt KB:
  - `llm_agent` with cutoff `2025-08-31`: 26,432 visible papers, max date `2025-08-31`, 0 papers after cutoff.
  - `llm_agent` with cutoff `2025-11-30`: 32,998 visible papers, max date `2025-11-30`, 0 papers after cutoff.
- v3.1 smoke passed with no `--future-kb-dir`:
  - Input: `data/releases/benchmark_researchagent_refined_smoke1_20260425/results.jsonl`
  - Output: `data/releases/benchmark_researchagent_refined_smoke1_20260425/final_metrics_cutoffaware_smoke2`
  - Summary confirms `future_kb_dir` is empty and future alignment evidence source is `embedded_future`.

## Caveats

- Methods are safe only if they continue passing `task["time_cutoff"]` into all KB retrievers. Current ResearchAgent, ARIS, and CoI retrieval paths already do this.
- `trace.future_evidence` is hidden evaluation evidence. It should be used only by evaluation code, never by method runners or public task views.
