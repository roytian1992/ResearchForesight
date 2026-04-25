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
- Updated direct and parallel v3/v3.1 evaluation entrypoints so `--kb-dir`, `--history-kb-dir`, and `--future-kb-dir` are optional and resolve to the canonical release defaults.
- Updated v3, v3.1, v4, aux, and final metrics evaluators to fail fast when a result row references a task ID missing from the refined release eval view.
- Updated evaluator fallback handling so missing optional fallback LLM config files do not abort runs when the primary judge config is valid.
- Updated the maintained release loader to require `task_refined.jsonl` and stop falling back to legacy split release files.
- Removed obsolete tracked split-release files from `benchmark_release/`; that directory is now documentation-only, while the canonical public release remains `data/releases/researchforesight_refined_422/`.
- Updated maintained runner/evaluator default LLM configs to `configs/llm/qwen3_235b_8002.local.yaml`, added public Qwen3 and BGE-M3 example configs, and ignored local experiment/backup outputs.

## KB Counts

- `llm_agent`: 32,998 papers, date range `2023-01-01` to `2025-11-30`, with 6,566 papers after `2025-08-31`.
- `llm_finetuning_post_training`: 23,072 papers, date range `2023-01-01` to `2025-11-30`, with 3,831 papers after `2025-08-31`.
- `rag_and_retrieval_structuring`: 5,524 papers, date range `2023-01-01` to `2025-11-30`, with 719 papers after `2025-08-31`.
- `visual_generative_modeling_and_diffusion`: 8,494 papers, date range `2023-01-01` to `2025-11-30`, with 948 papers after `2025-08-31`.

## Validation

- `python -m py_compile` passed for the changed export/eval modules.
- `scripts/validate_refined_release.py` was added as the canonical release validator.
- Validator initially found four future-evidence boundary violations where papers dated `2026-03-01` appeared after `future_end=2026-02-28`.
- The following tasks were corrected by removing the out-of-window future evidence from `trace.future_evidence`, `future_alignment_targets`, and `claim_bank.reference_paper_ids`: `RTLv3-0041`, `RTLv3-0043`, `RTLv3-0149`, `RTLv3-0396`.
- After correction, `python scripts/validate_refined_release.py --release-dir data/releases/researchforesight_refined_422` reports `error_count = 0`.
- Runtime cutoff check passed on the rebuilt KB:
  - `llm_agent` with cutoff `2025-08-31`: 26,432 visible papers, max date `2025-08-31`, 0 papers after cutoff.
  - `llm_agent` with cutoff `2025-11-30`: 32,998 visible papers, max date `2025-11-30`, 0 papers after cutoff.
- v3.1 smoke passed with no `--future-kb-dir`:
  - Input: `data/releases/benchmark_researchagent_refined_smoke1_20260425/results.jsonl`
  - Output: `data/releases/benchmark_researchagent_refined_smoke1_20260425/final_metrics_cutoffaware_smoke2`
  - Summary confirms `future_kb_dir` is empty and future alignment evidence source is `embedded_future`.
- Additional smoke checks passed after the evaluator hardening:
  - Direct v3.1: `.venv-researchforesight/bin/python scripts/evaluate_experiment_run_v3_1.py --results-jsonl data/releases/benchmark_researchagent_refined_smoke1_20260425/results.jsonl --release-dir data/releases/researchforesight_refined_422 --output-dir /tmp/rf_eval_v31_optional_future_smoke --judge-llm-config configs/llm/qwen3_235b_8002.local.yaml --task-limit 1`
  - Unified final metrics: `.venv-researchforesight/bin/python scripts/evaluate_experiment_final_metrics.py --results-jsonl data/releases/benchmark_researchagent_refined_smoke1_20260425/results.jsonl --release-dir data/releases/researchforesight_refined_422 --output-dir /tmp/rf_final_metrics_smoke --judge-llm-config configs/llm/qwen3_235b_8002.local.yaml --metrics factuality --task-limit 1`
  - Both runs completed with no `--future-kb-dir` and no valid fallback LLM config file.
- Strict-loader smoke passed with unified final metrics after removing legacy fallback:
  - `.venv-researchforesight/bin/python scripts/evaluate_experiment_final_metrics.py --results-jsonl data/releases/benchmark_researchagent_refined_smoke1_20260425/results.jsonl --release-dir data/releases/researchforesight_refined_422 --output-dir /tmp/rf_final_metrics_loader_strict_smoke --judge-llm-config configs/llm/qwen3_235b_8002.local.yaml --metrics factuality --task-limit 1`
- Default-config smoke passed with unified final metrics:
  - `.venv-researchforesight/bin/python scripts/evaluate_experiment_final_metrics.py --results-jsonl data/releases/benchmark_researchagent_refined_smoke1_20260425/results.jsonl --release-dir data/releases/researchforesight_refined_422 --output-dir /tmp/rf_default_config_final_metrics_smoke --metrics factuality --task-limit 1`

## Metric Logic Audit

- FactScore evidence now carries `published_date` for paper, structure, section, and pageindex retrieval rows wherever available.
- History evidence is filtered to each task's `history_cutoff`; future evidence is filtered to each task's `[future_start, future_end]` window.
- Embedded future evidence used by future-alignment evaluation is also filtered to `[future_start, future_end]`, so omitting `--future-kb-dir` does not bypass the future window.
- Empty evidence no longer goes through an LLM judge path. Fact verification returns `insufficient`; future alignment returns `not_aligned`.
- Judge-returned `cited_evidence_ids` are filtered to IDs that were actually provided in the evidence bundle.
- FactScore precision, coverage, final score, and supported counts now exclude temporally inconsistent support.
- v3 task fulfillment, v3 insight, v4 traceability, and aux family scores now clamp judge scores to `[0, 1]` and tolerate non-numeric score fields.
- Unified final metrics still resolves `--metrics factuality` and `--metrics future_alignment` to the shared `eval_v31` bundle. This avoids duplicating the expensive shared pass, but the root `summary.json` records the bundle-resolution note explicitly.
- Metric hardening smoke passed:
  - `.venv-researchforesight/bin/python -m py_compile src/researchworld/factscore_eval_v3.py src/researchworld/future_alignment_eval_v3_1.py src/researchworld/experiment_eval_v3.py src/researchworld/experiment_eval_v4.py src/researchworld/experiment_eval_aux.py scripts/evaluate_experiment_final_metrics.py`
  - `.venv-researchforesight/bin/python scripts/validate_refined_release.py --release-dir data/releases/researchforesight_refined_422`
  - `.venv-researchforesight/bin/python scripts/evaluate_experiment_final_metrics.py --results-jsonl data/releases/benchmark_researchagent_refined_smoke1_20260425/results.jsonl --release-dir data/releases/researchforesight_refined_422 --output-dir /tmp/rf_metric_logic_smoke --metrics factuality --task-limit 1`
  - Smoke output `summary.json` resolved to `v31`, used empty `future_kb_dir`, and completed with 0 future evidence date-window violations for `RTLv3-0001`.

## Clean Task-Refined Contract

- Maintained prediction and evaluation loaders now use explicit `load_task_refined_*` APIs.
- Removed the maintained `load_hidden_eval`, `load_release_tasks`, `load_release_*` refined-loader APIs so current code paths cannot imply or silently use split release files.
- Removed the `--hidden-eval-v3` override from `scripts/run_factscore_eval_v3.py`; FactScore now reads evaluation targets only from `task_refined.jsonl`.
- Updated maintained runners, final metrics, direct metric scripts, pairwise scripts, adapters, and non-agent judging to resolve public/eval views from `task_refined.jsonl`.
- Updated KB export and the checked-in KB manifest so the data contract says only `task_refined.jsonl` plus `kb/` are valid maintained runtime inputs.
- Clean-contract validation passed:
  - `.venv-researchforesight/bin/python -m py_compile src/researchworld/refined_release.py src/researchworld/baseline_runner.py src/researchworld/research_arc_skills.py src/researchworld/coi_agent_offline.py scripts/validate_refined_release.py scripts/export_offline_kb.py scripts/run_researchagent_offline.py scripts/run_aris_offline.py scripts/run_coi_agent_offline.py scripts/run_coi_agent_offline_sharded.py scripts/prepare_coi_fulltext_cache.py scripts/run_offline_kb_baseline.py scripts/run_research_arc.py scripts/run_research_arc_kb.py scripts/run_research_arc_skills.py scripts/run_research_arc_v2.py scripts/run_research_arc_v3.py scripts/run_research_arc_v4.py scripts/run_research_arc_v5.py scripts/run_research_arc_v6.py scripts/run_factscore_eval_v3.py scripts/evaluate_experiment_run_v3.py scripts/evaluate_experiment_run_v3_1.py scripts/evaluate_experiment_run_v4.py scripts/evaluate_experiment_aux.py scripts/evaluate_experiment_final_metrics.py scripts/run_pairwise_judge_v3.py scripts/run_pairwise_bestofk_v3.py scripts/rerun_pairwise_conflicts_v3.py scripts/aggregate_pairwise_bestofk_v3.py scripts/adapt_experiment_answers.py scripts/judge_nonagent_baseline.py scripts/export_canonical_method_results.py`
  - `.venv-researchforesight/bin/python scripts/validate_refined_release.py --release-dir data/releases/researchforesight_refined_422`
  - `.venv-researchforesight/bin/python scripts/evaluate_experiment_final_metrics.py --results-jsonl data/releases/benchmark_researchagent_refined_smoke1_20260425/results.jsonl --release-dir data/releases/researchforesight_refined_422 --output-dir /tmp/rf_clean_contract_eval_smoke --metrics factuality --task-limit 1`
  - Load-only prediction smokes passed for `run_researchagent_offline.py`, `run_aris_offline.py`, and `run_coi_agent_offline.py` with `--task-limit 0`.

## Caveats

- Methods are safe only if they continue passing `task["time_cutoff"]` into all KB retrievers. Current ResearchAgent, ARIS, and CoI retrieval paths already do this.
- `trace.future_evidence` is hidden evaluation evidence. It should be used only by evaluation code, never by method runners or public task views.
