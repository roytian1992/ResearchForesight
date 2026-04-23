# Experiment Log 2026-04-17

## Experiment100 Final Table Snapshot

- update_time: `2026-04-17 Asia/Shanghai`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- evaluation_scope:
  - Primary:
    - `Evidence-Grounded Factuality`
    - `Future Alignment`
    - `Evidence Traceability`
  - Family-specific auxiliary:
    - `Opportunity Grounding`
    - `Forecast Grounding`
    - `Strategic Execution Grounding`
    - `Venue Positioning Grounding`
- exclusion_note:
  - `ResearchArc` is intentionally omitted from this `2026-04-17` snapshot per current reporting preference.

## Included Runs

- `Native LLM`
  - run_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4`
  - eval_v31_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_v31/summary.json`
  - eval_v4_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_v4/summary.json`
  - eval_aux_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_aux/summary.json`
- `Hybrid RAG`
  - run_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4`
  - eval_v31_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_v31/summary.json`
  - eval_v4_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_v4/summary.json`
  - eval_aux_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_aux/summary.json`
- `ARIS`
  - run_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4`
  - eval_v31_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4/eval_v31/summary.json`
  - eval_v4_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4/eval_v4/summary.json`
  - eval_aux_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4/eval_aux/summary.json`
- `ResearchAgent`
  - run_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4`
  - eval_v31_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4/eval_v31/summary.json`
  - eval_v4_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4/eval_v4/summary.json`
  - eval_aux_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4/eval_aux/summary.json`
- `CoI`
  - run_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel1_sharded`
  - eval_v31_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel1_sharded/eval_v31/summary.json`
  - eval_v4_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel1_sharded/eval_v4/summary.json`
  - eval_aux_summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel1_sharded/eval_aux/summary.json`

## Current Comparison Table

| Method | Evidence-Grounded Factuality | Future Alignment | Evidence Traceability | Opportunity Grounding | Forecast Grounding | Strategic Execution Grounding | Venue Positioning Grounding |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native LLM | 0.4470 | 0.4653 | 0.0597 | 0.6589 | 0.4536 | 0.9015 | 0.8742 |
| Hybrid RAG | 0.4644 | 0.4453 | 0.4599 | 0.6388 | 0.4756 | 0.9037 | 0.8467 |
| ARIS | 0.4408 | 0.3924 | 0.5836 | 0.5581 | 0.5108 | 0.7858 | 0.6026 |
| ResearchAgent | 0.4571 | 0.3765 | 0.3063 | 0.3657 | 0.2648 | 0.2818 | 0.3132 |
| CoI | 0.4227 | 0.4622 | 0.6656 | 0.5961 | 0.3880 | 0.2320 | 0.3040 |

## Short Notes

- `CoI` is strongest on `Evidence Traceability`, but weak on `Strategic Execution Grounding` and `Venue Positioning Grounding`.
- `Hybrid RAG` remains unusually strong on `Strategic Execution Grounding` and `Venue Positioning Grounding`, so those metrics should still be treated with caution for method ranking.
- `ResearchAgent` task-native rerun improved benchmark alignment of the pipeline structure, but its current aggregate scores are still behind `Hybrid RAG`, `ARIS`, and `CoI` on most retained metrics.
- This file is intended to be the `2026-04-17` reporting snapshot and should supersede ad hoc chat-side tables for these five methods.

## ARIS Worst-25 Pilot Setup

- update_time: `2026-04-17 Asia/Shanghai`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_worst25_20260417_task_ids.txt`
- subset_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_aris_worst25_20260417`
- subset_task_count: `25`
- subset_family_counts:
  - `bottleneck_opportunity_discovery`: `6`
  - `direction_forecasting`: `6`
  - `strategic_research_planning`: `3`
  - `venue_aware_research_positioning`: `10`
- subset_kb_note:
  - `kb/` and `future_kb/` are symlinked to the source release because this pilot only changes the task subset, not the corpus snapshot.
- selection_rule:
  - worst-25 by `ARIS` deficit against the strongest current comparator on the same task, using the mean of:
    - `Evidence-Grounded Factuality`
    - `Future Alignment`
    - `Evidence Traceability`
    - family-specific auxiliary metric
- top_gap_examples:
  - `RTLv3-EXP-1049`
  - `RTLv3-0115`
  - `RTLv3-0034`
  - `RTLv3-0410`
  - `RTLv3-1195`
  - `RTLv3-0164`
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/aris_offline.py`
- current_modification_focus:
  - split `venue` and `strategic` away from the old shared generic agenda schema
  - tighten hard contract so `venue` / `strategic` must rank all listed candidate directions exactly once
  - strengthen `forecasting` guardrails around historically grounded successor labels
  - replace final free-form rewrite with deterministic structured rendering to prevent render-stage contract drift
  - soften bottleneck reranker override so title-overlap does not overrule a better review candidate as easily
- smoke_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_worst25_smoke4_20260417_task_ids.txt`
- smoke_output_dirs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_worst25_smoke4_20260417`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_worst25_smoke4_20260417_v2`
- status: `smoke running`

## ARIS Worst-25 Pilot Launch

- launch_time: `2026-04-17 Asia/Shanghai`
- method: `aris_worst25_20260417_contractv2_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_aris_worst25_20260417`
- release_task_count: `25`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_worst25_20260417_contractv2_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `aris_worst25_20260417_p4`
- status: `launching`
- notes:
  - Uses the new `benchmark_aris_worst25_20260417` subset built from the current `experiment100` release.
  - This pilot is intended to validate the new family-specific contract enforcement and deterministic final rendering in `aris_offline.py`.
  - The wrapper will automatically run answer generation, `eval_v31`, `eval_v4`, and `eval_aux`.
