# Evidence-Chain Expansion Build — 2026-04-16

## Goal

Expand the current LLM-future-cleaned strict release using evidence-chain reuse, not paraphrase-only rewriting, and then rebuild a stronger 100-task experiment subset from the expanded release.

## Source Release

- source release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002`
- source `task_count`:
  - `528`
- source `strict_task_count`:
  - `290`
- source `strict_family_counts`:
  - bottleneck: `71`
  - forecasting: `94`
  - strategic: `82`
  - venue: `43`

## Scripts Used

- expansion script:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/build_evidence_chain_expansion_from_release.py`
- subset rebuild script:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/build_core100_subset.py`

## Expansion Policy

- Only derive new tasks from already strict-valid source rows.
- Do not generate new seed packets in this pass.
- Expansion types:
  - `direction_forecasting -> venue_aware_direction_forecast`
  - `strategic_research_planning -> venue_targeted_planning`
  - `strategic_research_planning -> comparative_opportunity_prioritization`
- Comparative pair selection settings:
  - `min_common_prefix = 2`
  - `min_score_gap = 0.05`
  - `max_occurrence_per_node = 3`
  - `max_per_domain = 8`
- Important fix:
  - `venue_aware_direction_forecast` additions now include `ground_truth.direction_records`, so they are counted by the current strict rule for the venue family.
- Important fix:
  - comparative strategic additions also include `ground_truth.direction_records`, so they remain strict-compatible under the current metric-corresponding GT rule.

## Expanded Full Release

- output release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- output `task_count`:
  - `615`
- output `strict_task_count`:
  - `377`
- strict delta:
  - `+87`

## Added Tasks

- total added:
  - `87`
- by subtype:
  - `venue_aware_direction_forecast`: `46`
  - `venue_targeted_planning`: `9`
  - `comparative_opportunity_prioritization`: `32`

## Expanded Strict Distribution

- family counts:
  - bottleneck: `71`
  - forecasting: `94`
  - strategic: `114`
  - venue: `98`
- domain counts:
  - LLM agents: `108`
  - LLM fine-tuning and post-training: `106`
  - RAG and retrieval structuring: `61`
  - Visual generative modeling and diffusion: `102`

## Rebuilt Experiment100

- output release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- `task_count`:
  - `100`
- family counts:
  - bottleneck: `25`
  - forecasting: `25`
  - strategic: `25`
  - venue: `25`
- domain counts:
  - LLM agents: `25`
  - LLM fine-tuning and post-training: `25`
  - RAG and retrieval structuring: `25`
  - Visual generative modeling and diffusion: `25`
- subtype counts:
  - `pageindex_grounded_bottleneck`: `25`
  - `chain_terminal_forecast`: `25`
  - `agenda_priority_selection`: `12`
  - `comparative_opportunity_prioritization`: `13`
  - `venue_aware_direction_forecast`: `13`
  - `venue_targeted_planning`: `12`

## Files Written

- expanded release bundle:
  - `tasks.jsonl`
  - `tasks_hidden_eval.jsonl`
  - `tasks_build_trace.jsonl`
  - `tasks_internal_full.jsonl`
  - `tasks_hidden_eval_v3.jsonl`
  - `tasks_hidden_eval_v3_1.jsonl`
  - `task_ids.txt`
  - `strict_task_ids.txt`
  - `strict_summary.json`
  - `manifest.json`
- added-task artifacts:
  - `added_tasks_public.jsonl`
  - `added_tasks_hidden_eval.jsonl`
  - `added_tasks_build_trace.jsonl`
  - `added_tasks_internal_full.jsonl`

## Caveats

- This pass improves strict coverage substantially, but it is still derivation-heavy rather than new-seed-heavy.
- `RAG and retrieval structuring` remains the smallest strict domain after expansion.
- Bottleneck count did not increase in this pass; further growth will likely require new seed packets or more deliberate reuse of historical future-work clusters / support packets.
