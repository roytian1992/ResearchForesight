# Schema Contract Pilot 2026-04-16

## Purpose

Build an isolated `25`-task pilot release for schema simplification and answer-contract experiments, without modifying the current `100`-task experiment release.

## Source And Output

- source release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- output release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment25_schema_contract_pilot_v1`
- source `task_count`: `100`
- output `task_count`: `25`
- source future novelty cleanup status: `not applied` in source `manifest.json`
- supplementation run for this pilot: `no`
- future novelty cleanup run for this pilot: `no`

## Scripts Used

- subset builder: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/build_schema_contract_pilot_release.py`
- task-id seed file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/schema_contract_pilot25_task_ids.txt`

## Public Schema Pilot

`tasks.jsonl` in the pilot release now uses a simplified public schema:

- `task_id`
- `family`
- `domain`
- `horizon`
- `title`
- `question`
- `time_cutoff`
- `deliverable_spec`
- `answer_contract`

`answer_contract` is the new structured field for method-side answer shaping. It includes:

- `shape`
- `topic_text`
- `ranking_required`
- `max_items`
- optional `candidate_directions`
- `must_cover`
- `style_requirements`
- `disallowed_patterns`

The original public rows are preserved in `tasks_public_legacy.jsonl`.

## Hidden Eval Views

The pilot release keeps all legacy hidden eval files for compatibility:

- `tasks_hidden_eval.jsonl`
- `tasks_hidden_eval_v3.jsonl`
- `tasks_hidden_eval_v3_1.jsonl`

It also adds a simplified canonical GT view:

- `tasks_hidden_eval_canonical.jsonl`

Canonical hidden rows contain:

- `task_id`
- `family`
- `domain`
- `topic`
- `gold_answer`
- `expected_answer_points`
- `answer_contract`
- `primary_metric_gt`
- `family_aux_gt`

## Counts

### Family Counts

- `bottleneck_opportunity_discovery`: `6`
- `direction_forecasting`: `6`
- `strategic_research_planning`: `7`
- `venue_aware_research_positioning`: `6`

### Domain Counts

- `LLM agents`: `7`
- `LLM fine-tuning and post-training`: `6`
- `RAG and retrieval structuring`: `6`
- `Visual generative modeling and diffusion`: `6`

### Answer Contract Shape Counts

- `single_paragraph`: `12`
- `ranked_list`: `3`
- `compare_ranked_list`: `4`
- `venue_ranked_list`: `6`

## Files Written

Inside `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment25_schema_contract_pilot_v1`:

- `README.md`
- `manifest.json`
- `task_ids.txt`
- `tasks.jsonl`
- `tasks_public_legacy.jsonl`
- `tasks_hidden_eval.jsonl`
- `tasks_hidden_eval_v3.jsonl`
- `tasks_hidden_eval_v3_1.jsonl`
- `tasks_hidden_eval_canonical.jsonl`
- `tasks_build_trace.jsonl`
- `tasks_internal_full.jsonl`
- `kb ->` symlink to source KB
- `future_kb ->` symlink to source future KB

## Code Adjustments Supporting The Pilot

Method-side support for `answer_contract` was added in:

- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/research_arc_v2.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/answer_adapter.py`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`

Builder-side canonical hidden fallback was also tightened so `topic` and `family_aux_gt.topic` do not stay null when `slot_targets` is empty.

## Strict Count Note

This pilot is a pure subset of the source release. No new `strict` recomputation was run, because this action does not alter GT contents or release lineage for future novelty cleanup. If we later use this pilot for official reporting, strict logic should be discussed separately from this schema experiment.

## Caveats

- This pilot does not replace the current `100`-task experiment release.
- Legacy evaluation compatibility is preserved, but the new canonical GT view is experimental and intended for metric redesign work.
- Some canonical hidden fields remain intentionally lightweight and should be treated as a redesign surface rather than a frozen benchmark schema.
