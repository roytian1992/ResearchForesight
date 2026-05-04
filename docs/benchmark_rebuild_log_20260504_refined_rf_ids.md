# Refined Task Wording And Stable RF IDs

Date: 2026-05-04

## Scope

- Source release: `researchforesight_refined_422_goldexp_full_20260430_question_polished_manual_renumbered_20260504`
- Output release: `data/releases/researchforesight_refined_422`
- Task file: `data/releases/researchforesight_refined_422/task_refined.jsonl`
- Task count before: 422
- Task count after: 422
- Strict count before: not recomputed in this publishing pass
- Strict count after: not recomputed in this publishing pass
- Future novelty cleanup: not run in this publishing pass
- Supplementation: not run in this publishing pass

## Changes

- Replaced the public canonical 422-task release with the manually refined task wording.
- Replaced the legacy `RTLv3-*` task ids with the stable `RF-0001` to `RF-0422` id scheme.
- Added legacy-id mapping files so existing historical result rows can be migrated:
  - `data/releases/researchforesight_refined_422/task_id_mapping_old_to_rf_20260504.json`
  - `data/releases/researchforesight_refined_422/task_id_mapping_old_to_rf_20260504.jsonl`
- Added release audit summaries:
  - `data/releases/researchforesight_refined_422/renumbering_report_20260504.json`
  - `data/releases/researchforesight_refined_422/question_polish_manual_report_20260504.json`
- Added the current Research Judgment evaluator and rubric files:
  - `src/researchworld/research_judgment_eval_v8.py`
  - `src/researchworld/research_judgment_rubrics.py`
- Updated the maintained final metric evaluator so `--metrics all` includes Research Judgment.
- Updated README documentation for the refined task wording, stable id mapping, and Research Judgment metric.

## Known Caveats

- The checked-in `kb/` directory is unchanged. It remains the cutoff-aware KB for `data/releases/researchforesight_refined_422`.
- The validator now treats `evaluation_rubric` as a derived evaluation field supplied by `research_judgment_rubrics.py` when it is not embedded in `task_refined.jsonl`.
- This publishing pass changes the public task ids. Consumers with old `RTLv3-*` result files should remap ids with `task_id_mapping_old_to_rf_20260504.json` before running current evaluators.

## Validation

- `python scripts/validate_refined_release.py --release-dir data/releases/researchforesight_refined_422 --output-json /tmp/rf_publish_validate_20260504.json`
  - `error_count = 0`
  - `task_count = 422`
  - family counts: 135 bottleneck, 67 forecasting, 120 strategic, 100 venue
  - time cutoffs: 380 tasks at `2025-08-31`, 42 tasks at `2025-11-30`
- Task-id check passed:
  - first id: `RF-0001`
  - last id: `RF-0422`
  - exact sequential ids: yes
  - legacy `RTLv3-*` ids in `task_refined.jsonl`: 0
  - `old_to_new` mapping count: 422
  - `new_to_old` mapping count: 422
- Import/help smoke passed:
  - `PYTHONPATH=src python - <<'PY' ... import researchworld.research_judgment_eval_v8 ...`
  - `python scripts/evaluate_experiment_final_metrics.py --help`
- Compile smoke passed:
  - `python -m py_compile scripts/evaluate_experiment_final_metrics.py scripts/evaluate_refined422_final.py scripts/validate_refined_release.py src/researchworld/offline_kb.py src/researchworld/research_judgment_eval_v8.py src/researchworld/research_judgment_rubrics.py src/researchworld/factscore_eval_v5.py src/researchworld/future_alignment_eval_v5.py src/researchworld/refined_release.py`
