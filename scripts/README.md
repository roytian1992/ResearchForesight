# Scripts

## Current Refined-Release Flow

Use these entrypoints for the public `data/releases/researchforesight_refined_422` release:

- `validate_refined_release.py`: validates the unified `task_refined.jsonl` plus cutoff-aware `kb/`.
- `run_researchagent_offline.py`: runs ResearchAgent-Offline predictions.
- `run_aris_offline.py`: runs ARIS-Offline predictions.
- `run_coi_agent_offline.py`: runs CoI-Agent-Offline predictions.
- `evaluate_experiment_final_metrics.py`: runs the maintained final metric bundle from `task_refined.jsonl`.
- `run_pairwise_judge_v3.py` and `run_pairwise_bestofk_v3.py`: pairwise comparison utilities using the refined public task view.

## Legacy/Provenance Scripts

Many older build, augmentation, and post-processing scripts still reference `tasks.jsonl`, `tasks_hidden_eval*.jsonl`, `tasks_build_trace.jsonl`, or `tasks_internal_full.jsonl`. They are retained to preserve release-construction provenance, but they are not part of the current public benchmark flow.

For new experiments, do not build dependencies on those split files. Treat `task_refined.jsonl` as the single source of task, target, and evaluation metadata.
