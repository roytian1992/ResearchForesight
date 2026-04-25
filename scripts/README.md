# Scripts

## Current Refined-Release Flow

Use these entrypoints for the public `data/releases/researchforesight_refined_422` release:

- `validate_refined_release.py`: validates the unified `task_refined.jsonl` plus cutoff-aware `kb/`.
- `run_researchagent_offline.py`: runs ResearchAgent-Offline predictions.
- `run_aris_offline.py`: runs ARIS-Offline predictions.
- `run_coi_agent_offline.py`: runs CoI-Agent-Offline predictions.
- `evaluate_experiment_final_metrics.py`: runs the maintained final metric bundle from `task_refined.jsonl`.
- `run_pairwise_judge_v3.py` and `run_pairwise_bestofk_v3.py`: pairwise comparison utilities using the refined public task view.

## Current Data Contract

All maintained prediction and evaluation scripts must treat `task_refined.jsonl` as the single source of task, target, and evaluation metadata. The only other release input they may read is the cutoff-aware `kb/` directory.

The maintained `researchworld.refined_release` loader requires `task_refined.jsonl` and fails fast if it is absent.
