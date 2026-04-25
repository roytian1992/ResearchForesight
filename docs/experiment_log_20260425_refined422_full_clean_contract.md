# Refined422 Full Clean-Contract Experiment

Date: 2026-04-25

## Release

- Release path: `data/releases/researchforesight_refined_422`
- Task file: `data/releases/researchforesight_refined_422/task_refined.jsonl`
- KB path: `data/releases/researchforesight_refined_422/kb`
- Validator command:
  - `.venv-researchforesight/bin/python scripts/validate_refined_release.py --release-dir data/releases/researchforesight_refined_422`
- Validator result:
  - `error_count = 0`
  - `task_count = 422`
  - family counts: bottleneck opportunity 135, direction forecasting 67, strategic planning 120, venue positioning 100
  - cutoff counts: `2025-08-31`: 380, `2025-11-30`: 42

## Experiment

- Output root: `results/refined422_full_clean_contract_20260425`
- Orchestrator: `scripts/run_refined422_full_experiment.py`
- Methods:
  - `researchagent`: `scripts/run_researchagent_offline.py`
  - `aris`: `scripts/run_aris_offline.py`
  - `coi`: `scripts/run_coi_agent_offline_sharded.py`
- Shards per method: 4
- Final metric workers: 4
- LLM config: `configs/llm/qwen3_235b_8002.local.yaml`
- Embedding config for CoI: `configs/embedding/bge_m3.local.yaml`
- Evaluation release path: `data/releases/researchforesight_refined_422`

## Launch Command

```bash
nohup .venv-researchforesight/bin/python -u scripts/run_refined422_full_experiment.py \
  --release-dir data/releases/researchforesight_refined_422 \
  --output-dir results/refined422_full_clean_contract_20260425 \
  --methods researchagent aris coi \
  --num-shards 4 \
  --eval-workers 4 \
  --llm-config configs/llm/qwen3_235b_8002.local.yaml \
  --embedding-config configs/embedding/bge_m3.local.yaml \
  --poll-seconds 60 \
  > results/refined422_full_clean_contract_20260425/nohup.out 2>&1 &
```

## Monitoring

- Orchestrator log: `results/refined422_full_clean_contract_20260425/orchestrator.log`
- Top-level stdout/stderr: `results/refined422_full_clean_contract_20260425/nohup.out`
- Per-shard method logs:
  - `results/refined422_full_clean_contract_20260425/researchagent/shards/shard_*/run.log`
  - `results/refined422_full_clean_contract_20260425/aris/shards/shard_*/run.log`
  - `results/refined422_full_clean_contract_20260425/coi/shard_*/run.log`
- Final metrics:
  - `results/refined422_full_clean_contract_20260425/<method>/final_metrics/summary.json`

## Status

- Initial state: prepared, launch pending.
