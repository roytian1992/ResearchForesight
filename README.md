# ResearchInsightBenchmark

This repository contains the cleaned core code, public benchmark release files, and aggregated experiment results for the current Research Insight Benchmark.

## Included assets

### 1. Core benchmark construction and evaluation code
- `src/researchworld/`: benchmark construction, taxonomy induction, offline retrieval, agent/baseline runners, and evaluation utilities
- `scripts/`: end-to-end construction, release building, evaluation, and aggregation scripts
- `configs/`, `prompts/`, `schemas/`: configuration files, prompts, and schemas used by the pipeline

### 2. Public benchmark release
- `benchmark_release/benchmark_v3_20260407_venue/README.md`
- `benchmark_release/benchmark_v3_20260407_venue/manifest.json`
- `benchmark_release/benchmark_v3_20260407_venue/tasks.jsonl`
- `benchmark_release/benchmark_v3_20260407_venue/venue_task_summary.csv`

Release summary:
- task count: **168**
- families: **3**
- domains: **4**
- history cutoff: **2025-08-31**
- forecast window: **2025-09-01 to 2026-02-28**

### 3. Aggregated experiment results
- `results/final_metrics/`: merged metric tables for the full 168-task benchmark
- `results/pairwise_round_robin/`: round-robin pairwise comparison summaries

## Current task families
1. `bottleneck_opportunity_discovery`
2. `direction_forecasting`
3. `strategic_research_planning`

## Current domains
1. `LLM agents`
2. `LLM fine-tuning and post-training`
3. `RAG and retrieval structuring`
4. `Visual generative modeling and diffusion`

## Main files for quick inspection
- Public tasks: `benchmark_release/benchmark_v3_20260407_venue/tasks.jsonl`
- Metric summary: `results/final_metrics/final_metric_results_summary.md`
- Pairwise summary: `results/pairwise_round_robin/pairwise_round_robin_summary.md`

## Notes
- The internal hidden evaluation targets are intentionally not included in this cleaned repository snapshot.
- The Python package namespace remains `researchworld` to avoid breaking existing imports.
