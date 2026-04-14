# ResearchForesight

ResearchForesight is a time-sliced benchmark for evaluating whether research agents can reason from pre-cutoff literature and produce forward-looking research judgment. More specifically, it targets evidence-grounded, cutoff-controlled research foresight rather than only retrieval, summarization, or retrospective answer recovery.

This repository contains the current benchmark code, official benchmark releases, the offline knowledge base used by offline methods, and the latest prompt assets used for task construction and evaluation.

## Repository structure

### 1. Core code
- `src/researchworld/`: benchmark construction, offline retrieval, agent runners, and evaluation logic
- `scripts/`: release building, evaluation, aggregation, and utility scripts
- `configs/`: benchmark, domain, and model configuration files

### 2. Named benchmark releases
We maintain three named releases under `data/releases/`:
- `benchmark_halfyear`: 437 tasks, cutoff `2025-08-31`
- `benchmark_quarter`: 96 tasks, cutoff `2025-11-30`
- `benchmark_full`: 533 tasks combining the half-year and quarter settings

These folders store task files, hidden evaluation views, build traces, and release metadata. Experiment outputs are intentionally kept outside the release folders.

### 3. Official offline benchmark bundle
The current public offline bundle is `benchmark_release/benchmark_v3_20260407_venue/`.

It contains:
- `tasks.jsonl`: public task file
- `manifest.json`: release metadata
- `kb/`: the offline historical knowledge base used by offline methods

Offline runners default to `release_dir / kb`, so the bundle is directly runnable without passing a separate knowledge-base path.

Example:

```bash
python ResearchForesight/scripts/run_researchagent_offline.py \
  --release-dir ResearchForesight/benchmark_release/benchmark_v3_20260407_venue \
  --output-dir ResearchForesight/results/researchagent_offline_example \
  --reasoning-llm-config ResearchForesight/tmp/local_llm_configs/qwen_235b_8002.local.yaml \
  --render-llm-config ResearchForesight/tmp/local_llm_configs/qwen_235b_8002.local.yaml \
  --fallback-llm-config ''
```

The same `release_dir / kb` convention is used by `ARIS-Offline`, `ResearchAgent-Offline`, and other offline runners in `scripts/`.

### 4. Prompt assets
We only keep the **latest** prompt inventory in-repo.

- `prompts/task_generation/`: latest task-construction prompts
- `prompts/metrics/`: latest evaluation and comparison prompts
- `src/researchworld/prompting.py`: YAML prompt loader

#### Task-generation prompt inventory
- Generic candidate polish/judging
- Family-specific judge/rewrite prompts for:
  - `bottleneck_opportunity_discovery`
  - `direction_forecasting`
  - `strategic_research_planning`
  - `venue_aware_research_positioning`

#### Metric prompt inventory
- `fact_claim_extraction.yaml`
- `fact_verification.yaml`
- `future_alignment.yaml`
- `evidence_traceability.yaml`
- `family_auxiliary.yaml`
- `pairwise_round_robin.yaml`

These prompt files are the canonical latest snapshots we want visible on GitHub for transparency and reproducibility.

### 5. Results and summaries
- `results/final_metrics/`: merged metric tables for the current standardized comparison setting
- `results/pairwise_round_robin/`: pairwise round-robin comparison summaries

## Current task families
1. `bottleneck_opportunity_discovery`
2. `direction_forecasting`
3. `strategic_research_planning`
4. `venue_aware_research_positioning`

## Current domains
1. `LLM agents`
2. `LLM fine-tuning and post-training`
3. `RAG and retrieval structuring`
4. `Visual generative modeling and diffusion`

## Current evaluation metrics
### Primary
- `Evidence-Grounded Factuality`
- `Future Alignment`
- `Evidence Traceability`

### Family-specific auxiliary metrics
- `Opportunity Grounding`
- `Forecast Grounding`
- `Technical Dependency Grounding`

## Benchmark positioning
ResearchForesight is designed to study whether a system can exhibit useful forward-looking research judgment under a strict temporal cutoff. The target capability is prospective reasoning from bounded historical evidence: can an agent anticipate bottlenecks, directions, and plans before subsequent developments are known?

## Notes
- The Python package namespace remains `researchworld` to avoid breaking existing imports.
- The current paper workspace lives outside this repository under `papers/ResearchBenchmark`.
