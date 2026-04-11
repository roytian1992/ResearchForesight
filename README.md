# ResearchInsightBenchmark

ResearchInsightBenchmark is a time-sliced benchmark for evaluating whether research agents can reason from pre-cutoff literature and produce forward-looking research insight.

This repository contains the current benchmark code, official benchmark releases, and the latest prompt assets used for task construction and evaluation.

## Repository structure

### 1. Core code
- `src/researchworld/`: benchmark construction, offline retrieval, agent runners, and evaluation logic
- `scripts/`: release building, evaluation, aggregation, and utility scripts
- `configs/`: benchmark, domain, and model configuration files

### 2. Official benchmark releases
We currently maintain three official releases under `data/releases/`:
- `benchmark_v1_halfyear_440`
- `benchmark_v1_quarter_131`
- `benchmark_v1_mixed_571`

These correspond to:
- **halfyear**: 440 tasks, cutoff `2025-08-31`
- **quarter**: 131 tasks, cutoff `2025-11-30`
- **mixed**: 571 tasks = halfyear + quarter

Each release stores benchmark tasks, hidden evaluation views, build traces, and release metadata. Experiment output files are intentionally kept outside the release folders.

### 3. Prompt assets
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

### 4. Results and summaries
- `results/final_metrics/`: merged metric tables for the standardized 168-task comparison subset
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
- `Fact`
- `Future Alignment`
- `Evidence Traceability`

### Family-specific auxiliary metrics
- `Opportunity Grounding`
- `Forecast Grounding`
- `Technical Dependency Grounding`

## Notes
- The Python package namespace remains `researchworld` to avoid breaking existing imports.
- The current paper workspace lives outside this repository under `papers/ResearchBenchmark`.
