# ResearchForesight

ResearchForesight is a time-sliced benchmark for evaluating whether research agents can reason from pre-cutoff literature and produce forward-looking research judgment. More specifically, it targets evidence-grounded, cutoff-controlled research foresight rather than only retrieval, summarization, or retrospective answer recovery.

This repository contains the current benchmark code, official benchmark releases, the offline knowledge base used by offline methods, and the latest prompt assets used for task construction and evaluation.

## Repository structure

### 1. Core code
- `src/researchworld/`: benchmark construction, offline retrieval, agent runners, and evaluation logic
- `scripts/`: release building, evaluation, aggregation, and utility scripts
- `configs/`: benchmark, domain, and model configuration files

### 2. Official benchmark release
The current official public release is the unified 422-task `ResearchForesight` release:
- `benchmark_release/`: flattened public release directory at the repository root
- `data/releases/benchmark_full_curated_polished/`: source release bundle used to assemble the public release

This release does not split public tasks into separate half-year and quarter sub-releases. Instead, it exposes a single mixed-horizon benchmark with per-task temporal cutoffs.

Historical intermediate folders and smaller internal subsets under `data/releases/` are retained as
build artifacts, not as the recommended public entry point.

### 3. Public release contents
The public release directory is `benchmark_release/`.

It contains:
- `tasks.jsonl`: public task file
- `manifest.json`: release metadata
- `tasks_hidden_eval.jsonl`: hidden evaluation targets
- `tasks_build_trace.jsonl`: build traces
- `tasks_internal_full.jsonl`: internal construction view

Because the official 422-task release mixes multiple time cutoffs, we do not bundle a single shared `kb/` directory inside `benchmark_release/`. A single frozen corpus would either leak future information for earlier-cutoff tasks or artificially underpower later-cutoff tasks.

If you want to run offline agents, use a cutoff-aligned task slice and pass the corresponding historical corpus explicitly with `--kb-dir`.

Example:

```bash
python ResearchForesight/scripts/run_researchagent_offline.py \
  --release-dir ResearchForesight/benchmark_release \
  --kb-dir /path/to/cutoff_aligned_kb \
  --output-dir ResearchForesight/results/researchagent_offline_example \
  --reasoning-llm-config ResearchForesight/tmp/local_llm_configs/qwen_235b_8002.local.yaml \
  --render-llm-config ResearchForesight/tmp/local_llm_configs/qwen_235b_8002.local.yaml \
  --fallback-llm-config ''
```

The same explicit `--kb-dir` pattern can be used with `ARIS-Offline`, `ResearchAgent-Offline`, and other offline runners in `scripts/`.

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
