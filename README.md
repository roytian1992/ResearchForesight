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
- `data/releases/researchforesight_refined_422/`: canonical release directory

This release does not split public tasks into separate half-year and quarter sub-releases. Instead, it exposes a single mixed-horizon benchmark with per-task temporal cutoffs.

Historical intermediate folders and smaller internal subsets may exist in local workspaces as build artifacts, but they are not part of the recommended public entry point.

### 3. Public release contents
The public release directory is `data/releases/researchforesight_refined_422/`.

It contains:
- `task_refined.jsonl`: unified task file containing public task fields and embedded evaluation targets
- `kb/`: cutoff-aware offline knowledge base

The `kb/` directory is exported up to the maximum task cutoff, but all runners filter retrieval by each task's own `time_cutoff`. This lets one KB serve mixed-cutoff tasks without leaking later history to earlier-cutoff tasks.

Validate the release:

```bash
python scripts/validate_refined_release.py \
  --release-dir data/releases/researchforesight_refined_422
```

Create a local OpenAI-compatible LLM config. Local configs are ignored by git:

```bash
cp configs/llm/qwen3_235b_8002.example.yaml \
  configs/llm/qwen3_235b_8002.local.yaml
```

If you run CoI with a local embedding server, create the optional embedding config:

```bash
cp configs/embedding/bge_m3.example.yaml \
  configs/embedding/bge_m3.local.yaml
```

Run an offline agent:

```bash
python scripts/run_researchagent_offline.py \
  --release-dir data/releases/researchforesight_refined_422 \
  --output-dir /path/to/local_run_output
```

Evaluate predictions:

```bash
python scripts/evaluate_experiment_final_metrics.py \
  --results-jsonl /path/to/local_run_output/results.jsonl \
  --release-dir data/releases/researchforesight_refined_422 \
  --output-dir /path/to/local_eval_output \
  --metrics all
```

The same release directory can be used with `ARIS-Offline`, `CoI-Agent-Offline`, non-agent baselines, pairwise judges, and final metrics. Legacy scripts that build old split releases are retained for provenance but are not the recommended user path.

Generated experiment outputs such as intermediate run directories, evaluation summaries, and pairwise comparison artifacts are expected to live in local working directories and are not part of the versioned benchmark asset surface.

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
- `Opportunity Grounding`: used for `bottleneck_opportunity_discovery`
- `Forecast Grounding`: used for `direction_forecasting`
- `Strategic Execution Grounding`: used for `strategic_research_planning`
- `Venue Positioning Grounding`: used for `venue_aware_research_positioning`

## Benchmark positioning
ResearchForesight is designed to study whether a system can exhibit useful forward-looking research judgment under a strict temporal cutoff. The target capability is prospective reasoning from bounded historical evidence: can an agent anticipate bottlenecks, directions, and plans before subsequent developments are known?

## Notes
- The Python package namespace remains `researchworld` to avoid breaking existing imports.
- The current paper workspace lives outside this repository under `papers/ResearchBenchmark`.
