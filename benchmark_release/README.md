# ResearchForesight Public Release

## Summary
- tasks: 422
- families: 4
- domains: 4
- release form: unified mixed-horizon benchmark
- canonical release path: `data/releases/researchforesight_refined_422`

This directory is documentation-only. The actual public release files live under the canonical release path above.

## Task families
1. bottleneck_opportunity_discovery
2. direction_forecasting
3. strategic_research_planning
4. venue_aware_research_positioning

## Time-cutoff policy
This public release does not split tasks into separate half-year and quarter subfolders.
Instead, each task carries its own `time_cutoff` field in `task_refined.jsonl`.

## Canonical release files
- `data/releases/researchforesight_refined_422/task_refined.jsonl`: unified benchmark task file, including public fields and embedded evaluation targets
- `data/releases/researchforesight_refined_422/kb/`: cutoff-aware offline knowledge base exported to the maximum task cutoff

## Offline evaluation note
This 422-task public release mixes multiple time cutoffs. The packaged `kb/` is a cutoff-aware history pool, not a single global visible corpus.

Offline agents must filter retrieval by each task's `time_cutoff`; the maintained runners in `scripts/` do this by default.

Validate the release before running experiments:

```bash
python scripts/validate_refined_release.py \
  --release-dir data/releases/researchforesight_refined_422
```

The maintained evaluation entrypoint is:

```bash
python scripts/evaluate_experiment_final_metrics.py \
  --results-jsonl /path/to/run/results.jsonl \
  --release-dir data/releases/researchforesight_refined_422 \
  --output-dir /path/to/eval \
  --metrics all
```
