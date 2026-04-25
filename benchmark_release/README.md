# ResearchForesight Public Release

## Summary
- tasks: 422
- families: 4
- domains: 4
- release form: unified mixed-horizon benchmark
- canonical release path: `data/releases/researchforesight_refined_422`

## Task families
1. bottleneck_opportunity_discovery
2. direction_forecasting
3. strategic_research_planning
4. venue_aware_research_positioning

## Time-cutoff policy
This public release does not split tasks into separate half-year and quarter subfolders.
Instead, each task carries its own `time_cutoff` field in `task_refined.jsonl`.

## Files
- `task_refined.jsonl`: unified benchmark task file, including public fields and embedded evaluation targets
- `kb/`: cutoff-aware offline knowledge base exported to the maximum task cutoff

## Offline evaluation note
This 422-task public release mixes multiple time cutoffs. The packaged `kb/` is a cutoff-aware history pool, not a single global visible corpus.

Offline agents must filter retrieval by each task's `time_cutoff`; the maintained runners in `scripts/` do this by default.

Validate the release before running experiments:

```bash
python scripts/validate_refined_release.py \
  --release-dir data/releases/researchforesight_refined_422
```
