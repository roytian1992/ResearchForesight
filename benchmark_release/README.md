# ResearchForesight Public Release

## Summary
- tasks: 422
- families: 4
- domains: 4
- release form: unified mixed-horizon benchmark

## Task families
1. bottleneck_opportunity_discovery
2. direction_forecasting
3. strategic_research_planning
4. venue_aware_research_positioning

## Time-cutoff policy
This public release does not split tasks into separate half-year and quarter subfolders.
Instead, each task carries its own `time_cutoff` field in `tasks.jsonl`.

## Files
- `tasks.jsonl`: public benchmark tasks
- `tasks_hidden_eval.jsonl`: hidden evaluation targets
- `tasks_build_trace.jsonl`: build traces
- `tasks_internal_full.jsonl`: internal construction view
- `manifest.json`: release metadata

## Offline evaluation note
This 422-task public release mixes multiple time cutoffs. For that reason, we do not package a single shared `kb/` directory here.

If you want to run offline agents, use a cutoff-aligned task slice and pass an explicit historical corpus with `--kb-dir`.
