# Benchmark Refine Log 20260423 Round 12

## Scope

- This round redesigns the subtype taxonomy instead of only renaming labels.
- Principle: `family` and `horizon` already exist as separate fields, so `subtype` should only express the task form.
- Result: subtype labels are shorter, more natural, and orthogonal to horizon.

## New Subtype Definitions

- `bottleneck_and_opportunity`: identify a historically grounded bottleneck and infer the downstream opportunity.
- `direction_forecast`: predict the next concrete research direction and/or near-term trajectory.
- `agenda_prioritization`: prioritize a research agenda or ordered strategic plan.
- `opportunity_prioritization`: compare and rank candidate opportunities or directions.
- `venue_aligned_planning`: build a venue-oriented research plan or priority ordering.
- `venue_aligned_direction_forecast`: forecast a likely next direction together with venue fit.

## Mapping From Previous Labels

- `historical_bottleneck_to_half_year_opportunity` -> `bottleneck_and_opportunity`
- `historical_bottleneck_to_quarter_opportunity` -> `bottleneck_and_opportunity`
- `next_direction_and_trajectory_forecast` -> `direction_forecast`
- `quarter_ahead_direction_and_trajectory_forecast` -> `direction_forecast`
- `research_agenda_prioritization` -> `agenda_prioritization`
- `comparative_opportunity_ranking` -> `opportunity_prioritization`
- `venue_strategy_planning` -> `venue_aligned_planning`
- `venue_specific_direction_positioning` -> `venue_aligned_direction_forecast`

## Files Updated

- `benchmark_release/task_refined.jsonl`
- `scripts/build_task_refined_pilot_20260423.py`
- `scripts/build_task_refined_batch2_20260423.py`
- `scripts/build_task_refined_full_20260423.py`
- `docs/benchmark_refine_log_20260423_round12.md`

## Counts

- Before: {'bottleneck_and_opportunity': 135, 'direction_forecast': 67, 'agenda_prioritization': 76, 'venue_aligned_direction_forecast': 40, 'venue_aligned_planning': 60, 'opportunity_prioritization': 44}
- After: {'bottleneck_and_opportunity': 135, 'direction_forecast': 67, 'agenda_prioritization': 76, 'venue_aligned_direction_forecast': 40, 'venue_aligned_planning': 60, 'opportunity_prioritization': 44}

- New `subtype_taxonomy_version`: `2026-04-23-taxonomy-v3`
