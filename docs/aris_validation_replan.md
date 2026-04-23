# ARIS Validation Replan

This note replaces the failed `4-task -> full-evaluation` validation path with balanced mid-sized validation splits.

## Goal

Use validation subsets that are large enough to surface cross-family regressions before launching a full evaluation run.

## Recommended Splits

- `tmp/pilot24_balanced_core_ids.txt`
  - `24` tasks
  - `2` tasks per `(family, domain)` cell
  - Covers only the three core families:
    - `bottleneck_opportunity_discovery`
    - `direction_forecasting`
    - `strategic_research_planning`

- `tmp/pilot36_balanced_core_ids.txt`
  - `36` tasks
  - `3` tasks per `(family, domain)` cell
  - Same three-family scope as above

## Why Exclude Venue Tasks

`venue_aware_research_positioning` has only a small number of tasks in the current benchmark subset and behaved differently from the other families in the failed ARIS v8 run. We should validate core forward-reasoning behavior first, then test venue-specific changes separately.

## Upgrade Gate Before Full Evaluation

Any ARIS change should first clear one of the balanced core splits with:

- target metric improves on the split
- `Fact` and `Future Alignment` do not materially regress
- no large collapse in family-specific grounding outside the target family

## Immediate Recommendation

For forecast-only ARIS changes, start from `tmp/pilot36_balanced_core_ids.txt` rather than a tiny forecasting-only probe.
