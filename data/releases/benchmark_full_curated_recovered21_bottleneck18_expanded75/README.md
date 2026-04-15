# benchmark_full_curated_recovered21_bottleneck18_expanded75

## Summary
- base release: benchmark_full_curated_recovered21_bottleneck18
- total tasks: 518
- strict tasks: 365
- added tasks: 75

## Expansion policy
- start from the current curated full release with recovered q1 strategic tasks and bottleneck future-descendant repairs
- add only strict-ready candidates from judged q1/cluster pools
- dedupe by normalized title against the base release and previously selected additions
- accept candidates when either:
  - `judge.overall_score >= 0.55`
  - or the judge output looks corrupted (`overall_score == 0`) but the mean numeric subscore remains >= 0.85

## Notes
- this release prioritizes expansion over conservative legacy accept-only filtering
- obvious title corruption from candidate pools was repaired during import
- hidden eval v3 / v3.1 should be regenerated after this release is built
