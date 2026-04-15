# benchmark_full_curated_recovered21_bottleneck18

## Summary
- tasks: 443
- base release: benchmark_full_curated_recovered21
- bottleneck future-descendant recoveries: 18

## Recovery notes
- only bottleneck tasks with empty `future_descendants` and non-empty `historical_future_work_cluster` were modified
- recovered labels are human-readable direction strings copied from `historical_future_work_cluster.direction`
- `realized_opportunity_directions` and `public_metadata.future_themes` were backfilled together to keep eval artifacts aligned
