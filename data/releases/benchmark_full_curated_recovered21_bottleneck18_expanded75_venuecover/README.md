# benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover

## Summary
- base release: benchmark_full_curated_recovered21_bottleneck18_expanded75
- total tasks: 528
- strict tasks: 375
- added venue tasks: 10

## Policy
- fill missing strict venue buckets first using strategic-planning tasks with non-empty `direction_records`
- allow bucket-targeted venue derivation from any supported bucket in `target_window_stats.top_venue_buckets`, not only the dominant bucket
- then add quarter venue coverage when a domain still has none
- candidate-pool sources are allowed only when they are strict-ready and pass the judged quality threshold or suspicious-zero recovery rule
