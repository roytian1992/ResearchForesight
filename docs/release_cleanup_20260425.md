# Release Cleanup 2026-04-25

- source repo: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight`
- source dataset build: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight_DEV/benchmark_release/task_refined_v2_pilot.jsonl`
- output release path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/researchforesight_refined_422`
- migration mode: simplify git-tracked release assets to one final dataset plus offline KB

## Files Kept

- `data/releases/researchforesight_refined_422/task_refined.jsonl`
- `data/releases/researchforesight_refined_422/kb/`

## Files Removed From Git Tracking

- `data/releases/benchmark_full/`
- `data/releases/benchmark_full_curated_polished/`
- `data/releases/benchmark_full_curated_recovered21/`
- `data/releases/benchmark_full_curated_recovered21_bottleneck18/`
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75/`
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover/`
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover_future_novelty_v1/`
- `data/releases/benchmark_halfyear/`
- `data/releases/benchmark_quarter/`

## Notes

- The final refined dataset remains the 422-task manually repaired file built in `ResearchForesight_DEV`.
- The git-tracked release surface is now intentionally minimal.
- This cleanup does not rely on additional hidden eval files inside `data/releases/`; the intended public artifact is the single refined `jsonl` plus offline `kb`.
