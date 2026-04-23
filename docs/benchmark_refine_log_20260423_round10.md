# Benchmark Refine Log 20260423 Round 10

## Scope

- This round replaces the last three weak questions instead of trying to preserve under-supported originals.
- Constraint preserved: same task family and horizon, but with new bottleneck/opportunity formulations that better match the literature.
- Method: targeted manual paper search plus question replacement when the original evidence chain could not be made robust enough.

## Files Updated

- `benchmark_release/task_refined.jsonl`
- `docs/benchmark_refine_log_20260423_round10.md`
- `scripts/manual_refine_round10_20260423.py`

## Status Counts

- Before: {'validated_with_refinement': 323, 'substantially_corrected': 96, 'provisional_low_evidence': 3}
- After: {'substantially_corrected': 99, 'validated_with_refinement': 323}

## Replaced Tasks

- `RTLv3-0405`: substantially_corrected (historical=strong, future=moderate)
- `RTLv3-0505`: substantially_corrected (historical=strong, future=moderate)
- `RTLv3-0516`: substantially_corrected (historical=strong, future=moderate)

## Notes

- `RTLv3-0405` was replaced with a tighter unified audio-video diffusion task because the earlier cross-modal formulation was too heterogeneous.
- `RTLv3-0505` was replaced with a long-term personalized dialogue grounding task supported by direct early-2026 persona-memory work.
- `RTLv3-0516` was replaced with an educational RAG retrieval-quality task supported by direct in-window entity-aware follow-on work.
