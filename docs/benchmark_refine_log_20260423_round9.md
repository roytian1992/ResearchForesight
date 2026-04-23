# Benchmark Refine Log 20260423 Round 9

## Scope

- This round focuses on the last four weak items after round 8.
- Goal: either upgrade items with newly found direct evidence, or make the remaining provisional items more honest and better specified.
- Method: targeted manual paper search plus question/gold/evidence replacement where the earlier wording was too broad.

## Files Updated

- `benchmark_release/task_refined.jsonl`
- `docs/benchmark_refine_log_20260423_round9.md`
- `scripts/manual_refine_round9_20260423.py`

## Status Counts

- Before: {'validated_with_refinement': 322, 'substantially_corrected': 96, 'provisional_low_evidence': 4}
- After: {'validated_with_refinement': 323, 'substantially_corrected': 96, 'provisional_low_evidence': 3}

## Manually Re-audited Tasks

- `RTLv3-0405`: provisional_low_evidence (historical=moderate, future=weak)
- `RTLv3-0504`: validated_with_refinement (historical=strong, future=moderate)
- `RTLv3-0505`: provisional_low_evidence (historical=strong, future=weak)
- `RTLv3-0516`: provisional_low_evidence (historical=moderate, future=weak)

## Notes

- `RTLv3-0504` was upgraded after adding direct early-2026 biomedical GraphRAG / EHR-RAG follow-ons.
- `RTLv3-0505` and `RTLv3-0516` were kept provisional deliberately; weak or out-of-window future papers were removed instead of overstating support.
- `RTLv3-0405` remains provisional because the evidence still spans multiple loosely connected modality families even after narrowing the prompt.
