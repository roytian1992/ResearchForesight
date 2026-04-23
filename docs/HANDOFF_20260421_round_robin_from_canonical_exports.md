# Handoff 2026-04-21: Round-Robin From Canonical Exports

## Scope

This handoff covers the current post-experiment state after:

- synchronizing the latest `experiment100` method table
- updating the paper to remove the `ResearchArc` section from the compiled manuscript
- moving venue prior-profile discussion into benchmark construction
- building canonical machine-readable result packages for the five current comparison methods

The next session should start from the canonical export packages and use them to run round-robin
pairwise judging.

## Current State

- canonical export root:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421`
- release used by the canonical export:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- canonical methods currently exported:
  - `native_llm`
  - `hybrid_rag`
  - `coi`
  - `researchagent`
  - `aris`
- each method package now contains:
  - final task-level merged answers + metrics
  - score summary
  - dimension summary
  - provenance
  - formal code snapshot
- paper status:
  - `ResearchArc` is no longer included in the compiled manuscript path
  - venue prior profiles are now described in benchmark construction rather than only in experiment interpretation

## Trusted Results

- current master comparison table:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/docs/experiment100_method_table_20260418.md`
- canonical export root:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421`
- canonical export README:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/README.md`
- top-level score summary:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/all_methods_score_summary.json`
- top-level dimension summary:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/all_methods_dimension_summary.json`
- method index:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/method_index.json`
- latest CoI full-100 canonical row:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_focusfixv3_20260420_parallel4/final_metrics/summary.json`
- latest ResearchAgent full-100 canonical row:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_qwen8002_20260420_parallel4/final_metrics/summary.json`
- paper source:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/paper.tex`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/sections/03_benchmark.tex`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/sections/05_experiments.tex`
- compiled paper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/paper.pdf`

## Untrusted Or Failed Results

- do not use stale `ResearchArc` paper content for the current manuscript:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/sections/04_researcharc.tex`
  - file still exists on disk, but it is no longer included by `paper.tex`
- do not assume any old non-canonical method run is still the reporting source
  - use each method package's `provenance.json`
- do not use the old `CoI contractalign` row as the current CoI main row
  - it has been superseded by `focusfixv3`

## Important Paths

- canonical export root:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421`
- per-method task-level files for round-robin input:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/native_llm/task_level_results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/hybrid_rag/task_level_results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/coi/task_level_results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/researchagent/task_level_results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/aris/task_level_results.jsonl`
- canonical export script:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/export_canonical_method_results.py`
- pairwise scripts:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/run_pairwise_judge_v3.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/aggregate_pairwise_v3.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/run_pairwise_bestofk_v3.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/aggregate_pairwise_bestofk_v3.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/rerun_pairwise_conflicts_v3.py`
- quick judged-run comparison helper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/compare_judged_runs.py`
- paper sources:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/paper.tex`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/sections/01_introduction.tex`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/sections/02_related_work.tex`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/sections/03_benchmark.tex`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark/sections/05_experiments.tex`

## Important Scripts

- `export_canonical_method_results.py`
  - rebuilds the canonical method packages from the currently adopted trusted source mapping
- `run_pairwise_judge_v3.py`
  - round-robin pairwise judging over multiple method answer files
  - accepts `method_id=path/to/results.jsonl`
  - canonical `task_level_results.jsonl` files are directly compatible because they contain `task_id` and `answer`
- `aggregate_pairwise_v3.py`
  - collapses raw pairwise instances into per-comparison winners and Bradley-Terry summary
- `run_pairwise_bestofk_v3.py`
  - stronger best-of-k pairwise judging path if stability is a concern
- `rerun_pairwise_conflicts_v3.py`
  - reruns unstable pairwise conflicts after an initial pass

## What Was Tried

- synchronized the latest `CoI focusfixv3` full-100 row into the master table and paper
- built canonical method packages that merge trusted family-level patch sources instead of relying on one monolithic result directory per method
- added formal code snapshots to each canonical method package
- removed `ResearchArc` from the compiled paper path
- moved venue prior-profile discussion into the benchmark construction section

## What Was Learned

- the canonical export is now the correct machine-readable starting point for any downstream comparison
- `task_level_results.jsonl` is the right round-robin input abstraction
  - it already contains the final answer plus all adopted metric outputs and provenance
- `Hybrid RAG` and `ARIS` still rely on controlled multi-source merge policy
  - do not try to reconstruct their final row from a single run directory
- paper state is now cleaner:
  - no active `ResearchArc` section in the compiled manuscript
  - venue prior knowledge is treated as benchmark methodology, not just experiment-side adjustment

## Recommended Next Steps

1. Start from the canonical export root, not from old raw result folders.
2. Run a first round-robin pairwise pass on the five canonical method packages.
3. Aggregate the raw pairwise outputs.
4. If instability is high, rerun conflicts with the best-of-k or conflict-rerun path.
5. Write the round-robin summary back into `docs/` rather than mixing it into the current method table directly.

## Command Template

First-pass round-robin:

```bash
python /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/run_pairwise_judge_v3.py \
  --release-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1 \
  --output-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/round_robin_from_canonical_20260421 \
  --judge-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8002.local.yaml \
  --workers 8 \
  --mirror \
  --input \
    native_llm=/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/native_llm/task_level_results.jsonl \
    hybrid_rag=/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/hybrid_rag/task_level_results.jsonl \
    coi=/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/coi/task_level_results.jsonl \
    researchagent=/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/researchagent/task_level_results.jsonl \
    aris=/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421/aris/task_level_results.jsonl
```

Aggregation:

```bash
python /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/aggregate_pairwise_v3.py \
  --input-jsonl /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/round_robin_from_canonical_20260421/pairwise_results.jsonl \
  --output-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/round_robin_from_canonical_20260421/aggregated
```

## Validation

- canonical export root exists
- all five method packages exist
- all handoff paths listed above exist at write time
- `task_level_results.jsonl` files are present for all five methods
