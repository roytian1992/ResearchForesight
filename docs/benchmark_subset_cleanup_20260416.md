# Benchmark Subset Cleanup 2026-04-16

## Purpose

Remove the `core98` / `core100` benchmark subset artifacts and stop all related runs so the workspace no longer uses those reporting subsets.

## Stopped Sessions

- `research_arc_core98_v6_parallel4`

## Removed Release Directories

- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core98`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core100`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core100_future_novelty_v1`

## Removed Result Directories

- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_core100_v1`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_core100_v2_retrievalfusion_parallel4`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_core100_v1`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_core100_v2_retrievalfusion_parallel4`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_core100_strict251_parallel4`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core100_v1`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core100_v2_rebuilt_parallel4`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_core100_v1`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_core100_strict251_parallel4`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_core100_v1`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_core100_v2_covanchor`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_core100_v3_retrievalfusion`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_core100_v4_futurefix_parallel4`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_core98_v6_parallel4`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_core100_v1`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/final_metrics/core100_20260415_partial`

## Removed Tmp Files

- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/benchmark_core100_strict251_task_ids.txt`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_core100_eval_only.sh`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_core100_method_eval.sh`
- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_core100_method_eval_parallel.sh`

## Verification

After cleanup:

- `tmux ls | rg 'core98|core100'` returns no related sessions.
- `data/releases/` contains no `core98` / `core100` release directories.
- `results/` contains no `core98` / `core100` result directories or files.
- `tmp/` contains no `core98` / `core100` subset helpers.

## Remaining Scope

This cleanup removes the `core98` / `core100` subset line. Full-release benchmark artifacts remain untouched.
