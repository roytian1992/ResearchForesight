# Experiment Log 2026-04-16

## Default Evaluation Workflow Update

- update_time: 2026-04-16 Asia/Shanghai
- policy:
  - For benchmark experiments, default post-answer evaluation now includes all primary metrics and all family-specific auxiliary metrics.
  - Primary metrics: `Evidence-Grounded Factuality`, `Future Alignment`, `Evidence Traceability`
  - Family-specific metrics: `Opportunity Grounding`, `Forecast Grounding`, `Technical Dependency Grounding`, `Venue Positioning Grounding`
  - Auxiliary metrics are still only aggregated on their corresponding family.
- helper_script:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`

## Hybrid RAG on core98

- launch_time: 2026-04-16 Asia/Shanghai
- method: `hybrid_rag_core98_v2_rebuilt_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core98`
- release_task_count: `98`
- answer_results_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4/results.jsonl`
- answer_result_count: `98`
- history_kb_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core98/kb`
- future_kb_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core98/future_kb`
- judge_llm_config: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab/configs/llm/qwen_235b.local.yaml`
- judge_fallback_llm_config: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab/configs/llm/mimo_pro.local.yaml`
- eval_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_run_v3_1_parallel.py`
- eval_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4/eval_v31_core98`
- worker_count: `6`
- tmux_session: `hybrid_rag_core98_eval_20260416`
- status: `finished`
- notes:
  - Previous `eval_v31_core98` attempt failed because the wrapper was launched with a missing relative config path (`configs/llm/qwen_235b.local.yaml`).
  - Relaunch uses absolute release / KB / judge config paths and must run inside `tmux`.
  - Finished main eval output: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4/eval_v31_core98/results_eval_v3_1.jsonl`
  - Finished main eval summary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4/eval_v31_core98/summary.json`
  - Main summary snapshot: `task_count=98`, `mean_fact_precision_score=0.4433`, `mean_future_alignment_score=0.5276`

## Hybrid RAG on core98: Evidence Traceability

- launch_time: 2026-04-16 Asia/Shanghai
- method: `hybrid_rag_core98_v2_rebuilt_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core98`
- release_task_count: `98`
- answer_results_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4/results.jsonl`
- answer_result_count: `98`
- judge_llm_config: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab/configs/llm/qwen_235b.local.yaml`
- judge_fallback_llm_config: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab/configs/llm/mimo_pro.local.yaml`
- eval_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_run_v4_parallel.py`
- eval_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4/eval_v4_core98`
- worker_count: `6`
- tmux_session: `hybrid_rag_core98_v4_20260416`
- status: `running`
- notes:
  - Worker logs are writing under `eval_v4_core98/logs/`.
  - `tmux_run.log` was not created because the output directory did not exist before shell redirection, but the evaluation processes launched successfully.

## Hybrid RAG on core98: Family Auxiliary

- launch_time: 2026-04-16 Asia/Shanghai
- method: `hybrid_rag_core98_v2_rebuilt_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core98`
- release_task_count: `98`
- answer_results_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4/results.jsonl`
- answer_result_count: `98`
- judge_llm_config: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab/configs/llm/qwen_235b.local.yaml`
- judge_fallback_llm_config: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab/configs/llm/mimo_pro.local.yaml`
- eval_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_aux_parallel.py`
- eval_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_core98_v2_rebuilt_parallel4/eval_aux_core98`
- worker_count: `6`
- tmux_session: `hybrid_rag_core98_aux_20260416`
- status: `running`
- notes:
  - Worker logs are writing under `eval_aux_core98/logs/`.
  - `tmux_run.log` was not created because the output directory did not exist before shell redirection, but the evaluation processes launched successfully.

## ResearchArc on core98

- launch_time: 2026-04-16 Asia/Shanghai
- method: `research_arc_core98_v6_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_core98`
- release_task_count: `98`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_core98_v6_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `research_arc_core98_v6_parallel4`
- status: `running`
- notes:
  - Answer generation starts with 4 shards and then automatically runs `eval_v31`, `eval_v4`, and `eval_aux`.
  - Current shard split: `25 / 25 / 24 / 24`.

## Hybrid RAG on experiment100 evidenceexp v1

- launch_time: 2026-04-16 Asia/Shanghai
- method: `hybrid_rag_experiment100_evidenceexp_v1_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `hybrid_rag_experiment100_evidenceexp_v1_parallel4`
- status: `launched`
- notes:
  - Uses the rebuilt `experiment100` sourced from `...evidenceexp_v1`.
  - Answer generation runs in 4 shards and then automatically runs `eval_v31`, `eval_v4`, and `eval_aux`.

## ResearchArc on experiment100 evidenceexp v1

- launch_time: 2026-04-16 Asia/Shanghai
- method: `research_arc_experiment100_evidenceexp_v1_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `research_arc_experiment100_evidenceexp_v1_parallel4`
- status: `launched`
- notes:
  - Uses the rebuilt `experiment100` sourced from `...evidenceexp_v1`.
  - Answer generation runs in 4 shards and then automatically runs `eval_v31`, `eval_v4`, and `eval_aux`.
