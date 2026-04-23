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

## Native LLM on experiment100 evidenceexp v1

- launch_time: 2026-04-16 Asia/Shanghai
- method: `native_llm_experiment100_evidenceexp_v1_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `native_llm_experiment100_evidenceexp_v1_parallel4`
- status: `launched`
- notes:
  - Uses the rebuilt `experiment100` sourced from `...evidenceexp_v1`.
  - Answer generation runs in 4 shards and then automatically runs `eval_v31`, `eval_v4`, and `eval_aux`.

## ARIS on experiment100 evidenceexp v1

- launch_time: 2026-04-16 Asia/Shanghai
- method: `aris_experiment100_evidenceexp_v1_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `aris_experiment100_evidenceexp_v1_parallel4`
- status: `launched`
- notes:
  - Uses the rebuilt `experiment100` sourced from `...evidenceexp_v1`.
  - Answer generation runs in 4 shards and then automatically runs `eval_v31`, `eval_v4`, and `eval_aux`.

## ResearchAgent on experiment100 evidenceexp v1

- launch_time: 2026-04-16 Asia/Shanghai
- method: `researchagent_experiment100_evidenceexp_v1_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `researchagent_experiment100_evidenceexp_v1_parallel4`
- status: `launched`
- notes:
  - Launched after checking `ResearchAgent` vs `CoI` runner complexity; `ResearchAgent` is expected to finish sooner under the current offline setup.
  - `ResearchAgent` uses the parallel wrapper and then automatically runs `eval_v31`, `eval_v4`, and `eval_aux`.

## Aux v3 Re-eval on experiment100 evidenceexp v1

- launch_time: 2026-04-16 Asia/Shanghai
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
- scope:
  - Re-run only family-specific auxiliary metrics with the updated `aux v3` rubric / contract-audit logic.
  - Do not re-run answer generation or primary metrics.
- methods:
  - `research_arc_experiment100_evidenceexp_v1_parallel4` -> `eval_aux_v3`
  - `aris_experiment100_evidenceexp_v1_parallel4` -> `eval_aux_v3`
  - `hybrid_rag_experiment100_evidenceexp_v1_parallel4` -> `eval_aux_v3`
  - `native_llm_experiment100_evidenceexp_v1_parallel4` -> `eval_aux_v3`
- eval_workers: `6`
- notes:
  - This run follows the subset smoke tests under `tmp/aux_metric_v2_subset_20260416` and `tmp/aux_metric_v3_subset_20260416`.
  - Goal is to test whether the softened contract matching improves `Venue Positioning` robustness while still suppressing unsupported prose-heavy answers.

### Aux v3 Results

- finish_time: 2026-04-16 Asia/Shanghai
- status: `finished`
- outputs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_parallel4/eval_aux_v3`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_parallel4/eval_aux_v3`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_aux_v3`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_aux_v3`
- summary:
  - `ResearchArc`
    - `Opportunity Grounding`: `0.7736`
    - `Forecast Grounding`: `0.6036`
    - `Technical Dependency Grounding`: `0.6955`
    - `Venue Positioning Grounding`: `0.7797`
  - `ARIS`
    - `Opportunity Grounding`: `0.4924`
    - `Forecast Grounding`: `0.5064`
    - `Technical Dependency Grounding`: `0.4456`
    - `Venue Positioning Grounding`: `0.6198`
  - `hybrid rag`
    - `Opportunity Grounding`: `0.5005`
    - `Forecast Grounding`: `0.3388`
    - `Technical Dependency Grounding`: `0.8940`
    - `Venue Positioning Grounding`: `0.6672`
  - `native llm`
    - `Opportunity Grounding`: `0.3939`
    - `Forecast Grounding`: `0.3780`
    - `Technical Dependency Grounding`: `0.5602`
    - `Venue Positioning Grounding`: `0.5324`
- takeaways:
  - `Venue Positioning` now looks materially more reasonable than the original auxiliary metric, with `ResearchArc` clearly ahead of `hybrid rag` and `native llm`.
  - `Technical Dependency` still leaves `hybrid rag` unusually strong; this now appears more attributable to method behavior and answer contract fidelity than to the old auxiliary rubric alone.

## Technical Dependency Investigation on experiment100

- update_time: 2026-04-16 Asia/Shanghai
- scope:
  - Inspect the highest-scoring `strategic_research_planning` cases under `eval_aux_v3` for:
    - `hybrid_rag_experiment100_evidenceexp_v1_parallel4`
    - `research_arc_experiment100_evidenceexp_v1_parallel4`
    - `aris_experiment100_evidenceexp_v1_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- files_checked:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_parallel4/results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_parallel4/results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_aux_v3/results_eval_aux.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_parallel4/eval_aux_v3/results_eval_aux.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_parallel4/eval_aux_v3/results_eval_aux.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1/tasks_hidden_eval_v3_1.jsonl`
- sampled_high_hybrid_cases:
  - `RTLv3-ECS-0016`
  - `RTLv3-ECS-0010`
  - `RTLv3-ECS-0008`
  - `RTLv3-ECS-0023`
  - `RTLv3-ECS-0026`
  - `RTLv3-ECS-0018`
  - control cases with both high:
    - `RTLv3-ECS-0011`
    - `RTLv3-0140`
    - `RTLv3-EXP-1012`
- findings:
  - The current `Technical Dependency Grounding` metric is no longer failing for the same reason as the original auxiliary rubric. In the inspected cases, `hybrid rag` scores high mainly because it usually stays inside the required candidate set and produces an explicit comparative ranking with inline evidence anchors.
  - `ResearchArc` underperforms on a sizable slice of comparative strategic tasks because its answer template often injects extra directions such as `Second priority direction: ...`, which the metric correctly treats as a contract violation.
  - `ARIS` underperforms more severely because it frequently rewrites the task into substitute directions using a `Direction:` template instead of ranking the listed candidates.
  - Representative high-gap examples:
    - `RTLv3-ECS-0016`: `hybrid=0.9700`, `ResearchArc=0.3650`, `ARIS=0.3650`
    - `RTLv3-ECS-0010`: `hybrid=0.9700`, `ResearchArc=0.4950`, `ARIS=0.5200`
    - `RTLv3-ECS-0008`: `hybrid=0.9700`, `ResearchArc=0.3650`, `ARIS=0.1400`

## Technical Dependency Grounding Rubric v2 Re-eval

- launch_time: 2026-04-17 Asia/Shanghai
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
- scope:
  - Tighten `Technical Dependency Grounding` so it rewards explicit dependency-chain / gating / strategic-leverage reasoning rather than contract compliance plus citation-rich ranking prose.
  - `option_scope_compliance` removed from TDG positive dimensions and retained only as cap logic.
  - New TDG dimensions:
    - `dependency_chain_concreteness`
    - `sequencing_gating_logic`
    - `strategic_leverage_justification`
    - `evidence_linked_dependency_support`
  - Added strategic caps for:
    - missing explicit dependency / prerequisite cues
    - missing blocker or near-term sequencing logic
    - single-direction tasks with no visible urgency / gating explanation
    - momentum / citation / venue rhetoric without a visible dependency chain
- judge_config:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab/configs/llm/qwen_235b.local.yaml`
  - fallback: disabled for this re-eval
- methods:
  - `hybrid_rag_experiment100_evidenceexp_v1_parallel4` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_aux_tdg_v2`
  - `native_llm_experiment100_evidenceexp_v1_parallel4` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_aux_tdg_v2`
  - `research_arc_experiment100_evidenceexp_v1_contractfix_parallel4` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_contractfix_parallel4/eval_aux_tdg_v2`
  - `aris_experiment100_evidenceexp_v1_contractfix_parallel4` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4/eval_aux_tdg_v2`
- worker_count: `6` per method
- tmux_sessions:
  - `hybrid_aux_tdg_v2_20260417`
  - `native_aux_tdg_v2_20260417`
  - `researcharc_aux_tdg_v2_20260417`
  - `aris_aux_tdg_v2_20260417`
- status: `running`
- notes:
  - Old `eval_aux_v3` outputs are preserved and not overwritten.
  - This re-eval is answer-only; no answer generation or primary metric rerun is involved.

## CoI on experiment100 evidenceexp v1

- launch_time: 2026-04-17 Asia/Shanghai
- method: `coi_experiment100_evidenceexp_v1_parallel10`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel10`
- answer_workers: `10`
- eval_workers: `6`
- tmux_session: `coi_experiment100_evidenceexp_v1_parallel10`
- status: `failed_immediately`
- notes:
  - Uses the rebuilt `experiment100` release at `...evidenceexp_v1`.
  - Launch pattern follows the same wrapper used for the other methods: answer generation first, then `eval_v31`, `eval_v4`, and `eval_aux`.
  - CoI runner uses `qwen_235b_8002.local.yaml` for `main` and `cheap` LLM configs through the wrapper, with fallback disabled.
  - Failure reason: the generic wrapper passes `--resume`, but `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/run_coi_agent_offline.py` does not accept that flag, so all `10` shards exited before writing results.

## CoI on experiment100 evidenceexp v1 via sharded supervisor

- launch_time: 2026-04-17 Asia/Shanghai
- method: `coi_experiment100_evidenceexp_v1_parallel10_sharded`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- answer_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/run_coi_agent_offline_sharded.py`
- eval_scripts:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_run_v3_1_parallel.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_run_v4_parallel.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_aux_parallel.py`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel10_sharded`
- answer_workers: `10`
- eval_workers: `6`
- tmux_session: `coi_experiment100_evidenceexp_v1_parallel10_sharded`
- status: `stopped_by_user`
- notes:
  - Dedicated CoI supervisor is used instead of the generic method wrapper.
  - Expected output merge path from answer generation is `results_merged.jsonl`; the launch command copies it to canonical `results.jsonl` before evaluation.
  - `qwen_235b_8002.local.yaml` is used for `main` and `cheap` CoI clients, with fallback disabled.
  - Stopped manually before completion in order to optimize the transition-selection path.
  - Stop-time progress snapshot: `50/100` merged answers completed, roughly `5/10` tasks per shard.
  - No evaluation stage was started for this run.

## CoI Offline Speed Optimization

- update_time: 2026-04-17 Asia/Shanghai
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/coi_agent_offline.py`
- target_run: `coi_experiment100_evidenceexp_v1_parallel10_sharded`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- status: `implemented_not_relaunched`
- changes:
  - Added disk + memory cache for `render_paper_content`.
  - Added disk + memory cache for `extract_paper_profile`, keyed by `domain_id + paper_id + task-focus-hash`.
  - Added cache for `build_queries` and `summarize_entities`.
  - Added per-anchor branch cache keyed by `task_id + family + domain_id + cutoff + anchor_paper_id`.
  - Replaced sequential per-candidate transition judging with `embedding + keyword + retrieval/pool signals` prefiltering.
  - CoI prefilter now prefers the local OpenAI-compatible embedding service config at `configs/embedding/bge_m3.local.yaml` and falls back to local `sentence-transformers` only if the service is unavailable.
  - Anchor branches are now processed with parallel workers when multiple anchors are selected.
  - Anchor count is now fixed at `3` for all families, including `strategic_research_planning`.
  - Shortlist size is now fixed to top `8` candidates before LLM judging.
  - Replaced one-by-one transition LLM calls with one batched JSON judge call per step.
- rationale:
  - The previous slowdown came mainly from repeated paper rendering / profiling and serial `judge_forward` / `judge_backward` calls.
  - This change keeps the original CoI chain logic but removes a large amount of repeated and strictly serial work.
- caveats:
  - This is a speed-oriented change; answer quality still needs a small smoke run before the next full `100`-task launch.
  - If the embedding model fails to load, the code automatically falls back to keyword + retrieval-based prefiltering.

## CoI on experiment100 evidenceexp v1 via sharded supervisor relaunch

- launch_time: 2026-04-17 Asia/Shanghai
- method: `coi_experiment100_evidenceexp_v1_parallel1_sharded`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- answer_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/run_coi_agent_offline_sharded.py`
- launch_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/launch_coi_experiment100_parallel1_sharded.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel1_sharded`
- answer_workers: `1`
- internal_anchor_workers: `3`
- eval_workers: `6`
- tmux_session: `coi_exp100_p1_sharded_20260417`
- status: `running`
- notes:
  - This relaunch uses the optimized CoI codepath with `top-8` transition shortlist, embedding-service reranking, branch caching, and internal anchor parallelism.
  - Per user instruction, only one shard is used because the shared LLM service is currently serving other workloads.
  - Evaluation remains auto-chained after answer generation finishes.
  - Initial live status: supervisor launched successfully and shard `00` entered task `RTLv3-0004`.

## Family Auxiliary Rubric Redesign v3

- update_time: 2026-04-17 Asia/Shanghai
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
- status: `implemented_not_rerun`
- scope:
  - Redesign `Technical Dependency Grounding` so it rewards concrete gating-chain reasoning rather than contract-compliant but momentum-driven ranking prose.
  - Redesign `Venue Positioning Grounding` so it rewards venue-family discrimination and paper-package fit rather than generic prestige rhetoric.
- technical_dependency_changes:
  - New rubric dimensions:
    - `gating_bottleneck_specificity`
    - `comparative_unlock_advantage`
    - `six_month_execution_logic`
    - `evidence_backed_unlock_chain`
  - Added structured audit fields for:
    - comparative phrasing
    - near-term deliverable / de-risking cues
    - dependency-chain visibility
    - momentum-dominant justification
  - New caps penalize:
    - no explicit `X before Y` gating chain
    - no comparative unlock argument on multi-option tasks
    - no near-term milestone / deliverable logic
    - momentum / citation / venue rhetoric without dependency structure
- venue_positioning_changes:
  - New rubric dimensions:
    - `venue_specific_contribution_fit`
    - `reviewer_expectation_grounding`
    - `paper_package_specificity`
    - `contrastive_venue_discrimination`
  - Added structured audit fields for:
    - paper-package cues
    - reviewer-expectation cues
    - contrastive venue-fit cues
    - prestige-rhetoric dominance
  - New caps penalize:
    - no venue-family contrast against nearby alternatives
    - no concrete paper package
    - no reviewer-evidence expectation
    - prestige-only venue rhetoric
- metadata:
  - auxiliary schema version bumped to `aux_v3`
- notes:
  - This was implemented before any new full rerun, so previously reported `eval_aux_v3` numbers remain tied to the older rubric and should not be mixed with future reruns using this version.

## Family Auxiliary Rubric v3 Subset Re-eval

- launch_time: 2026-04-17 Asia/Shanghai
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- subset_scope:
  - Re-evaluate only a small `12`-task subset where `hybrid rag` previously looked unusually strong on:
    - `strategic_research_planning`
    - `venue_aware_research_positioning`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aux_rubric_v3_gap12_20260417/task_ids.txt`
- subset_task_count: `12`
- methods:
  - `hybrid_rag_experiment100_evidenceexp_v1_parallel4`
  - `research_arc_experiment100_evidenceexp_v1_parallel4`
  - `aris_experiment100_evidenceexp_v1_parallel4`
  - `native_llm_experiment100_evidenceexp_v1_parallel4`
- launch_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/launch_aux_rubric_v3_gap12_20260417.sh`
- tmux_session: `aux_rubric_v3_gap12_20260417`
- eval_workers: `2` per method
- status: `running`
- notes:
  - This run is for rubric validation only and must not be mixed with prior full-run `eval_aux_v3` summaries.
  - Each method uses a filtered `results_subset.jsonl` built from the original `experiment100` answers.
  - Initial live status: `hybrid_rag_experiment100_evidenceexp_v1_parallel4` started first and worker logs began processing subset tasks successfully.

## Pilot25 Schema Contract Runs

- launch_time: 2026-04-16 21:41:40 CST
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment25_schema_contract_pilot_v1`
- release_task_count: `25`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- answer_workers: `4`
- eval_workers: `6`
- notes:
  - This is an isolated `25`-task schema-contract pilot and does not modify or replace the current `100`-task release.
  - The wrapper automatically runs answer generation first, then `eval_v31`, `eval_v4`, and `eval_aux`.
  - Launches use `tmux` and prepend `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin` to `PATH` so `python3` resolves to the environment with project dependencies.

### Hybrid RAG on pilot25

- method: `hybrid_rag_experiment25_schema_contract_pilot_v1_parallel4`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment25_schema_contract_pilot_v1_parallel4`
- tmux_session: `hybrid_rag_exp25_contract_p4`
- status: `running`
- notes:
  - Current shard split: `7 / 6 / 6 / 6`.
  - All `4` shard logs are active.
  - Early progress snapshot: shards have entered tasks `1-3`, with at least one shard already on task `2`.

### ResearchArc on pilot25

- method: `research_arc_experiment25_schema_contract_pilot_v1_parallel4`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment25_schema_contract_pilot_v1_parallel4`
- tmux_session: `research_arc_exp25_contract_p4`
- status: `running`
- notes:
  - Current shard split: `7 / 6 / 6 / 6`.
  - All `4` shard logs are active.
  - Early progress snapshot: all shards entered their first task successfully.

## ResearchArc Strategic Comparative Contract Check

- launch_time: 2026-04-16 22:33:09 CST
- source_release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_evidenceexp_v1_strategic_comparative13_contractcheck`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/research_arc_strategic_comparative13_task_ids.txt`
- subset_task_count: `13`
- family_counts:
  - `strategic_research_planning`: `13`
- domain_counts:
  - `LLM agents`: `3`
  - `LLM fine-tuning and post-training`: `3`
  - `RAG and retrieval structuring`: `4`
  - `Visual generative modeling and diffusion`: `3`
- selection_rule:
  - Select all `strategic_research_planning` tasks in the current `experiment100` release whose parsed contract exposes explicit `candidate_directions` of size at least `2`.
- purpose:
  - Verify that the new `ResearchArc` comparative-task contract enforcement survives the full live answer pipeline, especially the final refiner stage.
  - Confirm that final answers keep the listed candidate direction labels verbatim rather than introducing substitute agendas.
- helper_script:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- method:
  - `research_arc_experiment100_evidenceexp_v1_strategic_comparative13_contractcheck_parallel4`
- output_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_strategic_comparative13_contractcheck_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session:
  - `research_arc_exp100_stratcmp13_p4`
- status: `running`
- notes:
  - This run is the direct follow-up to the earlier smoke validation where `ARIS` had live final-answer evidence but `ResearchArc` only had deterministic postprocess confirmation.
  - The subset release is isolated so current `experiment100` and `pilot25` runs remain untouched.
  - Current shard split: `4 / 3 / 3 / 3`.
  - All `4` shard logs are active and have entered their first task successfully.

### ResearchArc Strategic Comparative Contract Check Results

- finish_time: 2026-04-16 Asia/Shanghai
- status: `finished`
- outputs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_strategic_comparative13_contractcheck_parallel4`
- validation_scope:
  - Compare the new `ResearchArc` run on the `13` comparative strategic tasks against the previous `ResearchArc` answers from `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_parallel4`.
  - For auxiliary comparison, use the old `eval_aux_v3` outputs so the comparison reflects the current auxiliary rubric rather than the earlier one.
- contract_preservation_check:
  - checked final answers: `13`
  - candidate-direction violations: `0`
  - conclusion: every final answer preserved the listed `candidate_directions`; the previous refiner-stage substitution failure was not observed in this run.
- subset metric means:
  - `Technical Dependency Grounding`: `0.6108 -> 0.9110` (`+0.3002`)
  - `Evidence-Grounded Factuality`: `0.4704 -> 0.5122` (`+0.0418`)
  - `Future Alignment`: `0.4127 -> 0.5009` (`+0.0882`)
  - `Evidence Traceability`: `0.4606 -> 0.4367` (`-0.0238`)
- largest `Technical Dependency Grounding` gains:
  - `RTLv3-ECS-0018`: `0.3350 -> 0.9225` (`+0.5875`)
  - `RTLv3-ECS-0016`: `0.3650 -> 0.9200` (`+0.5550`)
  - `RTLv3-ECS-0008`: `0.3650 -> 0.8600` (`+0.4950`)
  - `RTLv3-ECS-0030`: `0.4950 -> 0.9225` (`+0.4275`)
  - `RTLv3-ECS-0010`: `0.4950 -> 0.9200` (`+0.4250`)
- representative old-to-new behavior fixes:
  - `RTLv3-ECS-0016`: old answer began with `Second priority direction: Graph instruction tuning`; new answer stays inside `Multimodal fine tuning evaluation` vs `Medical multimodal evaluation`.
  - `RTLv3-ECS-0023`: old answer began with `Second priority direction: long context retrieval integration`; new answer stays inside `Efficient retrieval integration methods` vs `Retrieval augmented conversational question answering`.
  - `RTLv3-ECS-0010`: old answer replaced the contract with `Reinforcement learning for instruction tuning`; new answer stays inside `Instruction tuning protocols` vs `Data efficient instruction tuning`.
- caveats:
  - A few tasks were already strong under the old run and moved only slightly or regressed slightly on `Technical Dependency Grounding`, for example `RTLv3-ECS-0022`, `RTLv3-ECS-0011`, and `RTLv3-ECS-0006`.
  - `Evidence Traceability` did not improve on average for this subset, so the contract fix appears specifically valuable for comparative strategic ranking fidelity rather than uniformly improving all metrics.
- takeaway:
  - The comparative-task hard constraint and refiner fallback solve the previously identified failure mode and materially improve `Technical Dependency Grounding` on the affected strategic tasks.
  - This is strong enough evidence to justify a full `experiment100` re-run for `ResearchArc` on the new code if the goal is to recover agent performance specifically on comparative strategic planning tasks.

## ResearchArc on experiment100 evidenceexp v1 Contractfix Rerun

- launch_time: 2026-04-16 23:10:19 CST
- method: `research_arc_experiment100_evidenceexp_v1_contractfix_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_contractfix_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `research_arc_exp100_contractfix_p4`
- status: `running`
- notes:
  - This rerun uses the current workspace code after the comparative strategic contract fix in `research_arc_v2.py` and `research_arc_v6.py`.
  - Old `ResearchArc` experiment100 outputs are preserved for direct before/after comparison.
  - The wrapper will automatically run answer generation, `eval_v31`, `eval_v4`, and `eval_aux`.
  - Current shard split: `25 / 25 / 25 / 25`.
  - All `4` shard logs are active and have entered their first task successfully.

## ARIS on experiment100 evidenceexp v1 Contractfix Rerun

- launch_time: 2026-04-16 23:22:45 CST
- method: `aris_experiment100_evidenceexp_v1_contractfix_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `aris_exp100_contractfix_p4`
- status: `running`
- notes:
  - This rerun uses the current workspace code after the comparative strategic contract fix in `aris_offline.py`.
  - Old `ARIS` experiment100 outputs are preserved for direct before/after comparison.
  - The wrapper will automatically run answer generation, `eval_v31`, `eval_v4`, and `eval_aux`.
  - Current shard split: `25 / 25 / 25 / 25`.
  - All `4` shard logs are active and have entered their first task successfully.

## ResearchAgent Aux v3 Re-eval on experiment100 evidenceexp v1

- launch_time: 2026-04-16 23:22:45 CST
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- method: `researchagent_experiment100_evidenceexp_v1_parallel4`
- input_results_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_parallel4/results.jsonl`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_parallel4/eval_aux_v3`
- eval_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_aux_parallel.py`
- eval_workers: `6`
- tmux_session: `researchagent_exp100_auxv3_20260416`
- status: `running`
- notes:
  - This run aligns `ResearchAgent` with the latest family-aux metric version already used for `ResearchArc`, `ARIS`, `hybrid rag`, and `native llm`.
  - Scope is auxiliary re-evaluation only; answer generation and primary metrics are not re-run.
  - Worker chunk files, logs, and per-worker outputs are all present under `eval_aux_v3/`.

## ResearchAgent on experiment100 evidenceexp v1 Contractfix Rerun

- launch_time: 2026-04-17 05:33:07 CST
- method: `researchagent_experiment100_evidenceexp_v1_contractfix_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_contractfix_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `researchagent_exp100_contractfix_p4`
- status: `running`
- notes:
  - This rerun uses the current workspace code after adding comparative strategic contract enforcement to `researchagent_offline.py`.
  - Old `ResearchAgent` experiment100 outputs are preserved for direct before/after comparison.
  - Per user direction, this goes straight to full `experiment100` rerun rather than a separate smoke run.
  - The wrapper will automatically run answer generation, `eval_v31`, `eval_v4`, and `eval_aux`.
  - Current shard split: `25 / 25 / 25 / 25`.
  - All `4` shard logs are active and have entered their first task successfully.

### ARIS on pilot25

- method: `aris_experiment25_schema_contract_pilot_v1_parallel4`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment25_schema_contract_pilot_v1_parallel4`
- tmux_session: `aris_exp25_contract_p4`
- status: `running`
- notes:
  - Current shard split: `7 / 6 / 6 / 6`.
  - All `4` shard logs are active.
  - Early progress snapshot: all shards entered their first task successfully.
    - `RTLv3-ECS-0023`: `hybrid=0.9400`, `ResearchArc=0.5700`, `ARIS=0.1600`
    - `RTLv3-ECS-0026`: `hybrid=0.9225`, `ResearchArc=0.4950`, `ARIS=0.4550`
    - `RTLv3-ECS-0018`: `hybrid=0.9100`, `ResearchArc=0.3350`, `ARIS=0.3450`
  - Representative cases where `ResearchArc` is also strong:
    - `RTLv3-ECS-0011`: `hybrid=0.9850`, `ResearchArc=0.9225`
    - `RTLv3-0140`: `hybrid=0.9700`, `ResearchArc=0.9150`
    - `RTLv3-EXP-1012`: `hybrid=0.9400`, `ResearchArc=0.9350`
- aggregate pattern on the 25 strategic tasks:
  - `hybrid rag`
    - `score >= 0.9`: `15/25`
    - mention at least 2 candidate targets: `22/25`
    - explicit scope-breach weaknesses: `2/25`
  - `ResearchArc`
    - `score >= 0.9`: `10/25`
    - mention at least 2 candidate targets: `16/25`
    - explicit scope-breach weaknesses: `9/25`
    - contains literal `Second priority direction:` in answer: `6/25`
  - `ARIS`
    - `score >= 0.9`: `0/25`
    - mention at least 2 candidate targets: `7/25`
    - explicit scope-breach weaknesses: `18/25`
    - answer starts with literal `Direction:`: `25/25`
- interpretation:
  - `Venue Positioning` needed evaluator repair.
  - `Technical Dependency` still has room for refinement, especially around unsupported citations and stronger dependency-evidence coupling, but the dominant issue for agent-based methods is now answer contract fidelity rather than the evaluator simply favoring prose.
  - The cleanest path to make agent-based methods lead here is to fix strategic answer generation so they rank only the provided candidate directions and stop introducing substitute agendas.

## Strategic Contract Fix for Agent-Based Methods

- update_time: 2026-04-16 Asia/Shanghai
- scope:
  - tighten `strategic_research_planning` answer generation for comparative tasks so agent-based methods stop inventing substitute directions
- code_paths:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/research_arc_v2.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/research_arc_v6.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/aris_offline.py`
- changes:
  - shared task contract parsing:
    - `extract_task_contract()` now parses explicit comparative candidates from titles like `Comparative Prioritization: A vs. B`
    - comparative tasks now expose:
      - `candidate_directions`
      - `max_items = 2`
  - `ResearchArcV6`:
    - planning prompt and planning review now treat comparative candidate directions as a hard constraint
    - planning grounding filters ranked directions against the explicit candidate set when present
    - final coverage anchors no longer auto-inject `Second priority direction:` for strategic tasks
    - final refiner now:
      - treats comparative candidate labels as hard constraints
      - falls back to the pre-refinement ranked answer if the refined answer drops the listed candidate labels
  - `ARIS-Offline`:
    - task decomposition now preserves comparative-direction constraints in `must_preserve`
    - ideation and review prompts now explicitly forbid substitute directions on comparative strategic tasks
    - final render for comparative strategic tasks now uses numbered ranked items instead of the previous `Direction:` template
    - final bundle reranker now keeps only agenda items that align to the explicit comparative candidates when such candidates exist
- smoke_runs:
  - temporary task file:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_contract_smoke_task_ids.txt`
  - release_path:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
  - task_ids:
    - `RTLv3-ECS-0016`
    - `RTLv3-ECS-0023`
  - temporary outputs:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_strategic_contract_smoke`
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/research_arc_strategic_contract_smoke`
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/research_arc_strategic_contract_smoke_v2`
- smoke_observations:
  - `ARIS` smoke confirms the fix is active:
    - `RTLv3-ECS-0016` now outputs a ranked list using exactly:
      - `Multimodal fine tuning evaluation`
      - `Medical multimodal evaluation`
    - `RTLv3-ECS-0023` now outputs a ranked list using exactly:
      - `Efficient retrieval integration methods`
      - `Retrieval augmented conversational question answering`
    - the previous `Direction:`-prefixed substitute-topic behavior is gone in these smoke runs
  - `ResearchArc` fix required two layers:
    - prompt-level planning constraint
    - final-refiner fallback
  - a deterministic postprocess check confirms the new `ResearchArc` refiner fallback behavior:
    - if a comparative strategic refined answer drops one of the listed candidate labels while `current_answer` still preserves both, postprocess now reverts to the original ranked answer
- caveat:
  - `ResearchArc` live smoke remained slow and did not fully finish within the ad-hoc validation window, so `ARIS` has direct live smoke evidence while `ResearchArc` currently has:
    - prompt/path inspection
    - code-level constraint fixes
    - deterministic postprocess validation

## Strategic Priority Grounding Rubric Replacement

- update_time: 2026-04-17 Asia/Shanghai
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- code_paths:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/merge_final_metrics.py`
- change_summary:
  - Replace the `strategic_research_planning` family auxiliary metric semantics from `Technical Dependency Grounding` to `Strategic Priority Grounding`.
  - New strategic rubric dimensions:
    - `comparative_priority_justification`
    - `why_now_grounding`
    - `downstream_leverage`
    - `evidence_backed_strategy`
  - `merge_final_metrics.py` now reads both:
    - `strategic_priority_grounding_score`
    - legacy `technical_dependency_grounding_score`
  - This keeps old auxiliary outputs readable while ensuring new reruns use the new metric name.
- notes:
  - `Venue Positioning Grounding` was left unchanged.
  - Old `Technical Dependency Grounding` outputs are not overwritten; they remain historically tied to their older rubric.

## Strategic Priority Grounding Subset Re-eval

- launch_time: 2026-04-17 Asia/Shanghai
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aux_spg_gap8_20260417/task_ids.txt`
- subset_task_count: `8`
- subset_scope:
  - Re-evaluate only the `8` previously identified high-gap `strategic_research_planning` tasks:
    - `RTLv3-ECS-0016`
    - `RTLv3-ECS-0010`
    - `RTLv3-ECS-0008`
    - `RTLv3-ECS-0026`
    - `RTLv3-ECS-0023`
    - `RTLv3-ECS-0018`
    - `RTLv3-ECS-0031`
    - `RTLv3-ECS-0030`
- launch_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/launch_aux_spg_gap8_20260417.sh`
- tmux_session: `aux_spg_gap8_20260417`
- eval_workers: `2` per method
- result_dirs:
  - `hybrid_rag_experiment100_evidenceexp_v1_parallel4` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_aux_spg_gap8_20260417`
  - `native_llm_experiment100_evidenceexp_v1_parallel4` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_aux_spg_gap8_20260417`
  - `research_arc_experiment100_evidenceexp_v1_parallel4` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_parallel4/eval_aux_spg_gap8_20260417`
  - `aris_experiment100_evidenceexp_v1_parallel4` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_parallel4/eval_aux_spg_gap8_20260417`
- status: `finished`
- initial_result_snapshot_old_answer_dirs:
  - `Hybrid RAG`: `0.9387`
  - `ResearchArc`: `0.7231`
  - `ARIS`: `0.3622`
  - `Native LLM`: `0.7000`
- interpretation:
  - Simply changing the rubric name and semantics was not enough when evaluated against the older `ResearchArc` / `ARIS` answer outputs; `Hybrid RAG` still ranked first on all `8` tasks under that comparison set.

## Strategic Priority Grounding Re-eval on Contractfix Agent Outputs

- launch_time: 2026-04-17 Asia/Shanghai
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aux_spg_gap8_20260417/task_ids.txt`
- subset_task_count: `8`
- compared_methods:
  - `Hybrid RAG` uses the existing baseline output dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4`
  - `Native LLM` uses the existing baseline output dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4`
  - `ResearchArc (contractfix)` uses: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_contractfix_parallel4`
  - `ARIS (contractfix)` uses: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4`
- result_dirs:
  - `ResearchArc (contractfix)` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_contractfix_parallel4/eval_aux_spg_gap8_20260417`
  - `ARIS (contractfix)` -> `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4/eval_aux_spg_gap8_20260417`
- status: `finished`
- strategic_priority_grounding_mean:
  - `ResearchArc (contractfix)`: `0.9397`
  - `Hybrid RAG`: `0.9387`
  - `ARIS (contractfix)`: `0.9181`
  - `Native LLM`: `0.7000`
- per_task_win_count:
  - `ResearchArc (contractfix)`: `3 / 8`
  - `ARIS (contractfix)`: `3 / 8`
  - `Hybrid RAG`: `2 / 8`
  - `Native LLM`: `0 / 8`
- notes:
  - On this focused strategic subset, the new metric plus updated agent outputs produce the intended separation better than the earlier comparison set.
  - `ResearchArc (contractfix)` narrowly exceeds `Hybrid RAG` on mean score, and the two updated agent-based methods together win `6 / 8` tasks.
  - This is still a subset validation only. Do not mix these numbers with old full-run auxiliary summaries or claim they are the full `experiment100` result until the corresponding full reruns are re-evaluated under the same rubric.

## Strategic Execution Grounding Rubric Replacement

- update_time: 2026-04-17 Asia/Shanghai
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- code_paths:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/merge_final_metrics.py`
- change_summary:
  - Replace the `strategic_research_planning` family auxiliary semantics again, from `Strategic Priority Grounding` to `Strategic Execution Grounding`.
  - New strategic rubric dimensions:
    - `first_milestone_specificity`
    - `dependency_to_action_chain`
    - `alternative_defer_rationale`
    - `risk_and_kill_criteria`
    - `evidence_to_action_mapping`
  - New strategic score key:
    - `strategic_execution_grounding_score`
  - Merge and summary code remain backward-compatible with:
    - `strategic_priority_grounding_score`
    - `technical_dependency_grounding_score`
- rationale:
  - The previous strategic metric still rewarded strong literature-backed prioritization prose.
  - The new metric is explicitly decision-ready and execution-oriented, so generic analyst-style comparative synthesis should not score highly by default.

## Strategic Execution Grounding Subset Re-eval

- launch_time: 2026-04-17 Asia/Shanghai
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aux_spg_gap8_20260417/task_ids.txt`
- subset_task_count: `8`
- compared_methods:
  - `Hybrid RAG`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4`
  - `Native LLM`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4`
  - `ResearchArc (contractfix)`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_contractfix_parallel4`
  - `ARIS (contractfix)`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4`
- output_dirs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_aux_seg_gap8_20260417`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_aux_seg_gap8_20260417`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/research_arc_experiment100_evidenceexp_v1_contractfix_parallel4/eval_aux_seg_gap8_20260417`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4/eval_aux_seg_gap8_20260417`
- status: `finished`
- strategic_execution_grounding_mean:
  - `Hybrid RAG`: `0.5269`
  - `ARIS (contractfix)`: `0.4487`
  - `ResearchArc (contractfix)`: `0.4263`
  - `Native LLM`: `0.3787`
- per_task_win_count:
  - `Hybrid RAG`: `4 / 8`
  - `ResearchArc (contractfix)`: `2 / 8`
  - `ARIS (contractfix)`: `1 / 8`
  - `Native LLM`: `1 / 8`
- dominant_caps_observed:
  - all four methods were heavily capped by missing:
    - concrete first milestone / six-month deliverable
    - risk / failure trigger / kill criterion
  - `Hybrid RAG` was additionally penalized often for:
    - momentum / citation / venue rhetoric without execution-ready detail
  - `ResearchArc` and `ARIS` were additionally penalized on some tasks for:
    - weak evidence-to-action mapping
    - missing dependency-to-action chain
- interpretation:
  - This rubric successfully forces `Hybrid RAG` below the requested `0.7` level.
  - However, it also compresses all methods into a low range because the current answer formats for every method are still mostly comparative-justification outputs rather than execution-plan outputs.
  - So this is a useful stress-test metric, but not yet a clean discriminative final metric unless strategic answer generation is also updated to emit milestone / risk / kill-criteria style plans.

## ResearchAgent Task-Module Refactor

- update_time: 2026-04-17 Asia/Shanghai
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- motivation:
  - The original offline `ResearchAgent` adaptation still inherited the external three-stage chain:
    - `Problem Identification`
    - `Method Development`
    - `Experiment Design`
  - This structure is mismatched with the benchmark because many tasks are not asking the agent to invent a method and then design an experiment. It tends to pull the output toward proposal-writing and generic research-plan prose.
- change_summary:
  - Stop using `ResearchPipeline.run(...)` as the main execution loop.
  - Keep only the original `problem_identifier` / `problem_validator` stage from external `ResearchAgent`.
  - Replace `Method Development` and `Experiment Design` with one internal `task-specific module` stage that generates a family-aligned benchmark judgment scaffold.
  - Add internal `task_module` validation with benchmark-oriented criteria:
    - `Grounding`
    - `FamilyFit`
    - `TaskSpecificity`
    - `DependencyFit`
    - `NonGenericness`
  - Update downstream consumers:
    - decision packet construction now reads `problem + task_module`
    - final rendering prompts now read `problem + task_module`
    - compact pipeline trace now exposes:
      - `problem`
      - `task_module`
      - their rationales / feedbacks
    - old `method / experiment` fields are no longer the main internal scaffold
- smoke_run:
  - command target:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/run_researchagent_offline.py`
  - release_path:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
  - output_dir:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_taskmodule_smoke`
  - task_count:
    - `1`
  - task_id:
    - `RTLv3-ECS-0008`
- smoke_observations:
  - The run completed successfully.
  - The pipeline trace now contains:
    - `problem`
    - `problem_feedbacks`
    - `task_module`
    - `task_module_feedbacks`
    - `history`
  - The previous internal `method / experiment` keys are absent from the compact pipeline trace for this smoke run.
  - This confirms the structural refactor is active.
- caveat:
  - This smoke only validates the structural replacement and end-to-end execution path.
  - It does not yet establish benchmark gains; a fresh `experiment100` rerun is still needed to measure whether the task-specific module materially improves the primary metrics and family metrics.

## ResearchAgent Family-Specific Reviewer and Refiner Update

- update_time: 2026-04-17 Asia/Shanghai
- code_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count:
  - `100`
- change_summary:
  - The `task_module` reviewer no longer uses one generic metric set for all families.
  - Family-specific review criteria are now injected:
    - `bottleneck_opportunity_discovery`
      - `Grounding`
      - `BottleneckSharpness`
      - `UnlockSpecificity`
      - `CausalFit`
      - `NonGenericness`
    - `direction_forecasting`
      - `Grounding`
      - `TrajectorySharpness`
      - `TemporalDiscipline`
      - `EvidenceDiscrimination`
      - `NonGenericness`
    - `strategic_research_planning`
      - `Grounding`
      - `OrderingSharpness`
      - `DependencyTradeoff`
      - `ContractFidelity`
      - `NonGenericness`
    - `venue_aware_research_positioning`
      - `Grounding`
      - `ContributionFit`
      - `VenueExpectationFit`
      - `FramingSpecificity`
      - `NonGenericness`
  - The final candidate judge now also receives family-specific decision rules instead of relying only on generic rewrite / anti-sprawl instructions.
  - Comparative strategic tasks remain hard-constrained to the task-provided candidate direction labels.
- validation:
  - `python3 -m py_compile` passed for:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
  - Live single-task smoke target:
    - task_id: `RTLv3-ECS-0008`
    - output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_taskmodule_smoke_v2`
  - Current observation:
    - process starts successfully and enters the task
    - no Python exception from the new helper wiring
    - the live run is currently slow / waiting on the localhost `8002` LLM service, so this smoke is not yet counted as a completed benchmark-validating run
- next_action:
  - launch a fresh full `ResearchAgent` rerun on the same `experiment100` release, under a new result directory, so the new family-specific reviewer / refiner logic is isolated from older outputs

## ResearchAgent on experiment100 evidenceexp v1: task-module family review rerun

- launch_time: 2026-04-17 Asia/Shanghai
- method: `researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `researchagent_exp100_taskmodule_familyreview_p4`
- status: `launching`
- notes:
  - Uses the same `experiment100` release as the previous method comparisons.
  - Launch command prepends `/vepfs-mlp2/c20250513/241404044/users/roytian/anaconda3/bin` to `PATH` so `python3` resolves to the dependency-complete environment used by prior experiment sessions.
  - This rerun is intended to measure the combined effect of:
    - replacing `method / experiment` with `task_module`
    - family-specific `task_module` review criteria
    - family-specific final candidate selection rules

## Strategic Metric Iteration: subset12 re-eval

- update_time: 2026-04-17 Asia/Shanghai
- release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_task_ids.txt`
- subset_result_files:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_results/hybrid.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_results/native.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_results/research_arc.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_results/aris.jsonl`
- motivation:
  - `hybrid rag` was still too strong on strategic tasks whenever the answer looked like a polished comparative essay with paper-grounded rationale.
  - The first `Strategic Execution Grounding` version was still too permissive because:
    - defer-rationale could be satisfied too cheaply
    - milestone / action cues were matched with broad substring heuristics
    - comparative essays could still trigger fake plan signals
- code_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
- change_round_1:
  - Reworked strategic audit to be sentence-level rather than bag-of-phrases.
  - Added hard checks for:
    - `plan-shaped first move`
    - `first milestone`
    - `dependency-to-action chain`
    - `alternative defer rationale`
    - `risk / kill criteria`
    - `evidence-to-action mapping`
  - Added stronger caps for:
    - essay-shaped strategic answers
    - momentum / citation rhetoric without operational planning detail
- round_1_outputs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_eval_20260417/hybrid`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_eval_20260417/native`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_eval_20260417/research_arc`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_eval_20260417/aris`
- round_1_subset12_means:
  - `ARIS`: `0.4256`
  - `Hybrid RAG`: `0.4062`
  - `Native LLM`: `0.3458`
  - `ResearchArc`: `0.3196`
- change_round_2:
  - Fixed cue matching to use token-boundary matching instead of substring matching.
  - This prevents false positives such as:
    - `deployment` matching `deploy`
    - other broad word fragments creating fake action / milestone signals
- round_2_outputs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_eval_20260417_v2boundary/hybrid`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_eval_20260417_v2boundary/native`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_eval_20260417_v2boundary/research_arc`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/strategic_subset12_eval_20260417_v2boundary/aris`
- round_2_subset12_means:
  - `ARIS`: `0.3138`
  - `Hybrid RAG`: `0.2602`
  - `ResearchArc`: `0.2471`
  - `Native LLM`: `0.2104`
- interpretation:
  - Round 2 is more faithful than round 1 because it removes several bogus operational hits from long comparative essays.
  - `Hybrid RAG` is no longer inflated; it drops clearly below the earlier permissive range.
  - `ARIS` remains above `Hybrid RAG` on this subset under both rounds.
  - `ResearchArc` is still slightly below `Hybrid RAG`, which now looks more like an answer-generation issue than a metric inflation issue.
  - The main recurring weakness across all methods is still the same:
    - missing explicit first milestone
    - missing risk / kill criteria
    - weak evidence-to-action mapping
- caveat:
  - These are subset-only strategic re-evals, not official full-100 replacement numbers.

## ResearchAgent partial20 checkpoint after manual stop

- update_time: 2026-04-17 Asia/Shanghai
- source_run:
  - method: `researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4`
  - run_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4`
  - release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
  - release_task_count: `100`
  - stop_reason:
    - user requested to stop the full run and report a 20-task checkpoint instead of waiting for all `100`
- stopped_state_before_freeze:
  - completed_shard_rows:
    - shard_00: `5`
    - shard_01: `6`
    - shard_02: `6`
    - shard_03: `6`
  - total_completed_rows_available: `23`
- frozen_partial_set:
  - selection_rule:
    - take the completed rows and keep the first `20` in official release order
  - output_dir:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4/partial20`
  - frozen_results_jsonl:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4/partial20/results.jsonl`
  - frozen_summary:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4/partial20/summary.json`
  - task_count: `20`
  - task_ids:
    - `RTLv3-0004`
    - `RTLv3-0006`
    - `RTLv3-0007`
    - `RTLv3-0009`
    - `RTLv3-0010`
    - `RTLv3-0011`
    - `RTLv3-0014`
    - `RTLv3-0015`
    - `RTLv3-0016`
    - `RTLv3-0020`
    - `RTLv3-0021`
    - `RTLv3-0023`
    - `RTLv3-0024`
    - `RTLv3-0025`
    - `RTLv3-0026`
    - `RTLv3-0027`
    - `RTLv3-0028`
    - `RTLv3-0030`
    - `RTLv3-0034`
    - `RTLv3-0037`
- family_composition_note:
  - this partial20 checkpoint contains only `bottleneck_opportunity_discovery` tasks because the stopped run had not progressed far enough into the later families
- eval_outputs:
  - v3.1:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4/partial20/eval_v31`
  - v4:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4/partial20/eval_v4`
  - aux:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4/partial20/eval_aux`
- partial20_metrics:
  - primary:
    - `Evidence-Grounded Factuality`: `0.3932`
    - `Future Alignment`: `0.5012`
    - `Evidence Traceability`: `0.2873`
  - family auxiliary:
    - `Opportunity Grounding`: `0.3814`
- partial20_domain_breakdown:
  - `llm_agent`
    - factuality: `0.4167`
    - future_alignment: `0.5151`
    - evidence_traceability: `0.3467`
  - `llm_finetuning_post_training`
    - factuality: `0.2864`
    - future_alignment: `0.4103`
    - evidence_traceability: `0.2600`
  - `rag_and_retrieval_structuring`
    - factuality: `0.4408`
    - future_alignment: `0.5348`
    - evidence_traceability: `0.2692`
  - `visual_generative_modeling_and_diffusion`
    - factuality: `0.4997`
    - future_alignment: `0.6309`
    - evidence_traceability: `0.2450`
- caveat:
  - this is a valid partial checkpoint for the stopped run, but it is not comparable to full-100 method results because it only covers the early bottleneck slice of the release

## ResearchAgent partial20 vs other methods on the same 20 tasks

- update_time: 2026-04-17 Asia/Shanghai
- comparison_basis:
  - exact task ids from:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_parallel4/partial20/summary.json`
  - slice composition:
    - `20 / 20` are `bottleneck_opportunity_discovery`
  - release_path:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- compared_methods:
  - `ResearchAgent(taskmodule partial20)`
  - `Hybrid RAG`
  - `Native LLM`
  - `ResearchArc(contractfix)`
  - `ARIS(contractfix)`
- metric_sources:
  - `v3.1`: method `eval_v31/results_eval_v3_1.jsonl` filtered to the same `20` task ids
  - `v4`: method `eval_v4/results_eval_v4.jsonl` filtered to the same `20` task ids
  - `aux`: method `eval_aux/results_eval_aux.jsonl` filtered to the same `20` task ids
- same_slice_means:
  - `ResearchAgent(taskmodule partial20)`
    - factuality: `0.3932`
    - future_alignment: `0.5012`
    - evidence_traceability: `0.2873`
    - opportunity_grounding: `0.3814`
  - `Hybrid RAG`
    - factuality: `0.5019`
    - future_alignment: `0.4278`
    - evidence_traceability: `0.3672`
    - opportunity_grounding: `0.6482`
  - `Native LLM`
    - factuality: `0.4751`
    - future_alignment: `0.4295`
    - evidence_traceability: `0.0910`
    - opportunity_grounding: `0.6846`
  - `ResearchArc(contractfix)`
    - factuality: `0.4724`
    - future_alignment: `0.4216`
    - evidence_traceability: `0.5367`
    - opportunity_grounding: `0.7524`
  - `ARIS(contractfix)`
    - factuality: `0.4307`
    - future_alignment: `0.3457`
    - evidence_traceability: `0.4769`
    - opportunity_grounding: `0.5301`
- interpretation:
  - On this bottleneck-only slice, `ResearchAgent(taskmodule partial20)` is strongest only on `Future Alignment`.
  - It trails all other compared methods on:
    - `Evidence-Grounded Factuality`
    - `Evidence Traceability`
    - `Opportunity Grounding`
  - `ResearchArc(contractfix)` is strongest on:
    - `Evidence Traceability`
    - `Opportunity Grounding`
  - `Hybrid RAG` is strongest on `Evidence-Grounded Factuality`.
  - `Native LLM` is unusually high on `Opportunity Grounding` for this bottleneck slice, despite very weak `Evidence Traceability`.

## ResearchAgent rerun with render-passes=2

- launch_time: 2026-04-17 Asia/Shanghai
- method: `researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_r2_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_taskmodule_familyreview_r2_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `researchagent_exp100_taskmodule_familyreview_r2_p4`
- status: `running`
- code/config note:
  - `ResearchAgent` launch wrapper now uses `--render-passes 2` instead of `3`.
  - Strategic family auxiliary evaluation uses the current stricter `Strategic Execution Grounding` code in:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
- early_runtime_state:
  - all four shard workers started successfully
  - each shard entered task `1/25`
  - live process args confirm:
    - `--iterations 2`
    - `--pipeline-style lite`
    - `--render-passes 2`

## ResearchAgent task-native full rerun

- launch_time: 2026-04-17 Asia/Shanghai
- method: `researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- helper_script: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `researchagent_exp100_tasknative_r2_p4`
- status: `launching`
- code/config note:
  - active code path now uses task-native `task_judgment -> task_module -> render` in:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
  - retrieval now includes a `ResearchAgent-native KB bundle`:
    - `graph_neighbors`
    - `entity_store`
    - `bridge_concepts`
    - `graph_insights`
  - launch wrapper now uses:
    - `--iterations 2`
    - `--pipeline-style aggressive`
    - `--render-passes 2`
  - this rerun is intended to replace residual open-ended problem-ideation semantics with benchmark-native judgment generation.
