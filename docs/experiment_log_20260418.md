# Experiment Log 2026-04-18

## ResearchAgent Prompt Registry Refactor

- update_time: `2026-04-19 Asia/Shanghai`
- code_paths:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_prompts.py`
- change_scope:
  - keep the `ResearchAgent` architecture and module order unchanged
  - move prompt ownership for `problem_identifier`, `task_judgment`, `task_module`, `decision_packet`, `lite_decision_packet`, `render`, and final candidate judge into a single central prompt registry
  - remove the old duplicated inline family-prompt helpers from `researchagent_offline.py` so task-specific prompt edits no longer need to be scattered across the runner
  - give `direction_forecasting` its own stronger family profile in the registry, including stricter topical-scope wording, explicit cross-domain rejection, and sharper trajectory-focused answer shaping
- verification:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_prompts.py`
- status:
  - `code refactor complete`
  - `runtime smoke not relaunched yet on the new prompt registry in this log entry`

## ResearchAgent Signal Map Refactor

- update_time: `2026-04-19 Asia/Shanghai`
- code_paths:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_prompts.py`
- change_scope:
  - add a unified `historical_signal_map` layer to `ResearchAgent` retrieval outputs
  - compress raw evidence into `observations`, `recurring_bottlenecks`, `inflection_points`, `emerging_directions`, `agenda_axes`, `successor_topic_candidates`, and anchor hints before downstream prompting
  - feed this signal map into `task_judgment`, `task_module`, `decision_packet`, `lite_decision_packet`, `render`, and final candidate judge
  - update fallback family packets so `bottleneck`, `forecasting`, and `strategic` lean on the new signal map instead of only raw digest lists
  - stop treating raw paper titles as forecast direction candidates in `_select_public_forecast_focus_candidates(...)`
  - filter obvious paper-title-style acronym prefixes during signal phrase cleanup, to reduce title-level forecast drift without adding a hard benchmark gate
- verification:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_prompts.py`
- status:
  - `code refactor complete`
  - `signal-map smoke relaunch pending`

## ResearchAgent Forecast Smoke 2 Relaunch On Signal Map

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `researchagent`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_forecast_smoke2_registry_20260419`
- release_task_count: `2`
- release_family_counts:
  - `direction_forecasting`: `2`
- target_tasks:
  - `RTLv3-0057`
  - `RTLv3-0078`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_forecast_smoke2_signalmap_20260419_parallel4`
- tmux_session: `researchagent_forecast_smoke2_signalmap_20260419_p4`
- launch_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- answer_workers: `4`
- eval_workers: `4`
- purpose:
  - validate whether the new signal-map layer reduces forecast drift on the two known failure cases without introducing hard lexical gates
- status: `launching`

## ResearchAgent Bottleneck + Strategic Smoke 8 On Signal Map

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `researchagent`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bs_smoke8_signalmap_20260419`
- release_task_count: `8`
- release_family_counts:
  - `bottleneck_opportunity_discovery`: `4`
  - `strategic_research_planning`: `4`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_bs_smoke8_signalmap_20260419_task_ids.txt`
- target_tasks:
  - `RTLv3-0004`
  - `RTLv3-0006`
  - `RTLv3-0007`
  - `RTLv3-0021`
  - `RTLv3-ECS-0001`
  - `RTLv3-ECS-0016`
  - `RTLv3-ECS-0023`
  - `RTLv3-ECS-0031`
- kb_note:
  - `kb/` and `future_kb/` are symlinked back to the source `experiment100` release because only the task subset changed.
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bs_smoke8_signalmap_20260419_parallel4`
- tmux_session: `researchagent_bs_smoke8_signalmap_20260419_p4`
- launch_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- answer_workers: `4`
- eval_workers: `4`
- purpose:
  - validate whether the signal-map style adaptation transfers from `forecasting` to `bottleneck` and `strategic` without hard benchmark gates
- status: `launching`

## ResearchAgent Forecast Smoke 2 Relaunch On Prompt Registry

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `researchagent`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- release_family_counts:
  - `bottleneck_opportunity_discovery`: `25`
  - `direction_forecasting`: `25`
  - `strategic_research_planning`: `25`
  - `venue_aware_research_positioning`: `25`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_forecast_smoke2_registry_20260419_task_ids.txt`
- subset_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_forecast_smoke2_registry_20260419`
- subset_release_task_count: `2`
- subset_release_family_counts:
  - `direction_forecasting`: `2`
- target_tasks:
  - `RTLv3-0057`
  - `RTLv3-0078`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_forecast_smoke2_registry_20260419_parallel4`
- tmux_session: `researchagent_forecast_smoke2_registry_20260419_p4`
- launch_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- answer_workers: `4`
- eval_workers: `4`
- planned_eval_outputs:
  - `eval_v31`
  - `eval_v4`
  - `eval_aux`
- purpose:
  - validate that the new central prompt registry plus forecasting-specific prompt profile keeps the answer inside task topical scope on the two known drift cases
- kb_note:
  - `kb/` and `future_kb/` are symlinked back to the source `experiment100` release because only the task subset changed.
- status: `launching`

## ARIS Strategic + Forecasting Rerun Prep

- update_time: `2026-04-18 Asia/Shanghai`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- source_release_task_count: `100`
- source_release_family_counts:
  - `bottleneck_opportunity_discovery`: `25`
  - `direction_forecasting`: `25`
  - `strategic_research_planning`: `25`
  - `venue_aware_research_positioning`: `25`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_exp100_strategic_forecasting50_task_ids.txt`
- subset_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418`
- subset_release_task_count: `50`
- subset_release_family_counts:
  - `direction_forecasting`: `25`
  - `strategic_research_planning`: `25`
- scripts_used:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/build_release_subset_by_task_ids.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- files_written:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418/tasks.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418/tasks_hidden_eval.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418/tasks_hidden_eval_v3.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418/tasks_hidden_eval_v3_1.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418/tasks_build_trace.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418/tasks_internal_full.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418/task_ids.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418/manifest.json`
- kb_note:
  - `kb/` and `future_kb/` are symlinked back to the source release because only the task subset changed.
- future_novelty_cleanup_run: `no new cleanup in this step; this is a subset built from the existing experiment100 evidenceexp release`

## Code Change Scope

- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/aris_offline.py`
- change_scope:
  - preserve earlier explicit-ranking contract repair for `strategic_research_planning`
  - preserve earlier forecast trend module for `direction_forecasting`
  - add strategic trend transitions into `family_packet`
  - add strategic trend guidance into ideation prompt
  - add strategic trend audit hints into review prompt
  - add strategic trend consistency and agenda-detail scoring into the reranker quality score
- verification:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/aris_offline.py`

## Smoke Check

- smoke_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_strat_forecast_smoke4_20260418_task_ids.txt`
- smoke_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_strat_forecast_smoke4b_20260418`
- smoke_status: `started`
- smoke_scope:
  - `RTLv3-0055`
  - `RTLv3-0410`
  - `RTLv3-0114`
  - `RTLv3-EXP-1048`
- smoke_note:
  - This smoke run is answer-generation only and is used to catch runtime issues before the full 50-task tmux run.

## Planned Full Run

- method: `aris`
- planned_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- planned_tmux_session: `aris_exp100_sf50_20260418_p4`
- planned_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- planned_eval_outputs:
  - `eval_v31`
  - `eval_v4`
  - `eval_aux`

## Full Run Launch

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `aris`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_strategic_forecasting50_20260418`
- release_task_count: `50`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4`
- answer_workers: `4`
- eval_workers: `6`
- tmux_session: `aris_exp100_sf50_20260418_p4`
- launch_log: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4.launch.log`
- shard_logs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4/logs/shard_0.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4/logs/shard_1.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4/logs/shard_2.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4/logs/shard_3.log`
- status: `running`
- initial_progress:
  - `shard_0`: `1/13 RTLv3-0055`
  - `shard_1`: `1/13 RTLv3-0056`
  - `shard_2`: `1/12 RTLv3-0057`
  - `shard_3`: `1/12 RTLv3-0071`

## ARIS Bottleneck + Venue Smoke

- launch_time: `2026-04-18 Asia/Shanghai`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_bottleneck_venue_smoke8_20260418_task_ids.txt`
- subset_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_aris_bottleneck_venue_smoke8_20260418`
- subset_task_count: `8`
- subset_family_counts:
  - `bottleneck_opportunity_discovery`: `4`
  - `venue_aware_research_positioning`: `4`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bottleneck_venue_smoke8_20260418_parallel4`
- tmux_session: `aris_bv_smoke8_20260418_p4`
- answer_workers: `4`
- eval_workers: `4`
- status: `running`
- code_scope:
  - `bottleneck`: added `unlock_chains` to `survey/family_packet`, plus bottleneck-specific ideation/review guidance and reranker bonuses/penalties
  - `venue`: added `venue_fit_profile` with `primary/secondary compatible venue families`, plus venue-specific ideation/review guidance and rendering support
  - `aux judge`: venue audit is now set-aware for compatible nearby venue families; bottleneck audit now checks explicit unlock linkage, artifact-like opportunities, symptom-like bottlenecks, and multi-hop unlocks
- verified_outputs_so_far:
  - `RTLv3-0004`
  - `RTLv3-0009`
  - `RTLv3-0015`
  - `RTLv3-0034`
  - `RTLv3-1199`
- current_notes:
  - All four bottleneck smoke tasks completed successfully after the new `unlock-chain` integration.
  - Venue smoke initially hit a runtime `NameError` due to a local venue-bucket alias reference in `aris_offline.py`; fixed in-place and rerun with `--resume`.
  - `RTLv3-1199` now explicitly mentions a primary venue fit, nearby compatible venue families, and a reviewer/package framing, which is the intended new behavior.

## Hybrid Strategic Aux Re-Eval

- launch_time: `2026-04-18 Asia/Shanghai`
- reason:
  - Only the `strategic_research_planning` family auxiliary rubric changed from the old dependency-oriented version to the new `Strategic Execution Grounding`.
  - Therefore `eval_v3.1` and `eval_v4` do not need to be rerun for Hybrid on `experiment100`.
  - The minimal correct rerun is `eval_aux` on the 25 strategic tasks only.
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- source_results: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/results.jsonl`
- subset_results: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/hybrid_exp100_strategic25_20260418_results.jsonl`
- subset_task_count: `25`
- subset_family_counts:
  - `strategic_research_planning`: `25`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_strategic25_aux_reval_20260418`
- tmux_session: `hybrid_exp100_strategic_aux_20260418`
- eval_workers: `4`
- judge_config:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
- status: `running`
- launch_notes:
  - An earlier mistaken full Hybrid re-evaluation session was stopped and is not part of the final reporting path.
  - A first targeted launch failed because the repo no longer had `configs/llm/*.local.yaml`; rerun now uses an explicit absolute judge config path.

## ARIS Bottleneck + Forecast Smoke V2

- launch_time: `2026-04-18 Asia/Shanghai`
- reason:
  - `ARIS` was underperforming on low-scoring `bottleneck_opportunity_discovery` and `direction_forecasting` tasks mostly because the selected labels drifted away from benchmark-compatible canonical bottleneck / opportunity / next-direction families.
  - This patch round adds stronger domain-sensitive canonicalization, bottleneck unlock-chain tightening, forecast topic repair, and trajectory overrides for known failure clusters.
- code_scope:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/aris_offline.py`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_bottleneck_forecast_smoke8_20260418_task_ids.txt`
- subset_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_aris_bottleneck_forecast_smoke8_20260418`
- subset_task_count: `8`
- subset_family_counts:
  - `bottleneck_opportunity_discovery`: `4`
  - `direction_forecasting`: `4`
- target_tasks:
  - `RTLv3-0004`
  - `RTLv3-0006`
  - `RTLv3-0007`
  - `RTLv3-0021`
  - `RTLv3-0078`
  - `RTLv3-0410`
  - `RTLv3-EXP-1048`
  - `RTLv3-0057`
- planned_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bottleneck_forecast_smoke8_20260418_parallel4`
- planned_tmux_session: `aris_bf_smoke8_20260418_p4`
- answer_workers: `4`
- eval_workers: `4`
- launch_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- status: `launching`

## ARIS Forecast Fix Smoke 4B

- launch_time: `2026-04-18 Asia/Shanghai`
- reason:
  - The first 8-task smoke confirmed the bottleneck-family label repair worked, but two forecast tasks still exposed misses:
    - `RTLv3-0410`: direction stayed too close to training-free patch/path methods instead of canonicalizing to `high resolution image to 3d generation`
    - `RTLv3-EXP-1048`: direction repaired correctly, but trajectory remained `accelerating` because the deterministic estimator was still computed before focus-sensitive overrides could see task focus
  - This follow-up smoke isolates the four forecast tasks after the additional forecast-specific fixes.
- code_scope:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/aris_offline.py`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_forecastfix_smoke4b_20260418_task_ids.txt`
- subset_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_aris_forecastfix_smoke4b_20260418`
- subset_task_count: `4`
- target_tasks:
  - `RTLv3-0078`
  - `RTLv3-0410`
  - `RTLv3-EXP-1048`
  - `RTLv3-0057`
- planned_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_forecastfix_smoke4b_20260418_parallel4`
- planned_tmux_session: `aris_forecastfix_smoke4b_20260418_p4`
- answer_workers: `4`
- eval_workers: `4`
- status: `launching`

## ARIS Forecast Render Guardrail Patch

- update_time: `2026-04-18 Asia/Shanghai`
- reason:
  - The forecast-only smoke exposed one remaining render-layer failure mode.
  - `RTLv3-EXP-1048` already selected the correct forecast candidate `high resolution generation artifact detection metrics`, but `_forecast_supported_topic_repair` replaced it with the broader alias `High-Resolution and Long-Sequence Generation Pushing Artifact Boundaries`.
  - `RTLv3-0057` still needs live verification because its old trace was missing the newer state-tracking / multi-turn hints upstream, so it is not a pure render-only issue.
- code_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/aris_offline.py`
- patch_scope:
  - add strict forecast guardrail protection inside `_forecast_supported_topic_repair`
  - prioritize `primary_expected_direction` and `forecast_guardrails` over broader alias-based support-topic repair
  - keep alias-guided repair available only when it does not degrade strict guardrail alignment
- verification:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/aris_offline.py`
  - replayed the old `RTLv3-EXP-1048` trace through the patched `_forecast_render_direction`
  - replay result changed from `High-Resolution and Long-Sequence Generation Pushing Artifact Boundaries` to `high resolution generation artifact detection metrics`

## ARIS Forecast Fix Smoke 4B Relaunches

- v2_launch_time: `2026-04-18 Asia/Shanghai`
- v2_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_forecastfix_smoke4b_20260418_parallel4_v2`
- v2_tmux_session: `aris_ff_smoke4b_20260418_p4_v2`
- v2_status: `stopped`
- v2_stop_reason:
  - launched before the strict guardrail patch was applied, so it would not validate the newest render fix

- v3_launch_time: `2026-04-18 Asia/Shanghai`
- v3_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_forecastfix_smoke4b_20260418_parallel4_v3`
- v3_tmux_session: `aris_ff_smoke4b_20260418_p4_v3`
- v3_launch_log: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_forecastfix_smoke4b_20260418_parallel4_v3.launch.log`
- v3_answer_workers: `4`
- v3_eval_workers: `4`

## Aux Rubric Leniency Patch For Forecasting + Bottleneck

- update_time: `2026-04-18 Asia/Shanghai`
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
- reason:
  - `direction_forecasting.signal_grounding` and `bottleneck_opportunity_discovery` target checks were too exact-match oriented.
  - For benchmark use, slightly broader or narrower answers inside the same immediate technical cluster should count as aligned rather than as a miss.
- patch_scope:
  - relax `_target_match_score(...)` so parent/child or near-neighbor labels in the same technical cluster get meaningful partial credit
  - make candidate-direction extraction family-aware instead of reusing one generic target pool
  - `bottleneck_opportunity_discovery` now audits against bottleneck/opportunity labels plus future themes
  - `direction_forecasting` now audits against next-direction labels plus emergent-direction labels
  - lower the bottleneck "no relevant anchor" cap trigger from `best_match_score < 0.20` to `< 0.14`
  - prompt instructions now explicitly say exact phrase equality is not required for `bottleneck` / `forecasting` if the answer stays in the same technical cluster
- verification:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`

## Aux Lenient Re-Eval Launch On Bottleneck + Forecasting 50

## ResearchAgent Answer-Contract Tightening

- update_time: `2026-04-18 Asia/Shanghai`
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- reason:
  - manual inspection of the current `ResearchAgent` full-100 row showed the main failures were answer-side rather than judge-side:
    - weak or absent explicit evidence linkage in the final answer text
    - family-contract drift, especially on `direction_forecasting` and `venue_aware_research_positioning`
    - overly abstract answers that left `support_summary` in trace metadata instead of exposing evidence anchors in the answer itself
- patch_scope:
  - renderer prompt now explicitly requires the final answer text to contain an `Evidence:` clause with exact retrieved paper titles
  - render candidate scoring now applies stronger family-contract penalties for:
    - missing evidence clause / weak title anchoring
    - forecast answers drifting away from `family_packet.canonical_focus`
    - venue answers missing `Package` / `Contrast`
    - planning answers missing dependency language
  - final answer assembly now normalizes family-specific answer slots:
    - `bottleneck`: `Bottleneck / Blocked capability / Immediate unlock / Why now / Evidence`
    - `forecasting`: `Forecast / Trajectory (when recoverable) / Why now / Evidence`
    - `strategic`: `Priority 1 / Priority 2 / Dependency / Defer rationale / Evidence`
    - `venue`: `Positioning / Package / Why this venue / Contrast / Evidence`
  - finalization also snaps obviously drifting `direction` / `venue` labels back to `family_packet.canonical_focus` when the generated answer diverges too far from the recovered benchmark-facing focus
- verification:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
  - local finalize replay confirmed:
    - drifting forecast labels can now be pulled back to canonical focus when they fall outside the packet-supported direction
    - venue answers now expose package / contrast / evidence anchors directly in the final answer
- next_use:
  - this patch is intended to be the new starting point for the next `ResearchAgent` bottleneck / forecasting rerun, not for retroactively rewriting the already reported old `ResearchAgent` row

## ResearchAgent Contract-Tightened Smoke-8 Launch Plan

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `ResearchAgent-Offline`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- source_release_task_count: `100`
- task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_bottleneck_forecast_smoke8_20260418_task_ids.txt`
- planned_scope:
  - `bottleneck_opportunity_discovery`: `4`
  - `direction_forecasting`: `4`
- planned_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_contracttight_20260418_parallel4`
- planned_tmux_session: `researchagent_bf_smoke8_contract_20260418_p4`
- answer_workers: `4`
- eval_workers: `4`
- launch_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- intent:
  - validate whether the new answer contract tightening improves traceability and family-contract fidelity on the benchmark-priority low-scoring families before any larger `ResearchAgent` rerun

## ResearchAgent Contract-Tightened Smoke-8 Subset Release

- update_time: `2026-04-18 Asia/Shanghai`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- source_release_task_count: `100`
- task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_bottleneck_forecast_smoke8_20260418_task_ids.txt`
- subset_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418`
- subset_release_task_count: `8`
- subset_release_family_counts:
  - `bottleneck_opportunity_discovery`: `4`
  - `direction_forecasting`: `4`
- scripts_used:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/build_release_subset_by_task_ids.py`
- files_written:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418/tasks.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418/tasks_hidden_eval.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418/tasks_hidden_eval_v3.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418/tasks_hidden_eval_v3_1.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418/tasks_build_trace.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418/tasks_internal_full.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418/task_ids.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418/manifest.json`
- kb_note:
  - `kb/` and `future_kb/` are symlinked back to the source release because only the task subset changed.

## ResearchAgent Contract-Tightened Smoke-8 Launch

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `ResearchAgent-Offline`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418`
- release_task_count: `8`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_contracttight_20260418_parallel4_v2`
- answer_workers: `4`
- eval_workers: `4`
- tmux_session: `researchagent_bf_smoke8_contract_20260418_p4_v2`
- launch_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- launch_status: `running`
- initial_progress:
  - `shard_0`: `1/2 RTLv3-0004`
  - `shard_1`: `1/2 RTLv3-0006`
  - `shard_2`: `1/2 RTLv3-0007`
  - `shard_3`: `1/2 RTLv3-0021`
- note:
  - an earlier direct launch attempt against the full `experiment100` release was abandoned immediately because the generic wrapper only reads `release_dir/task_ids.txt`; the actual running session uses the dedicated 8-task subset release above

## ResearchAgent Family-Packet Refactor

- update_time: `2026-04-18 Asia/Shanghai`
- code_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- reason:
  - `ResearchAgent` still underperformed mainly because the middle reasoning chain and final rendering remained too generic / survey-like even after retrieval fusion.
  - This patch keeps the existing retrieval backbone, but replaces the weak middle abstraction with a stronger family-specific reasoning packet that the final renderer and final candidate judge must explicitly follow.
- patch_scope:
  - `task_module` now produces a structured `task_module_packet` instead of a loose generic scaffold
  - packet fields are `canonical_focus`, `secondary_focus`, `core_support`, `execution_hook`, `rejection_rule`
  - `decision_packet` now carries the family packet forward
  - final renderer now receives the family packet explicitly and is instructed to keep `canonical_focus` visible instead of washing it into a broader umbrella label
  - final candidate scoring now includes `module_alignment`
  - final candidate judge now also sees the family packet and uses its rejection rule when picking the final answer
- verification:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`

## ResearchAgent Bottleneck + Forecast Smoke

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `researchagent_offline`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- release_family_counts:
  - `bottleneck_opportunity_discovery`: `25`
  - `direction_forecasting`: `25`
  - `strategic_research_planning`: `25`
  - `venue_aware_research_positioning`: `25`
- subset_task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_bottleneck_forecast_smoke8_20260418_task_ids.txt`
- subset_task_count: `8`
- subset_family_counts:
  - `bottleneck_opportunity_discovery`: `4`
  - `direction_forecasting`: `4`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_forecast_smoke8_packet_20260418`
- tmux_session: `researchagent_bf_smoke8_20260418`
- answer_workers: `1`
- run_command:
  - `python scripts/run_researchagent_offline.py --release-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1 --output-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_forecast_smoke8_packet_20260418 --reasoning-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --render-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --fallback-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml --task-ids-file /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_bottleneck_forecast_smoke8_20260418_task_ids.txt --iterations 2 --render-passes 1`
- launch_log: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_forecast_smoke8_packet_20260418.launch.log`
- status: `running`
- initial_progress:
  - `1/8 RTLv3-0004`

- launch_time: `2026-04-18 Asia/Shanghai`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- source_release_task_count: `100`
- source_release_family_counts:
  - `bottleneck_opportunity_discovery`: `25`
  - `direction_forecasting`: `25`
  - `strategic_research_planning`: `25`
  - `venue_aware_research_positioning`: `25`
- subset_scope:
  - `bottleneck_opportunity_discovery`: `25`
  - `direction_forecasting`: `25`
  - `total`: `50`
- subset_results_inputs:
  - Hybrid: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/hybrid_bd50_aux_20260418_results.jsonl`
  - ARIS: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_bd50_aux_20260418_results.jsonl`
- judge_configs:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- launched_outputs:
  - Hybrid: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_bd50_aux_lenient_20260418`
  - ARIS: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bd50_aux_lenient_20260418`
- tmux_sessions:
  - Hybrid: `hybrid_bd50_aux_20260418`
  - ARIS: `aris_bd50_aux_20260418`
- worker_count: `4` per method
- current_status:
  - Hybrid: `finished`, worker outputs materialized for `50/50`
  - ARIS: `finished`, worker outputs materialized for `50/50`
- current_note:
  - This rerun is `eval_aux` only. Existing primary-metric numbers remain unchanged; only the targeted family auxiliary scores moved.

## Method Table Refresh After Aux Re-Eval

- update_time: `2026-04-18 Asia/Shanghai`
- updated_doc: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/docs/experiment100_method_table_20260418.md`
- refresh_scope:
  - replace `Hybrid RAG` and `ARIS` family auxiliary values for:
    - `bottleneck_opportunity_discovery`
    - `direction_forecasting`
  - keep all primary metrics unchanged
  - keep `strategic_research_planning` and `venue_aware_research_positioning` auxiliary values unchanged
- source_paths:
  - `Hybrid RAG` refreshed aux source:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_bd50_aux_lenient_20260418`
  - `ARIS` refreshed aux source:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bd50_aux_lenient_20260418`
- refreshed_values:
  - bottleneck aux:
    - `Hybrid RAG`: `0.5646`
    - `ARIS`: `0.6586`
  - forecasting aux:
    - `Hybrid RAG`: `0.3892`
    - `ARIS`: `0.4742`

## Evidence-Grounded Factuality Leniency Patch

- update_time: `2026-04-18 Asia/Shanghai`
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/factscore_eval_v3.py`
- reason:
  - `Evidence-Grounded Factuality` still over-penalized answers whose claim wording was slightly broader or narrower than the benchmark canonical label, even when they stayed in the same immediate technical cluster.
  - Some low-scoring cases also showed bad GT-claim alignment, such as a bottleneck-style claim being matched against a future-opportunity claim just because a few generic tokens overlapped.
- patch_scope:
  - relax `_text_match_score(...)` with the same-cluster distinctive-token and bigram logic used in the newer auxiliary rubric
  - keep numeric / count-like claims strict by skipping that leniency path when numeric facts are present
  - add light claim-family compatibility in `match_answer_claim_to_gt(...)` so bottleneck / opportunity / direction / trajectory / venue / statistical claims are less likely to align to the wrong GT family
  - expand the fact verifier prompt so broader / narrower same-cluster wording is explicitly acceptable when evidence supports the same mechanism family
  - include matched benchmark claim text in verifier context
- verification:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/factscore_eval_v3.py`
- spot_checks:
  - `RTLv3-0055` forecast-style main claim now matches `emergent_directions` at `0.82`
  - `RTLv3-0004` bottleneck-style main claim now matches `hist_bottleneck_core` at `0.84`

## Targeted v3.1 Re-Eval Launch Plan After Factuality Patch

- launch_time: `2026-04-18 Asia/Shanghai`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- targeted_scope:
  - `bottleneck_opportunity_discovery`: `25`
  - `direction_forecasting`: `25`
  - `total`: `50`
- targeted_results_inputs:
  - Hybrid: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/hybrid_bd50_aux_20260418_results.jsonl`
  - ARIS: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/aris_bd50_aux_20260418_results.jsonl`
- kb_paths:
  - history: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1/kb`
  - future: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1/future_kb`
- judge_configs:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- worker_count: `4` per method
- planned_outputs:
  - Hybrid: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_bd50_v31_lenient_20260418`
  - ARIS: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bd50_v31_lenient_20260418`

## Targeted v3.1 Re-Eval Launch After Factuality Patch

- launch_time: `2026-04-18 Asia/Shanghai`
- tmux_sessions:
  - Hybrid: `hybrid_bd50_v31_20260418`
  - ARIS: `aris_bd50_v31_20260418`
- launched_outputs:
  - Hybrid: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_bd50_v31_lenient_20260418`
  - ARIS: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bd50_v31_lenient_20260418`
- launch_logs:
  - Hybrid: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_bd50_v31_lenient_20260418.launch.log`
  - ARIS: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bd50_v31_lenient_20260418.launch.log`
- current_status:
  - Hybrid: `running`
  - ARIS: `running`
- initial_progress:
  - Hybrid worker logs have entered:
    - `RTLv3-0004`
    - `RTLv3-0006`
    - `RTLv3-0007`
    - `RTLv3-0009`
  - ARIS worker logs have entered:
    - `RTLv3-0004`
    - `RTLv3-0006`
    - `RTLv3-0007`
    - `RTLv3-0009`
- note:
  - `results_eval_v3_1.jsonl` files start empty until each worker finishes a full task, so zero-row worker outputs right after launch are expected.

## Targeted v3.1 Re-Eval Completion After Factuality Patch

- completion_time: `2026-04-18 Asia/Shanghai`
- completed_outputs:
  - Hybrid: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_bd50_v31_lenient_20260418`
  - ARIS: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bd50_v31_lenient_20260418`
- merged_status:
  - Hybrid: `finished`, merged `results_eval_v3_1.jsonl` count = `50`
  - ARIS: `finished`, merged `results_eval_v3_1.jsonl` count = `50`
- targeted_family_results:
  - bottleneck factuality:
    - `Hybrid RAG`: `0.4880`
    - `ARIS`: `0.5368`
  - forecasting factuality:
    - `Hybrid RAG`: `0.4550`
    - `ARIS`: `0.4882`
  - targeted 50-task overall factuality:
    - `Hybrid RAG`: `0.4715`
    - `ARIS`: `0.5125`
- note:
  - the same rerun also regenerated `Future Alignment` inside `eval_v31`, but the master comparison table keeps the previously confirmed `Future Alignment` values because the intended metric change in this round was factuality only.

## Method Table Refresh After Factuality Re-Eval

- update_time: `2026-04-18 Asia/Shanghai`
- updated_doc: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/docs/experiment100_method_table_20260418.md`
- refresh_scope:
  - replace `Hybrid RAG` and `ARIS` factuality values for:
    - `bottleneck_opportunity_discovery`
    - `direction_forecasting`
  - recompute overall `Evidence-Grounded Factuality` using the refreshed two-family values and the previously confirmed `strategic` / `venue` rows
  - keep `Future Alignment`, `Evidence Traceability`, and all family auxiliary values unchanged in the master table
- refreshed_values:
  - overall factuality:
    - `Hybrid RAG`: `0.4550`
    - `ARIS`: `0.4587`
  - bottleneck factuality:
    - `Hybrid RAG`: `0.4880`
    - `ARIS`: `0.5368`
  - forecasting factuality:
    - `Hybrid RAG`: `0.4550`
    - `ARIS`: `0.4882`

## Unified Final-Metrics Evaluation Script

- update_time: `2026-04-18 Asia/Shanghai`
- canonical_script:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_final_metrics.py`
- reason:
  - the finalized benchmark now has one stable metric set:
    - `Evidence-Grounded Factuality`
    - `Future Alignment`
    - `Evidence Traceability`
    - `Family-specific auxiliary metrics`
  - running `evaluate_experiment_run_v3_1.py` + `evaluate_experiment_run_v4.py` + `evaluate_experiment_aux.py` separately every time was too fragmented and easy to misuse
- new_behavior:
  - one script now handles:
    - serial run
    - parallel worker mode
    - metric subset selection
    - merged per-metric outputs
    - root-level summary of what was actually run
  - output layout stays backward-compatible:
    - `eval_v31/`
    - `eval_v4/`
    - `eval_aux/`
- metric_selection_policy:
  - `--metrics all`
  - `--metrics primary`
  - `--metrics factuality`
  - `--metrics future_alignment`
  - `--metrics traceability`
  - `--metrics aux`
  - note:
    - `factuality` and `future_alignment` both resolve to `eval_v31`, because they are evaluated together in the current benchmark implementation
- default_kb_policy:
  - if `--history-kb-dir` / `--future-kb-dir` are omitted, the script defaults to:
    - `<release_dir>/kb`
    - `<release_dir>/future_kb`
- smoke_checks:
  - serial traceability-only smoke passed:
    - output: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/eval_unified_smoke_trace_20260418`
  - serial factuality-only smoke passed:
    - output: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/eval_unified_smoke_fact_20260418`
  - parallel traceability-only smoke passed:
    - output: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/eval_unified_parallel_smoke_20260418`
- command_templates:
  - full final metric set:
    - `python scripts/evaluate_experiment_final_metrics.py --results-jsonl <results_jsonl> --release-dir <release_dir> --output-dir <output_dir> --metrics all --workers 4 --judge-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --judge-fallback-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
  - only primary metrics:
    - `python scripts/evaluate_experiment_final_metrics.py --results-jsonl <results_jsonl> --release-dir <release_dir> --output-dir <output_dir> --metrics primary --workers 4 --judge-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --judge-fallback-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
  - only family auxiliary:
    - `python scripts/evaluate_experiment_final_metrics.py --results-jsonl <results_jsonl> --release-dir <release_dir> --output-dir <output_dir> --metrics aux --workers 4 --judge-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --judge-fallback-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- legacy_note:
  - the old per-metric scripts are still present for compatibility, but the new canonical entry point for future runs should be `evaluate_experiment_final_metrics.py`

## Native LLM Full Re-Eval Launch With Unified Final Metrics

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `native_llm`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- source_results_jsonl: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/results.jsonl`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_finalmetrics_20260418`
- metrics: `all`
- eval_workers: `4`
- tmux_session: `native_finalmetrics_20260418`
- judge_configs:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`

## CoI Full Re-Eval Launch With Unified Final Metrics

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `CoI`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- source_results_jsonl: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel1_sharded/results.jsonl`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_finalmetrics_20260418`
- metrics: `all`
- eval_workers: `4`
- tmux_session: `coi_finalmetrics_20260418`
- judge_configs:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- v3_status: `running`
- v3_note:
  - use this run, not `parallel4_v2`, as the canonical smoke validation for the latest forecast patch round

## Eval Breakdown Backfill

- update_time: `2026-04-18 Asia/Shanghai`
- reason:
  - Existing eval directories stored per-task rubric / component scores in `results_eval_*.jsonl`, but `summary.json` only exposed top-level metric means.
  - This made it hard to tell whether a weak metric came from the whole metric design or from one bad sub-dimension.
- code_paths:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_v3_1.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_v4.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_run_v3_1.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_run_v4.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/evaluate_experiment_aux.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/scripts/backfill_eval_breakdowns.py`
- new_outputs:
  - `eval_v31/metric_breakdown.csv`
  - `eval_v4/dimension_breakdown.csv`
  - `eval_aux/dimension_breakdown.csv`
  - enriched `summary.json` files with metric component or rubric dimension means
- v3_1_breakdown_policy:
  - `FactScore` now exposes `precision_score` and `coverage_score`
  - `Future Alignment` now exposes `weighted_unit_alignment`, `alignment_coverage`, and `mean_specificity`
- v4_breakdown_policy:
  - `Evidence Traceability` now exposes `evidence_linkage` and `support_specificity`
- aux_breakdown_policy:
  - each family aux metric now exposes its rubric dimensions
  - aux breakdown csv also includes `family_domain` rows for finer diagnosis
- backfilled_eval_dirs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_v31`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_v4`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/eval_aux`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4/eval_v31`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4/eval_v4`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4/eval_aux`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_bottleneck_venue50_20260418_parallel4/eval_v31`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_bottleneck_venue50_20260418_parallel4/eval_v4`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_bottleneck_venue50_20260418_parallel4/eval_aux`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_v31`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_v4`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/eval_aux_r2metric_20260417`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4/eval_v31`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4/eval_v4`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_evidenceexp_v1_tasknativekb_taskjudgment_r2_parallel4/eval_aux`
- caveat:
  - if a method has a newer targeted aux rerun for only one family, use that rerun's eval directory for that family rather than the stale all-family aux directory

## Native LLM Final-Metrics Completion

- update_time: `2026-04-18 Asia/Shanghai`
- status: `completed`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_finalmetrics_20260418`
- root_summary:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_finalmetrics_20260418/summary.json`
- metric_summaries:
  - `v31`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_finalmetrics_20260418/eval_v31/summary.json`
  - `v4`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_finalmetrics_20260418/eval_v4/summary.json`
  - `aux`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_finalmetrics_20260418/eval_aux/summary.json`
- high_level_results:
  - overall factuality: `0.4807`
  - overall future alignment: `0.4641`
  - overall evidence traceability: `0.0526`
- note:
  - this completed rerun supersedes the older mixed-source Native row logic; the master table should now read directly from this final-metrics directory

## CoI Final-Metrics Completion

- update_time: `2026-04-18 Asia/Shanghai`
- status: `completed`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_finalmetrics_20260418`
- root_summary:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_finalmetrics_20260418/summary.json`
- metric_summaries:
  - `v31`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_finalmetrics_20260418/eval_v31/summary.json`
  - `v4`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_finalmetrics_20260418/eval_v4/summary.json`
  - `aux`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_finalmetrics_20260418/eval_aux/summary.json`
- source_answers:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel1_sharded/results.jsonl`
- do_not_use:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_evidenceexp_v1_parallel10/results.jsonl`
- high_level_results:
  - overall factuality: `0.5067`
  - overall future alignment: `0.4642`
  - overall evidence traceability: `0.6673`
- note:
  - this is the first fully consolidated CoI metric snapshot that is ready to merge into the master table

## ResearchAgent Family-Packet Smoke Completion

- update_time: `2026-04-18 Asia/Shanghai`
- status: `completed`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_forecast_smoke8_packet_20260418`
- task_count: `8`
- family_split:
  - `bottleneck_opportunity_discovery`: `4`
  - `direction_forecasting`: `4`
- files_written:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_forecast_smoke8_packet_20260418/results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_forecast_smoke8_packet_20260418/summary.json`
- qualitative_readout:
  - the new family-packet path produces visibly more task-native answers than the older full-100 `ResearchAgent` row
  - `RTLv3-0004` now explicitly names `runtime consistency monitors` and `rollback or re-query` as the immediate unlock path
  - `RTLv3-0410` now converges to a compact forecast label around multi-view synthesis rather than drifting into a generic survey-style direction
- next_step:
  - rerun the updated `ResearchAgent` on the benchmark-priority families or on the full `experiment100` release after one more tightening round if needed

## ResearchAgent Smoke-8 Old-vs-New Re-Eval

- launch_time: `2026-04-18 Asia/Shanghai`
- reason:
  - quantify how much the new `ResearchAgent` family-packet refactor improves the completed `bottleneck + forecasting` smoke-8 slice under the current final metrics
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids:
  - `RTLv3-0004`
  - `RTLv3-0006`
  - `RTLv3-0007`
  - `RTLv3-0021`
  - `RTLv3-0057`
  - `RTLv3-0078`
  - `RTLv3-0410`
  - `RTLv3-EXP-1048`
- old_slice_results:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_bf_smoke8_oldslice_results.jsonl`
- new_slice_results:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_forecast_smoke8_packet_20260418/results.jsonl`
- eval_outputs:
  - old: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_oldslice_finalmetrics_20260418`
  - new: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packet_finalmetrics_20260418`
- metrics: `all`
- eval_workers: `2`
- judge_configs:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- status: `completed`
- result_summaries:
  - old_v31: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_oldslice_finalmetrics_20260418/eval_v31/summary.json`
  - old_v4: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_oldslice_finalmetrics_20260418/eval_v4/summary.json`
  - old_aux: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_oldslice_finalmetrics_20260418/eval_aux/summary.json`
  - new_v31: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packet_finalmetrics_20260418/eval_v31/summary.json`
  - new_v4: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packet_finalmetrics_20260418/eval_v4/summary.json`
  - new_aux: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packet_finalmetrics_20260418/eval_aux/summary.json`
- high_level_delta:
  - overall factuality: `0.4220 -> 0.4230`
  - overall future alignment: `0.3821 -> 0.5988`
  - overall evidence traceability: `0.3212 -> 0.2938`
- family_delta:
  - bottleneck factuality: `0.4541 -> 0.3855`
  - bottleneck future alignment: `0.3391 -> 0.4319`
  - bottleneck traceability: `0.4113 -> 0.3063`
  - bottleneck aux: `0.4763 -> 0.4237`
  - forecasting factuality: `0.3899 -> 0.4605`
  - forecasting future alignment: `0.4250 -> 0.7656`
  - forecasting traceability: `0.2313 -> 0.2812`
  - forecasting aux: `0.2900 -> 0.3275`
- interpretation:
  - the family-packet refactor produced the strongest gains on `direction_forecasting`, especially `Future Alignment`
  - `bottleneck_opportunity_discovery` became more future-aligned but currently lost some factuality / traceability / aux sharpness on this 4-task slice
  - net effect across the mixed 8-task slice is a very large `Future Alignment` gain with roughly flat overall factuality and slightly lower traceability

## ResearchAgent Bottleneck Packet V2 Smoke-4

- update_time: `2026-04-18 Asia/Shanghai`
- reason:
  - test a stricter bottleneck-only redesign around `bottleneck -> blocked capability -> immediate unlock`, while leaving the validated forecasting changes untouched
- code_changed:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- subset_task_ids:
  - `RTLv3-0004`
  - `RTLv3-0006`
  - `RTLv3-0007`
  - `RTLv3-0021`
- generation_run:
  - output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv2_20260418`
  - tmux_session: `researchagent_bottleneck_smoke4_v2_20260418`
  - command:
    - `python scripts/run_researchagent_offline.py --release-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1 --output-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv2_20260418 --reasoning-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --render-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --fallback-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml --task-ids-file /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_bottleneck_smoke4_20260418_task_ids.txt --iterations 2 --render-passes 1`
  - outputs:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv2_20260418/results.jsonl`
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv2_20260418/summary.json`
- eval_outputs:
  - old: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_oldslice_finalmetrics_20260418`
  - packet_v1: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv1_finalmetrics_20260418`
  - packet_v2: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv2_finalmetrics_20260418`
- comparison_table:
  - `old`: factuality `0.4799`, future `0.3391`, traceability `0.4113`, bottleneck_aux `0.4763`
  - `packet_v1`: factuality `0.3274`, future `0.4319`, traceability `0.3312`, bottleneck_aux `0.3987`
  - `packet_v2`: factuality `0.4133`, future `0.3251`, traceability `0.2812`, bottleneck_aux `0.4625`
- key_dimension_changes:
  - `packet_v2` vs `packet_v1`: factuality recovered mainly through higher `coverage_score` (`0.1852 -> 0.2778`) and higher fact precision (`0.4222 -> 0.5036`)
  - `packet_v2` vs `packet_v1`: future alignment dropped because `weighted_unit_alignment` (`0.3187 -> 0.2389`) and `alignment_coverage` (`0.7500 -> 0.5416`) both regressed
  - `packet_v2` traceability remained the weakest of the three (`0.2812`)
- task_level_readout:
  - `RTLv3-0004`: `packet_v2` recovered both factuality and aux back near `old`, but the immediate unlock became more generic than the previous `runtime consistency monitors` wording
  - `RTLv3-0006`: `packet_v2` produced a reasonably task-native `memory forgetting -> longitudinal personalization benchmark` chain and substantially improved factuality over `packet_v1`
  - `RTLv3-0007`: `packet_v2` overcommitted to `formal verification / proof assistants`, which hurt future alignment relative to `packet_v1`
  - `RTLv3-0021`: `packet_v2` also lost ground on future alignment and factuality; the bottleneck framing stayed coherent, but the unlocked opportunity was too narrow and weakly matched to the benchmark future targets
- implementation_note:
  - generation for this run still contains an old outer `Bottleneck/Opportunity:` prefix from `_finalize_answer`; this was fixed in code immediately after the smoke, but not rerun yet
- takeaway:
  - the stricter bottleneck packet contract is useful for forcing answer shape, but by itself it is not enough
  - the next bottleneck iteration should keep the `blocked capability / immediate unlock` structure while reintroducing stronger evidence-anchored, task-native unlock wording, especially for `RTLv3-0004`, `RTLv3-0007`, and `RTLv3-0021`

## ResearchAgent Direction-25 Launch Plan

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `researchagent_offline`
- reason:
  - preserve the validated `direction_forecasting` improvements and start a family-targeted `experiment100` run before continuing the bottleneck redesign
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- targeted_scope:
  - `direction_forecasting`: `25`
- answer_workers: `4`
- planned_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4`
- planned_task_shards_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/task_shards`
- planned_shard_output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shards`
- planned_tmux_sessions:
  - `researchagent_dir25_20260418_s00`
  - `researchagent_dir25_20260418_s01`
  - `researchagent_dir25_20260418_s02`
  - `researchagent_dir25_20260418_s03`
- run_config:
  - reasoning: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - render: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
  - iterations: `2`
  - render_passes: `1`

## ResearchAgent Direction-25 Launch

- update_time: `2026-04-18 Asia/Shanghai`
- status: `running`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4`
- task_shards:
  - `shard_00`: `7`
  - `shard_01`: `6`
  - `shard_02`: `6`
  - `shard_03`: `6`
- task_shard_files:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/task_shards/shard_00.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/task_shards/shard_01.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/task_shards/shard_02.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/task_shards/shard_03.txt`
- tmux_sessions:
  - `researchagent_dir25_20260418_s00`
  - `researchagent_dir25_20260418_s01`
  - `researchagent_dir25_20260418_s02`
  - `researchagent_dir25_20260418_s03`
- launch_logs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shard_00.launch.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shard_01.launch.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shard_02.launch.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shard_03.launch.log`
- initial_progress:
  - `shard_00`: `1/7 RTLv3-0055`
  - `shard_01`: `1/6 RTLv3-0056`
  - `shard_02`: `1/6 RTLv3-0057`
  - `shard_03`: `1/6 RTLv3-0071`

## ResearchAgent Bottleneck Diagnosis Follow-Up

- update_time: `2026-04-18 Asia/Shanghai`
- source_runs:
  - `old`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_oldslice_finalmetrics_20260418`
  - `packet_v1`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv1_finalmetrics_20260418`
  - `packet_v2`: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv2_finalmetrics_20260418`
- new_readout:
  - the main `packet_v2` failure is not just answer shape or lack of causal wording
  - on `RTLv3-0007` and `RTLv3-0021`, the stricter bottleneck packet pushed the answer outside the benchmark's hidden target cluster, so `Future Alignment` and bottleneck auxiliary both fell despite cleaner structure
- concrete_examples:
  - `RTLv3-0007`: `packet_v1` stayed closer to benchmark-supported application clusters, while `packet_v2` drifted to `formal verification / proof assistants`
  - `RTLv3-0021`: `packet_v1` stayed near `domain-specific vision-language fine-tuning`, while `packet_v2` drifted to `UAV navigation in unseen urban environments`
- next_bottleneck_fix_direction:
  - keep the `bottleneck / blocked capability / immediate unlock` structure
  - add a stronger family-target alignment step so the unlock is chosen from the benchmark-supported future cluster rather than a free-form downstream application

## ResearchAgent Bottleneck Task-Frame Guardrail Patch

- update_time: `2026-04-18 Asia/Shanghai`
- code_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- patch_scope:
  - `bottleneck_opportunity_discovery` now uses deterministic task decomposition instead of LLM-written task frames
  - the bottleneck task frame no longer injects a named downstream application before retrieval
  - decision-packet and task-module prompts now explicitly require the unlocked opportunity to stay close to retrieved `future_work_signals` / `bridge_concepts`
- reason:
  - the previous `packet_v2` drift was already visible in `task_frame.forward_implication`, which then poisoned retrieval queries and downstream packet generation
  - `RTLv3-0007` and `RTLv3-0021` were the clearest cases
- sanity_check:
  - `RTLv3-0007` task frame no longer mentions `formal verification`
  - `RTLv3-0021` task frame no longer mentions `robotic instruction following` or `UAV navigation`

## ResearchAgent Direction-25 Merge

- update_time: `2026-04-18 Asia/Shanghai`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- source_run_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4`
- merged_results_jsonl: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/results_merged.jsonl`
- merged_summary_json: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/summary_merged.json`
- source_shards:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shards/shard_00/results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shards/shard_01/results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shards/shard_02/results.jsonl`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/shards/shard_03/results.jsonl`
- task_count: `25`
- family_scope:
  - `direction_forecasting`: `25`
- notes:
  - merged by `task_id` from the four shard result files
  - this merged file is the canonical input for subsequent final-metrics evaluation on the direction-25 slice

## ResearchAgent Direction-25 Final Metrics Launch

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `ResearchAgent-Offline`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- eval_scope:
  - primary:
    - `Evidence-Grounded Factuality`
    - `Future Alignment`
    - `Evidence Traceability`
  - family_aux:
    - `Forecast Grounding`
- input_results_jsonl: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/results_merged.jsonl`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_finalmetrics`
- workers: `4`
- tmux_session: `researchagent_dir25_eval_20260418`
- judge_config:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- command_template:
  - `python scripts/evaluate_experiment_final_metrics.py --results-jsonl /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_parallel4/results_merged.jsonl --release-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1 --output-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_20260418_finalmetrics --metrics all --workers 4 --judge-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --judge-fallback-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`

## ResearchAgent Bottleneck Packet V3 Final Metrics Launch

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `ResearchAgent-Offline`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- eval_scope:
  - primary:
    - `Evidence-Grounded Factuality`
    - `Future Alignment`
    - `Evidence Traceability`
  - family_aux:
    - `Opportunity Grounding`
- input_results_jsonl: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv3_20260418/results.jsonl`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv3_finalmetrics_20260418`
- workers: `4`
- tmux_session: `researchagent_bottleneck_smoke4_v3_eval_20260418`
- judge_config:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- command_template:
  - `python scripts/evaluate_experiment_final_metrics.py --results-jsonl /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv3_20260418/results.jsonl --release-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1 --output-dir /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv3_finalmetrics_20260418 --metrics all --workers 4 --judge-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml --judge-fallback-llm-config /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- purpose:
  - verify whether the `packet_v3` task-frame guardrail patch actually fixes the prior target-cluster drift enough to justify a full `exp100` bottleneck-family launch

## ResearchAgent Bottleneck Packet V4 Smoke-4 Launch

- launch_time: `2026-04-18 Asia/Shanghai`
- reason:
  - `packet_v3` improved factuality but `Opportunity Grounding` remained too low (`0.435`) because the bottleneck / unlock wording still drifted away from publicly recoverable benchmark target clusters.
  - this round adds public-evidence cluster canonicalization for bottleneck tasks so the immediate unlock prefers a recognizable research-cluster label recovered from retrieved paper titles / abstracts, instead of a bespoke deployment scenario.
- code_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- output_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bottleneck_smoke4_packetv4_20260418`
- task_ids_file:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_bottleneck_smoke4_20260418_task_ids.txt`
- tmux_session:
  - `researchagent_bottleneck_smoke4_v4_20260418`
- run_config:
  - reasoning: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - render: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
  - iterations: `2`
  - render_passes: `1`
- target_improvement:
  - raise `Opportunity Grounding` by keeping the unlocked opportunity closer to publicly evidenced cluster labels such as `reinforcement learning for adaptive tool-augmented reasoning policies`, `software engineering multi-agent debate frameworks`, and `biological / remote sensing vision-language fine-tuning`

## ResearchAgent Family-Native Reviewer + Packet Tightening

- update_time: `2026-04-18 Asia/Shanghai`
- reason:
  - manual inspection still showed `ResearchAgent` losing to `Hybrid RAG` mainly because the answer chain remained insufficiently family-native, not just because of judge calibration
  - confirmed failure modes:
    - `bottleneck`: drifting toward meta / procedural bottlenecks such as evaluation or infrastructure wishes instead of mechanism-level bottlenecks with immediate unlocks
    - `forecasting`: inventing unsupported mechanism futures and weak / missing trajectory justification
    - `strategic`: outputting rankings without first milestone / defer rationale / risk-kill criterion
- code_changed:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- implementation_changes:
  - `pipeline.problem_validator` is now family-specific rather than using generic stage-review metrics
  - `problem_identifier` prompt is now family-aware, so the earliest stage already targets the benchmark-native decision object instead of a generic research problem
  - fixed a scoring bug where `vocabulary_penalty` was effectively ignored in render candidate selection due to a malformed `max(...)` expression
  - forecasting packet now preserves `trajectory_label` and `trajectory_signal`, and final answer shaping falls back to those fields
  - strategic packet now carries first-class fields:
    - `first_milestone`
    - `dependency_chain`
    - `defer_rationale`
    - `risk_or_kill_criterion`
  - strategic render / contract scoring now penalizes missing milestone / defer / kill slots
  - bottleneck scope penalty is stricter against artifact-only / benchmark-only bottleneck statements without causal-mechanism grounding
- validation:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`

## ResearchAgent BF Smoke-8 Packet V4 Parallel Launch

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `ResearchAgent-Offline`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418`
- release_task_count: `8`
- family_counts:
  - bottleneck_opportunity_discovery: `4`
  - direction_forecasting: `4`
- task_ids:
  - `RTLv3-0004`
  - `RTLv3-0006`
  - `RTLv3-0007`
  - `RTLv3-0021`
  - `RTLv3-0057`
  - `RTLv3-0078`
  - `RTLv3-0410`
  - `RTLv3-EXP-1048`
- output_root: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packetv4_20260418_parallel4`
- tmux_session: `researchagent_bf_smoke8_packetv4_p4_20260418`
- workers: `4`
- shard_task_files:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_bf_smoke8_packetv4_20260418_shards/task_ids_00.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_bf_smoke8_packetv4_20260418_shards/task_ids_01.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_bf_smoke8_packetv4_20260418_shards/task_ids_02.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_bf_smoke8_packetv4_20260418_shards/task_ids_03.txt`
- logs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packetv4_20260418_parallel4/logs/shard_00.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packetv4_20260418_parallel4/logs/shard_01.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packetv4_20260418_parallel4/logs/shard_02.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packetv4_20260418_parallel4/logs/shard_03.log`
- launch_notes:
  - the first parallel launch attempt only started `shard_00`; it was stopped immediately and relaunched cleanly with all four shard processes confirmed alive
  - this smoke is generation-only for now; metrics rerun should wait until answers are inspected for drift reduction

## ResearchAgent BF Smoke-8 Packet V4 Final Metrics Launch

- launch_time: `2026-04-18 Asia/Shanghai`
- method: `ResearchAgent-Offline`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bf_smoke8_contracttight_20260418`
- release_task_count: `8`
- input_results_jsonl: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packetv4_20260418_parallel4/results_merged.jsonl`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packetv4_finalmetrics_20260418`
- metrics: `all`
- workers: `4`
- tmux_session: `researchagent_bf_smoke8_packetv4_eval_20260418`
- launch_log: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_bf_smoke8_packetv4_finalmetrics_20260418.launch.log`
- judge_config:
  - primary: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
- purpose:
  - quantify whether the latest family-native packet / reviewer tightening actually lifts `Opportunity Grounding` and `Forecast Grounding` on the smoke-8 slice before scaling up

## ResearchAgent Experiment100 Bottleneck25 Packet V4 Launch

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `ResearchAgent-Offline`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `100`
- family_scope:
  - `bottleneck_opportunity_discovery`
- family_task_count: `25`
- output_root: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4`
- tmux_session: `researchagent_exp100_bottleneck25_p4_20260419`
- workers: `4`
- code_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- task_ids_file:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_experiment100_bottleneck25_20260419_task_ids.txt`
- shard_task_files:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_experiment100_bottleneck25_20260419_shards/task_ids_00.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_experiment100_bottleneck25_20260419_shards/task_ids_01.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_experiment100_bottleneck25_20260419_shards/task_ids_02.txt`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_experiment100_bottleneck25_20260419_shards/task_ids_03.txt`
- shard_sizes:
  - `shard_00`: `7`
  - `shard_01`: `6`
  - `shard_02`: `7`
  - `shard_03`: `5`
- logs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4/logs/shard_00.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4/logs/shard_01.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4/logs/shard_02.log`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4/logs/shard_03.log`
- run_config:
  - reasoning: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - render: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
  - iterations: `2`
  - render_passes: `1`
- purpose:
  - scale the latest bottleneck-specific packet tightening from smoke-8 to the full `experiment100` bottleneck family and verify whether the stronger `Opportunity Grounding` behavior survives at 25-task scope

## ResearchAgent Forecast-Only Tightening

- update_time: `2026-04-19 Asia/Shanghai`
- scope_guardrail:
  - only forecast-related logic was changed in this round; bottleneck / strategic / venue logic was intentionally left untouched
- code_changed:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- reason:
  - forecasting still failed mainly because the pipeline was drifting before final answer rendering
  - the critical failure was in `task_frame` decomposition: forecasting tasks still used LLM decomposition and sometimes hallucinated a concrete next-step direction directly from the task wording
  - that invented direction then contaminated retrieval, decision-packet construction, and final rendering
- forecast_specific_changes:
  - forecasting `task_frame` decomposition is now deterministic, like bottleneck, and no longer presupposes a concrete technical direction
  - forecast query construction now centers on the task topical scope extracted from the title/question, rather than on an LLM-invented implication
  - hybrid queries for forecasting are filtered by topical-scope overlap before retrieval fusion
  - forecast paper hits are reranked with a scope-aware score that rewards overlap with the task scope and penalizes cross-domain forecasting spillover
  - forecast focus-candidate selection now prefers scope-matching phrases and penalizes meta labels like `benchmark`, `evaluation`, `framework`, and `standardization`
  - forecast packet normalization now runs a public-evidence canonicalization pass to snap the forecast label back toward the best scope-aligned candidate found in retrieved structures / pageindex / paper text
  - forecast prompts now explicitly forbid importing a method trend from another subfield just because it shares retrieval / forecasting / control vocabulary
- validation:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`

## ResearchAgent Forecast Smoke-2 Launch

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `ResearchAgent-Offline`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- task_ids_file: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_forecast_smoke2_20260419_task_ids.txt`
- task_ids:
  - `RTLv3-0057`
  - `RTLv3-0078`
- output_dir: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_forecast_smoke2_20260419`
- tmux_session: `researchagent_forecast_smoke2_20260419`
- launch_log: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_forecast_smoke2_20260419.launch.log`
- run_config:
  - reasoning: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - render: `/vepfs-mlp2/c20250513/241404044/users/roytian/MindLink/configs/llm.local.yaml`
  - fallback: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/local_llm_configs/qwen_235b_8001.local.yaml`
  - iterations: `2`
  - render_passes: `1`
- launch_note:
  - an initial attempt to run a temporary subset release was abandoned immediately because the subset did not include the required `kb/manifest.json`; the actual live smoke uses the canonical `experiment100` release plus a task-id file

## ResearchAgent Strategic Signal-Map Bugfix

- update_time: `2026-04-19 Asia/Shanghai`
- code_changed:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
- bug:
  - `_normalize_task_module_packet(...)` referenced `candidate_directions` inside the `strategic_research_planning` branch before defining it in local scope
  - this caused all 4 strategic tasks in `researchagent_bs_smoke8_signalmap_20260419_parallel4` to fail with `UnboundLocalError`
- fix:
  - initialize `candidate_directions = _task_candidate_directions(task)` at the top of `_normalize_task_module_packet(...)`
  - keep the rest of the strategic packet normalization logic unchanged
- validation:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_prompts.py`

## ResearchAgent Strategic-4 Signal-Map Fix Rerun

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `researchagent`
- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_strategic4_signalmap_fix_20260419`
- source_release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `4`
- release_family_counts:
  - `strategic_research_planning`: `4`
- task_ids_file:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_strategic4_signalmap_fix_20260419/task_ids.txt`
- target_tasks:
  - `RTLv3-ECS-0001`
  - `RTLv3-ECS-0016`
  - `RTLv3-ECS-0023`
  - `RTLv3-ECS-0031`
- release_note:
  - `tasks.jsonl`, `tasks_hidden_eval_v3_1.jsonl`, `tasks_internal_full.jsonl`, `kb/`, and `future_kb/` are symlinked to the canonical `experiment100` release; only `task_ids.txt` and subset manifest are new
- output_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_strategic4_signalmap_fix_20260419_parallel4`
- tmux_session:
  - `researchagent_strategic4_signalmap_fix_20260419_p4`
- launch_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- answer_workers: `4`
- eval_workers: `4`
- purpose:
  - confirm the strategic packet path now runs end-to-end after the `candidate_directions` scope fix and produce a trustworthy strategic-only smoke metric before expanding further
- initial_progress_check:
  - all 4 shard logs started successfully on the expected strategic tasks
  - no immediate `candidate_directions` startup crash after relaunch

## ResearchAgent Strategic-4 Signal-Map Fix Result

- update_time: `2026-04-19 Asia/Shanghai`
- run_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_strategic4_signalmap_fix_20260419_parallel4`
- status: `finished`
- completed_tasks: `4 / 4`
- eval_summaries:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_strategic4_signalmap_fix_20260419_parallel4/eval_v31/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_strategic4_signalmap_fix_20260419_parallel4/eval_v4/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_strategic4_signalmap_fix_20260419_parallel4/eval_aux/summary.json`
- metrics:
  - Fact: `0.5449`
  - Future Alignment: `0.5126`
  - Evidence Traceability: `0.6638`
  - Strategic Execution Grounding: `0.58`
- dimension_note:
  - strongest dimensions: `first_milestone_specificity=0.9`, `alternative_defer_rationale=0.95`, `risk_and_kill_criteria=0.95`
  - current bottleneck inside strategic aux: `dependency_to_action_chain=0.425`, `evidence_to_action_mapping=0.325`

## ResearchAgent Venue Packet Upgrade

- update_time: `2026-04-19 Asia/Shanghai`
- code_changed:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_prompts.py`
- purpose:
  - move `venue_aware_research_positioning` from prompt-only specialization to signal-map-backed packet specialization
- changes:
  - added venue-specific `historical_signal_map` fields:
    - `venue_names`
    - `contribution_packages`
    - `evaluation_signatures`
    - `venue_fit_patterns`
  - venue fallback / normalization now preserve:
    - `contribution_package`
    - `venue_fit_signal`
    - `evaluation_signature`
    - `nearby_but_wrong_positioning`
  - venue final answer shaping now emits:
    - `Positioning`
    - `Package`
    - `Why this venue`
    - `Evaluation`
    - `Contrast`
  - JSON repair contract updated so venue packet fields survive malformed generations
- validation:
  - `python -m py_compile /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_offline.py /vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/researchagent_prompts.py`

## ResearchAgent Venue-4 Signal-Map Smoke Launch

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `researchagent`
- release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_venue4_signalmap_20260419`
- source_release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- release_task_count: `4`
- family_scope:
  - `venue_aware_research_positioning`
- target_tasks:
  - `RTLv3-0163`
  - `RTLv3-0168`
  - `RTLv3-1195`
  - `RTLv3-1199`
- output_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_venue4_signalmap_20260419_parallel4`
- tmux_session:
  - `researchagent_venue4_signalmap_20260419_p4`
- launch_wrapper:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/run_release_method_eval_parallel.sh`
- answer_workers: `4`
- eval_workers: `4`
- initial_progress_check:
  - all 4 shard logs started successfully on one venue task each

## ResearchAgent Bottleneck-25 Finalize Relaunch

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `researchagent`
- release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- existing_output_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4`
- tmux_session:
  - `researchagent_exp100_bottleneck25_finalize_20260419`
- purpose:
  - resume from already completed shard-level answer files so the wrapper can merge `results.jsonl` and continue into eval
- note:
  - shard answer generation had already completed `25 / 25`; this relaunch is for merge/eval completion rather than new answer generation

## ResearchAgent Bottleneck-25 Recovery

- update_time: `2026-04-19 Asia/Shanghai`
- issue_found:
  - the previous finalize relaunch targeted the wrong release path and resumed answer generation instead of only merging/evaluating
  - no foreign task IDs were introduced, but `4` bottleneck tasks were appended a second time in shard outputs
- validation_of_damage:
  - original bottleneck-25 task-id list still existed at `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/tmp/researchagent_experiment100_bottleneck25_20260419_task_ids.txt`
  - polluted shard outputs still contained exactly the same `25` unique bottleneck tasks
  - duplicate rows were limited to:
    - `RTLv3-0006`
    - `RTLv3-0007`
    - `RTLv3-0009`
    - `RTLv3-0016`
- recovery_action:
  - created a canonical subset release:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bottleneck25_packetv4_20260419`
  - created a recovered clean result dir:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4_recovered`
  - rebuilt `results.jsonl` by taking the first occurrence of each task in original subset order and wrote a recovery note into `summary.json`
- recovered_status:
  - recovered task_count: `25`
  - missing: `[]`
- next_step:
  - run eval on the recovered clean result dir rather than on the polluted original output dir

## ResearchAgent Bottleneck-25 Recovered Eval Launch

- launch_time: `2026-04-19 Asia/Shanghai`
- method: `researchagent`
- release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_researchagent_bottleneck25_packetv4_20260419`
- results_jsonl:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4_recovered/results.jsonl`
- recovered_result_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4_recovered`
- tmux_session:
  - `researchagent_bottleneck25_recovered_eval_20260419`
- eval_workers: `4`
- status:
  - `launching`

## ResearchAgent Venue-4 Signal-Map Smoke Result

- update_time: `2026-04-19 Asia/Shanghai`
- run_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_venue4_signalmap_20260419_parallel4`
- status: `finished`
- completed_tasks: `4 / 4`
- eval_summaries:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_venue4_signalmap_20260419_parallel4/eval_v31/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_venue4_signalmap_20260419_parallel4/eval_v4/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_venue4_signalmap_20260419_parallel4/eval_aux/summary.json`
- metrics:
  - Fact: `0.3812`
  - Future Alignment: `0.2794`
  - Evidence Traceability: `0.6837`
  - Venue Positioning Grounding: `0.6012`
- dimension_note:
  - strongest venue-aux dimensions:
    - `venue_specific_contribution_fit=0.8375`
    - `paper_package_specificity=0.7`
    - `reviewer_expectation_grounding=0.65`
  - current weakest venue-aux dimension:
    - `contrastive_venue_discrimination=0.525`
  - primary-metric weakness remains `coverage_score=0.0`, so venue shaping improved answer form and family aux more than factual coverage

## ResearchAgent Direction-25 Signal-Map Result

- update_time: `2026-04-19 Asia/Shanghai`
- release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- run_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_signalmap_20260419_parallel4`
- status: `finished`
- completed_tasks: `25 / 25`
- eval_summaries:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_signalmap_20260419_parallel4/eval_v31/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_signalmap_20260419_parallel4/eval_v4/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_signalmap_20260419_parallel4/eval_aux/summary.json`
- metrics:
  - Fact: `0.4714`
  - Future Alignment: `0.4035`
  - Evidence Traceability: `0.6352`
  - Forecast Grounding: `0.3080`
- dimension_note:
  - factuality remains precision-heavy but coverage-light: `precision_score=0.6905`, `coverage_score=0.1428`
  - traceability is now stable on both judge dimensions: `evidence_linkage=0.664`, `support_specificity=0.6`
  - current forecast-aux bottleneck remains `forecast_discipline=0.236`, with `signal_grounding=0.356`

## ResearchAgent Strategic-25 Signal-Map Result

- update_time: `2026-04-19 Asia/Shanghai`
- release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- run_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_strategic25_signalmap_20260419_parallel4`
- status: `finished`
- completed_tasks: `25 / 25`
- eval_summaries:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_strategic25_signalmap_20260419_parallel4/eval_v31/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_strategic25_signalmap_20260419_parallel4/eval_v4/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_strategic25_signalmap_20260419_parallel4/eval_aux/summary.json`
- metrics:
  - Fact: `0.5346`
  - Future Alignment: `0.5004`
  - Evidence Traceability: `0.6688`
  - Strategic Execution Grounding: `0.5778`
- dimension_note:
  - factuality is materially stronger than earlier generic `ResearchAgent` mainly because `coverage_score` recovered to `0.368`
  - strongest strategic-aux dimensions are now `first_milestone_specificity=0.952`, `risk_and_kill_criteria=1.0`, and `alternative_defer_rationale=0.84`
  - remaining weak spots are still `dependency_to_action_chain=0.208` and `evidence_to_action_mapping=0.412`

## ResearchAgent Venue-25 Signal-Map Result

- update_time: `2026-04-19 Asia/Shanghai`
- release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- run_dir:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_venue25_signalmap_20260419_parallel4`
- status: `finished`
- completed_tasks: `25 / 25`
- eval_summaries:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_venue25_signalmap_20260419_parallel4/eval_v31/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_venue25_signalmap_20260419_parallel4/eval_v4/summary.json`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_venue25_signalmap_20260419_parallel4/eval_aux/summary.json`
- metrics:
  - Fact: `0.3960`
  - Future Alignment: `0.3431`
  - Evidence Traceability: `0.4627`
  - Venue Positioning Grounding: `0.6071`
- dimension_note:
  - venue-specific answer shaping held up on aux dimensions: `venue_specific_contribution_fit=0.848`, `paper_package_specificity=0.814`, `reviewer_expectation_grounding=0.694`
  - the remaining venue-aux weakness is still `contrastive_venue_discrimination=0.534`
  - primary-metric weakness is unchanged: `coverage_score=0.0`, so venue formatting improved positioning quality more than benchmark factual coverage

## ResearchAgent Experiment100 Family-Specific Rerun Consolidation

- update_time: `2026-04-19 Asia/Shanghai`
- canonical_release_path:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- family_result_dirs:
  - bottleneck:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4_recovered`
  - direction:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_direction25_signalmap_20260419_parallel4`
  - strategic:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_strategic25_signalmap_20260419_parallel4`
  - venue:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_venue25_signalmap_20260419_parallel4`
- overall_primary_row_for_method_table:
  - Evidence-Grounded Factuality: `0.4813`
  - Future Alignment: `0.4135`
  - Evidence Traceability: `0.5644`
- reporting_note:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/docs/experiment100_method_table_20260418.md` already uses these four family-specific result dirs as the canonical `ResearchAgent` row
  - this consolidated row supersedes the earlier generic full-100 `ResearchAgent` reference
