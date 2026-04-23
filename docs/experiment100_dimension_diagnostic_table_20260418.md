# Experiment100 Dimension Diagnostic Table (2026-04-18, refreshed 2026-04-21)

## Scope

- benchmark release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- methods included:
  - `Native LLM`
  - `Hybrid RAG`
  - `CoI`
  - `ResearchAgent`
  - `ARIS`
- purpose:
  - separate overall metric weakness from sub-dimension weakness
  - expose which part of a metric is actually driving the ranking
  - keep the paper-facing dimension readout aligned with the current `exp100` comparison
  - allow family-targeted diagnostic reruns when a method's current main full-100 row and its best diagnostic slice are not the same run

## Interpretation Note

- this file is a diagnostic companion to the master comparison table, not a replacement for it
- the canonical full-100 method comparison lives in:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/docs/experiment100_method_table_20260418.md`
- when a method's current main row comes from a later full-100 rerun, this diagnostic table may still use family-targeted reruns to expose clearer sub-dimension behavior
- in particular, `ResearchAgent` main-row reporting now uses the refreshed mixed-source canonical export:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_canonical_method_exports_20260421_researchagent_renderfix/researchagent`
- the family-level sub-dimension readouts below still use targeted diagnostic sources where helpful, but the forecasting and venue rows now follow the `2026-04-21` renderfix rerun that is also used in the refreshed canonical row

## Source Selection Policy

- `Native LLM`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_finalmetrics_20260418`
- `CoI`
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_finalmetrics_20260418`
- `ResearchAgent`
  - bottleneck:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_bottleneck25_packetv4_20260419_parallel4_recovered`
  - forecasting:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_forecast_venue50_renderfix_20260421_parallel4/final_metrics`
  - strategic:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_strategic25_signalmap_20260419_parallel4`
  - venue:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_forecast_venue50_renderfix_20260421_parallel4/final_metrics`
- `Hybrid RAG`
  - primary + traceability + venue mean:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4`
  - bottleneck + forecasting factuality rerun:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_bd50_v31_lenient_20260418`
  - bottleneck + forecasting auxiliary rerun:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_bd50_aux_lenient_20260418`
  - strategic auxiliary rerun:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_strategic25_aux_reval_20260418`
- `ARIS`
  - strategic + forecasting:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_strategic_forecasting50_20260418_parallel4`
  - bottleneck + venue:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_bottleneck_venue50_20260418_parallel4`
  - bottleneck + forecasting factuality rerun:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bd50_v31_lenient_20260418`
  - bottleneck + forecasting auxiliary rerun:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_bd50_aux_lenient_20260418`

## Bottleneck Opportunity Discovery

| Method | Fact Precision | Fact Coverage | Future Unit Align | Future Coverage | Future Specificity | ET Linkage | ET Specificity | Causal Linkage | Technical Plausibility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native LLM | 0.6860 | 0.2815 | 0.3661 | 0.7667 | 0.6067 | 0.0000 | 0.2080 | 0.5120 | 0.5600 |
| Hybrid RAG | 0.6355 | 0.2667 | 0.3644 | 0.7000 | 0.5880 | 0.3640 | 0.3360 | 0.5920 | 0.5960 |
| CoI | 0.5650 | 0.2519 | 0.1943 | 0.4000 | 0.4280 | 0.7420 | 0.6780 | 0.6580 | 0.6780 |
| ResearchAgent | 0.7237 | 0.2222 | 0.3202 | 0.6400 | 0.5473 | 0.4800 | 0.5040 | 0.4660 | 0.5020 |
| ARIS | 0.6577 | 0.3556 | 0.2839 | 0.5600 | 0.5137 | 0.6040 | 0.5480 | 0.7760 | 0.7600 |

**Readout**

- `ARIS` is strongest on the bottleneck-specific subdimensions the benchmark is intended to reward: `causal_linkage=0.7760` and `technical_plausibility=0.7600`.
- `ResearchAgent` now has the strongest factual precision in this family (`0.7237`), but the answer still degrades at the bottleneck-to-opportunity bridge itself (`0.4660 / 0.5020`).
- `CoI` remains the clearest evidence exposer (`ET linkage=0.7420`, `ET specificity=0.6780`), but that does not translate into good future alignment on this family.

## Direction Forecasting

| Method | Fact Precision | Fact Coverage | Future Unit Align | Future Coverage | Future Specificity | ET Linkage | ET Specificity | Signal Grounding | Forecast Discipline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native LLM | 0.5981 | 0.1857 | 0.3160 | 0.6400 | 0.5460 | 0.0000 | 0.0360 | 0.3140 | 0.4580 |
| Hybrid RAG | 0.6631 | 0.1428 | 0.3370 | 0.6400 | 0.5440 | 0.4460 | 0.4140 | 0.4260 | 0.3340 |
| CoI | 0.7272 | 0.1571 | 0.5311 | 0.8600 | 0.7010 | 0.6440 | 0.6020 | 0.4680 | 0.3120 |
| ResearchAgent | 0.7189 | 0.1543 | 0.2400 | 0.5200 | 0.4680 | 0.7040 | 0.6260 | 0.3900 | 0.2480 |
| ARIS | 0.7108 | 0.1543 | 0.3155 | 0.6200 | 0.5380 | 0.7500 | 0.6900 | 0.5068 | 0.4252 |

**Readout**

- `ARIS` is the strongest forecasting method on both family-specific subdimensions: `signal_grounding=0.5068` and `forecast_discipline=0.4252`.
- `CoI` is strongest on the three future-alignment components (`0.5311 / 0.8600 / 0.7010`), which means it often lands in the right future cluster even when its answer form is less benchmark-native than `ARIS`.
- `ResearchAgent` now exposes forecasting evidence more clearly than the earlier signalmap slice (`ET linkage=0.7040`, `ET specificity=0.6260`), and its forecast-grounding score also recovers somewhat, but forecasting still breaks at the task-specific layer, especially on the future-label side (`signal_grounding=0.3900`, `forecast_discipline=0.2480`).

## Strategic Research Planning

| Method | Fact Precision | Fact Coverage | Future Unit Align | Future Coverage | Future Specificity | ET Linkage | ET Specificity | Milestone | Dependency Chain | Defer Rationale | Risk / Kill | Evidence Map |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native LLM | 0.7923 | 0.4000 | 0.5678 | 0.6100 | 0.6565 | 0.0000 | 0.0880 | 0.0000 | 0.0480 | 0.7440 | 0.2440 | 0.0400 |
| Hybrid RAG | 0.4978 | 0.3360 | 0.5487 | 0.6400 | 0.6530 | 0.5396 | 0.5228 | 0.0000 | 0.0960 | 0.7880 | 0.1400 | 0.4560 |
| CoI | 0.8005 | 0.4000 | 0.5543 | 0.6200 | 0.6510 | 0.6940 | 0.6060 | 0.0320 | 0.0560 | 0.7680 | 0.1640 | 0.3240 |
| ResearchAgent | 0.6457 | 0.3680 | 0.4803 | 0.5300 | 0.5820 | 0.7120 | 0.6160 | 0.9520 | 0.2080 | 0.8400 | 1.0000 | 0.4120 |
| ARIS | 0.5041 | 0.3680 | 0.7922 | 0.9800 | 0.8600 | 0.6880 | 0.6460 | 1.0000 | 0.4960 | 1.0000 | 1.0000 | 0.2960 |

**Readout**

- The current strategic metric is doing real discrimination. `ARIS` and `ResearchAgent` are high because they actually provide executable planning structure, not because they merely sound strategic.
- `ARIS` is strongest on the execution-facing dimensions: `milestone=1.0000`, `dependency_chain=0.4960`, `defer_rationale=1.0000`, `risk/kill=1.0000`.
- `ResearchAgent` is now very competitive on the same family and is especially strong on `milestone=0.9520` and `risk/kill=1.0000`, but still weak on `dependency_chain=0.2080`.
- `Hybrid RAG` and `CoI` gain much of their strategic score from `defer_rationale`, while remaining near-zero on the core milestone / dependency structure that this family is meant to test.

## Venue-Aware Research Positioning

| Method | Fact Precision | Fact Coverage | Future Unit Align | Future Coverage | Future Specificity | ET Linkage | ET Specificity | Aux Mean | Venue Fit | Reviewer Grounding | Package Specificity | Contrastive Discrimination |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native LLM | 0.5500 | 0.0000 | 0.3208 | 0.6000 | 0.5277 | 0.0000 | 0.1360 | 0.5577 | 0.6880 | 0.4820 | 0.5340 | 0.5640 |
| Hybrid RAG | 0.7400 | 0.0000 | 0.3040 | 0.5267 | 0.4990 | 0.5300 | 0.5180 | 0.5771 | 0.7600 | 0.5240 | 0.6400 | 0.4640 |
| CoI | 0.7462 | 0.0000 | 0.3538 | 0.5333 | 0.5290 | 0.7100 | 0.6360 | 0.5146 | 0.6760 | 0.5880 | 0.7060 | 0.3720 |
| ResearchAgent | 0.6379 | 0.0000 | 0.2734 | 0.5400 | 0.4923 | 0.4640 | 0.4820 | 0.5602 | 0.7240 | 0.7320 | 0.7780 | 0.2880 |
| ARIS | 0.6000 | 0.0000 | 0.3327 | 0.5867 | 0.5317 | 0.7900 | 0.7620 | 0.7281 | 0.8320 | 0.8820 | 0.9320 | 0.6260 |

**Readout**

- after the venue prior-knowledge refresh, `ARIS` is the strongest synchronized venue-packaging method, with all four venue-specific dimensions in a healthy range.
- `Hybrid RAG` no longer dominates this family once reviewer expectation grounding and contrastive primary-vs-secondary venue reasoning are scored explicitly; its refreshed venue auxiliary is `0.5771`.
- `ResearchAgent` is now much more competitive on venue packaging itself (`0.7240 / 0.7320 / 0.7780 / 0.2880`), but its weakest point is still clearly `contrastive_venue_discrimination`.
- `CoI` remains a good reminder that strong evidence traceability does not automatically imply strong venue positioning; even after the refresh, its weakest dimension is still `contrastive_venue_discrimination=0.3720`.

## High-Value Conclusions

- `bottleneck_opportunity_discovery`:
  - the main problem is not evidence exposure alone
  - the main problem is canonical target selection plus causal bottleneck-to-opportunity bridging
- `direction_forecasting`:
  - the main bottleneck is still `signal_grounding`
  - `ARIS` is strongest because it is better at abstracting the right historical signal cluster, not because it simply cites more evidence
- `strategic_research_planning`:
  - the current `Strategic Execution Grounding` metric is useful
  - it cleanly separates executable agendas from generic high-level recommendations
- `venue_aware_research_positioning`:
  - this family is more retrieval-favorable than the other three
  - even so, synchronized dimension tables still distinguish methods that produce real venue-specific packaging from those that only surface relevant papers
