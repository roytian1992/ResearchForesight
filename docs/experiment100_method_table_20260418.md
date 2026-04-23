# Experiment100 Method Table (2026-04-18, refreshed 2026-04-22)

## Benchmark

- release_path: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1`
- task_count: `100`
- family_split:
  - `bottleneck_opportunity_discovery`: `25`
  - `direction_forecasting`: `25`
  - `strategic_research_planning`: `25`
  - `venue_aware_research_positioning`: `25`

## Current Synchronized Artifact Set

- benchmark-side rewritten agent outputs:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_agent_final_renderer_v2_qwen8001_20260422`
- merged final metrics:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_full_eval_20260422_agent_final_renderer_v2_qwen8001_rewrite_mimoeval/merged_final_metrics`
- round-robin aggregate:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/round_robin_structured_idea_arena_evidence_light_full100_agent_final_renderer_v2_qwen8001_rewrite_20260422/aggregated`

## Source Policy

- `Native LLM`
  - source answers:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/native_llm_experiment100_evidenceexp_v1_parallel4/results.jsonl`
- `Hybrid RAG`
  - source answers:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/results.jsonl`
- `CoI`
  - original answers:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/coi_experiment100_focusfixv3_20260420_parallel4/results_merged.jsonl`
  - benchmark-side rewritten answers:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_agent_final_renderer_v2_qwen8001_20260422/coi/results.jsonl`
- `ResearchAgent`
  - original answers:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/researchagent_experiment100_qwen8002_20260420_parallel4/results_merged.jsonl`
  - benchmark-side rewritten answers:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_agent_final_renderer_v2_qwen8001_20260422/researchagent/results.jsonl`
- `ARIS`
  - original answers:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4/results.jsonl`
  - benchmark-side rewritten answers:
    - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/results/experiment100_agent_final_renderer_v2_qwen8001_20260422/aris/results.jsonl`

## Current Consolidated Comparison

### Overall Primary Metrics

| Method | Evidence-Grounded Factuality | Future Alignment | Evidence Traceability |
| --- | ---: | ---: | ---: |
| Native LLM | 0.5724 | 0.4381 | 0.0015 |
| Hybrid RAG | 0.5714 | 0.4137 | 0.7224 |
| CoI | 0.6015 | 0.4636 | 0.7661 |
| ResearchAgent | 0.5732 | 0.3698 | 0.7004 |
| ARIS | 0.5847 | 0.4007 | 0.8094 |

### Family Breakdown

| Family | Method | Factuality | Future Alignment | Traceability | Family Aux |
| --- | --- | ---: | ---: | ---: | ---: |
| bottleneck_opportunity_discovery | native_llm | 0.6644 | 0.4222 | 0.0000 | 0.7029 |
| bottleneck_opportunity_discovery | hybrid_rag | 0.6110 | 0.4120 | 0.5791 | 0.7686 |
| bottleneck_opportunity_discovery | coi | 0.6827 | 0.4616 | 0.7511 | 0.7602 |
| bottleneck_opportunity_discovery | researchagent | 0.6667 | 0.4305 | 0.7202 | 0.7164 |
| bottleneck_opportunity_discovery | aris | 0.6101 | 0.3302 | 0.8088 | 0.7376 |
| direction_forecasting | native_llm | 0.5279 | 0.3698 | 0.0000 | 0.4224 |
| direction_forecasting | hybrid_rag | 0.5287 | 0.3000 | 0.7191 | 0.5448 |
| direction_forecasting | coi | 0.5751 | 0.5333 | 0.7720 | 0.6276 |
| direction_forecasting | researchagent | 0.5722 | 0.2261 | 0.7919 | 0.5876 |
| direction_forecasting | aris | 0.6053 | 0.3048 | 0.7853 | 0.7588 |
| strategic_research_planning | native_llm | 0.6804 | 0.5666 | 0.0000 | 0.1662 |
| strategic_research_planning | hybrid_rag | 0.6150 | 0.6144 | 0.8521 | 0.1673 |
| strategic_research_planning | coi | 0.6472 | 0.5524 | 0.8196 | 0.5771 |
| strategic_research_planning | researchagent | 0.5618 | 0.4781 | 0.6853 | 0.5039 |
| strategic_research_planning | aris | 0.6465 | 0.5823 | 0.8196 | 0.5649 |
| venue_aware_research_positioning | native_llm | 0.4170 | 0.3937 | 0.0062 | 0.4963 |
| venue_aware_research_positioning | hybrid_rag | 0.5310 | 0.3283 | 0.7391 | 0.4583 |
| venue_aware_research_positioning | coi | 0.5010 | 0.3070 | 0.7218 | 0.6749 |
| venue_aware_research_positioning | researchagent | 0.4920 | 0.3446 | 0.6043 | 0.6226 |
| venue_aware_research_positioning | aris | 0.4770 | 0.3856 | 0.8240 | 0.6466 |

### Family Aux Metric Names

| Family | Auxiliary Metric |
| --- | --- |
| bottleneck_opportunity_discovery | Opportunity Grounding |
| direction_forecasting | Forecast Grounding |
| strategic_research_planning | Strategic Execution Grounding |
| venue_aware_research_positioning | Venue Positioning Grounding |

### Overall Round-Robin ELO

| Method | Pairwise Preference ELO |
| --- | ---: |
| ResearchAgent | 1677.18 |
| CoI | 1640.30 |
| ARIS | 1538.62 |
| Hybrid RAG | 1439.64 |
| Native LLM | 1204.27 |

### Family Round-Robin ELO

| Family | ELO Ranking |
| --- | --- |
| bottleneck_opportunity_discovery | `aris > researchagent > coi > hybrid_rag > native_llm` |
| direction_forecasting | `researchagent > hybrid_rag > coi > aris > native_llm` |
| strategic_research_planning | `coi > researchagent > aris > hybrid_rag > native_llm` |
| venue_aware_research_positioning | `aris > coi > researchagent > hybrid_rag > native_llm` |

## Current Readout

- the current synchronized benchmark-side rewrite materially changed the ranking landscape for the three agent methods:
  - overall round-robin is now `researchagent > coi > aris > hybrid_rag > native_llm`
- `CoI` is now the strongest method on both overall `Evidence-Grounded Factuality` and overall `Future Alignment`:
  - `0.6015 / 0.4636`
- `ARIS` is now the strongest method on overall `Evidence Traceability`:
  - `0.8094`
- `Hybrid RAG` is still the strongest method on `Opportunity Grounding`:
  - `0.7686`
  - but it is no longer the top overall pairwise method, nor does it remain ahead of the three agent methods in overall ELO
- `direction_forecasting` remains the hardest family to fully fix:
  - `Hybrid RAG` is still second on forecasting-family ELO and remains competitive on forecasting-family metrics
- `ResearchAgent` is the clearest beneficiary of the final-answer rewrite:
  - overall pairwise ELO rises above `Hybrid RAG`
  - direct head-to-head against `Hybrid RAG` is now `0.64`
- the current direct head-to-head win rates against `Hybrid RAG` are:
  - `ResearchAgent`: `0.64`
  - `CoI`: `0.59`
  - `ARIS`: `0.79`

## Main Optimization Priority

- highest priority:
  - `direction_forecasting`
- rationale:
  - this is still the only family where `Hybrid RAG` remains second in ELO
  - it is also the family where `ResearchAgent` still shows a clear mismatch between evidence exposure and future-target correctness
