# benchmark_v3_20260407_venue

## Summary
- tasks: 168
- families: 3
- domains: 4
- per family × domain cap: None
- history cutoff: 2025-08-31
- future windows:
  - quarterly: 2025-09-01 ~ 2025-11-30, 2025-12-01 ~ 2026-02-28
  - half-year: 2025-09-01 ~ 2026-02-28

## Task families
1. direction_forecasting
2. bottleneck_opportunity_discovery
3. strategic_research_planning

## Domains
- LLM agents
- LLM fine-tuning and post-training
- RAG and retrieval structuring
- Visual generative modeling and diffusion

## Construction notes
This release is built from:
- quarterly TaxoAdapt taxonomy snapshots
- node-level venue/citation aggregates
- CoI-style support packets
- selective paper-structure extraction
- LLM rewrite and LLM-as-judge filtering

## Environment caveat
During construction, direct arXiv source/html fetching was unavailable from the current runtime environment.
Therefore the selective paper evidence layer was instantiated with abstract-derived normalized content and abstract-conditioned structure extraction rather than source-tex/html full text.
The benchmark release remains temporally valid, but this evidence layer should be upgraded to full-text extraction in a later refresh when arXiv connectivity is available.
