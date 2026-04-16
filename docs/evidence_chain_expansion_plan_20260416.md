# Evidence-Chain Expansion Plan — 2026-04-16

## Current Stable Base

- strict source release:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002`
  - `strict_task_count = 290`
- current experiment subset:
  - `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002`
  - `task_count = 100`

## Why Evidence-Chain Reuse Is The Right Expansion Path

The current benchmark construction pipeline already works from structured support packets rather than from free-form question rewriting:

- support packet build:
  - `scripts/build_coi_support_packets.py`
- family packet derivation:
  - `scripts/build_family_packets.py`
  - `src/researchworld/family_pipelines.py`
- family candidate generation:
  - `scripts/build_family_task_candidates_v3.py`
- rewrite / judge:
  - `scripts/rewrite_v3_family_task_candidates.py`
  - `scripts/judge_v3_family_task_candidates.py`

The most reusable assets are not the public task wordings. They are:

- `history_chain`
- `history_representative_papers`
- `historical_limitation_cluster`
- `historical_future_work_cluster`
- `candidate_directions`
- `direction_records`
- `future_windows` / `future_half_stats` / `target_window_stats`
- venue bucket signals derived from future venue stats

## Reusable Expansion Patterns Already Visible In Code

### 1. One Support Packet -> Multiple Families

`src/researchworld/family_pipelines.py` already derives different family packets from the same packet backbone:

- `build_opportunity_packets`
- `build_chain_packets`
- `build_planning_packets`

This means we can expand by selecting new seed packets or underused seed nodes, without rewriting old questions.

### 2. Planning / Forecast Evidence -> New Venue Tasks

`scripts/build_full_expansion_four_domains.py` already contains a good reuse pattern:

- `make_new_venue_task_from_source`

It converts:

- `direction_forecasting` -> `venue_aware_direction_forecast`
- `strategic_research_planning` -> `venue_targeted_planning`

using the same evidence chain plus venue bucket statistics.

This is a strong pattern because it creates a genuinely different task contract from the same support evidence.

### 3. Planning Evidence -> Comparative Strategic Tasks

`scripts/build_full_expansion_four_domains.py` also already contains:

- `select_comparative_pairs`
- `make_comparative_task`

This reuses planning evidence from two related nodes and turns them into:

- `comparative_opportunity_prioritization`

This is especially valuable because the current strict source has `agenda_priority_selection` but currently lacks enough comparative tasks.

### 4. Historical Future-Work Cluster -> Bottleneck / Planning Variants

`src/researchworld/family_pipelines.py` shows that when descendant evidence is weak, `build_planning_packets` already falls back to:

- `historical_future_work_cluster`

So another strong expansion path is:

- take packets where descendant coverage is sparse but historical future-work signals are rich
- generate alternative planning or bottleneck tasks around those evidence clusters
- keep the support trace fixed, but change the task contract

## Recommended Expansion Order

### A. Recover More Strategic Diversity First

Best immediate move:

- generate more `comparative_opportunity_prioritization` tasks from existing planning evidence

Why:

- current strict 290 includes only `agenda_priority_selection` on the strategic side
- the code for comparative construction already exists
- this adds subtype diversity without fabricating new evidence

### B. Recover More Venue Diversity Next

Best immediate move:

- generate more `venue_aware_direction_forecast` tasks from accepted `direction_forecasting` sources
- continue `venue_targeted_planning` generation from accepted planning sources

Why:

- the venue family is currently only `43`
- the code for venue derivation from existing source rows already exists
- venue tasks benefit directly from the same future evidence but impose a different answer contract

### C. Expand Seed Coverage Only After Derivation Reuse

After the two steps above:

- add new seed packets from under-covered nodes / clusters
- then run the same family packet -> candidate -> rewrite -> judge pipeline

This should be done only after we exhaust high-quality derivations from existing accepted evidence chains.

## Concrete Next Implementation

The most practical next build is a dedicated expansion script that:

1. starts from the stable strict release `...llmfuture_vote_v3_qwen8002`
2. extracts accepted strategic planning rows and builds comparative pairs using:
   - `build_planning_node_record`
   - `select_comparative_pairs`
   - `make_comparative_task`
3. extracts accepted forecasting / planning rows and derives venue tasks using:
   - `make_new_venue_task_from_source`
4. writes these as a new candidate release
5. runs the normal rewrite / judge / curation gates

## What Not To Prioritize

- simple paraphrase-only rewrites of existing public questions
- synthetic question templates that do not add a new family contract
- generating more tasks from rows whose future labels are already empty or structurally weak

## Success Criterion

A good expansion pass should raise benchmark size by introducing:

- new seed nodes, or
- new family/subtype contracts over the same seed evidence,

while preserving:

- the original cutoff
- the same support evidence lineage
- strict GT validation after future cleanup
