#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHONPATH=src python scripts/build_v2_support_packet_cases.py

# If arXiv access is unavailable in the runtime, fall back to abstract-only content.
PYTHONPATH=src python scripts/build_case_abstract_content.py \
  --cases data/support_packets/fulltext_cases/llm_agent/support_packet_cases.jsonl \
  --papers data/domains/llm_agent/interim/papers_merged.publication_enriched.semanticscholar.jsonl \
  --output data/support_packets/fulltext_content/llm_agent/content.jsonl
PYTHONPATH=src python scripts/build_case_abstract_content.py \
  --cases data/support_packets/fulltext_cases/llm_finetuning_post_training/support_packet_cases.jsonl \
  --papers data/domains/llm_finetuning_post_training/clean/core_papers.publication_enriched.semanticscholar.jsonl \
  --output data/support_packets/fulltext_content/llm_finetuning_post_training/content.jsonl
PYTHONPATH=src python scripts/build_case_abstract_content.py \
  --cases data/support_packets/fulltext_cases/rag_and_retrieval_structuring/support_packet_cases.jsonl \
  --papers data/domains/rag_and_retrieval_structuring/clean/core_papers.publication_enriched.semanticscholar.jsonl \
  --output data/support_packets/fulltext_content/rag_and_retrieval_structuring/content.jsonl
PYTHONPATH=src python scripts/build_case_abstract_content.py \
  --cases data/support_packets/fulltext_cases/visual_generative_modeling_and_diffusion/support_packet_cases.jsonl \
  --papers data/domains/visual_generative_modeling_and_diffusion/clean/core_papers.publication_enriched.semanticscholar.jsonl \
  --output data/support_packets/fulltext_content/visual_generative_modeling_and_diffusion/content.jsonl

PYTHONPATH=src python scripts/build_pageindex_compat.py \
  --content data/support_packets/fulltext_content/llm_agent/content.jsonl \
  --output data/support_packets/pageindex/llm_agent/pageindex.jsonl
PYTHONPATH=src python scripts/build_pageindex_compat.py \
  --content data/support_packets/fulltext_content/llm_finetuning_post_training/content.jsonl \
  --output data/support_packets/pageindex/llm_finetuning_post_training/pageindex.jsonl
PYTHONPATH=src python scripts/build_pageindex_compat.py \
  --content data/support_packets/fulltext_content/rag_and_retrieval_structuring/content.jsonl \
  --output data/support_packets/pageindex/rag_and_retrieval_structuring/pageindex.jsonl
PYTHONPATH=src python scripts/build_pageindex_compat.py \
  --content data/support_packets/fulltext_content/visual_generative_modeling_and_diffusion/content.jsonl \
  --output data/support_packets/pageindex/visual_generative_modeling_and_diffusion/pageindex.jsonl

PYTHONPATH=src python scripts/extract_paper_structures.py \
  --cases data/support_packets/fulltext_cases/llm_agent/support_packet_cases.jsonl \
  --papers data/domains/llm_agent/interim/papers_merged.publication_enriched.semanticscholar.jsonl \
  --labels data/domains/llm_agent/annotations/paper_labels.jsonl \
  --paper-content data/support_packets/fulltext_content/llm_agent/content.jsonl \
  --paper-index data/support_packets/pageindex/llm_agent/pageindex.jsonl \
  --llm-config configs/llm/mimo_flash.local.yaml \
  --output data/support_packets/paper_structures/llm_agent/paper_structures.jsonl \
  --error-output data/support_packets/paper_structures/llm_agent/errors.jsonl \
  --workers 8

PYTHONPATH=src python scripts/extract_paper_structures.py \
  --cases data/support_packets/fulltext_cases/llm_finetuning_post_training/support_packet_cases.jsonl \
  --papers data/domains/llm_finetuning_post_training/clean/core_papers.publication_enriched.semanticscholar.jsonl \
  --labels data/domains/llm_finetuning_post_training/annotations/paper_labels.jsonl \
  --paper-content data/support_packets/fulltext_content/llm_finetuning_post_training/content.jsonl \
  --paper-index data/support_packets/pageindex/llm_finetuning_post_training/pageindex.jsonl \
  --llm-config configs/llm/mimo_flash.local.yaml \
  --output data/support_packets/paper_structures/llm_finetuning_post_training/paper_structures.jsonl \
  --error-output data/support_packets/paper_structures/llm_finetuning_post_training/errors.jsonl \
  --workers 8

PYTHONPATH=src python scripts/extract_paper_structures.py \
  --cases data/support_packets/fulltext_cases/rag_and_retrieval_structuring/support_packet_cases.jsonl \
  --papers data/domains/rag_and_retrieval_structuring/clean/core_papers.publication_enriched.semanticscholar.jsonl \
  --labels data/domains/rag_and_retrieval_structuring/annotations/paper_labels.jsonl \
  --paper-content data/support_packets/fulltext_content/rag_and_retrieval_structuring/content.jsonl \
  --paper-index data/support_packets/pageindex/rag_and_retrieval_structuring/pageindex.jsonl \
  --llm-config configs/llm/mimo_flash.local.yaml \
  --output data/support_packets/paper_structures/rag_and_retrieval_structuring/paper_structures.jsonl \
  --error-output data/support_packets/paper_structures/rag_and_retrieval_structuring/errors.jsonl \
  --workers 8

PYTHONPATH=src python scripts/extract_paper_structures.py \
  --cases data/support_packets/fulltext_cases/visual_generative_modeling_and_diffusion/support_packet_cases.jsonl \
  --papers data/domains/visual_generative_modeling_and_diffusion/clean/core_papers.publication_enriched.semanticscholar.jsonl \
  --labels data/domains/visual_generative_modeling_and_diffusion/annotations/paper_labels.jsonl \
  --paper-content data/support_packets/fulltext_content/visual_generative_modeling_and_diffusion/content.jsonl \
  --paper-index data/support_packets/pageindex/visual_generative_modeling_and_diffusion/pageindex.jsonl \
  --llm-config configs/llm/mimo_flash.local.yaml \
  --output data/support_packets/paper_structures/visual_generative_modeling_and_diffusion/paper_structures.jsonl \
  --error-output data/support_packets/paper_structures/visual_generative_modeling_and_diffusion/errors.jsonl \
  --workers 8

PYTHONPATH=src python scripts/build_v2_task_candidates.py
PYTHONPATH=src python scripts/rewrite_v2_task_candidates.py --llm-config configs/llm/mimo_flash.local.yaml --workers 16
PYTHONPATH=src python scripts/judge_v2_task_candidates.py --llm-config configs/llm/mimo_flash.local.yaml --workers 16
PYTHONPATH=src python scripts/build_v2_release.py \
  --accept-threshold 0.82 \
  --min-heuristic 0.4 \
  --max-per-family-domain 18 \
  --out-dir data/releases/benchmark_v2_20260329

echo "benchmark_v2_20260329 ready"
