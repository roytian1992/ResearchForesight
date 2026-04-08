#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

stamp() { date '+%F %T %Z'; }
run() {
  echo "[RUN  $(stamp)] $*"
  "$@"
  echo "[DONE $(stamp)] $1"
}

wait_for_downloads() {
  while pgrep -f 'scripts/download_papers_monthly.py --domain-id llm_agent --start 2026-02-01T00:00:00 --end 2026-03-01T00:00:00' >/dev/null \
     || pgrep -f 'scripts/download_papers_monthly.py --domain-id llm_finetuning_post_training --start 2026-02-01T00:00:00 --end 2026-03-01T00:00:00' >/dev/null; do
    echo "[WAIT $(stamp)] waiting for 2026-02 download jobs to finish"
    sleep 60
  done
}

count_month() {
  python3 - "$1" "$2" <<'PY'
import json, sys
path, prefix = sys.argv[1], sys.argv[2]
count=0
latest=None
with open(path) as f:
    for line in f:
        try:o=json.loads(line)
        except: continue
        dt=o.get('published') or o.get('published_at') or o.get('updated') or o.get('created')
        if isinstance(dt, str) and dt.startswith(prefix):
            count += 1
            latest = max(latest, dt) if latest else dt
print(f"{path} prefix={prefix} count={count} latest={latest}")
PY
}

echo "[START $(stamp)] quarterly refresh pipeline"
wait_for_downloads
count_month data/raw/paper_metadata_llm_agent.jsonl 2026-02
count_month data/raw/paper_metadata_llm_post_training.jsonl 2026-02
count_month data/raw/paper_metadata_rag.jsonl 2026-02

for domain_cfg in \
  configs/domains/llm_agent.yaml \
  configs/domains/llm_finetuning_post_training.yaml \
  configs/domains/rag_and_retrieval_structuring.yaml
  do
  run python3 scripts/build_domain_corpus.py --domain-config "$domain_cfg"
done

run python3 -u scripts/enrich_publication_semanticscholar.py \
  --input data/domains/llm_agent/interim/papers_merged.jsonl \
  --output data/domains/llm_agent/interim/publication_enrichment.semanticscholar.all.jsonl \
  --summary-output data/domains/llm_agent/interim/publication_enrichment.semanticscholar.all.summary.json \
  --batch-size 500 --sleep 0.1 --sort-by-published --resume
run python3 scripts/merge_publication_enrichment.py \
  --input data/domains/llm_agent/interim/papers_merged.jsonl \
  --enrichment data/domains/llm_agent/interim/publication_enrichment.semanticscholar.all.jsonl \
  --output data/domains/llm_agent/interim/papers_merged.publication_enriched.semanticscholar.jsonl

run python3 -u scripts/enrich_publication_semanticscholar.py \
  --input data/domains/llm_finetuning_post_training/interim/papers_merged.jsonl \
  --output data/domains/llm_finetuning_post_training/interim/publication_enrichment.semanticscholar.all.jsonl \
  --summary-output data/domains/llm_finetuning_post_training/interim/publication_enrichment.semanticscholar.all.summary.json \
  --batch-size 500 --sleep 0.1 --sort-by-published --resume
run python3 scripts/merge_publication_enrichment.py \
  --input data/domains/llm_finetuning_post_training/interim/papers_merged.jsonl \
  --enrichment data/domains/llm_finetuning_post_training/interim/publication_enrichment.semanticscholar.all.jsonl \
  --output data/domains/llm_finetuning_post_training/interim/papers_merged.publication_enriched.semanticscholar.jsonl

run python3 -u scripts/enrich_publication_semanticscholar.py \
  --input data/domains/rag_and_retrieval_structuring/interim/papers_merged.jsonl \
  --output data/domains/rag_and_retrieval_structuring/interim/publication_enrichment.semanticscholar.all.jsonl \
  --summary-output data/domains/rag_and_retrieval_structuring/interim/publication_enrichment.semanticscholar.all.summary.json \
  --batch-size 500 --sleep 0.1 --sort-by-published --resume
run python3 scripts/merge_publication_enrichment.py \
  --input data/domains/rag_and_retrieval_structuring/interim/papers_merged.jsonl \
  --enrichment data/domains/rag_and_retrieval_structuring/interim/publication_enrichment.semanticscholar.all.jsonl \
  --output data/domains/rag_and_retrieval_structuring/interim/papers_merged.publication_enriched.semanticscholar.jsonl

for domain_cfg in \
  configs/domains/llm_agent.yaml \
  configs/domains/llm_finetuning_post_training.yaml \
  configs/domains/rag_and_retrieval_structuring.yaml
  do
  run python3 -u scripts/llm_label_domain_papers.py --domain-config "$domain_cfg" --resume --progress-every 50
done

for domain_cfg in \
  configs/domains/llm_agent.yaml \
  configs/domains/llm_finetuning_post_training.yaml \
  configs/domains/rag_and_retrieval_structuring.yaml
  do
  run python3 scripts/extract_core_domain_corpus.py --domain-config "$domain_cfg"
done

run python3 scripts/merge_publication_enrichment.py \
  --input data/domains/llm_finetuning_post_training/clean/core_papers.jsonl \
  --enrichment data/domains/llm_finetuning_post_training/interim/publication_enrichment.semanticscholar.all.jsonl \
  --output data/domains/llm_finetuning_post_training/clean/core_papers.publication_enriched.semanticscholar.jsonl
run python3 scripts/merge_publication_enrichment.py \
  --input data/domains/rag_and_retrieval_structuring/clean/core_papers.jsonl \
  --enrichment data/domains/rag_and_retrieval_structuring/interim/publication_enrichment.semanticscholar.all.jsonl \
  --output data/domains/rag_and_retrieval_structuring/clean/core_papers.publication_enriched.semanticscholar.jsonl

nohup python3 -u scripts/run_taxoadapt_temporal.py --config configs/taxoadapt_quarterly.yaml --domain-id rag_and_retrieval_structuring --run-name quarterly_v1 > "$LOG_DIR/taxoadapt_quarterly_rag.log" 2>&1 &
echo $! > "$LOG_DIR/taxoadapt_quarterly_rag.pid"
nohup python3 -u scripts/run_taxoadapt_temporal.py --config configs/taxoadapt_quarterly.yaml --domain-id llm_agent --run-name quarterly_v1 > "$LOG_DIR/taxoadapt_quarterly_llm_agent.log" 2>&1 &
echo $! > "$LOG_DIR/taxoadapt_quarterly_llm_agent.pid"
nohup python3 -u scripts/run_taxoadapt_temporal.py --config configs/taxoadapt_quarterly.yaml --domain-id llm_finetuning_post_training --run-name quarterly_v1 > "$LOG_DIR/taxoadapt_quarterly_post_training.log" 2>&1 &
echo $! > "$LOG_DIR/taxoadapt_quarterly_post_training.pid"

echo "[DONE $(stamp)] quarterly refresh pipeline launched taxonomy jobs"
