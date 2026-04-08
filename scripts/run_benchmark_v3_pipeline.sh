#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PACKETS_ROOT="${PACKETS_ROOT:-data/family_packets/v3}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-data/task_candidates_v3}"
REWRITE_LLM_CONFIG="${REWRITE_LLM_CONFIG:-configs/llm/mimo_flash.local.yaml}"
JUDGE_LLM_CONFIG="${JUDGE_LLM_CONFIG:-configs/llm/mimo_flash.local.yaml}"
RELEASE_DIR="${RELEASE_DIR:-data/releases/benchmark_v3_$(date +%Y%m%d)}"
WORKERS="${WORKERS:-8}"

echo "[v3] build family packets -> $PACKETS_ROOT"
PYTHONPATH=src python scripts/build_family_packets.py --out-root "$PACKETS_ROOT"

echo "[v3] build family task candidates -> $CANDIDATE_ROOT"
rm -rf "$CANDIDATE_ROOT"
PYTHONPATH=src python scripts/build_family_task_candidates_v3.py --packets-root "$PACKETS_ROOT" --out-dir "$CANDIDATE_ROOT"

echo "[v3] rewrite family candidates"
PYTHONPATH=src python scripts/rewrite_v3_family_task_candidates.py \
  --input "$CANDIDATE_ROOT/all_candidates.jsonl" \
  --output "$CANDIDATE_ROOT/all_candidates.rewritten.jsonl" \
  --errors "$CANDIDATE_ROOT/all_candidates.rewrite_errors.jsonl" \
  --llm-config "$REWRITE_LLM_CONFIG" \
  --workers "$WORKERS"

echo "[v3] judge family candidates"
PYTHONPATH=src python scripts/judge_v3_family_task_candidates.py \
  --input "$CANDIDATE_ROOT/all_candidates.rewritten.jsonl" \
  --output "$CANDIDATE_ROOT/all_candidates.judged.jsonl" \
  --errors "$CANDIDATE_ROOT/all_candidates.judge_errors.jsonl" \
  --llm-config "$JUDGE_LLM_CONFIG" \
  --workers "$WORKERS"

echo "[v3] build release -> $RELEASE_DIR"
PYTHONPATH=src python scripts/build_v3_release.py \
  --input "$CANDIDATE_ROOT/all_candidates.judged.jsonl" \
  --out-dir "$RELEASE_DIR"

echo "[v3] build hidden eval v3"
python scripts/build_hidden_eval_v3.py --release-dir "$RELEASE_DIR"

echo "[v3] postprocess release bundle"
python scripts/postprocess_release_bundle.py --release-dir "$RELEASE_DIR"

echo "[v3] done -> $RELEASE_DIR"
