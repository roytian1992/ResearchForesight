#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RELEASE_DIR="${RELEASE_DIR:-$ROOT/data/releases/benchmark_experiment100_llmfuture_vote_v3_qwen8002_evidenceexp_v1}"
ADAPTER_PRIMARY_LLM="${ADAPTER_PRIMARY_LLM:-$ROOT/tmp/local_llm_configs/qwen_235b_8001.local.yaml}"
ADAPTER_FALLBACK_LLM="${ADAPTER_FALLBACK_LLM:-$ROOT/tmp/local_llm_configs/qwen_235b_8001.local.yaml}"
EVAL_PRIMARY_LLM="${EVAL_PRIMARY_LLM:-$ROOT/tmp/local_llm_configs/mimo_v2_pro_xiaomimimo_20260421.local.yaml}"
EVAL_FALLBACK_LLM="${EVAL_FALLBACK_LLM:-$ROOT/tmp/local_llm_configs/qwen_235b_8001.local.yaml}"
ADAPTER_WORKERS="${ADAPTER_WORKERS:-6}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
RR_WORKERS="${RR_WORKERS:-4}"

ADAPT_ROOT="${ADAPT_ROOT:-$ROOT/results/experiment100_agent_final_renderer_v2_qwen8001_20260422}"
METRICS_ROOT="${METRICS_ROOT:-$ROOT/results/experiment100_full_eval_20260422_agent_final_renderer_v2_qwen8001_rewrite_mimoeval}"
ROUND_ROOT="${ROUND_ROOT:-$ROOT/results/round_robin_structured_idea_arena_evidence_light_full100_agent_final_renderer_v2_qwen8001_rewrite_20260422}"

mkdir -p "$ADAPT_ROOT/logs" "$METRICS_ROOT/logs" "$ROUND_ROOT"

COI_INPUT="$ROOT/results/coi_experiment100_focusfixv3_20260420_parallel4/results_merged.jsonl"
RA_INPUT="$ROOT/results/researchagent_experiment100_qwen8002_20260420_parallel4/results_merged.jsonl"
ARIS_INPUT="$ROOT/results/aris_experiment100_evidenceexp_v1_contractfix_parallel4/results.jsonl"

COI_OUTPUT="$ADAPT_ROOT/coi/results.jsonl"
RA_OUTPUT="$ADAPT_ROOT/researchagent/results.jsonl"
ARIS_OUTPUT="$ADAPT_ROOT/aris/results.jsonl"

NATIVE_INPUT="$ROOT/results/native_llm_experiment100_evidenceexp_v1_parallel4/results.jsonl"
HYBRID_INPUT="$ROOT/results/hybrid_rag_experiment100_evidenceexp_v1_parallel4/results.jsonl"

run_adapter() {
  local method="$1"
  local input_jsonl="$2"
  local output_jsonl="$3"
  local log_path="$ADAPT_ROOT/logs/${method}.log"
  mkdir -p "$(dirname "$output_jsonl")"
  python -u "$ROOT/scripts/adapt_experiment_answers.py" \
    --results-jsonl "$input_jsonl" \
    --release-dir "$RELEASE_DIR" \
    --output-jsonl "$output_jsonl" \
    --adapter-llm-config "$ADAPTER_PRIMARY_LLM" \
    --adapter-fallback-llm-config "$ADAPTER_FALLBACK_LLM" \
    --workers "$ADAPTER_WORKERS" \
    --resume \
    >"$log_path" 2>&1
}

run_eval() {
  local method="$1"
  local input_jsonl="$2"
  local output_dir="$METRICS_ROOT/$method"
  local log_path="$METRICS_ROOT/logs/${method}.log"
  mkdir -p "$output_dir"
  python -u "$ROOT/scripts/evaluate_experiment_final_metrics.py" \
    --results-jsonl "$input_jsonl" \
    --release-dir "$RELEASE_DIR" \
    --output-dir "$output_dir" \
    --metrics all \
    --workers "$EVAL_WORKERS" \
    --judge-llm-config "$EVAL_PRIMARY_LLM" \
    --judge-fallback-llm-config "$EVAL_FALLBACK_LLM" \
    >"$log_path" 2>&1
}

echo "[pipeline] adapting agent answers"
run_adapter "coi" "$COI_INPUT" "$COI_OUTPUT"
run_adapter "researchagent" "$RA_INPUT" "$RA_OUTPUT"
run_adapter "aris" "$ARIS_INPUT" "$ARIS_OUTPUT"

echo "[pipeline] evaluating full metrics"
run_eval "native_llm" "$NATIVE_INPUT" &
pid_native=$!
run_eval "hybrid_rag" "$HYBRID_INPUT" &
pid_hybrid=$!
run_eval "coi" "$COI_OUTPUT" &
pid_coi_eval=$!
run_eval "researchagent" "$RA_OUTPUT" &
pid_ra_eval=$!
run_eval "aris" "$ARIS_OUTPUT" &
pid_aris_eval=$!
wait "$pid_native" "$pid_hybrid" "$pid_coi_eval" "$pid_ra_eval" "$pid_aris_eval"

echo "[pipeline] merging metrics"
python -u "$ROOT/scripts/merge_final_metrics.py" \
  --v31-dir "native_llm=$METRICS_ROOT/native_llm/eval_v31" \
  --v31-dir "hybrid_rag=$METRICS_ROOT/hybrid_rag/eval_v31" \
  --v31-dir "coi=$METRICS_ROOT/coi/eval_v31" \
  --v31-dir "researchagent=$METRICS_ROOT/researchagent/eval_v31" \
  --v31-dir "aris=$METRICS_ROOT/aris/eval_v31" \
  --v4-dir "native_llm=$METRICS_ROOT/native_llm/eval_v4" \
  --v4-dir "hybrid_rag=$METRICS_ROOT/hybrid_rag/eval_v4" \
  --v4-dir "coi=$METRICS_ROOT/coi/eval_v4" \
  --v4-dir "researchagent=$METRICS_ROOT/researchagent/eval_v4" \
  --v4-dir "aris=$METRICS_ROOT/aris/eval_v4" \
  --aux-dir "native_llm=$METRICS_ROOT/native_llm/eval_aux" \
  --aux-dir "hybrid_rag=$METRICS_ROOT/hybrid_rag/eval_aux" \
  --aux-dir "coi=$METRICS_ROOT/coi/eval_aux" \
  --aux-dir "researchagent=$METRICS_ROOT/researchagent/eval_aux" \
  --aux-dir "aris=$METRICS_ROOT/aris/eval_aux" \
  --output-dir "$METRICS_ROOT/merged_final_metrics" \
  >"$METRICS_ROOT/logs/merge_final_metrics.log" 2>&1

echo "[pipeline] running round-robin"
python -u "$ROOT/scripts/run_pairwise_bestofk_v3.py" \
  --release-dir "$RELEASE_DIR" \
  --output-dir "$ROUND_ROOT" \
  --judge-llm-config "$EVAL_PRIMARY_LLM" \
  --fallback-judge-llm-config "$EVAL_FALLBACK_LLM" \
  --workers "$RR_WORKERS" \
  --min-rounds 3 \
  --max-rounds 3 \
  --judge-profile structured_idea_arena_evidence_light \
  --input \
    "native_llm=$NATIVE_INPUT" \
    "hybrid_rag=$HYBRID_INPUT" \
    "coi=$COI_OUTPUT" \
    "researchagent=$RA_OUTPUT" \
    "aris=$ARIS_OUTPUT" \
  >"$ROUND_ROOT/run.log" 2>&1

echo "[pipeline] aggregating round-robin"
python -u "$ROOT/scripts/aggregate_pairwise_bestofk_v3.py" \
  --base-jsonl "$ROUND_ROOT/pairwise_bestofk_results.jsonl" \
  --output-dir "$ROUND_ROOT/aggregated" \
  >"$ROUND_ROOT/aggregate.log" 2>&1

echo "[pipeline] complete"
echo "adapt_root=$ADAPT_ROOT"
echo "metrics_root=$METRICS_ROOT"
echo "round_root=$ROUND_ROOT"
