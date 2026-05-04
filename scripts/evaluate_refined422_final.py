from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.experiment_eval_v3 import infer_domain_id
from researchworld.experiment_eval_aux import AUX_NAME_BY_FAMILY, evaluate_family_auxiliary
from researchworld.experiment_eval_v4 import evaluate_evidence_traceability_v4
from researchworld.factscore_eval_v3 import FactScoreV3Config
from researchworld.factscore_eval_v5 import evaluate_answer_expanded_factscore_v5, evaluate_answer_factscore_v5
from researchworld.future_alignment_eval_v3_1 import FutureAlignmentV3_1Config
from researchworld.future_alignment_eval_v5 import evaluate_expanded_future_alignment_v5, evaluate_future_alignment_v5
from researchworld.llm import (
    FallbackOpenAICompatChatClient,
    OpenAICompatChatClient,
    OpenAICompatEmbeddingClient,
    load_openai_compat_config,
    load_openai_compat_embedding_config,
)
from researchworld.refined_release import load_task_refined_views
from researchworld.research_judgment_rubrics import default_evaluation_rubric
from researchworld.research_judgment_eval_v8 import evaluate_research_judgment_v8


FINAL_METRICS = ("factuality", "future_alignment", "traceability", "research_judgment", "family_metric")
EXPANDED_GOLD_METRICS = ("evidence_grounded_factuality", "expanded_future_alignment")
METRIC_ORDER = FINAL_METRICS + EXPANDED_GOLD_METRICS
METRIC_ALIASES = {
    "all": set(FINAL_METRICS),
    "all_with_expanded_gold": set(METRIC_ORDER),
    "primary": {"factuality", "future_alignment", "traceability"},
    "primary_metrics": {"factuality", "future_alignment", "traceability"},
    "expanded_primary": {"evidence_grounded_factuality", "expanded_future_alignment", "traceability"},
    "expanded_gold": {"evidence_grounded_factuality", "expanded_future_alignment"},
    "candidate": {"research_judgment"},
    "candidate_metric": {"research_judgment"},
    "factuality": {"factuality"},
    "fact": {"factuality"},
    "evidence_grounded_factuality": {"evidence_grounded_factuality"},
    "expanded_factuality": {"evidence_grounded_factuality"},
    "expanded_fact": {"evidence_grounded_factuality"},
    "future_alignment": {"future_alignment"},
    "future": {"future_alignment"},
    "expanded_future_alignment": {"expanded_future_alignment"},
    "expanded_future": {"expanded_future_alignment"},
    "traceability": {"traceability"},
    "evidence_traceability": {"traceability"},
    "research_judgment": {"research_judgment"},
    "judgment": {"research_judgment"},
    "family_metric": {"family_metric"},
    "family": {"family_metric"},
    "family_aux": {"family_metric"},
    "aux": {"family_metric"},
    "task_spec": {"research_judgment"},
    "task_specification": {"research_judgment"},
    "contract": {"research_judgment"},
}
FINAL_CACHE_VERSION = "refined422_final_metrics_20260430_research_judgment_v8_traceability_v5_1_unit_fas_v7_2_expanded_gold_pilot_v2"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _task_id(row: Dict[str, Any]) -> str:
    return str(row.get("task_id") or "").strip()


def _method_name(row: Dict[str, Any]) -> str:
    return str(row.get("agent") or row.get("baseline") or row.get("method") or "unknown")


def _clamp01(value: Any) -> float:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(obj: Any) -> str:
    return hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()


def _parse_metrics(raw_values: Sequence[str]) -> tuple[List[str], List[str]]:
    requested: List[str] = []
    selected: Set[str] = set()
    for raw in raw_values:
        for part in [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]:
            requested.append(part)
            aliases = METRIC_ALIASES.get(part)
            if aliases is None:
                valid = ", ".join(sorted(METRIC_ALIASES))
                raise SystemExit(f'unknown metric "{part}". valid values: {valid}')
            selected.update(aliases)
    if not selected:
        requested = ["all"]
        selected = set(FINAL_METRICS)
    return [metric for metric in METRIC_ORDER if metric in selected], requested


def _build_judge_client(primary_config: Path, fallback_config: str) -> FallbackOpenAICompatChatClient:
    primary = OpenAICompatChatClient(load_openai_compat_config(primary_config))
    fallback = None
    fallback_path = Path(fallback_config) if str(fallback_config or "").strip() else None
    if fallback_path and fallback_path.exists():
        fallback = OpenAICompatChatClient(load_openai_compat_config(fallback_path))
    return FallbackOpenAICompatChatClient(primary, fallback)


def _metric_cache_key(metric: str, public_task: Dict[str, Any], eval_row: Dict[str, Any], result_row: Dict[str, Any]) -> str:
    answer = str(result_row.get("answer") or "")
    payload = {
        "version": FINAL_CACHE_VERSION,
        "metric": metric,
        "task_id": public_task.get("task_id") or eval_row.get("task_id"),
        "family": public_task.get("family") or eval_row.get("family"),
        "question": public_task.get("question"),
        "answer": answer,
    }
    if metric == "factuality":
        payload["claim_bank"] = eval_row.get("claim_bank") or []
        payload["temporal_policy"] = eval_row.get("temporal_policy") or {}
    elif metric == "evidence_grounded_factuality":
        payload["claim_bank"] = eval_row.get("claim_bank") or []
        payload["gold_sets"] = eval_row.get("gold_sets") or {}
        payload["temporal_policy"] = eval_row.get("temporal_policy") or {}
    elif metric == "future_alignment":
        payload["future_alignment_targets"] = eval_row.get("future_alignment_targets") or {}
        payload["temporal_policy"] = eval_row.get("temporal_policy") or {}
        payload["gold_answer"] = eval_row.get("gold_answer")
        payload["expected_answer_points"] = eval_row.get("expected_answer_points") or []
    elif metric == "expanded_future_alignment":
        payload["future_alignment_targets"] = eval_row.get("future_alignment_targets") or {}
        payload["gold_sets"] = eval_row.get("gold_sets") or {}
        payload["temporal_policy"] = eval_row.get("temporal_policy") or {}
        payload["gold_answer"] = eval_row.get("gold_answer")
        payload["expected_answer_points"] = eval_row.get("expected_answer_points") or []
    elif metric == "traceability":
        payload["result_trace"] = result_row.get("trace") or {}
        payload["result_evidence"] = result_row.get("evidence") or {}
    elif metric == "research_judgment":
        payload["answer_contract"] = public_task.get("answer_contract") or {}
        payload["gold_answer"] = eval_row.get("gold_answer")
        payload["expected_answer_points"] = eval_row.get("expected_answer_points") or []
        payload["evaluation_rubric"] = eval_row.get("evaluation_rubric") or default_evaluation_rubric(
            str(public_task.get("family") or eval_row.get("family") or "")
        )
        payload["component_targets"] = eval_row.get("component_targets") or {}
    elif metric == "family_metric":
        payload["slot_targets"] = eval_row.get("slot_targets") or {}
        payload["component_targets"] = eval_row.get("component_targets") or {}
        payload["answer_contract"] = public_task.get("answer_contract") or {}
        payload["deliverable_spec"] = public_task.get("deliverable_spec") or {}
    return _sha256(payload)


def _cache_path(cache_dir: Optional[Path], key: str) -> Optional[Path]:
    if cache_dir is None:
        return None
    return cache_dir / key[:2] / f"{key}.json"


def _read_cache(cache_dir: Optional[Path], key: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(cache_dir, key)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(cache_dir: Optional[Path], key: str, payload: Dict[str, Any]) -> None:
    path = _cache_path(cache_dir, key)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _compact_metric_detail(metric: str, detail: Dict[str, Any]) -> Dict[str, Any]:
    if metric in {"factuality", "evidence_grounded_factuality"}:
        return {
            "precision_score": detail.get("precision_score"),
            "coverage_score": detail.get("coverage_score"),
            "claim_count": detail.get("claim_count"),
            "supported_count": detail.get("supported_count"),
            "weighted_supported": detail.get("weighted_supported"),
            "weighted_total": detail.get("weighted_total"),
            "weighted_gt_covered": detail.get("weighted_gt_covered"),
            "weighted_gt_total": detail.get("weighted_gt_total"),
            "claims": detail.get("claims") or [],
            "claim_bank_count": detail.get("claim_bank_count"),
            "expanded_claim_bank_count": detail.get("expanded_claim_bank_count"),
            "claim_bank_mode": detail.get("claim_bank_mode"),
            "evaluator_scope": detail.get("evaluator_scope"),
        }
    if metric in {"future_alignment", "expanded_future_alignment"}:
        return {
            "weighted_unit_alignment": detail.get("weighted_unit_alignment"),
            "alignment_coverage": detail.get("alignment_coverage"),
            "mean_specificity": detail.get("mean_specificity"),
            "canonical_fas_score": detail.get("canonical_fas_score"),
            "reference_answer_similarity": detail.get("reference_answer_similarity"),
            "reference_answer_raw_cosine": detail.get("reference_answer_raw_cosine"),
            "reference_answer_text": detail.get("reference_answer_text"),
            "scope_calibrated_future_signal_fas": detail.get("scope_calibrated_future_signal_fas"),
            "raw_weighted_embedding_fas": detail.get("raw_weighted_embedding_fas"),
            "fas_mode": detail.get("fas_mode"),
            "unit_count": len(detail.get("units") or []),
            "units": detail.get("units") or [],
            "strict_future_alignment_score": detail.get("strict_future_alignment_score"),
            "best_acceptable_neighbor_score": detail.get("best_acceptable_neighbor_score"),
            "acceptable_neighbors": detail.get("acceptable_neighbors") or [],
            "negative_confusions": detail.get("negative_confusions") or [],
            "candidate_selection": detail.get("candidate_selection") or {},
            "negative_confusion_cap": detail.get("negative_confusion_cap"),
            "evaluator_scope": detail.get("evaluator_scope"),
        }
    if metric == "traceability":
        return {
            "rubric_scores": detail.get("rubric_scores") or {},
            "strengths": detail.get("strengths") or [],
            "weaknesses": detail.get("weaknesses") or [],
        }
    if metric == "research_judgment":
        return {
            "dimension_scores": detail.get("dimension_scores") or {},
            "relation_to_gold": detail.get("relation_to_gold"),
            "neighborhood_advancement": detail.get("neighborhood_advancement"),
            "more_forward_than_gold": detail.get("more_forward_than_gold"),
            "scope_relation": detail.get("scope_relation"),
            "critical_core_error": detail.get("critical_core_error"),
            "wrong_required_label_or_ranking": detail.get("wrong_required_label_or_ranking"),
            "kitchen_sink_or_noncommittal": detail.get("kitchen_sink_or_noncommittal"),
            "non_answer_or_refusal": detail.get("non_answer_or_refusal"),
            "has_explicit_comparison": detail.get("has_explicit_comparison"),
            "has_risk_or_dependency": detail.get("has_risk_or_dependency"),
            "has_actionable_decision_rule": detail.get("has_actionable_decision_rule"),
            "has_future_specificity": detail.get("has_future_specificity"),
            "decision_strengths": detail.get("decision_strengths") or [],
            "decision_weaknesses": detail.get("decision_weaknesses") or [],
            "reference_units": detail.get("reference_units") or {},
            "candidate_units": detail.get("candidate_units") or {},
            "missing_gold_units": detail.get("missing_gold_units") or [],
            "overbroad_or_drift_units": detail.get("overbroad_or_drift_units") or [],
            "rationale": detail.get("rationale"),
            "evaluator_scope": detail.get("evaluator_scope"),
            "rubric_version": detail.get("rubric_version"),
        }
    if metric == "family_metric":
        return {
            "family_aux_metric_name": detail.get("family_aux_metric_name"),
            "rubric_scores": detail.get("rubric_scores") or {},
            "strengths": detail.get("strengths") or [],
            "weaknesses": detail.get("weaknesses") or [],
            "contract_audit": detail.get("contract_audit") or {},
        }
    return detail


def _evaluate_metric(
    metric: str,
    *,
    judge_client: FallbackOpenAICompatChatClient,
    embedding_client: Optional[OpenAICompatEmbeddingClient],
    public_task: Dict[str, Any],
    eval_row: Dict[str, Any],
    result_row: Dict[str, Any],
    cache_dir: Optional[Path],
) -> tuple[float, Dict[str, Any], Dict[str, Any]]:
    key = _metric_cache_key(metric, public_task, eval_row, result_row)
    cached = _read_cache(cache_dir, key)
    if cached is not None:
        return _clamp01(cached.get("score")), cached.get("detail") or {}, {
            "enabled": cache_dir is not None,
            "status": "hit",
            "cache_key": key,
            "cache_version": FINAL_CACHE_VERSION,
        }

    if metric == "factuality":
        detail = evaluate_answer_factscore_v5(
            judge_client=judge_client,
            result_row=result_row,
            gt_row=eval_row,
            cfg=FactScoreV3Config(),
        )
        score = _clamp01(detail.get("benchmark_factscore"))
    elif metric == "evidence_grounded_factuality":
        detail = evaluate_answer_expanded_factscore_v5(
            judge_client=judge_client,
            result_row=result_row,
            gt_row=eval_row,
            cfg=FactScoreV3Config(),
        )
        score = _clamp01(detail.get("benchmark_factscore"))
    elif metric == "future_alignment":
        detail = evaluate_future_alignment_v5(
            judge_client=judge_client,
            embedding_client=embedding_client,
            public_task=public_task,
            result_row=result_row,
            hidden_row=eval_row,
            cfg=FutureAlignmentV3_1Config(),
        )
        score = _clamp01(detail.get("future_alignment_score"))
    elif metric == "expanded_future_alignment":
        detail = evaluate_expanded_future_alignment_v5(
            judge_client=judge_client,
            embedding_client=embedding_client,
            public_task=public_task,
            result_row=result_row,
            hidden_row=eval_row,
            cfg=FutureAlignmentV3_1Config(),
        )
        score = _clamp01(detail.get("future_alignment_score"))
    elif metric == "traceability":
        detail = evaluate_evidence_traceability_v4(
            judge_client,
            public_task=public_task,
            family=str(public_task.get("family") or eval_row.get("family") or ""),
            candidate_answer=str(result_row.get("answer") or ""),
            result_row=result_row,
        )
        score = _clamp01(detail.get("evidence_traceability_score"))
    elif metric == "research_judgment":
        detail = evaluate_research_judgment_v8(
            judge_client=judge_client,
            public_task=public_task,
            hidden_row=eval_row,
            result_row=result_row,
        )
        score = _clamp01(detail.get("research_judgment_score"))
    elif metric == "family_metric":
        detail = evaluate_family_auxiliary(
            judge_client,
            public_task=public_task,
            hidden_row=eval_row,
            result_row=result_row,
        )
        family = str(public_task.get("family") or eval_row.get("family") or "")
        score_key = AUX_NAME_BY_FAMILY.get(family) or "family_aux_score"
        detail = dict(detail)
        detail["family_metric_score_key"] = score_key
        score = _clamp01(detail.get(score_key))
    else:
        raise RuntimeError(f"unsupported metric: {metric}")

    _write_cache(
        cache_dir,
        key,
        {
            "cache_version": FINAL_CACHE_VERSION,
            "metric": metric,
            "task_id": public_task.get("task_id"),
            "answer_sha256": hashlib.sha256(str(result_row.get("answer") or "").encode("utf-8")).hexdigest(),
            "score": score,
            "detail": detail,
        },
    )
    return score, detail, {
        "enabled": cache_dir is not None,
        "status": "miss",
        "cache_key": key,
        "cache_version": FINAL_CACHE_VERSION,
    }


def _result_row(
    *,
    run_id: str,
    public_task: Dict[str, Any],
    eval_row: Dict[str, Any],
    source_row: Dict[str, Any],
    selected_metrics: Sequence[str],
    judge_client: FallbackOpenAICompatChatClient,
    embedding_client: Optional[OpenAICompatEmbeddingClient],
    cache_dir: Optional[Path],
) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    details: Dict[str, Dict[str, Any]] = {}
    cache: Dict[str, Dict[str, Any]] = {}
    for metric in selected_metrics:
        score, detail, cache_meta = _evaluate_metric(
            metric,
            judge_client=judge_client,
            embedding_client=embedding_client,
            public_task=public_task,
            eval_row=eval_row,
            result_row=source_row,
            cache_dir=cache_dir,
        )
        scores[f"{metric}_score"] = score
        details[metric] = _compact_metric_detail(metric, detail)
        cache[metric] = cache_meta

    return {
        "run_id": run_id,
        "task_id": public_task.get("task_id"),
        "family": public_task.get("family"),
        "domain": infer_domain_id(source_row) or public_task.get("domain"),
        "method": _method_name(source_row),
        "answer": str(source_row.get("answer") or ""),
        "scores": scores,
        "metric_details": details,
        "metadata": {
            "schema_version": FINAL_CACHE_VERSION,
            "task_title": public_task.get("title"),
            "time_cutoff": public_task.get("time_cutoff"),
            "evaluator_scope": "task_refined_json_only_no_kb_for_fact_future_research_judgment",
            "traceability_scope": "method_output_support_artifacts_plus_answer_internal_rationale",
            "requested_metrics": list(selected_metrics),
            "eval_cache": cache,
        },
    }


def _validate_inputs(rows: List[Dict[str, Any]], public_by_id: Dict[str, Dict[str, Any]], eval_by_id: Dict[str, Dict[str, Any]]) -> None:
    seen: Set[str] = set()
    bad: List[str] = []
    duplicates: List[str] = []
    for row in rows:
        task_id = _task_id(row)
        if not task_id or task_id not in public_by_id or task_id not in eval_by_id:
            bad.append(task_id or "<missing-task-id>")
        if task_id in seen:
            duplicates.append(task_id)
        seen.add(task_id)
    if bad:
        raise RuntimeError(f"results contain task IDs missing from task_refined.jsonl: count={len(bad)} first={bad[:5]}")
    if duplicates:
        raise RuntimeError(f"results contain duplicate task IDs: count={len(duplicates)} first={duplicates[:5]}")


def _evaluate_rows(
    *,
    rows: List[Dict[str, Any]],
    release_dir: Path,
    output_dir: Path,
    run_id: str,
    selected_metrics: Sequence[str],
    judge_client: FallbackOpenAICompatChatClient,
    embedding_client: Optional[OpenAICompatEmbeddingClient],
    resume: bool,
    cache_dir: Optional[Path],
) -> List[Dict[str, Any]]:
    public_by_id, eval_by_id = load_task_refined_views(release_dir)
    _validate_inputs(rows, public_by_id, eval_by_id)

    output_path = output_dir / "results_final_metrics.jsonl"
    existing_rows = _load_jsonl(output_path) if resume else []
    completed = {_task_id(row) for row in existing_rows if _task_id(row)}
    mode = "a" if resume and completed else "w"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_rows = list(existing_rows)
    with output_path.open(mode, encoding="utf-8") as handle:
        for idx, source_row in enumerate(rows, start=1):
            task_id = _task_id(source_row)
            if not task_id or task_id in completed:
                continue
            public_task = public_by_id[task_id]
            eval_row = eval_by_id[task_id]
            print(f"[eval_final_clean] {idx}/{len(rows)} {task_id} metrics={','.join(selected_metrics)}", flush=True)
            row = _result_row(
                run_id=run_id,
                public_task=public_task,
                eval_row=eval_row,
                source_row=source_row,
                selected_metrics=selected_metrics,
                judge_client=judge_client,
                embedding_client=embedding_client,
                cache_dir=cache_dir,
            )
            out_rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
    return out_rows


def _summarizable_metrics(selected_metrics: Sequence[str], *, include_family_metric: bool) -> List[str]:
    if include_family_metric:
        return list(selected_metrics)
    return [metric for metric in selected_metrics if metric != "family_metric"]


def _group_summary(rows: List[Dict[str, Any]], selected_metrics: Sequence[str], *, include_family_metric: bool = False) -> Dict[str, Any]:
    metrics = _summarizable_metrics(selected_metrics, include_family_metric=include_family_metric)
    return {
        "count": len(rows),
        "mean_scores": {
            f"{metric}_score": _mean([float((row.get("scores") or {}).get(f"{metric}_score") or 0.0) for row in rows])
            for metric in metrics
        },
    }


def _summarize(rows: List[Dict[str, Any]], selected_metrics: Sequence[str]) -> Dict[str, Any]:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method") or "")].append(row)
        by_family[str(row.get("family") or "")].append(row)
        by_domain[str(row.get("domain") or "")].append(row)
    return {
        "task_count": len(rows),
        "metrics": list(selected_metrics),
            "metric_roles": {
                "primary_metrics": ["factuality", "future_alignment", "traceability"],
                "expanded_gold_metrics": ["evidence_grounded_factuality", "expanded_future_alignment"],
                "candidate_metric": "research_judgment",
            "family_metric": {
                "name": "family_metric",
                "aggregation_policy": "Report only within each task family; do not report or compare an overall cross-family mean.",
            },
            "aggregation_policy": "Do not average different metrics into a single headline score. Do not average family_metric across task families.",
        },
        "overall": _group_summary(rows, selected_metrics, include_family_metric=False),
        "method_summary": {key: _group_summary(group, selected_metrics, include_family_metric=False) for key, group in sorted(by_method.items())},
        "family_summary": {key: _group_summary(group, selected_metrics, include_family_metric=True) for key, group in sorted(by_family.items())},
        "domain_summary": {key: _group_summary(group, selected_metrics, include_family_metric=False) for key, group in sorted(by_domain.items())},
    }


def _write_summary_files(output_dir: Path, rows: List[Dict[str, Any]], selected_metrics: Sequence[str], args: argparse.Namespace, run_id: str, requested: Sequence[str]) -> None:
    summary = _summarize(rows, selected_metrics)
    summary.update(
        {
            "run_id": run_id,
            "release_dir": str(args.release_dir),
            "results_jsonl": str(args.results_jsonl),
            "requested_metrics": list(requested),
            "output_results_jsonl": str(output_dir / "results_final_metrics.jsonl"),
            "judge_llm_config": str(args.judge_llm_config),
            "embedding_config": str(args.embedding_config or ""),
            "eval_cache_dir": str(args.eval_cache_dir or ""),
        }
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / "main_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["method"] + [f"{metric}_score" for metric in selected_metrics]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, group in sorted(summary["method_summary"].items()):
            row = {"method": method}
            row.update(group.get("mean_scores") or {})
            writer.writerow(row)


def _split_rows(rows: List[Dict[str, Any]], workers: int) -> List[List[Dict[str, Any]]]:
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, workers))]
    for idx, row in enumerate(rows):
        buckets[idx % len(buckets)].append(row)
    return buckets


def _spawn_workers(args: argparse.Namespace, selected_metrics: Sequence[str]) -> None:
    output_dir = Path(args.output_dir)
    chunk_root = output_dir / "chunks"
    log_root = output_dir / "logs"
    rows = list(iter_jsonl(Path(args.results_jsonl)))
    if args.task_limit is not None:
        rows = rows[: args.task_limit]
    chunks = _split_rows(rows, args.workers)
    chunk_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    processes: List[subprocess.Popen[str]] = []
    for idx, chunk in enumerate(chunks):
        chunk_file = chunk_root / f"chunk_{idx:02d}.jsonl"
        _write_jsonl(chunk_file, chunk)
        worker_out = output_dir / f"worker_{idx:02d}"
        if worker_out.exists() and not args.resume:
            shutil.rmtree(worker_out)
        worker_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--results-jsonl",
            str(chunk_file),
            "--release-dir",
            args.release_dir,
            "--output-dir",
            str(worker_out),
            "--metrics",
            ",".join(selected_metrics),
            "--workers",
            "1",
            "--judge-llm-config",
            args.judge_llm_config,
            "--judge-fallback-llm-config",
            args.judge_fallback_llm_config,
            "--run-id",
            f"{args.run_id or output_dir.name}_w{idx:02d}",
            "--_worker-mode",
        ]
        if str(args.embedding_config or "").strip():
            cmd.extend(["--embedding-config", args.embedding_config])
        if str(args.eval_cache_dir or "").strip():
            cmd.extend(["--eval-cache-dir", args.eval_cache_dir])
        if args.resume:
            cmd.append("--resume")
        log_f = (log_root / f"worker_{idx:02d}.log").open("a" if args.resume else "w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, text=True)
        proc._log_f = log_f  # type: ignore[attr-defined]
        processes.append(proc)
        print(f"[eval_final_clean_parallel] launched worker={idx} tasks={len(chunk)} pid={proc.pid}", flush=True)
    failures = []
    for idx, proc in enumerate(processes):
        code = proc.wait()
        proc._log_f.close()  # type: ignore[attr-defined]
        print(f"[eval_final_clean_parallel] worker={idx} exit={code}", flush=True)
        if code != 0:
            failures.append(idx)
    if failures:
        raise SystemExit(f"workers failed: {failures}")


def _merge_workers(output_dir: Path, source_rows: List[Dict[str, Any]], selected_metrics: Sequence[str], args: argparse.Namespace, run_id: str, requested: Sequence[str]) -> None:
    by_task: Dict[str, Dict[str, Any]] = {}
    for idx in range(1000):
        worker_out = output_dir / f"worker_{idx:02d}"
        if not worker_out.exists():
            break
        for row in _load_jsonl(worker_out / "results_final_metrics.jsonl"):
            task_id = _task_id(row)
            if task_id:
                by_task[task_id] = row
    ordered: List[Dict[str, Any]] = []
    missing: List[str] = []
    for source in source_rows:
        task_id = _task_id(source)
        if task_id in by_task:
            ordered.append(by_task[task_id])
        else:
            missing.append(task_id)
    if missing:
        raise SystemExit(f"missing merged final metric rows: count={len(missing)} first={missing[:5]}")
    _write_jsonl(output_dir / "results_final_metrics.jsonl", ordered)
    _write_summary_files(output_dir, ordered, selected_metrics, args, run_id, requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final clean evaluator for task_refined.jsonl releases.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--judge-llm-config", default="configs/llm/qwen3_235b_8002.local.yaml")
    parser.add_argument("--judge-fallback-llm-config", default="")
    parser.add_argument("--embedding-config", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval-cache-dir", default="results/eval_cache/refined422_final_metrics")
    parser.add_argument("--_worker-mode", action="store_true")
    args = parser.parse_args()

    selected_metrics, requested = _parse_metrics(args.metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id.strip() or output_dir.name or Path(args.results_jsonl).stem
    source_rows = list(iter_jsonl(Path(args.results_jsonl)))
    if args.task_limit is not None:
        source_rows = source_rows[: args.task_limit]

    if args.workers > 1 and not args._worker_mode:
        _spawn_workers(args, selected_metrics)
        _merge_workers(output_dir, source_rows, selected_metrics, args, run_id, requested)
        print(json.dumps({"run_id": run_id, "metrics": selected_metrics, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))
        return

    judge_client = _build_judge_client(Path(args.judge_llm_config), args.judge_fallback_llm_config)
    embedding_path = Path(args.embedding_config) if str(args.embedding_config or "").strip() else None
    embedding_client = (
        OpenAICompatEmbeddingClient(load_openai_compat_embedding_config(embedding_path))
        if embedding_path and embedding_path.exists()
        else None
    )
    rows = _evaluate_rows(
        rows=source_rows,
        release_dir=Path(args.release_dir),
        output_dir=output_dir,
        run_id=run_id,
        selected_metrics=selected_metrics,
        judge_client=judge_client,
        embedding_client=embedding_client,
        resume=args.resume,
        cache_dir=Path(args.eval_cache_dir) if str(args.eval_cache_dir or "").strip() else None,
    )
    _write_summary_files(output_dir, rows, selected_metrics, args, run_id, requested)
    print(json.dumps({"run_id": run_id, "metrics": selected_metrics, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
