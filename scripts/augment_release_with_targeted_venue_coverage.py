from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.eval_v3 import build_hidden_eval_v3_row
from researchworld.eval_v3_1 import build_hidden_eval_v3_1_row


DEFAULT_SRC_RELEASE = ROOT / "data" / "releases" / "benchmark_full_curated_recovered21_bottleneck18_expanded75"
DEFAULT_OUT_RELEASE = ROOT / "data" / "releases" / "benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover"
DEFAULT_Q1 = ROOT / "tmp" / "q1_short_candidates" / "all_candidates.judged.jsonl"
DEFAULT_CLUSTER = ROOT / "tmp" / "cluster_expansion_v1" / "all_candidates.judged.jsonl"

DOMAIN_PUBLIC = {
    "llm_agent": "LLM agents",
    "llm_finetuning_post_training": "LLM fine-tuning and post-training",
    "rag_and_retrieval_structuring": "RAG and retrieval structuring",
    "visual_generative_modeling_and_diffusion": "Visual generative modeling and diffusion",
}
DOMAIN_ORDER = list(DOMAIN_PUBLIC)
TARGET_BUCKETS = {
    "llm_agent": ["AAAI", "ACL", "EMNLP", "ICML", "ICLR", "NeurIPS", "IJCAI", "WSDM"],
    "llm_finetuning_post_training": ["AAAI", "ACL", "EMNLP", "ICML", "ICLR", "NeurIPS"],
    "rag_and_retrieval_structuring": ["SIGIR", "WWW", "KDD", "WSDM", "EMNLP", "ACL", "ICML", "AAAI", "IJCAI"],
    "visual_generative_modeling_and_diffusion": ["CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "AAAI", "ICML"],
}
STRICT_RULES = {
    "bottleneck_opportunity_discovery": "future_descendants",
    "direction_forecasting": "emergent_descendants",
    "strategic_research_planning": "direction_records",
    "venue_aware_research_positioning": "direction_records",
}
CANONICAL_VENUE_BY_BUCKET = {
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "ACL": "Annual Meeting of the Association for Computational Linguistics",
    "CVPR": "Computer Vision and Pattern Recognition",
    "ECCV": "European Conference on Computer Vision",
    "EMNLP": "Conference on Empirical Methods in Natural Language Processing",
    "ICCV": "International Conference on Computer Vision",
    "ICLR": "International Conference on Learning Representations",
    "ICML": "International Conference on Machine Learning",
    "IJCAI": "International Joint Conference on Artificial Intelligence",
    "KDD": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
    "NAACL": "North American Chapter of the Association for Computational Linguistics",
    "NeurIPS": "Neural Information Processing Systems",
    "SIGIR": "International ACM SIGIR Conference on Research and Development in Information Retrieval",
    "WSDM": "ACM International Conference on Web Search and Data Mining",
    "WWW": "The Web Conference",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment a release with bucket-targeted venue planning tasks.")
    parser.add_argument("--src-release", default=str(DEFAULT_SRC_RELEASE))
    parser.add_argument("--out-release", default=str(DEFAULT_OUT_RELEASE))
    parser.add_argument("--q1-candidates", default=str(DEFAULT_Q1))
    parser.add_argument("--cluster-candidates", default=str(DEFAULT_CLUSTER))
    parser.add_argument("--pool-min-score", type=float, default=0.55)
    parser.add_argument("--pool-suspicious-min-mean", type=float, default=0.85)
    parser.add_argument("--max-missing-per-bucket", type=int, default=1)
    parser.add_argument("--quarter-target-per-domain", type=int, default=1)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def title_case(text: Any) -> str:
    raw = normalize_text(text).replace("_", " ")
    return raw[:1].upper() + raw[1:] if raw else ""


def strict_keep(row: Dict[str, Any]) -> bool:
    gt = row.get("ground_truth") or {}
    key = STRICT_RULES.get(str(row.get("family") or ""))
    return bool(key and gt.get(key))


def judge_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("candidate_quality_judge") or row.get("judge") or {}


def judge_score(row: Dict[str, Any]) -> float:
    score = judge_obj(row).get("overall_score")
    return float(score) if isinstance(score, (int, float)) else 0.0


def suspicious_zero_judge(row: Dict[str, Any], min_mean: float) -> Tuple[bool, float]:
    judge = judge_obj(row)
    if judge.get("overall_score") not in (0, 0.0):
        return False, 0.0
    scores = judge.get("scores")
    values: List[float]
    if isinstance(scores, dict):
        values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    else:
        values = [
            float(v)
            for k, v in judge.items()
            if k not in {"decision", "overall_score", "strengths", "weaknesses", "suggested_fix"}
            and isinstance(v, (int, float))
        ]
    if len(values) < 4:
        return False, 0.0
    mean_value = sum(values) / len(values)
    return mean_value >= min_mean, mean_value


def supported_bucket_map(row: Dict[str, Any]) -> Dict[str, int]:
    stats = ((row.get("ground_truth") or {}).get("target_window_stats") or {})
    return {
        str(k): int(v)
        for k, v in (stats.get("top_venue_buckets") or {}).items()
        if str(k) not in {"other", "unknown"} and int(v or 0) > 0
    }


def topic_title(row: Dict[str, Any]) -> str:
    meta = row.get("public_metadata") or {}
    for key in ("topic_title", "topic"):
        value = normalize_text(meta.get(key))
        if value:
            return value
    seed = row.get("seed") or {}
    node_id = normalize_text(seed.get("node_id"))
    if node_id:
        leaf = node_id.split("/")[-1].replace("_", " ").strip()
        if leaf:
            return title_case(leaf)
    return normalize_text(row.get("title"))


def venue_title(topic: str, bucket: str) -> str:
    return f"Venue-Targeted Prioritization of Research Directions in {topic} for {bucket}-Like Venues"


def venue_question(topic: str, bucket: str) -> str:
    return (
        f"Suppose a research team wants to maximize its relevance for {bucket}-like venues in the next submission cycle. "
        f"Based only on literature available before September 1, 2025, which one or two next-step directions in {topic} should be prioritized, "
        f"and what evidence-based rationale supports that ranking for {bucket}-style publication dynamics?"
    )


def public_deliverable_spec() -> Dict[str, Any]:
    return {
        "format": "free_form_research_analysis",
        "requirements": [
            "Use only evidence available up to the stated cutoff.",
            "Provide a small ranked set of concrete directions rather than an unstructured list.",
            "Explicitly justify why the ranking fits the named venue bucket.",
            "Ground the ranking in literature-based reasoning rather than unsupported conjecture.",
        ],
    }


def next_public_task_id(existing_task_ids: Iterable[str]) -> int:
    max_value = 0
    for task_id in existing_task_ids:
        text = str(task_id)
        if text.startswith("RTLv3-"):
            try:
                max_value = max(max_value, int(text.split("-")[-1]))
            except Exception:
                pass
    return max_value + 1


def candidate_signature(topic: str, bucket: str) -> Tuple[str, str]:
    return (normalize_text(topic).lower(), normalize_text(bucket).upper())


def rank_block(direction_records: List[Dict[str, Any]], fallback_themes: List[str]) -> Tuple[List[str], str]:
    ordered = sorted(direction_records, key=lambda row: int((row or {}).get("future_paper_count") or 0), reverse=True)
    top = [title_case((row or {}).get("display_name") or "") for row in ordered if title_case((row or {}).get("display_name") or "")]
    if not top:
        top = [title_case(x) for x in fallback_themes if title_case(x)]
    top = top[:2]
    block = "; then ".join(top) if len(top) > 1 else (top[0] if top else "the strongest emerging directions")
    return top, block


def canonical_venue_name(bucket: str) -> str:
    return CANONICAL_VENUE_BY_BUCKET.get(bucket, "")


def build_variant(
    *,
    source_public: Dict[str, Any],
    source_hidden: Dict[str, Any],
    source_trace: Dict[str, Any],
    source_internal: Dict[str, Any],
    public_task_id: str,
    internal_task_id: str,
    target_bucket: str,
    bucket_support_count: int,
    bucket_rank: int,
    source_kind: str,
    source_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    gt = json.loads(json.dumps(source_hidden.get("ground_truth") or {}, ensure_ascii=False))
    trace_gt = json.loads(json.dumps(source_trace.get("ground_truth") or {}, ensure_ascii=False))
    topic = topic_title(source_hidden)
    top_buckets = supported_bucket_map(source_hidden)
    top, block = rank_block(list(gt.get("direction_records") or []), list((source_hidden.get("public_metadata") or {}).get("future_themes") or []))
    venue_name = canonical_venue_name(target_bucket)

    gt["target_venue_bucket"] = target_bucket
    gt["target_venue_name"] = venue_name
    gt["venue_forecast"] = {
        "likely_bucket": target_bucket,
        "likely_venue": venue_name,
        "target_bucket_support_count": int(bucket_support_count),
        "target_bucket_rank": int(bucket_rank),
        "top_venue_buckets": top_buckets,
    }
    gt["bucket_targeting_metadata"] = {
        "target_bucket": target_bucket,
        "target_bucket_support_count": int(bucket_support_count),
        "target_bucket_rank": int(bucket_rank),
        "source_kind": source_kind,
        "source_name": source_name,
    }
    trace_gt.update(
        {
            "target_venue_bucket": gt["target_venue_bucket"],
            "target_venue_name": gt["target_venue_name"],
            "venue_forecast": gt["venue_forecast"],
            "bucket_targeting_metadata": gt["bucket_targeting_metadata"],
        }
    )

    question = venue_question(topic, target_bucket)
    top_conf_count = int(((gt.get("target_window_stats") or {}).get("top_conf_count") or 0))
    gold_answer = (
        f"For a team targeting {target_bucket}-like venues, the highest-priority directions should be {block}. "
        f"This ranking is supported by the pre-cutoff record: these directions sit closest to the strongest direction records, "
        f"they align with the topic's historical future-work and evaluation signals, and the realized target window later showed "
        f"{int(bucket_support_count)} top-tier papers or bucket hits consistent with {target_bucket}-style publication dynamics within a total top-tier window of {top_conf_count} papers."
    )
    expected_points = [
        "Produces a ranked research plan rather than an unstructured list.",
        "Targets the named venue bucket explicitly and links the ranking to that venue style.",
        "Uses pre-cutoff evidence such as direction records, historical future-work signals, or evaluation profile to justify the ranking.",
    ]

    hidden = {
        "task_id": public_task_id,
        "internal_task_id": internal_task_id,
        "family": "venue_aware_research_positioning",
        "domain": source_hidden.get("domain"),
        "title": venue_title(topic, target_bucket),
        "gold_answer": gold_answer,
        "expected_answer_points": expected_points,
        "evaluation_rubric": source_hidden.get("evaluation_rubric"),
        "judge": source_hidden.get("judge"),
        "ground_truth": gt,
        "public_metadata": {
            **(source_hidden.get("public_metadata") or {}),
            "task_variant": "venue_targeted",
            "target_bucket_override": target_bucket,
            "source_planning_task_id": source_hidden.get("task_id"),
            "source_planning_family": source_hidden.get("family"),
        },
    }
    trace = {
        "task_id": public_task_id,
        "internal_task_id": internal_task_id,
        "family": "venue_aware_research_positioning",
        "domain": source_trace.get("domain"),
        "seed": source_trace.get("seed"),
        "time_context": source_trace.get("time_context"),
        "support_context": {
            **(source_trace.get("support_context") or {}),
            "target_venue_bucket": target_bucket,
            "target_venue_name": venue_name,
            "bucket_targeting_metadata": gt["bucket_targeting_metadata"],
        },
        "ground_truth": trace_gt,
        "quality_signals": {
            **(source_trace.get("quality_signals") or {}),
            "derived_variant": "venue_targeted_bucket_specific",
            "target_bucket_override": target_bucket,
            "target_bucket_support_count": int(bucket_support_count),
            "target_bucket_rank": int(bucket_rank),
            "source_kind": source_kind,
            "source_name": source_name,
        },
        "rewrite": source_trace.get("rewrite"),
        "rewrite_leakage_check": source_trace.get("rewrite_leakage_check"),
        "rewrite_surface_check": source_trace.get("rewrite_surface_check"),
        "judge": source_trace.get("judge"),
        "public_metadata": hidden["public_metadata"],
    }
    public = {
        "task_id": public_task_id,
        "family": "venue_aware_research_positioning",
        "subtype": "venue_targeted_planning",
        "domain": source_public.get("domain"),
        "horizon": source_public.get("horizon"),
        "title": hidden["title"],
        "question": question,
        "time_cutoff": source_public.get("time_cutoff"),
        "deliverable_spec": public_deliverable_spec(),
    }
    internal = {
        **source_internal,
        "task_id": public_task_id,
        "internal_task_id": internal_task_id,
        "family": "venue_aware_research_positioning",
        "subtype": "venue_targeted_planning",
        "title": hidden["title"],
        "question": question,
        "draft_question": question,
        "gold_answer": gold_answer,
        "draft_reference_answer": gold_answer,
        "expected_answer_points": expected_points,
        "ground_truth": gt,
        "public_metadata": hidden["public_metadata"],
        "support_context": trace["support_context"],
        "quality_signals": trace["quality_signals"],
        "derived_from": source_hidden.get("task_id"),
        "public_task_id": public_task_id,
    }
    return public, hidden, trace, internal


def pseudo_public_from_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": str(row.get("task_id") or ""),
        "family": row.get("family"),
        "subtype": row.get("subtype"),
        "domain": DOMAIN_PUBLIC.get(str(row.get("domain") or ""), str(row.get("domain") or "")),
        "horizon": row.get("horizon") or (row.get("time_context") or {}).get("horizon") or "half_year",
        "title": row.get("title"),
        "question": row.get("question") or row.get("draft_question"),
        "time_cutoff": (row.get("time_context") or {}).get("history_end") or "2025-08-31",
    }


def pseudo_hidden_from_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": str(row.get("task_id") or ""),
        "internal_task_id": str(row.get("task_id") or ""),
        "family": row.get("family"),
        "domain": row.get("domain"),
        "title": row.get("title"),
        "gold_answer": row.get("gold_answer") or row.get("draft_reference_answer"),
        "expected_answer_points": row.get("expected_answer_points") or [],
        "evaluation_rubric": row.get("evaluation_rubric"),
        "judge": row.get("judge"),
        "ground_truth": row.get("ground_truth") or {},
        "public_metadata": row.get("public_metadata") or {},
    }


def pseudo_trace_from_candidate(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": str(row.get("task_id") or ""),
        "internal_task_id": str(row.get("task_id") or ""),
        "family": row.get("family"),
        "domain": row.get("domain"),
        "seed": row.get("seed"),
        "time_context": row.get("time_context"),
        "support_context": row.get("support_context"),
        "ground_truth": row.get("ground_truth") or {},
        "quality_signals": row.get("quality_signals") or {},
        "rewrite": row.get("rewrite"),
        "rewrite_leakage_check": row.get("rewrite_leakage_check"),
        "rewrite_surface_check": row.get("rewrite_surface_check"),
        "judge": row.get("judge"),
        "public_metadata": row.get("public_metadata") or {},
    }


def collect_existing_venue_state(
    public_rows: List[Dict[str, Any]],
    hidden_rows: List[Dict[str, Any]],
) -> Tuple[Counter[Tuple[str, str]], Counter[str], set[Tuple[str, str]]]:
    public_by_id = {str(row["task_id"]): row for row in public_rows}
    bucket_counts: Counter[Tuple[str, str]] = Counter()
    quarter_counts: Counter[str] = Counter()
    signatures: set[Tuple[str, str]] = set()
    for hidden_row in hidden_rows:
        if str(hidden_row.get("family") or "") != "venue_aware_research_positioning":
            continue
        gt = hidden_row.get("ground_truth") or {}
        bucket = normalize_text(gt.get("target_venue_bucket") or (gt.get("venue_forecast") or {}).get("likely_bucket") or "")
        if not bucket:
            continue
        domain = str(hidden_row.get("domain") or "")
        topic = topic_title(hidden_row)
        signatures.add(candidate_signature(topic, bucket))
        if strict_keep(hidden_row):
            bucket_counts[(domain, bucket)] += 1
            task_id = str(hidden_row.get("task_id") or "")
            if (public_by_id.get(task_id) or {}).get("horizon") == "quarter":
                quarter_counts[domain] += 1
    return bucket_counts, quarter_counts, signatures


def candidate_sort_key(row: Dict[str, Any], *, need_quarter: bool) -> Tuple[int, int, int, float, int, str]:
    return (
        0 if (need_quarter and row.get("horizon") == "quarter") else 1,
        0 if row.get("source_type") == "release" else 1,
        -int(row.get("bucket_support_count") or 0),
        -float(row.get("score") or 0.0),
        0 if row.get("horizon") == "quarter" else 1,
        str(row.get("source_task_id") or ""),
    )


def main() -> None:
    args = parse_args()
    src_release = Path(args.src_release)
    out_release = Path(args.out_release)

    if out_release.exists():
        shutil.rmtree(out_release)
    shutil.copytree(src_release, out_release)

    public_rows = list(iter_jsonl(src_release / "tasks.jsonl"))
    hidden_rows = list(iter_jsonl(src_release / "tasks_hidden_eval.jsonl"))
    trace_rows = list(iter_jsonl(src_release / "tasks_build_trace.jsonl"))
    internal_rows = list(iter_jsonl(src_release / "tasks_internal_full.jsonl"))

    public_by_id = {str(row["task_id"]): row for row in public_rows}
    hidden_by_id = {str(row["task_id"]): row for row in hidden_rows}
    trace_by_id = {str(row["task_id"]): row for row in trace_rows}
    internal_by_id = {str(row["task_id"]): row for row in internal_rows}

    venue_bucket_counts, venue_quarter_counts, existing_signatures = collect_existing_venue_state(public_rows, hidden_rows)

    candidates_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    quarter_candidates_by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for hidden_row in hidden_rows:
        if str(hidden_row.get("family") or "") != "strategic_research_planning" or not strict_keep(hidden_row):
            continue
        task_id = str(hidden_row.get("task_id") or "")
        source_public = public_by_id[task_id]
        source_trace = trace_by_id[task_id]
        source_internal = internal_by_id[task_id]
        topic = topic_title(hidden_row)
        bucket_map = supported_bucket_map(hidden_row)
        ranked_buckets = sorted(bucket_map.items(), key=lambda kv: (-kv[1], kv[0]))
        for bucket_rank, (bucket, support_count) in enumerate(ranked_buckets, start=1):
            if bucket not in TARGET_BUCKETS.get(str(hidden_row.get("domain") or ""), []):
                continue
            sig = candidate_signature(topic, bucket)
            if sig in existing_signatures:
                continue
            cand = {
                "source_type": "release",
                "source_name": src_release.name,
                "source_task_id": task_id,
                "domain": str(hidden_row.get("domain") or ""),
                "bucket": bucket,
                "horizon": str(source_public.get("horizon") or ""),
                "score": 1.0,
                "bucket_support_count": int(support_count),
                "bucket_rank": int(bucket_rank),
                "topic_title": topic,
                "selection_reason": "missing_bucket_fill",
                "source_public": source_public,
                "source_hidden": hidden_row,
                "source_trace": source_trace,
                "source_internal": source_internal,
            }
            candidates_by_key[(cand["domain"], cand["bucket"])].append(cand)
            if cand["horizon"] == "quarter":
                quarter_candidates_by_domain[cand["domain"]].append(cand)

    for pool_name, pool_path in (("q1", Path(args.q1_candidates)), ("cluster", Path(args.cluster_candidates))):
        for row in iter_jsonl(pool_path):
            if str(row.get("family") or "") != "strategic_research_planning" or not strict_keep(row):
                continue
            score = judge_score(row)
            suspicious, suspicious_mean = suspicious_zero_judge(row, args.pool_suspicious_min_mean)
            if not (score >= args.pool_min_score or suspicious):
                continue
            source_public = pseudo_public_from_candidate(row)
            source_hidden = pseudo_hidden_from_candidate(row)
            source_trace = pseudo_trace_from_candidate(row)
            source_internal = json.loads(json.dumps(row, ensure_ascii=False))
            topic = topic_title(source_hidden)
            bucket_map = supported_bucket_map(source_hidden)
            ranked_buckets = sorted(bucket_map.items(), key=lambda kv: (-kv[1], kv[0]))
            for bucket_rank, (bucket, support_count) in enumerate(ranked_buckets, start=1):
                if bucket not in TARGET_BUCKETS.get(str(row.get("domain") or ""), []):
                    continue
                sig = candidate_signature(topic, bucket)
                if sig in existing_signatures:
                    continue
                cand = {
                    "source_type": "pool",
                    "source_name": pool_name,
                    "source_task_id": str(row.get("task_id") or ""),
                    "domain": str(row.get("domain") or ""),
                    "bucket": bucket,
                    "horizon": str(source_public.get("horizon") or ""),
                    "score": round(max(score, suspicious_mean), 4),
                    "bucket_support_count": int(support_count),
                    "bucket_rank": int(bucket_rank),
                    "topic_title": topic,
                    "selection_reason": "missing_bucket_fill",
                    "source_public": source_public,
                    "source_hidden": source_hidden,
                    "source_trace": source_trace,
                    "source_internal": source_internal,
                }
                candidates_by_key[(cand["domain"], cand["bucket"])].append(cand)
                if cand["horizon"] == "quarter":
                    quarter_candidates_by_domain[cand["domain"]].append(cand)

    selected_candidates: List[Dict[str, Any]] = []
    used_source_bucket: set[Tuple[str, str, str]] = set()

    for domain in DOMAIN_ORDER:
        for bucket in TARGET_BUCKETS.get(domain, []):
            key = (domain, bucket)
            if venue_bucket_counts.get(key, 0) >= args.max_missing_per_bucket:
                continue
            options = sorted(
                candidates_by_key.get(key, []),
                key=lambda row: candidate_sort_key(row, need_quarter=(venue_quarter_counts.get(domain, 0) < args.quarter_target_per_domain)),
            )
            for cand in options:
                source_key = (cand["source_type"], str(cand["source_task_id"]), bucket)
                if source_key in used_source_bucket:
                    continue
                selected_candidates.append(cand)
                used_source_bucket.add(source_key)
                existing_signatures.add(candidate_signature(cand["topic_title"], bucket))
                if cand["horizon"] == "quarter":
                    venue_quarter_counts[domain] += 1
                venue_bucket_counts[key] += 1
                break

    for domain in DOMAIN_ORDER:
        while venue_quarter_counts.get(domain, 0) < args.quarter_target_per_domain:
            options = sorted(
                quarter_candidates_by_domain.get(domain, []),
                key=lambda row: candidate_sort_key(row, need_quarter=True),
            )
            picked = None
            for cand in options:
                source_key = (cand["source_type"], str(cand["source_task_id"]), cand["bucket"])
                sig = candidate_signature(cand["topic_title"], cand["bucket"])
                if source_key in used_source_bucket or sig in existing_signatures:
                    continue
                picked = cand
                break
            if picked is None:
                break
            picked = dict(picked)
            picked["selection_reason"] = "quarter_fill"
            selected_candidates.append(picked)
            used_source_bucket.add((picked["source_type"], str(picked["source_task_id"]), picked["bucket"]))
            existing_signatures.add(candidate_signature(picked["topic_title"], picked["bucket"]))
            venue_quarter_counts[domain] += 1
            venue_bucket_counts[(domain, picked["bucket"])] += 1

    next_id = next_public_task_id(row["task_id"] for row in public_rows)
    added_public: List[Dict[str, Any]] = []
    added_hidden: List[Dict[str, Any]] = []
    added_trace: List[Dict[str, Any]] = []
    added_internal: List[Dict[str, Any]] = []
    selection_report: List[Dict[str, Any]] = []

    for cand in selected_candidates:
        public_task_id = f"RTLv3-{next_id:04d}"
        next_id += 1
        internal_task_id = f"venue_bucket::{cand['source_type']}::{cand['source_name']}::{cand['source_task_id']}::{cand['bucket']}"
        public, hidden, trace, internal = build_variant(
            source_public=cand["source_public"],
            source_hidden=cand["source_hidden"],
            source_trace=cand["source_trace"],
            source_internal=cand["source_internal"],
            public_task_id=public_task_id,
            internal_task_id=internal_task_id,
            target_bucket=cand["bucket"],
            bucket_support_count=int(cand["bucket_support_count"]),
            bucket_rank=int(cand["bucket_rank"]),
            source_kind=str(cand["source_type"]),
            source_name=str(cand["source_name"]),
        )
        added_public.append(public)
        added_hidden.append(hidden)
        added_trace.append(trace)
        added_internal.append(internal)
        selection_report.append(
            {
                "task_id": public_task_id,
                "source_type": cand["source_type"],
                "source_name": cand["source_name"],
                "source_task_id": cand["source_task_id"],
                "domain": cand["domain"],
                "bucket": cand["bucket"],
                "horizon": cand["horizon"],
                "bucket_support_count": cand["bucket_support_count"],
                "bucket_rank": cand["bucket_rank"],
                "topic_title": cand["topic_title"],
                "selection_reason": cand.get("selection_reason") or "missing_bucket_fill",
            }
        )

    merged_public = public_rows + added_public
    merged_hidden = hidden_rows + added_hidden
    merged_trace = trace_rows + added_trace
    merged_internal = internal_rows + added_internal

    dump_jsonl(out_release / "tasks.jsonl", merged_public)
    dump_jsonl(out_release / "tasks_hidden_eval.jsonl", merged_hidden)
    dump_jsonl(out_release / "tasks_build_trace.jsonl", merged_trace)
    dump_jsonl(out_release / "tasks_internal_full.jsonl", merged_internal)
    (out_release / "task_ids.txt").write_text("\n".join(str(row["task_id"]) for row in merged_public) + "\n", encoding="utf-8")

    strict_ids = [str(row["task_id"]) for row in merged_internal if strict_keep(row)]
    (out_release / "strict_task_ids.txt").write_text("\n".join(strict_ids) + "\n", encoding="utf-8")

    trace_by_id_all = {str(row["task_id"]): row for row in merged_trace}
    hidden_v3_rows = [build_hidden_eval_v3_row(hidden, trace_by_id_all[str(hidden["task_id"])]) for hidden in merged_hidden]
    dump_jsonl(out_release / "tasks_hidden_eval_v3.jsonl", hidden_v3_rows)
    hidden_v31_rows = [build_hidden_eval_v3_1_row(row, trace_by_id_all[str(row["task_id"])]) for row in hidden_v3_rows]
    dump_jsonl(out_release / "tasks_hidden_eval_v3_1.jsonl", hidden_v31_rows)
    dump_json(
        out_release / "tasks_hidden_eval_v3_manifest.json",
        {
            "release_dir": str(out_release),
            "output": str(out_release / "tasks_hidden_eval_v3.jsonl"),
            "task_count": len(hidden_v3_rows),
            "notes": [
                "Rebuilt after targeted venue bucket augmentation.",
            ],
        },
    )
    dump_json(
        out_release / "tasks_hidden_eval_v3_1_manifest.json",
        {
            "release_dir": str(out_release),
            "input_hidden_v3": str(out_release / "tasks_hidden_eval_v3.jsonl"),
            "input_trace": str(out_release / "tasks_build_trace.jsonl"),
            "output": str(out_release / "tasks_hidden_eval_v3_1.jsonl"),
            "task_count": len(hidden_v31_rows),
            "notes": [
                "Rebuilt after targeted venue bucket augmentation.",
            ],
        },
    )

    family_counts = Counter(row["family"] for row in merged_public)
    added_bucket_counts = Counter((row["domain"], row["bucket"]) for row in selection_report)
    added_horizon_counts = Counter(str(row["horizon"] or "") for row in selection_report)
    added_source_counts = Counter(str(row["source_name"] or "") for row in selection_report)
    strict_family_counts = Counter(row["family"] for row in merged_internal if strict_keep(row))

    manifest = {
        "release_name": out_release.name,
        "base_release": str(src_release),
        "task_count": len(merged_public),
        "strict_task_count": len(strict_ids),
        "added_task_count": len(selection_report),
        "family_counts": dict(family_counts),
        "strict_family_counts": dict(strict_family_counts),
        "added_horizon_counts": dict(added_horizon_counts),
        "added_source_counts": dict(added_source_counts),
        "selection_policy": {
            "goal": "fill missing venue strict buckets first, then add quarter venue coverage when possible",
            "target_buckets": TARGET_BUCKETS,
            "pool_min_score": args.pool_min_score,
            "pool_suspicious_zero_judge_mean_threshold": args.pool_suspicious_min_mean,
            "max_missing_per_bucket": args.max_missing_per_bucket,
            "quarter_target_per_domain": args.quarter_target_per_domain,
            "bucket_support_definition": "any positive entry in ground_truth.target_window_stats.top_venue_buckets",
        },
        "added_bucket_counts": {f"{domain}::{bucket}": count for (domain, bucket), count in sorted(added_bucket_counts.items())},
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "task_ids": "task_ids.txt",
            "strict_task_ids": "strict_task_ids.txt",
            "tasks_hidden_eval_v3": "tasks_hidden_eval_v3.jsonl",
            "tasks_hidden_eval_v3_1": "tasks_hidden_eval_v3_1.jsonl",
            "venue_coverage_selection_report": "venue_coverage_selection_report.json",
        },
    }
    dump_json(out_release / "manifest.json", manifest)
    dump_json(out_release / "venue_coverage_selection_report.json", selection_report)

    readme = f"""# {out_release.name}

## Summary
- base release: {src_release.name}
- total tasks: {len(merged_public)}
- strict tasks: {len(strict_ids)}
- added venue tasks: {len(selection_report)}

## Policy
- fill missing strict venue buckets first using strategic-planning tasks with non-empty `direction_records`
- allow bucket-targeted venue derivation from any supported bucket in `target_window_stats.top_venue_buckets`, not only the dominant bucket
- then add quarter venue coverage when a domain still has none
- candidate-pool sources are allowed only when they are strict-ready and pass the judged quality threshold or suspicious-zero recovery rule
"""
    (out_release / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_release": str(out_release),
                "task_count": len(merged_public),
                "strict_task_count": len(strict_ids),
                "added_task_count": len(selection_report),
                "added_bucket_counts": manifest["added_bucket_counts"],
                "added_horizon_counts": dict(added_horizon_counts),
                "added_source_counts": dict(added_source_counts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
