from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "data" / "releases" / "benchmark_full_curated_recovered21_bottleneck18_expanded75"
DEFAULT_Q1 = ROOT / "tmp" / "q1_short_candidates" / "all_candidates.judged.jsonl"
DEFAULT_CLUSTER = ROOT / "tmp" / "cluster_expansion_v1" / "all_candidates.judged.jsonl"
DEFAULT_OUT = ROOT / "tmp" / "venue_coverage_audit"

DOMAIN_LABELS = {
    "llm_agent": "LLM agents",
    "llm_finetuning_post_training": "LLM fine-tuning and post-training",
    "rag_and_retrieval_structuring": "RAG and retrieval structuring",
    "visual_generative_modeling_and_diffusion": "Visual generative modeling and diffusion",
}
DOMAIN_ORDER = list(DOMAIN_LABELS)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit full-release venue coverage and augmentation opportunities.")
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE))
    parser.add_argument("--q1-candidates", default=str(DEFAULT_Q1))
    parser.add_argument("--cluster-candidates", default=str(DEFAULT_CLUSTER))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pool-min-score", type=float, default=0.55)
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


def normalize_title(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def judge_score(row: Dict[str, Any]) -> float:
    judge = row.get("candidate_quality_judge") or row.get("judge") or {}
    score = judge.get("overall_score")
    return float(score) if isinstance(score, (int, float)) else 0.0


def likely_bucket(stats: Dict[str, Any]) -> str:
    buckets = {
        str(k): int(v)
        for k, v in (stats.get("top_venue_buckets") or {}).items()
        if str(k) not in {"other", "unknown"} and int(v or 0) > 0
    }
    if not buckets:
        return ""
    return max(buckets.items(), key=lambda kv: (kv[1], kv[0]))[0]


def strict_keep(row: Dict[str, Any]) -> bool:
    gt = row.get("ground_truth") or {}
    key = STRICT_RULES.get(str(row.get("family") or ""))
    return bool(key and gt.get(key))


def topic_title(row: Dict[str, Any]) -> str:
    meta = row.get("public_metadata") or {}
    for key in ("topic_title", "topic"):
        value = normalize_title(meta.get(key))
        if value:
            return value
    title = normalize_title(row.get("title"))
    return title


def source_bucket(row: Dict[str, Any]) -> str:
    gt = row.get("ground_truth") or {}
    family = str(row.get("family") or "")
    if family == "direction_forecasting":
        return likely_bucket(gt.get("future_half_stats") or {})
    if family == "strategic_research_planning":
        return likely_bucket(gt.get("target_window_stats") or {})
    if family == "venue_aware_research_positioning":
        return str(gt.get("target_venue_bucket") or (gt.get("venue_forecast") or {}).get("likely_bucket") or "")
    return ""


def all_supported_buckets(row: Dict[str, Any]) -> Dict[str, int]:
    gt = row.get("ground_truth") or {}
    family = str(row.get("family") or "")
    if family == "strategic_research_planning":
        stats = gt.get("target_window_stats") or {}
        return {
            str(k): int(v)
            for k, v in (stats.get("top_venue_buckets") or {}).items()
            if str(k) not in {"other", "unknown"} and int(v or 0) > 0
        }
    if family == "venue_aware_research_positioning":
        bucket = str(gt.get("target_venue_bucket") or (gt.get("venue_forecast") or {}).get("likely_bucket") or "")
        return {bucket: 1} if bucket else {}
    return {}


def venue_subtype(public_row: Dict[str, Any], hidden_row: Dict[str, Any]) -> str:
    subtype = str(public_row.get("subtype") or "").strip()
    if subtype:
        return subtype
    variant = str((hidden_row.get("public_metadata") or {}).get("task_variant") or "").strip()
    if variant == "venue_targeted":
        return "venue_targeted_planning"
    if variant == "venue_aware":
        return "venue_aware_direction_forecast"
    return "unknown"


def collect_pool_candidates(path: Path, pool_name: str, min_score: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in iter_jsonl(path):
        if str(row.get("family") or "") != "strategic_research_planning":
            continue
        if not strict_keep(row):
            continue
        bucket_map = all_supported_buckets(row)
        if not bucket_map:
            continue
        score = judge_score(row)
        if score < min_score:
            continue
        for bucket, support_count in bucket_map.items():
            rows.append(
                {
                    "source_type": "pool_candidate",
                    "source_name": pool_name,
                    "task_id": str(row.get("task_id") or ""),
                    "domain": str(row.get("domain") or ""),
                    "horizon": str(row.get("horizon") or (row.get("time_context") or {}).get("horizon") or ""),
                    "bucket": bucket,
                    "bucket_support_count": int(support_count),
                    "score": round(score, 4),
                    "title": normalize_title(row.get("title") or ""),
                    "topic_title": topic_title(row),
                }
            )
    return rows


def sort_candidate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("score") or 0.0),
            row.get("horizon") != "quarter",
            -len(str(row.get("topic_title") or "")),
            str(row.get("title") or ""),
        ),
    )


def build_markdown_report(
    *,
    release_name: str,
    release_summary: Dict[str, Any],
    venue_summary: Dict[str, Any],
    matrix_rows: List[Dict[str, Any]],
    missing_rows: List[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    lines.append(f"# Venue Coverage Audit: {release_name}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- total tasks: `{release_summary['task_count']}`")
    lines.append(f"- strict tasks: `{release_summary['strict_task_count']}`")
    lines.append(f"- venue public tasks: `{venue_summary['public_count']}`")
    lines.append(f"- venue strict tasks: `{venue_summary['strict_count']}`")
    lines.append(f"- venue public but not strict: `{venue_summary['public_not_strict_count']}`")
    lines.append(f"- venue strict quarterly tasks: `{venue_summary['strict_quarter_count']}`")
    lines.append("")
    lines.append("## Strict Venue By Domain")
    for domain_key in DOMAIN_ORDER:
        domain_label = DOMAIN_LABELS[domain_key]
        count = venue_summary["strict_by_domain"].get(domain_key, 0)
        lines.append(f"- {domain_label}: `{count}`")
    lines.append("")
    lines.append("## Coverage Matrix")
    lines.append("| Domain | Bucket | Venue Public | Venue Strict | Release Supported Sources | Release Quarter Sources | Pool Supported Sources | Pool Quarter Sources |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in matrix_rows:
        lines.append(
            f"| {DOMAIN_LABELS.get(row['domain'], row['domain'])} | {row['bucket']} | {row['venue_public']} | "
            f"{row['venue_strict']} | {row['release_planning_sources']} | {row['release_planning_quarter_sources']} | "
            f"{row['pool_planning_sources']} | {row['pool_planning_quarter_sources']} |"
        )
    lines.append("")
    lines.append("## Missing Or Thin Buckets")
    for row in missing_rows:
        lines.append(
            f"### {DOMAIN_LABELS.get(row['domain'], row['domain'])} / {row['bucket']}"
        )
        lines.append(
            f"- current strict venue count: `{row['venue_strict']}`; current public venue count: `{row['venue_public']}`"
        )
        lines.append(
            f"- recoverable from current release planning support: `{row['release_planning_sources']}` (quarter: `{row['release_planning_quarter_sources']}`)"
        )
        lines.append(
            f"- recoverable from candidate-pool support: `{row['pool_planning_sources']}` (quarter: `{row['pool_planning_quarter_sources']}`)"
        )
        top_candidates = row.get("top_pool_candidates") or []
        if top_candidates:
            lines.append("- top pool candidates:")
            for cand in top_candidates:
                lines.append(
                    f"  - `{cand['source_name']}` / `{cand['task_id']}` / score `{cand['score']}` / `{cand['horizon']}` / {cand['title']}"
                )
        else:
            lines.append("- top pool candidates: none above threshold")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (DEFAULT_OUT / release_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)

    public_rows = list(iter_jsonl(release_dir / "tasks.jsonl"))
    hidden_rows = list(iter_jsonl(release_dir / "tasks_hidden_eval.jsonl"))
    public_by_id = {str(row["task_id"]): row for row in public_rows}

    strict_rows = [row for row in hidden_rows if strict_keep(row)]
    release_summary = {
        "release_name": release_dir.name,
        "task_count": len(public_rows),
        "strict_task_count": len(strict_rows),
    }

    venue_public_counts: Counter[Tuple[str, str]] = Counter()
    venue_strict_counts: Counter[Tuple[str, str]] = Counter()
    venue_public_by_domain: Counter[str] = Counter()
    venue_strict_by_domain: Counter[str] = Counter()
    venue_public_by_subtype: Counter[str] = Counter()
    venue_strict_by_subtype: Counter[str] = Counter()
    venue_strict_quarter_count = 0

    for hidden_row in hidden_rows:
        if str(hidden_row.get("family") or "") != "venue_aware_research_positioning":
            continue
        task_id = str(hidden_row.get("task_id") or "")
        public_row = public_by_id[task_id]
        domain = str(hidden_row.get("domain") or "")
        bucket = source_bucket(hidden_row)
        subtype = venue_subtype(public_row, hidden_row)
        horizon = str(public_row.get("horizon") or "")
        venue_public_counts[(domain, bucket)] += 1
        venue_public_by_domain[domain] += 1
        venue_public_by_subtype[subtype] += 1
        if strict_keep(hidden_row):
            venue_strict_counts[(domain, bucket)] += 1
            venue_strict_by_domain[domain] += 1
            venue_strict_by_subtype[subtype] += 1
            if horizon == "quarter":
                venue_strict_quarter_count += 1

    release_planning_candidates: List[Dict[str, Any]] = []
    for hidden_row in hidden_rows:
        if str(hidden_row.get("family") or "") != "strategic_research_planning":
            continue
        if not strict_keep(hidden_row):
            continue
        task_id = str(hidden_row.get("task_id") or "")
        public_row = public_by_id[task_id]
        bucket_map = all_supported_buckets(hidden_row)
        if not bucket_map:
            continue
        for bucket, support_count in bucket_map.items():
            release_planning_candidates.append(
                {
                    "source_type": "release_planning",
                    "source_name": release_dir.name,
                    "task_id": task_id,
                    "domain": str(hidden_row.get("domain") or ""),
                    "horizon": str(public_row.get("horizon") or ""),
                    "bucket": bucket,
                    "bucket_support_count": int(support_count),
                    "score": 1.0,
                    "title": normalize_title(hidden_row.get("title") or ""),
                    "topic_title": topic_title(hidden_row),
                }
            )

    pool_candidates = collect_pool_candidates(Path(args.q1_candidates), "q1", args.pool_min_score)
    pool_candidates += collect_pool_candidates(Path(args.cluster_candidates), "cluster", args.pool_min_score)

    release_planning_counts: Counter[Tuple[str, str]] = Counter()
    release_planning_quarter_counts: Counter[Tuple[str, str]] = Counter()
    pool_planning_counts: Counter[Tuple[str, str]] = Counter()
    pool_planning_quarter_counts: Counter[Tuple[str, str]] = Counter()
    pool_candidates_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in release_planning_candidates:
        key = (row["domain"], row["bucket"])
        release_planning_counts[key] += 1
        if row["horizon"] == "quarter":
            release_planning_quarter_counts[key] += 1

    for row in pool_candidates:
        key = (row["domain"], row["bucket"])
        pool_planning_counts[key] += 1
        if row["horizon"] == "quarter":
            pool_planning_quarter_counts[key] += 1
        pool_candidates_by_key[key].append(row)

    venue_summary = {
        "public_count": sum(venue_public_by_domain.values()),
        "strict_count": sum(venue_strict_by_domain.values()),
        "public_not_strict_count": sum(venue_public_by_domain.values()) - sum(venue_strict_by_domain.values()),
        "strict_quarter_count": venue_strict_quarter_count,
        "public_by_domain": dict(venue_public_by_domain),
        "strict_by_domain": dict(venue_strict_by_domain),
        "public_by_subtype": dict(venue_public_by_subtype),
        "strict_by_subtype": dict(venue_strict_by_subtype),
    }

    matrix_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    for domain in DOMAIN_ORDER:
        target_buckets = TARGET_BUCKETS.get(domain, [])
        for bucket in target_buckets:
            key = (domain, bucket)
            row = {
                "domain": domain,
                "bucket": bucket,
                "venue_public": venue_public_counts.get(key, 0),
                "venue_strict": venue_strict_counts.get(key, 0),
                "release_planning_sources": release_planning_counts.get(key, 0),
                "release_planning_quarter_sources": release_planning_quarter_counts.get(key, 0),
                "pool_planning_sources": pool_planning_counts.get(key, 0),
                "pool_planning_quarter_sources": pool_planning_quarter_counts.get(key, 0),
            }
            matrix_rows.append(row)
            if row["venue_strict"] > 0 and row["venue_public"] > 0:
                continue
            top_pool = sort_candidate_rows(pool_candidates_by_key.get(key, []))[:5]
            missing_rows.append({**row, "top_pool_candidates": top_pool})

    missing_rows.sort(
        key=lambda row: (
            -row["pool_planning_sources"],
            -row["release_planning_sources"],
            row["venue_strict"],
            row["domain"],
            row["bucket"],
        )
    )

    report = {
        "release_summary": release_summary,
        "venue_summary": venue_summary,
        "matrix_rows": matrix_rows,
        "missing_rows": missing_rows,
        "target_buckets": TARGET_BUCKETS,
    }
    dump_json(output_dir / "venue_coverage_audit.json", report)
    markdown = build_markdown_report(
        release_name=release_dir.name,
        release_summary=release_summary,
        venue_summary=venue_summary,
        matrix_rows=matrix_rows,
        missing_rows=missing_rows,
    )
    (output_dir / "venue_coverage_audit.md").write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "release_name": release_dir.name,
                "venue_public_count": venue_summary["public_count"],
                "venue_strict_count": venue_summary["strict_count"],
                "missing_bucket_rows": len(missing_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
