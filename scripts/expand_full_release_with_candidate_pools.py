from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data" / "releases" / "benchmark_full_curated_recovered21_bottleneck18"
DEFAULT_Q1 = ROOT / "tmp" / "q1_short_candidates" / "all_candidates.judged.jsonl"
DEFAULT_CLUSTER = ROOT / "tmp" / "cluster_expansion_v1" / "all_candidates.judged.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "releases" / "benchmark_full_curated_recovered21_bottleneck18_expanded75"

DOMAIN_PUBLIC = {
    "llm_agent": "LLM agents",
    "llm_finetuning_post_training": "LLM fine-tuning and post-training",
    "rag_and_retrieval_structuring": "RAG and retrieval structuring",
    "visual_generative_modeling_and_diffusion": "Visual generative modeling and diffusion",
}

STRICT_RULES = {
    "bottleneck_opportunity_discovery": "future_descendants",
    "direction_forecasting": "emergent_descendants",
    "strategic_research_planning": "direction_records",
    "venue_aware_research_positioning": "direction_records",
}

TASK_ID_RE = re.compile(r"^RTLv3-(\d+)$")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_title(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def strict_keep(row: Dict[str, Any]) -> bool:
    family = str(row.get("family") or "")
    gt_key = STRICT_RULES.get(family)
    if not gt_key:
        return False
    gt = row.get("ground_truth") or {}
    return bool(gt.get(gt_key))


def judge_obj(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("candidate_quality_judge") or row.get("judge") or {}


def judge_score(row: Dict[str, Any]) -> Optional[float]:
    score = judge_obj(row).get("overall_score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def suspicious_zero_judge(row: Dict[str, Any], min_mean: float) -> Tuple[bool, float, int]:
    judge = judge_obj(row)
    if judge.get("overall_score") not in (0, 0.0):
        return False, 0.0, 0

    scores = judge.get("scores")
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
        return False, 0.0, len(values)
    mean_value = sum(values) / len(values)
    return mean_value >= min_mean, mean_value, len(values)


def topic_title(row: Dict[str, Any]) -> str:
    meta = row.get("public_metadata") or {}
    for key in ("topic_title", "topic"):
        value = " ".join(str(meta.get(key) or "").split()).strip()
        if value:
            return value
    seed = row.get("seed") or {}
    node_id = str(seed.get("node_id") or "").strip()
    if node_id:
        leaf = node_id.split("/")[-1].replace("_", " ").strip()
        if leaf:
            return leaf[:1].upper() + leaf[1:]
    return ""


def repaired_title(row: Dict[str, Any]) -> str:
    raw = " ".join(str(row.get("title") or "").split()).strip()
    if raw and len(raw) >= 20 and raw.lower() != "forecasting next":
        return raw

    topic = topic_title(row)
    family = str(row.get("family") or "")
    if family == "direction_forecasting":
        return f"Forecasting Next-Step Direction in {topic or 'Emerging Research Directions'}"
    if family == "bottleneck_opportunity_discovery":
        return f"Bottleneck and Opportunity Discovery in {topic or 'Emerging Research Topics'}"
    if family == "strategic_research_planning":
        return f"Research Direction Prioritization for {topic or 'Emerging Research Topics'}"
    if family == "venue_aware_research_positioning":
        return f"Venue-Aware Positioning for {topic or 'Emerging Research Topics'}"
    return raw or str(row.get("task_id") or "")


def public_domain_label(domain: str) -> str:
    return DOMAIN_PUBLIC.get(str(domain or ""), str(domain or ""))


def deliverable_spec(family: str) -> Dict[str, Any]:
    requirements = [
        "Use only evidence available up to the stated cutoff.",
        "State a concrete conclusion instead of a vague trend summary.",
        "Ground the conclusion in literature-based reasoning.",
    ]
    if family == "direction_forecasting":
        requirements.append("Name one specific next-step direction and characterize the trajectory.")
    elif family == "bottleneck_opportunity_discovery":
        requirements.append("Identify one concrete bottleneck and connect it to one concrete downstream opportunity.")
    elif family == "strategic_research_planning":
        requirements.append("Select and justify a small ranked set of priority directions.")
    elif family == "venue_aware_research_positioning":
        requirements.append("Provide venue-aware positioning and justify the fit to venue dynamics.")
    return {
        "format": "free_form_research_analysis",
        "requirements": requirements,
    }


def public_from_candidate(row: Dict[str, Any], public_task_id: str, title: str) -> Dict[str, Any]:
    return {
        "task_id": public_task_id,
        "family": row.get("family"),
        "subtype": row.get("subtype"),
        "domain": public_domain_label(str(row.get("domain") or "")),
        "horizon": row.get("horizon"),
        "title": title,
        "question": row.get("question") or row.get("draft_question"),
        "time_cutoff": (row.get("time_context") or {}).get("history_end"),
        "deliverable_spec": deliverable_spec(str(row.get("family") or "")),
    }


def hidden_from_candidate(row: Dict[str, Any], public_task_id: str, title: str) -> Dict[str, Any]:
    return {
        "task_id": public_task_id,
        "internal_task_id": row.get("task_id"),
        "family": row.get("family"),
        "domain": row.get("domain"),
        "title": title,
        "gold_answer": row.get("gold_answer") or row.get("draft_reference_answer"),
        "expected_answer_points": row.get("expected_answer_points") or [],
        "evaluation_rubric": row.get("evaluation_rubric"),
        "judge": row.get("judge"),
        "candidate_quality_judge": row.get("candidate_quality_judge") or row.get("judge") or {},
        "ground_truth": row.get("ground_truth") or {},
        "public_metadata": row.get("public_metadata") or {},
    }


def trace_from_candidate(row: Dict[str, Any], public_task_id: str, title: str, selection: Dict[str, Any]) -> Dict[str, Any]:
    quality_signals = dict(row.get("quality_signals") or {})
    quality_signals["expansion_selection"] = selection
    return {
        "task_id": public_task_id,
        "internal_task_id": row.get("task_id"),
        "family": row.get("family"),
        "domain": row.get("domain"),
        "title": title,
        "seed": row.get("seed"),
        "time_context": row.get("time_context"),
        "support_context": row.get("support_context"),
        "ground_truth": row.get("ground_truth") or {},
        "quality_signals": quality_signals,
        "rewrite": row.get("rewrite"),
        "rewrite_leakage_check": row.get("rewrite_leakage_check"),
        "rewrite_surface_check": row.get("rewrite_surface_check"),
        "judge": row.get("judge"),
        "candidate_quality_judge": row.get("candidate_quality_judge") or row.get("judge") or {},
        "public_metadata": row.get("public_metadata") or {},
    }


def internal_from_candidate(row: Dict[str, Any], public_task_id: str, title: str, selection: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out["task_id"] = public_task_id
    out["internal_task_id"] = row.get("task_id")
    out["title"] = title
    out["quality_signals"] = dict(row.get("quality_signals") or {})
    out["quality_signals"]["expansion_selection"] = selection
    return out


def parse_next_numeric_task_id(existing_task_ids: Iterable[str]) -> int:
    max_value = 0
    for task_id in existing_task_ids:
        match = TASK_ID_RE.match(str(task_id))
        if match:
            max_value = max(max_value, int(match.group(1)))
    return max_value + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand the curated full release with strict-ready judged candidate pools.")
    parser.add_argument("--base-release", default=str(DEFAULT_BASE))
    parser.add_argument("--q1-candidates", default=str(DEFAULT_Q1))
    parser.add_argument("--cluster-candidates", default=str(DEFAULT_CLUSTER))
    parser.add_argument("--output-release", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--suspicious-min-mean", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_release = Path(args.base_release)
    q1_path = Path(args.q1_candidates)
    cluster_path = Path(args.cluster_candidates)
    output_release = Path(args.output_release)

    if output_release.exists():
        shutil.rmtree(output_release)
    output_release.mkdir(parents=True, exist_ok=True)

    base_public = list(iter_jsonl(base_release / "tasks.jsonl"))
    base_hidden = list(iter_jsonl(base_release / "tasks_hidden_eval.jsonl"))
    base_trace = list(iter_jsonl(base_release / "tasks_build_trace.jsonl"))
    base_internal = list(iter_jsonl(base_release / "tasks_internal_full.jsonl"))

    selected_rows: List[Dict[str, Any]] = []
    selected_report: List[Dict[str, Any]] = []
    seen_titles = {normalize_title(row.get("title")) for row in base_public}
    next_numeric_id = parse_next_numeric_task_id(row["task_id"] for row in base_public)

    for source_name, source_path in (("q1", q1_path), ("cluster", cluster_path)):
        for row in iter_jsonl(source_path):
            if not strict_keep(row):
                continue

            title = repaired_title(row)
            question = str(row.get("question") or row.get("draft_question") or "").strip()
            gold_answer = str(row.get("gold_answer") or row.get("draft_reference_answer") or "").strip()
            title_key = normalize_title(title)
            if not title_key or title_key in seen_titles or not question or not gold_answer:
                continue

            score = judge_score(row)
            suspicious, suspicious_mean, suspicious_score_count = suspicious_zero_judge(row, args.suspicious_min_mean)
            if not ((score is not None and score >= args.min_score) or suspicious):
                continue

            public_task_id = f"RTLv3-{next_numeric_id:04d}"
            next_numeric_id += 1
            seen_titles.add(title_key)

            selection = {
                "source_pool": source_name,
                "source_task_id": row.get("task_id"),
                "selection_mode": "score_threshold" if (score is not None and score >= args.min_score) else "suspicious_zero_judge_recovery",
                "judge_overall_score": score,
                "suspicious_zero_judge": suspicious,
                "suspicious_zero_judge_mean_subscore": round(suspicious_mean, 4) if suspicious else 0.0,
                "suspicious_zero_judge_subscore_count": suspicious_score_count if suspicious else 0,
                "judge_decision": judge_obj(row).get("decision"),
                "title_repaired": title != str(row.get("title") or ""),
                "strict_reason_key": STRICT_RULES.get(str(row.get("family") or "")),
            }

            selected_rows.append(
                {
                    "public": public_from_candidate(row, public_task_id, title),
                    "hidden": hidden_from_candidate(row, public_task_id, title),
                    "trace": trace_from_candidate(row, public_task_id, title, selection),
                    "internal": internal_from_candidate(row, public_task_id, title, selection),
                }
            )
            selected_report.append(
                {
                    "task_id": public_task_id,
                    "source_task_id": row.get("task_id"),
                    "source_pool": source_name,
                    "family": row.get("family"),
                    "domain": row.get("domain"),
                    "horizon": row.get("horizon"),
                    "title": title,
                    "selection_mode": selection["selection_mode"],
                    "judge_overall_score": score,
                    "judge_decision": selection["judge_decision"],
                    "suspicious_zero_judge_mean_subscore": selection["suspicious_zero_judge_mean_subscore"],
                    "strict_reason_key": selection["strict_reason_key"],
                }
            )

    merged_public = list(base_public) + [row["public"] for row in selected_rows]
    merged_hidden = list(base_hidden) + [row["hidden"] for row in selected_rows]
    merged_trace = list(base_trace) + [row["trace"] for row in selected_rows]
    merged_internal = list(base_internal) + [row["internal"] for row in selected_rows]

    dump_jsonl(output_release / "tasks.jsonl", merged_public)
    dump_jsonl(output_release / "tasks_hidden_eval.jsonl", merged_hidden)
    dump_jsonl(output_release / "tasks_build_trace.jsonl", merged_trace)
    dump_jsonl(output_release / "tasks_internal_full.jsonl", merged_internal)
    (output_release / "task_ids.txt").write_text(
        "\n".join(str(row["task_id"]) for row in merged_public) + "\n",
        encoding="utf-8",
    )

    strict_ids = [str(row["task_id"]) for row in merged_internal if strict_keep(row)]
    (output_release / "strict_task_ids.txt").write_text("\n".join(strict_ids) + "\n", encoding="utf-8")

    for name in [
        "dropped_tasks.json",
        "venue_repairs.json",
        "recovered_tasks.json",
        "language_polish_audit.json",
        "recovered_bottleneck_future_descendants.json",
    ]:
        src = base_release / name
        if src.exists():
            shutil.copy2(src, output_release / name)

    dump_json(output_release / "expanded_candidates_report.json", selected_report)

    public_by_id = {str(row["task_id"]): row for row in merged_public}
    family_counts = Counter(row["family"] for row in merged_public)
    horizon_counts = Counter(str(row.get("horizon") or "") for row in merged_public)
    strict_family_counts = Counter(row["family"] for row in merged_internal if strict_keep(row))
    strict_horizon_counts = Counter(
        str((public_by_id.get(str(row["task_id"])) or {}).get("horizon") or "")
        for row in merged_internal
        if strict_keep(row)
    )
    added_family_counts = Counter(row["family"] for row in selected_report)
    added_horizon_counts = Counter(str(row.get("horizon") or "") for row in selected_report)
    added_source_counts = Counter(str(row.get("source_pool") or "") for row in selected_report)

    manifest = {
        "release_name": output_release.name,
        "base_release": str(base_release),
        "task_count": len(merged_public),
        "strict_task_count": len(strict_ids),
        "strict_quarter_task_count": strict_horizon_counts.get("quarter", 0),
        "added_task_count": len(selected_rows),
        "family_counts": dict(family_counts),
        "horizon_counts": dict(horizon_counts),
        "strict_family_counts": dict(strict_family_counts),
        "strict_horizon_counts": dict(strict_horizon_counts),
        "added_family_counts": dict(added_family_counts),
        "added_horizon_counts": dict(added_horizon_counts),
        "added_source_counts": dict(added_source_counts),
        "selection_policy": {
            "dedupe_key": "normalized title against base release plus already selected additions",
            "strict_requirement": STRICT_RULES,
            "min_score": args.min_score,
            "suspicious_zero_judge_recovery": {
                "enabled": True,
                "mean_subscore_threshold": args.suspicious_min_mean,
                "minimum_numeric_subscores": 4,
                "purpose": "recover likely judge-parser failures where overall_score is zero but subscores remain high",
            },
            "field_requirements": [
                "non-empty repaired title",
                "non-empty question or draft_question",
                "non-empty gold_answer or draft_reference_answer",
            ],
        },
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "task_ids": "task_ids.txt",
            "strict_task_ids": "strict_task_ids.txt",
            "expanded_candidates_report": "expanded_candidates_report.json",
            "tasks_hidden_eval_v3": "tasks_hidden_eval_v3.jsonl",
            "tasks_hidden_eval_v3_manifest": "tasks_hidden_eval_v3_manifest.json",
            "tasks_hidden_eval_v3_1": "tasks_hidden_eval_v3_1.jsonl",
            "tasks_hidden_eval_v3_1_manifest": "tasks_hidden_eval_v3_1_manifest.json",
        },
    }
    dump_json(output_release / "manifest.json", manifest)

    readme = f"""# {output_release.name}

## Summary
- base release: {base_release.name}
- total tasks: {len(merged_public)}
- strict tasks: {len(strict_ids)}
- added tasks: {len(selected_rows)}

## Expansion policy
- start from the current curated full release with recovered q1 strategic tasks and bottleneck future-descendant repairs
- add only strict-ready candidates from judged q1/cluster pools
- dedupe by normalized title against the base release and previously selected additions
- accept candidates when either:
  - `judge.overall_score >= {args.min_score}`
  - or the judge output looks corrupted (`overall_score == 0`) but the mean numeric subscore remains >= {args.suspicious_min_mean}

## Notes
- this release prioritizes expansion over conservative legacy accept-only filtering
- obvious title corruption from candidate pools was repaired during import
- hidden eval v3 / v3.1 should be regenerated after this release is built
"""
    (output_release / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "release_name": output_release.name,
                "task_count": len(merged_public),
                "strict_task_count": len(strict_ids),
                "added_task_count": len(selected_rows),
                "added_source_counts": dict(added_source_counts),
                "added_family_counts": dict(added_family_counts),
                "added_horizon_counts": dict(added_horizon_counts),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
