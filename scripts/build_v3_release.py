from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl


DOMAIN_PUBLIC = {
    "llm_agent": "LLM agents",
    "llm_finetuning_post_training": "LLM fine-tuning and post-training",
    "rag_and_retrieval_structuring": "RAG and retrieval structuring",
    "visual_generative_modeling_and_diffusion": "Visual generative modeling and diffusion",
}


def public_domain_label(domain: str) -> str:
    return DOMAIN_PUBLIC.get(str(domain or ""), str(domain or ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble a benchmark v3 release bundle.")
    parser.add_argument("--input", default=str(ROOT / "data" / "task_candidates_v3" / "all_candidates.judged.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "releases" / "benchmark_v3_20260401"))
    parser.add_argument("--accept-threshold", type=float, default=0.78)
    parser.add_argument("--min-heuristic", type=float, default=0.60)
    parser.add_argument("--max-per-family-domain", type=int, default=18)
    return parser.parse_args()


def family_requirements_met(row: Dict[str, Any]) -> bool:
    family = str(row.get("family") or "")
    gt = row.get("ground_truth") or {}
    public = row.get("public_metadata") or {}
    support = row.get("support_context") or {}
    if family == "direction_forecasting":
        return bool(public.get("future_themes")) and bool((gt.get("future_half_stats") or {}).get("paper_count"))
    if family == "bottleneck_opportunity_discovery":
        return bool(gt.get("historical_limitation_signals")) and bool(public.get("future_themes"))
    if family == "strategic_research_planning":
        return bool(support.get("candidate_directions")) and bool((gt.get("target_window_stats") or {}).get("paper_count"))
    return False


def keep_row(row: Dict[str, Any], args: argparse.Namespace) -> bool:
    judge = row.get("judge") or {}
    decision = str(judge.get("decision") or "").lower()
    overall = float(judge.get("overall_score") or 0.0)
    heuristic = float((row.get("quality_signals") or {}).get("heuristic_score") or 0.0)
    if decision == "reject":
        return False
    if overall < args.accept_threshold:
        return False
    if heuristic < args.min_heuristic:
        return False
    return family_requirements_met(row)


def deliverable_spec(family: str) -> Dict[str, Any]:
    base = {
        "format": "free_form_research_analysis",
        "requirements": [
            "Use only evidence available up to the cutoff unless the benchmark setting explicitly provides an offline knowledge base.",
            "State a concrete conclusion rather than vague trend language.",
            "Support the conclusion with literature-based reasoning.",
        ],
    }
    if family == "direction_forecasting":
        base["requirements"].append("Name one specific next-step direction and characterize the trajectory.")
    elif family == "bottleneck_opportunity_discovery":
        base["requirements"].append("Identify one concrete bottleneck and connect it to one concrete downstream opportunity.")
    elif family == "strategic_research_planning":
        base["requirements"].append("Select and justify a small ranked set of priority directions.")
    return base


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = defaultdict(list)
    for row in iter_jsonl(args.input):
        if not keep_row(row, args):
            continue
        buckets[(row.get("family"), row.get("domain"))].append(row)

    selected = []
    per_family = Counter()
    per_domain = Counter()
    for key, rows in sorted(buckets.items()):
        rows.sort(
            key=lambda row: (
                -float((row.get("judge") or {}).get("overall_score") or 0.0),
                -float((row.get("quality_signals") or {}).get("heuristic_score") or 0.0),
                row.get("task_id"),
            )
        )
        for row in rows[: args.max_per_family_domain]:
            selected.append(row)
            per_family[str(row.get("family") or "")] += 1
            per_domain[str(row.get("domain") or "")] += 1

    selected.sort(key=lambda row: (row.get("family"), row.get("domain"), row.get("task_id")))
    tasks_path = out_dir / "tasks.jsonl"
    internal_full_path = out_dir / "tasks_internal_full.jsonl"
    hidden_eval_path = out_dir / "tasks_hidden_eval.jsonl"
    build_trace_path = out_dir / "tasks_build_trace.jsonl"

    with open(tasks_path, "w", encoding="utf-8") as public_handle, open(
        internal_full_path, "w", encoding="utf-8"
    ) as internal_handle, open(hidden_eval_path, "w", encoding="utf-8") as hidden_handle, open(
        build_trace_path, "w", encoding="utf-8"
    ) as trace_handle:
        for idx, row in enumerate(selected, start=1):
            public_task_id = f"RTLv3-{idx:04d}"
            internal_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            public_row = {
                "task_id": public_task_id,
                "family": row.get("family"),
                "subtype": row.get("subtype"),
                "domain": public_domain_label(row.get("domain")),
                "horizon": row.get("horizon"),
                "title": row.get("title") or row.get("task_id"),
                "question": row.get("question") or row.get("draft_question"),
                "time_cutoff": (row.get("time_context") or {}).get("history_end"),
                "deliverable_spec": deliverable_spec(str(row.get("family") or "")),
            }
            public_handle.write(json.dumps(public_row, ensure_ascii=False) + "\n")

            hidden_row = {
                "task_id": public_task_id,
                "internal_task_id": row.get("task_id"),
                "family": row.get("family"),
                "domain": row.get("domain"),
                "title": row.get("title") or row.get("task_id"),
                "gold_answer": row.get("gold_answer") or row.get("draft_reference_answer"),
                "expected_answer_points": row.get("expected_answer_points") or [],
                "evaluation_rubric": row.get("evaluation_rubric"),
                "judge": row.get("judge"),
                "ground_truth": row.get("ground_truth"),
                "public_metadata": row.get("public_metadata"),
            }
            hidden_handle.write(json.dumps(hidden_row, ensure_ascii=False) + "\n")

            trace_row = {
                "task_id": public_task_id,
                "internal_task_id": row.get("task_id"),
                "family": row.get("family"),
                "domain": row.get("domain"),
                "seed": row.get("seed"),
                "time_context": row.get("time_context"),
                "support_context": row.get("support_context"),
                "ground_truth": row.get("ground_truth"),
                "quality_signals": row.get("quality_signals"),
                "rewrite": row.get("rewrite"),
                "rewrite_leakage_check": row.get("rewrite_leakage_check"),
                "rewrite_surface_check": row.get("rewrite_surface_check"),
                "judge": row.get("judge"),
                "public_metadata": row.get("public_metadata"),
            }
            trace_handle.write(json.dumps(trace_row, ensure_ascii=False) + "\n")

    manifest = {
        "release_name": out_dir.name,
        "task_count": len(selected),
        "family_counts": dict(per_family),
        "domain_counts": dict(per_domain),
        "files": {
            "tasks_public": str(tasks_path.relative_to(out_dir)),
            "tasks_internal_full": str(internal_full_path.relative_to(out_dir)),
            "tasks_hidden_eval": str(hidden_eval_path.relative_to(out_dir)),
            "tasks_build_trace": str(build_trace_path.relative_to(out_dir)),
        },
        "selection_policy": {
            "accept_threshold": args.accept_threshold,
            "min_heuristic": args.min_heuristic,
            "max_per_family_domain": args.max_per_family_domain,
        },
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
