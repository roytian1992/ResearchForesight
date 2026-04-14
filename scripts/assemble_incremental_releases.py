from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]

DOMAIN_PUBLIC = {
    "llm_agent": "LLM agents",
    "llm_finetuning_post_training": "LLM fine-tuning and post-training",
    "rag_and_retrieval_structuring": "RAG and retrieval structuring",
    "visual_generative_modeling_and_diffusion": "Visual generative modeling and diffusion",
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def public_domain_label(domain: str) -> str:
    return DOMAIN_PUBLIC.get(str(domain or ""), str(domain or ""))


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
    elif family == "venue_aware_research_positioning":
        base["requirements"].append("Provide venue-aware positioning and justify why the proposed direction fits the target venue dynamics.")
    return base


def hidden_from_candidate(row: Dict[str, Any], public_task_id: str) -> Dict[str, Any]:
    return {
        "task_id": public_task_id,
        "internal_task_id": row.get("task_id"),
        "family": row.get("family"),
        "domain": row.get("domain"),
        "title": row.get("title") or row.get("task_id"),
        "gold_answer": row.get("gold_answer") or row.get("draft_reference_answer"),
        "expected_answer_points": row.get("expected_answer_points") or [],
        "evaluation_rubric": row.get("evaluation_rubric"),
        "judge": row.get("judge"),
        "candidate_quality_judge": row.get("candidate_quality_judge") or row.get("judge") or {},
        "ground_truth": row.get("ground_truth"),
        "public_metadata": row.get("public_metadata"),
    }


def trace_from_candidate(row: Dict[str, Any], public_task_id: str) -> Dict[str, Any]:
    return {
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
        "candidate_quality_judge": row.get("candidate_quality_judge") or row.get("judge") or {},
        "public_metadata": row.get("public_metadata"),
    }


def public_from_hidden_and_trace(hidden: Dict[str, Any], trace: Dict[str, Any], *, public_task_id: str | None = None) -> Dict[str, Any]:
    task_id = public_task_id or hidden["task_id"]
    return {
        "task_id": task_id,
        "family": hidden.get("family"),
        "subtype": (hidden.get("public_metadata") or {}).get("subtype") or (trace.get("public_metadata") or {}).get("subtype") or (trace.get("ground_truth") or {}).get("subtype") or hidden.get("subtype") or trace.get("subtype"),
        "domain": public_domain_label(hidden.get("domain")),
        "horizon": (trace.get("time_context") or {}).get("setting_id") if False else (trace.get("public_metadata") or {}).get("horizon") or (trace.get("ground_truth") or {}).get("horizon") or None,
        "title": hidden.get("title") or hidden.get("internal_task_id") or task_id,
        "question": (hidden.get("public_metadata") or {}).get("question") or None,
        "time_cutoff": (trace.get("time_context") or {}).get("history_end"),
        "deliverable_spec": deliverable_spec(str(hidden.get("family") or "")),
    }


def public_from_candidate(row: Dict[str, Any], public_task_id: str) -> Dict[str, Any]:
    return {
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


def build_release(out_dir: Path, public_rows, internal_rows, hidden_rows, trace_rows, readme_text: str, sources: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(out_dir / "tasks.jsonl", public_rows)
    dump_jsonl(out_dir / "tasks_internal_full.jsonl", internal_rows)
    dump_jsonl(out_dir / "tasks_hidden_eval.jsonl", hidden_rows)
    dump_jsonl(out_dir / "tasks_build_trace.jsonl", trace_rows)
    (out_dir / "README.md").write_text(readme_text, encoding="utf-8")

    family_counts = Counter(row.get("family") for row in public_rows)
    domain_counts = Counter(row.get("domain") for row in public_rows)
    subtype_counts = Counter(row.get("subtype") for row in public_rows)
    manifest = {
        "release_name": out_dir.name,
        "task_count": len(public_rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "subtype_counts": dict(subtype_counts),
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
        },
        "sources": sources,
    }
    dump_json(out_dir / "manifest.json", manifest)


def main() -> None:
    releases = ROOT / "data" / "releases"
    base_release = releases / "benchmark_v3_20260408_expanded"

    base_public = list(iter_jsonl(base_release / "tasks.jsonl"))
    base_hidden = list(iter_jsonl(base_release / "tasks_hidden_eval.jsonl"))
    base_trace = list(iter_jsonl(base_release / "tasks_build_trace.jsonl"))
    base_internal = list(iter_jsonl(base_release / "tasks_internal_full.jsonl"))

    cluster_path = ROOT / "tmp" / "cluster_expansion_v1" / "all_candidates.judged.jsonl"
    cluster_rows = [
        row for row in iter_jsonl(cluster_path)
        if ((row.get("candidate_quality_judge") or row.get("judge") or {}).get("decision") == "accept")
    ]
    cluster_rows.sort(key=lambda r: (r.get("domain", ""), r.get("family", ""), r.get("task_id", "")))

    q1_path = ROOT / "tmp" / "q1_short_candidates" / "all_candidates.judged.jsonl"
    q1_rows = [
        row for row in iter_jsonl(q1_path)
        if ((row.get("candidate_quality_judge") or row.get("judge") or {}).get("decision") == "accept")
    ]
    q1_rows.sort(key=lambda r: (r.get("domain", ""), r.get("family", ""), r.get("task_id", "")))

    # 440 half-year release
    hy_public = list(base_public)
    hy_hidden = list(base_hidden)
    hy_trace = list(base_trace)
    hy_internal = list(base_internal)
    next_id = len(hy_public) + 1
    for row in cluster_rows:
        pid = f"RTLv3-{next_id:04d}"
        next_id += 1
        hy_public.append(public_from_candidate(row, pid))
        hy_hidden.append(hidden_from_candidate(row, pid))
        hy_trace.append(trace_from_candidate(row, pid))
        hy_internal.append(row)

    hy_readme = """# benchmark_halfyear

## Summary
- tasks: 440
- families: 4
- domains: 4
- setting: half-year only
- history cutoff: 2025-08-31
- future window: 2025-09-01 ~ 2026-02-28

## Notes
This release preserves the stable 2025-08-31 cutoff benchmark and appends 84 accept-only high-quality additions from the incremental cluster expansion pass.
The release is intended as the finalized half-year slice before introducing the separate quarterly setting.
"""
    build_release(
        releases / "benchmark_halfyear",
        hy_public,
        hy_internal,
        hy_hidden,
        hy_trace,
        hy_readme,
        {
            "base_release": "data/releases/benchmark_v3_20260408_expanded",
            "cluster_expansion_accepts": "tmp/cluster_expansion_v1/all_candidates.judged.jsonl",
            "policy": "accept_only",
        },
    )

    # 131 q1 release
    q1_public, q1_hidden, q1_trace, q1_internal = [], [], [], []
    for i, row in enumerate(q1_rows, start=1):
        pid = f"RTLv3Q1-{i:04d}"
        q1_public.append(public_from_candidate(row, pid))
        q1_hidden.append(hidden_from_candidate(row, pid))
        q1_trace.append(trace_from_candidate(row, pid))
        q1_internal.append(row)

    q1_readme = """# benchmark_quarter

## Summary
- tasks: 131
- families: 3
- domains: 4
- setting: quarterly only
- history cutoff: 2025-11-30
- future window: 2025-12-01 ~ 2026-02-28

## Notes
This release is the new short-horizon benchmark slice. It was built from 2025Q4 taxonomy/support packets and filtered with the same rewrite-and-judge pipeline, keeping accept-only tasks.
"""
    build_release(
        releases / "benchmark_quarter",
        q1_public,
        q1_internal,
        q1_hidden,
        q1_trace,
        q1_readme,
        {
            "support_packets": "tmp/q1_support_packets_20251130",
            "q1_candidates": "tmp/q1_short_candidates/all_candidates.judged.jsonl",
            "policy": "accept_only",
        },
    )

    # merged release 571
    merged_public = list(hy_public)
    merged_hidden = list(hy_hidden)
    merged_trace = list(hy_trace)
    merged_internal = list(hy_internal)
    next_id = len(merged_public) + 1
    for row in q1_rows:
        pid = f"RTLv3-{next_id:04d}"
        next_id += 1
        merged_public.append(public_from_candidate(row, pid))
        merged_hidden.append(hidden_from_candidate(row, pid))
        merged_trace.append(trace_from_candidate(row, pid))
        merged_internal.append(row)

    merged_readme = """# benchmark_full

## Summary
- tasks: 571
- families: 4 in the half-year slice, 3 in the quarterly slice
- domains: 4
- settings:
  - hy_6m: history cutoff 2025-08-31, future window 2025-09-01 ~ 2026-02-28
  - q1_3m: history cutoff 2025-11-30, future window 2025-12-01 ~ 2026-02-28

## Notes
This merged release combines the finalized 440-task half-year benchmark with the 131-task quarterly benchmark.
Users can filter by `time_cutoff`, `horizon`, `subtype`, or hidden `time_context.setting_id` for setting-specific evaluation.
"""
    build_release(
        releases / "benchmark_full",
        merged_public,
        merged_internal,
        merged_hidden,
        merged_trace,
        merged_readme,
        {
            "hy_6m_release": "data/releases/benchmark_halfyear",
            "q1_3m_release": "data/releases/benchmark_quarter",
        },
    )

    print(json.dumps({
        "hy_6m_440": len(hy_public),
        "q1_3m_131": len(q1_public),
        "full": len(merged_public),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
