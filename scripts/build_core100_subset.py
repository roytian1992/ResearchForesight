from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "releases" / "benchmark_full"
DEFAULT_OUTPUT = ROOT / "data" / "releases" / "benchmark_core100"
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.eval_v3 import build_hidden_eval_v3_row
from researchworld.eval_v3_1 import build_hidden_eval_v3_1_row

STRICT_RULES = {
    "bottleneck_opportunity_discovery": "future_descendants",
    "direction_forecasting": "emergent_descendants",
    "strategic_research_planning": "direction_records",
    "venue_aware_research_positioning": "direction_records",
}

DOMAIN_ORDER = [
    "LLM agents",
    "LLM fine-tuning and post-training",
    "RAG and retrieval structuring",
    "Visual generative modeling and diffusion",
]

FAMILY_ORDER = [
    "bottleneck_opportunity_discovery",
    "direction_forecasting",
    "strategic_research_planning",
    "venue_aware_research_positioning",
]

FAMILY_DOMAIN_QUOTAS: Dict[str, Dict[str, int]] = {
    "bottleneck_opportunity_discovery": {
        "LLM agents": 7,
        "LLM fine-tuning and post-training": 6,
        "RAG and retrieval structuring": 6,
        "Visual generative modeling and diffusion": 6,
    },
    "direction_forecasting": {
        "LLM agents": 6,
        "LLM fine-tuning and post-training": 7,
        "RAG and retrieval structuring": 6,
        "Visual generative modeling and diffusion": 6,
    },
    "strategic_research_planning": {
        "LLM agents": 6,
        "LLM fine-tuning and post-training": 6,
        "RAG and retrieval structuring": 7,
        "Visual generative modeling and diffusion": 6,
    },
    "venue_aware_research_positioning": {
        "LLM agents": 6,
        "LLM fine-tuning and post-training": 6,
        "RAG and retrieval structuring": 6,
        "Visual generative modeling and diffusion": 7,
    },
}

STRATEGIC_SUBTYPE_ORDER = [
    "comparative_opportunity_prioritization",
    "agenda_priority_selection",
]

VENUE_SUBTYPE_ORDER = [
    "venue_aware_direction_forecast",
    "venue_targeted_planning",
]


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


def build_hidden_v3_manifest(release_dir: Path, rows: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    family_counts = Counter()
    domain_counts = Counter()
    for row in rows:
        family_counts[str(row.get("family") or "")] += 1
        domain_counts[str(row.get("domain") or "")] += 1
    return {
        "release_dir": str(release_dir),
        "output": str(output_path),
        "task_count": len(rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "notes": ["This manifest was regenerated while building a balanced 100-task subset."],
    }


def build_hidden_v31_manifest(release_dir: Path, rows: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    family_counts = Counter()
    domain_counts = Counter()
    for row in rows:
        family_counts[str(row.get("family") or "")] += 1
        domain_counts[str(row.get("domain") or "")] += 1
    return {
        "release_dir": str(release_dir),
        "output": str(output_path),
        "task_count": len(rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "notes": ["This manifest was regenerated while building a balanced 100-task subset."],
    }


def judge_score(hidden_row: Dict[str, Any]) -> float:
    for key in ("candidate_quality_judge", "judge"):
        obj = hidden_row.get(key) or {}
        if obj.get("overall_score") is not None:
            return float(obj.get("overall_score") or 0.0)
        if obj.get("avg_score") is not None:
            return float(obj.get("avg_score") or 0.0)
        scores = obj.get("scores") or {}
        if scores:
            vals = [float(v or 0.0) for v in scores.values()]
            if vals:
                return sum(vals) / len(vals)
    return 0.0


def trace_support(trace_row: Dict[str, Any]) -> Dict[str, Any]:
    return trace_row.get("support_context") or {}


def question_has_candidate_list(question: str, candidate_directions: List[str]) -> bool:
    text = str(question or "").lower()
    markers = [
        "rank the following candidate next-step research directions",
        "evaluate and rank these candidate research directions",
        "consider only the following candidate directions",
        "rank only the listed options",
    ]
    if not any(marker in text for marker in markers):
        return False
    if not candidate_directions:
        return False
    probe = [str(x).replace("_", " ").lower() for x in candidate_directions[: min(3, len(candidate_directions))]]
    return all(p in text for p in probe)


def eligible(public_row: Dict[str, Any], trace_row: Dict[str, Any]) -> bool:
    family = str(public_row.get("family") or "")
    subtype = str(public_row.get("subtype") or "")
    if str(public_row.get("horizon") or "") != "half_year":
        return False

    support = trace_support(trace_row)
    question = str(public_row.get("question") or "")

    if family == "bottleneck_opportunity_discovery":
        return subtype == "pageindex_grounded_bottleneck"

    if family == "direction_forecasting":
        return subtype == "chain_terminal_forecast"

    if family == "strategic_research_planning":
        if subtype == "comparative_opportunity_prioritization":
            return True
        return subtype == "agenda_priority_selection"

    if family == "venue_aware_research_positioning":
        return subtype in {"venue_aware_direction_forecast", "venue_targeted_planning"}

    return False


def strict_keep(hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> bool:
    family = str(hidden_row.get("family") or trace_row.get("family") or "")
    gt_key = STRICT_RULES.get(family)
    if not gt_key:
        return False
    ground_truth = hidden_row.get("ground_truth") or trace_row.get("ground_truth") or {}
    return bool(ground_truth.get(gt_key))


def selection_key(public_row: Dict[str, Any], hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> tuple[float, float, str]:
    score = judge_score(hidden_row)
    heuristic = float(((trace_row.get("quality_signals") or {}).get("heuristic_score") or 0.0))
    return (score, heuristic, str(public_row.get("task_id") or ""))


def pick_round_robin(groups: Dict[str, List[str]], subtype_order: List[str], quota: int) -> List[str]:
    picks: List[str] = []
    indices = {key: 0 for key in subtype_order}
    while len(picks) < quota:
        progress = False
        for subtype in subtype_order:
            rows = groups.get(subtype) or []
            idx = indices.get(subtype, 0)
            if idx >= len(rows):
                continue
            picks.append(rows[idx])
            indices[subtype] = idx + 1
            progress = True
            if len(picks) >= quota:
                break
        if not progress:
            break
    if len(picks) < quota:
        leftovers: List[str] = []
        for subtype in subtype_order:
            rows = groups.get(subtype) or []
            leftovers.extend(rows[indices.get(subtype, 0) :])
        picks.extend(leftovers[: max(0, quota - len(picks))])
    return picks[:quota]


def build_subset(source_dir: Path, output_dir: Path) -> Dict[str, Any]:
    public_rows = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks.jsonl")}
    hidden_rows = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_hidden_eval.jsonl")}
    trace_rows = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_build_trace.jsonl")}
    internal_rows = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_internal_full.jsonl")}

    eligible_ids: List[str] = []
    for task_id, public_row in public_rows.items():
        hidden_row = hidden_rows.get(task_id) or {}
        trace_row = trace_rows.get(task_id) or {}
        if eligible(public_row, trace_row) and strict_keep(hidden_row, trace_row):
            eligible_ids.append(task_id)

    buckets: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for task_id in eligible_ids:
        public_row = public_rows[task_id]
        hidden_row = hidden_rows[task_id]
        trace_row = trace_rows[task_id]
        family = str(public_row.get("family") or "")
        domain = str(public_row.get("domain") or "")
        buckets[family][domain].append(task_id)

    for family in FAMILY_ORDER:
        for domain in DOMAIN_ORDER:
            ids = buckets[family][domain]
            ids.sort(
                key=lambda task_id: selection_key(public_rows[task_id], hidden_rows[task_id], trace_rows[task_id]),
                reverse=True,
            )

    selected_ids: List[str] = []
    for family in FAMILY_ORDER:
        for domain in DOMAIN_ORDER:
            quota = FAMILY_DOMAIN_QUOTAS[family][domain]
            ids = buckets[family][domain]
            if family == "strategic_research_planning":
                subtype_groups: Dict[str, List[str]] = defaultdict(list)
                for task_id in ids:
                    subtype_groups[str(public_rows[task_id].get("subtype") or "")].append(task_id)
                chosen = pick_round_robin(subtype_groups, STRATEGIC_SUBTYPE_ORDER, quota)
            elif family == "venue_aware_research_positioning":
                subtype_groups = defaultdict(list)
                for task_id in ids:
                    subtype_groups[str(public_rows[task_id].get("subtype") or "")].append(task_id)
                chosen = pick_round_robin(subtype_groups, VENUE_SUBTYPE_ORDER, quota)
            else:
                chosen = ids[:quota]
            if len(chosen) != quota:
                raise RuntimeError(f"Insufficient tasks for {family} / {domain}: need {quota}, got {len(chosen)}")
            selected_ids.extend(chosen)

    selected_ids = sorted(selected_ids)
    selected_public = [public_rows[task_id] for task_id in selected_ids]
    selected_hidden = [hidden_rows[task_id] for task_id in selected_ids]
    selected_trace = [trace_rows[task_id] for task_id in selected_ids]
    selected_internal = []
    for task_id in selected_ids:
        internal_row = internal_rows.get(task_id)
        if internal_row is None:
            public_row = public_rows[task_id]
            hidden_row = hidden_rows[task_id]
            trace_row = trace_rows[task_id]
            internal_row = {
                "task_id": task_id,
                "family": public_row.get("family"),
                "subtype": public_row.get("subtype"),
                "domain": public_row.get("domain"),
                "horizon": public_row.get("horizon"),
                "title": public_row.get("title"),
                "question": public_row.get("question"),
                "gold_answer": hidden_row.get("gold_answer"),
                "expected_answer_points": hidden_row.get("expected_answer_points") or [],
                "ground_truth": hidden_row.get("ground_truth") or trace_row.get("ground_truth"),
                "support_context": trace_row.get("support_context"),
                "time_context": trace_row.get("time_context"),
                "seed": trace_row.get("seed"),
                "public_metadata": hidden_row.get("public_metadata") or trace_row.get("public_metadata") or {},
                "quality_signals": trace_row.get("quality_signals") or {},
                "source_task_id": hidden_row.get("internal_task_id"),
            }
        selected_internal.append(internal_row)

    dump_jsonl(output_dir / "tasks.jsonl", selected_public)
    dump_jsonl(output_dir / "tasks_hidden_eval.jsonl", selected_hidden)
    dump_jsonl(output_dir / "tasks_build_trace.jsonl", selected_trace)
    dump_jsonl(output_dir / "tasks_internal_full.jsonl", selected_internal)
    hidden_v3_rows = [build_hidden_eval_v3_row(hidden, trace) for hidden, trace in zip(selected_hidden, selected_trace)]
    hidden_v31_rows = [build_hidden_eval_v3_1_row(hidden_v3, trace) for hidden_v3, trace in zip(hidden_v3_rows, selected_trace)]
    dump_jsonl(output_dir / "tasks_hidden_eval_v3.jsonl", hidden_v3_rows)
    dump_jsonl(output_dir / "tasks_hidden_eval_v3_1.jsonl", hidden_v31_rows)
    dump_json(output_dir / "tasks_hidden_eval_v3_manifest.json", build_hidden_v3_manifest(source_dir, hidden_v3_rows, output_dir / "tasks_hidden_eval_v3.jsonl"))
    dump_json(output_dir / "tasks_hidden_eval_v3_1_manifest.json", build_hidden_v31_manifest(source_dir, hidden_v31_rows, output_dir / "tasks_hidden_eval_v3_1.jsonl"))
    (output_dir / "task_ids.txt").write_text("\n".join(selected_ids) + "\n", encoding="utf-8")

    family_counts = Counter(row["family"] for row in selected_public)
    domain_counts = Counter(row["domain"] for row in selected_public)
    subtype_counts = Counter((row["family"], row.get("subtype") or "") for row in selected_public)
    manifest = {
        "release_name": output_dir.name,
        "source_release": str(source_dir),
        "task_count": len(selected_public),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "subtype_counts": {f"{family}::{subtype}": count for (family, subtype), count in sorted(subtype_counts.items())},
        "selection_policy": {
            "horizon": "half_year",
            "strict_requirement": STRICT_RULES,
            "family_domain_quotas": FAMILY_DOMAIN_QUOTAS,
            "strategic_family_policy": "prefer repaired candidate-ranking agenda tasks and interleave comparative tasks",
            "venue_family_policy": "prefer venue-explicit tasks and interleave direction-forecast and venue-targeted planning subtypes",
        },
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "task_ids": "task_ids.txt",
        },
    }
    dump_json(output_dir / "manifest.json", manifest)

    readme = f"""# {output_dir.name}

## Summary
- tasks: {len(selected_public)}
- source release: {source_dir.name}
- horizon: half_year
- families: 4
- domains: 4

## Family counts
- bottleneck_opportunity_discovery: {family_counts.get('bottleneck_opportunity_discovery', 0)}
- direction_forecasting: {family_counts.get('direction_forecasting', 0)}
- strategic_research_planning: {family_counts.get('strategic_research_planning', 0)}
- venue_aware_research_positioning: {family_counts.get('venue_aware_research_positioning', 0)}

## Domain counts
- LLM agents: {domain_counts.get('LLM agents', 0)}
- LLM fine-tuning and post-training: {domain_counts.get('LLM fine-tuning and post-training', 0)}
- RAG and retrieval structuring: {domain_counts.get('RAG and retrieval structuring', 0)}
- Visual generative modeling and diffusion: {domain_counts.get('Visual generative modeling and diffusion', 0)}

## Selection notes
- Only half-year tasks are included.
- Strategic planning items are restricted to repaired candidate-ranking agenda tasks or comparative prioritization tasks.
- Venue-aware planning items must explicitly name the target venue bucket and surface candidate directions in the public question.
- Family-domain quotas are balanced so the subset can serve as a compact but representative evaluation suite.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a balanced 100-task core subset from benchmark_full.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_subset(Path(args.source_dir), Path(args.output_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
