from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.eval_v3 import build_hidden_eval_v3_row
from researchworld.eval_v3_1 import build_hidden_eval_v3_1_row
from researchworld.research_arc_v2 import extract_task_contract


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


def _answer_contract(public_row: Dict[str, Any]) -> Dict[str, Any]:
    family = str(public_row.get("family") or "")
    subtype = str(public_row.get("subtype") or "")
    contract = extract_task_contract(public_row)
    candidate_directions = [str(x).strip() for x in (contract.get("candidate_directions") or []) if str(x).strip()]
    topic_text = str(contract.get("topic_text") or public_row.get("title") or "").strip()
    if family == "bottleneck_opportunity_discovery":
        return {
            "shape": "single_paragraph",
            "topic_text": topic_text,
            "ranking_required": False,
            "max_items": 1,
            "must_cover": [
                "one concrete unresolved bottleneck",
                "one concrete downstream opportunity",
                "explicit causal linkage between bottleneck and opportunity",
            ],
            "style_requirements": [
                "commit to the bottleneck early",
                "use concrete technical language rather than broad trend language",
                "keep the answer compact and evidence-grounded",
            ],
            "disallowed_patterns": [
                "multiple unrelated bottlenecks",
                "multiple unrelated opportunities",
                "generic future trend summary",
            ],
        }
    if family == "direction_forecasting":
        return {
            "shape": "single_paragraph",
            "topic_text": topic_text,
            "ranking_required": False,
            "max_items": 1,
            "must_cover": [
                "one trajectory call",
                "one primary next direction",
                "one explicit why-now trigger",
            ],
            "style_requirements": [
                "state the trajectory and next direction early",
                "prefer one concrete successor theme over a broad area",
                "keep the answer compact and temporally disciplined",
            ],
            "disallowed_patterns": [
                "multiple disconnected next directions",
                "generic trend recap without a concrete call",
            ],
        }
    if family == "strategic_research_planning":
        return {
            "shape": "compare_ranked_list" if candidate_directions else "ranked_list",
            "topic_text": topic_text,
            "ranking_required": True,
            "max_items": int(contract.get("max_items") or 3),
            "candidate_directions": candidate_directions,
            "must_cover": [
                "explicit ranking",
                "why-now rationale for each ranked item",
                "one dependency or trade-off per ranked item",
            ],
            "style_requirements": [
                "use a short ranked list",
                "keep direction labels concrete",
                "prioritize executable near-term directions over broad agendas",
            ],
            "disallowed_patterns": (
                [
                    "introduce a third direction outside the listed candidate directions",
                    "replace listed candidate labels with narrower substitute labels",
                ]
                if candidate_directions
                else ["unranked brainstorm", "broad survey-style agenda"]
            ),
        }
    return {
        "shape": "venue_ranked_list",
        "topic_text": topic_text,
        "ranking_required": True,
        "max_items": 1 if subtype == "venue_aware_direction_forecast" else 2,
        "must_cover": [
            "one concrete direction or short ranked plan",
            "explicit technical rationale",
            "explicit venue-fit rationale",
        ],
        "style_requirements": [
            "make the venue family explicit",
            "separate technical merit from venue-fit logic",
            "prefer submission-ready framing over generic praise",
        ],
        "disallowed_patterns": [
            "implicit venue fit without explanation",
            "generic interestingness argument without venue framing",
        ],
    }


def _simplify_public_row(public_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": public_row.get("task_id"),
        "family": public_row.get("family"),
        "domain": public_row.get("domain"),
        "horizon": public_row.get("horizon"),
        "title": public_row.get("title"),
        "question": public_row.get("question"),
        "time_cutoff": public_row.get("time_cutoff"),
        "deliverable_spec": public_row.get("deliverable_spec") or {},
        "answer_contract": _answer_contract(public_row),
    }


def _canonical_component_targets(hidden_v31_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for component in ((hidden_v31_row.get("component_targets") or {}).get("components") or []):
        out.append(
            {
                "name": component.get("name"),
                "importance": component.get("importance"),
                "canonical_values": list(component.get("canonical_values") or []),
                "notes": component.get("notes"),
            }
        )
    return out


def _canonical_claim_bank(hidden_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for claim in (hidden_row.get("claim_bank") or []):
        out.append(
            {
                "claim_id": claim.get("claim_id"),
                "text": claim.get("text"),
                "claim_type": claim.get("claim_type"),
                "time_scope": claim.get("time_scope"),
                "importance": claim.get("importance"),
                "canonical_objects": list(claim.get("canonical_objects") or []),
            }
        )
    return out


def _family_aux_targets(hidden_row: Dict[str, Any]) -> Dict[str, Any]:
    slots = hidden_row.get("slot_targets") or {}
    family = str(hidden_row.get("family") or "")
    if family == "bottleneck_opportunity_discovery":
        return {
            "topic": slots.get("topic_title") or slots.get("topic"),
            "historical_bottleneck_labels": list(slots.get("core_bottleneck_labels") or []),
            "future_opportunity_labels": list(slots.get("core_opportunity_labels") or slots.get("future_themes") or []),
        }
    if family == "direction_forecasting":
        return {
            "topic": slots.get("topic_title") or slots.get("topic"),
            "future_direction_labels": list(slots.get("future_themes") or []),
            "trajectory_support": slots.get("trajectory_support") or {},
        }
    if family == "strategic_research_planning":
        return {
            "topic": slots.get("topic_title") or slots.get("topic"),
            "priority_direction_labels": list(slots.get("priority_direction_labels") or slots.get("future_themes") or []),
        }
    return {
        "topic": slots.get("topic_title") or slots.get("topic"),
        "priority_direction_labels": list(slots.get("priority_direction_labels") or slots.get("future_themes") or []),
        "target_window_stats": slots.get("target_window_stats") or {},
    }


def _canonical_hidden_row(public_row: Dict[str, Any], hidden_row: Dict[str, Any], hidden_v31_row: Dict[str, Any]) -> Dict[str, Any]:
    slots = hidden_row.get("slot_targets") or {}
    answer_contract = public_row.get("answer_contract") or {}
    canonical_topic = (
        slots.get("topic_title")
        or slots.get("topic")
        or answer_contract.get("topic_text")
        or public_row.get("title")
    )
    family_aux_gt = _family_aux_targets(hidden_row)
    if not family_aux_gt.get("topic"):
        family_aux_gt["topic"] = canonical_topic
    return {
        "task_id": hidden_row.get("task_id"),
        "family": hidden_row.get("family"),
        "domain": public_row.get("domain") or hidden_row.get("domain"),
        "topic": canonical_topic,
        "gold_answer": hidden_row.get("gold_answer"),
        "expected_answer_points": list(hidden_row.get("expected_answer_points") or []),
        "answer_contract": answer_contract,
        "primary_metric_gt": {
            "fact_claims": _canonical_claim_bank(hidden_row),
            "component_targets": _canonical_component_targets(hidden_v31_row),
            "future_alignment_targets": hidden_v31_row.get("future_alignment_targets") or {},
        },
        "family_aux_gt": family_aux_gt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a schema-contract pilot release from a source release and task-id list.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-ids-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    requested_ids = [line.strip() for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    requested_set = set(requested_ids)

    public_rows = list(iter_jsonl(source_dir / "tasks.jsonl"))
    hidden_by_id = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_hidden_eval.jsonl")}
    trace_by_id = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_build_trace.jsonl")}
    internal_by_id = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_internal_full.jsonl")}

    selected_public_legacy = [row for row in public_rows if str(row.get("task_id") or "") in requested_set]
    ordered_ids = [str(row.get("task_id") or "") for row in selected_public_legacy]
    missing = sorted(requested_set - set(ordered_ids))
    if missing:
        raise SystemExit(f"Missing task IDs in source release: {missing[:10]}")

    selected_public = [_simplify_public_row(row) for row in selected_public_legacy]
    selected_hidden = [hidden_by_id[row["task_id"]] for row in selected_public_legacy]
    selected_trace = [trace_by_id[row["task_id"]] for row in selected_public_legacy]
    selected_internal = [internal_by_id[row["task_id"]] for row in selected_public_legacy]
    hidden_v3_rows = [build_hidden_eval_v3_row(hidden, trace) for hidden, trace in zip(selected_hidden, selected_trace)]
    hidden_v31_rows = [build_hidden_eval_v3_1_row(hidden_v3, trace) for hidden_v3, trace in zip(hidden_v3_rows, selected_trace)]
    canonical_hidden_rows = [
        _canonical_hidden_row(public_row, hidden_row, hidden_v31_row)
        for public_row, hidden_row, hidden_v31_row in zip(selected_public, selected_hidden, hidden_v31_rows)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_dir / "tasks.jsonl", selected_public)
    dump_jsonl(output_dir / "tasks_public_legacy.jsonl", selected_public_legacy)
    dump_jsonl(output_dir / "tasks_hidden_eval.jsonl", selected_hidden)
    dump_jsonl(output_dir / "tasks_hidden_eval_v3.jsonl", hidden_v3_rows)
    dump_jsonl(output_dir / "tasks_hidden_eval_v3_1.jsonl", hidden_v31_rows)
    dump_jsonl(output_dir / "tasks_hidden_eval_canonical.jsonl", canonical_hidden_rows)
    dump_jsonl(output_dir / "tasks_build_trace.jsonl", selected_trace)
    dump_jsonl(output_dir / "tasks_internal_full.jsonl", selected_internal)
    (output_dir / "task_ids.txt").write_text("\n".join(ordered_ids) + "\n", encoding="utf-8")

    for name in ("kb", "future_kb"):
        src = source_dir / name
        dst = output_dir / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    family_counts = Counter(str(row.get("family") or "") for row in selected_public)
    domain_counts = Counter(str(row.get("domain") or "") for row in selected_public)
    shape_counts = Counter(str((row.get("answer_contract") or {}).get("shape") or "") for row in selected_public)
    manifest = {
        "release_name": output_dir.name,
        "source_release": str(source_dir),
        "task_count": len(selected_public),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "answer_contract_shape_counts": dict(shape_counts),
        "schema_pilot": {
            "public_schema": "public_v2_minimal_plus_answer_contract",
            "hidden_schema": "hidden_canonical_v1_plus_legacy_eval_views",
            "compatibility_note": "Legacy hidden eval files are retained so existing evaluation scripts still run.",
        },
        "files": {
            "tasks_public_v2": "tasks.jsonl",
            "tasks_public_legacy": "tasks_public_legacy.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_hidden_eval_v3": "tasks_hidden_eval_v3.jsonl",
            "tasks_hidden_eval_v3_1": "tasks_hidden_eval_v3_1.jsonl",
            "tasks_hidden_eval_canonical": "tasks_hidden_eval_canonical.jsonl",
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
- public schema: minimal public task + `answer_contract`
- hidden schema: canonical hidden GT + legacy eval files for compatibility

## Family counts
- bottleneck_opportunity_discovery: {family_counts.get('bottleneck_opportunity_discovery', 0)}
- direction_forecasting: {family_counts.get('direction_forecasting', 0)}
- strategic_research_planning: {family_counts.get('strategic_research_planning', 0)}
- venue_aware_research_positioning: {family_counts.get('venue_aware_research_positioning', 0)}

## Notes
- `tasks.jsonl` is the pilot public schema used for future method-side contract experiments.
- `tasks_public_legacy.jsonl` preserves the original public rows for comparison.
- `tasks_hidden_eval_canonical.jsonl` is a simplified GT view for metric-side redesign.
- Existing hidden eval files are retained so old evaluation scripts continue to work.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
