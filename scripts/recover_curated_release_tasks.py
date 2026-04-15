from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data" / "releases" / "benchmark_full_curated_polished"
DEFAULT_SOURCE = ROOT / "data" / "releases" / "benchmark_full"
DEFAULT_OUTPUT = ROOT / "data" / "releases" / "benchmark_full_curated_recovered21"


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


def load_jsonl_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["task_id"]): row for row in iter_jsonl(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover selected tasks back into a curated release.")
    parser.add_argument("--base-release", default=str(DEFAULT_BASE))
    parser.add_argument("--source-release", default=str(DEFAULT_SOURCE))
    parser.add_argument("--recover-task-ids-file", required=True)
    parser.add_argument("--output-release", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def synthesize_internal_row(
    task_id: str,
    public_row: Dict[str, Any],
    hidden_row: Dict[str, Any],
    trace_row: Dict[str, Any],
) -> Dict[str, Any]:
    return {
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


def main() -> None:
    args = parse_args()
    base_release = Path(args.base_release)
    source_release = Path(args.source_release)
    output_release = Path(args.output_release)
    recover_ids = [
        line.strip()
        for line in Path(args.recover_task_ids_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    recover_id_set = set(recover_ids)

    if output_release.exists():
        shutil.rmtree(output_release)
    output_release.mkdir(parents=True, exist_ok=True)

    base_public = load_jsonl_by_id(base_release / "tasks.jsonl")
    base_hidden = load_jsonl_by_id(base_release / "tasks_hidden_eval.jsonl")
    base_trace = load_jsonl_by_id(base_release / "tasks_build_trace.jsonl")
    base_internal = load_jsonl_by_id(base_release / "tasks_internal_full.jsonl")

    source_public_rows = list(iter_jsonl(source_release / "tasks.jsonl"))
    source_public = {str(row["task_id"]): row for row in source_public_rows}
    source_hidden = load_jsonl_by_id(source_release / "tasks_hidden_eval.jsonl")
    source_trace = load_jsonl_by_id(source_release / "tasks_build_trace.jsonl")
    source_internal = load_jsonl_by_id(source_release / "tasks_internal_full.jsonl")

    missing = [
        task_id
        for task_id in recover_ids
        if task_id not in source_public
        or task_id not in source_hidden
        or task_id not in source_trace
    ]
    if missing:
        raise SystemExit(f"Missing recovery rows in source release: {missing[:10]}")

    merged_ids = set(base_public) | recover_id_set
    ordered_ids = [str(row["task_id"]) for row in source_public_rows if str(row["task_id"]) in merged_ids]

    merged_public = [base_public.get(task_id) or source_public[task_id] for task_id in ordered_ids]
    merged_hidden = [base_hidden.get(task_id) or source_hidden[task_id] for task_id in ordered_ids]
    merged_trace = [base_trace.get(task_id) or source_trace[task_id] for task_id in ordered_ids]
    merged_internal = []
    for task_id in ordered_ids:
        internal_row = base_internal.get(task_id) or source_internal.get(task_id)
        if internal_row is None:
            internal_row = synthesize_internal_row(
                task_id=task_id,
                public_row=base_public.get(task_id) or source_public[task_id],
                hidden_row=base_hidden.get(task_id) or source_hidden[task_id],
                trace_row=base_trace.get(task_id) or source_trace[task_id],
            )
        merged_internal.append(internal_row)

    dump_jsonl(output_release / "tasks.jsonl", merged_public)
    dump_jsonl(output_release / "tasks_hidden_eval.jsonl", merged_hidden)
    dump_jsonl(output_release / "tasks_build_trace.jsonl", merged_trace)
    dump_jsonl(output_release / "tasks_internal_full.jsonl", merged_internal)
    (output_release / "task_ids.txt").write_text("\n".join(ordered_ids) + "\n", encoding="utf-8")

    base_dropped_path = base_release / "dropped_tasks.json"
    recovered_records: List[Dict[str, Any]] = []
    if base_dropped_path.exists():
        dropped_rows = json.loads(base_dropped_path.read_text(encoding="utf-8"))
        kept_dropped = [row for row in dropped_rows if str(row.get("task_id") or "") not in recover_id_set]
        recovered_records = [row for row in dropped_rows if str(row.get("task_id") or "") in recover_id_set]
        dump_json(output_release / "dropped_tasks.json", kept_dropped)

    venue_repairs_path = base_release / "venue_repairs.json"
    venue_repairs: List[Dict[str, Any]] = []
    if venue_repairs_path.exists():
        venue_repairs = json.loads(venue_repairs_path.read_text(encoding="utf-8"))
        dump_json(output_release / "venue_repairs.json", venue_repairs)

    recovery_rows: List[Dict[str, Any]] = []
    for task_id in recover_ids:
        public_row = source_public[task_id]
        hidden_row = source_hidden[task_id]
        trace_row = source_trace[task_id]
        gt = (
            (source_internal.get(task_id) or {}).get("ground_truth")
            or hidden_row.get("ground_truth")
            or trace_row.get("ground_truth")
            or {}
        )
        support = trace_row.get("support_context") or {}
        recovery_rows.append(
            {
                "task_id": task_id,
                "family": public_row.get("family"),
                "subtype": public_row.get("subtype"),
                "domain": public_row.get("domain"),
                "horizon": public_row.get("horizon"),
                "title": public_row.get("title"),
                "candidate_directions_count": len(support.get("candidate_directions") or gt.get("candidate_directions") or []),
                "direction_records_count": len(gt.get("direction_records") or []),
            }
        )
    dump_json(output_release / "recovered_tasks.json", recovery_rows)

    base_polish_audit_path = base_release / "language_polish_audit.json"
    if base_polish_audit_path.exists():
        base_polish_audit = json.loads(base_polish_audit_path.read_text(encoding="utf-8"))
        base_polish_audit["release_dir"] = str(output_release.relative_to(ROOT))
        base_polish_audit["base_release"] = str(base_release)
        base_polish_audit["recovered_unpolished_task_count"] = len(recover_ids)
        base_polish_audit["recovered_unpolished_task_ids"] = recover_ids
        base_polish_audit["notes"] = list(base_polish_audit.get("notes") or []) + [
            "This recovered release inherits the prior language-polished rows from the base release.",
            "Recovered q1 strategic tasks were appended from benchmark_full and were not re-run through the language polish pass.",
        ]
        dump_json(output_release / "language_polish_audit.json", base_polish_audit)

    family_counts = Counter(row["family"] for row in merged_public)
    domain_counts = Counter(row["domain"] for row in merged_public)
    subtype_counts = Counter((row["family"], row.get("subtype") or "") for row in merged_public)
    horizon_counts = Counter(str(row.get("horizon") or "") for row in merged_public)

    manifest = {
        "release_name": output_release.name,
        "base_release": str(base_release),
        "recovery_source_release": str(source_release),
        "task_count": len(merged_public),
        "recovered_task_count": len(recover_ids),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "subtype_counts": {f"{family}::{subtype}": count for (family, subtype), count in sorted(subtype_counts.items())},
        "horizon_counts": dict(horizon_counts),
        "remaining_dropped_task_count": len(json.loads((output_release / "dropped_tasks.json").read_text(encoding="utf-8"))) if (output_release / "dropped_tasks.json").exists() else 0,
        "recovery_policy": {
            "base_release_lineage": "inherit benchmark_full_curated_polished and restore a targeted subset of previously dropped q1 strategic tasks",
            "selection_requirement": "restore only dropped q1 agenda tasks that still expose candidate_directions or direction_records in source traces",
            "notes": [
                "Recovered tasks are quarter-horizon strategic planning items.",
                "Recovered tasks were not re-polished by the language cleanup pass.",
            ],
        },
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "task_ids": "task_ids.txt",
            "dropped_tasks": "dropped_tasks.json",
            "venue_repairs": "venue_repairs.json",
            "recovered_tasks": "recovered_tasks.json",
            "language_polish_audit": "language_polish_audit.json",
        },
    }
    dump_json(output_release / "manifest.json", manifest)

    readme = f"""# {output_release.name}

## Summary
- tasks: {len(merged_public)}
- base release: {base_release.name}
- recovery source: {source_release.name}
- recovered tasks: {len(recover_ids)}

## Recovery scope
- restore a low-risk subset of previously dropped q1 strategic planning tasks
- retain the rest of the curated polished release unchanged
- preserve prior venue repairs and prior language-polished rows

## Recovery notes
- all recovered tasks are `q1_agenda_priority_selection`
- all recovered tasks are `quarter` horizon
- recovered tasks were selected because source traces still retained candidate directions or direction records
- recovered tasks were not re-run through the later language polish pass
"""
    (output_release / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "release_name": output_release.name,
        "task_count": len(merged_public),
        "recovered_task_count": len(recover_ids),
        "remaining_dropped_task_count": len(json.loads((output_release / "dropped_tasks.json").read_text(encoding="utf-8"))) if (output_release / "dropped_tasks.json").exists() else 0,
        "recovered_task_ids": recover_ids,
        "recovered_drop_records": recovered_records,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
