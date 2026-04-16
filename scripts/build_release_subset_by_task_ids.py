from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "releases" / "benchmark_full_curated_polished"
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
        "notes": ["This manifest was regenerated while building a release subset."],
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
        "notes": ["This manifest was regenerated while building a release subset."],
    }


def strict_keep(internal_row: Dict[str, Any]) -> bool:
    family = str(internal_row.get("family") or "")
    key = STRICT_RULES.get(family)
    if not key:
        return False
    ground_truth = internal_row.get("ground_truth") or {}
    return bool(ground_truth.get(key))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a release subset by task IDs or strict filtering rules.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--task-ids-file", default="")
    parser.add_argument("--strict-251", action="store_true")
    parser.add_argument("--record-task-ids-out", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    public_rows = list(iter_jsonl(source_dir / "tasks.jsonl"))
    hidden_by_id = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_hidden_eval.jsonl")}
    trace_by_id = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_build_trace.jsonl")}
    internal_rows = list(iter_jsonl(source_dir / "tasks_internal_full.jsonl"))
    internal_by_id = {row["task_id"]: row for row in internal_rows}

    if args.task_ids_file:
        requested_ids = [line.strip() for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        selected_ids = set(requested_ids)
        selection_mode = "task_ids_file"
    elif args.strict_251:
        selected_ids = {row["task_id"] for row in internal_rows if strict_keep(row)}
        selection_mode = "strict_251"
    else:
        raise SystemExit("Either --task-ids-file or --strict-251 is required.")

    selected_public = [row for row in public_rows if row["task_id"] in selected_ids]
    selected_hidden = [hidden_by_id[row["task_id"]] for row in selected_public]
    selected_trace = [trace_by_id[row["task_id"]] for row in selected_public]
    selected_internal = [internal_by_id[row["task_id"]] for row in selected_public]
    ordered_ids = [row["task_id"] for row in selected_public]

    missing_ids = sorted(selected_ids - set(ordered_ids))
    if missing_ids:
        raise SystemExit(f"Missing task IDs in source release: {missing_ids[:10]}")

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
    (output_dir / "task_ids.txt").write_text("\n".join(ordered_ids) + "\n", encoding="utf-8")

    if args.record_task_ids_out:
        Path(args.record_task_ids_out).write_text("\n".join(ordered_ids) + "\n", encoding="utf-8")

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
        "selection_mode": selection_mode,
        "strict_filter_rules": STRICT_RULES if selection_mode == "strict_251" else {},
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
- selection mode: {selection_mode}

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

## Strict filtering
- bottleneck_opportunity_discovery requires non-empty `ground_truth.future_descendants`
- direction_forecasting requires non-empty `ground_truth.emergent_descendants`
- strategic_research_planning requires non-empty `ground_truth.direction_records`
- venue_aware_research_positioning requires non-empty `ground_truth.direction_records`
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
