from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "data" / "releases" / "benchmark_full_curated_recovered21"
DEFAULT_OUTPUT = ROOT / "data" / "releases" / "benchmark_full_curated_recovered21_bottleneck18"


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


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def slugify(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return value or "recovered_direction"


def candidate_directions_from_cluster(gt: Dict[str, Any]) -> List[str]:
    directions: List[str] = []
    for row in list(gt.get("historical_future_work_cluster") or []):
        text = str((row or {}).get("direction") or (row or {}).get("name") or "").strip()
        if text and text not in directions:
            directions.append(text)
    return directions


def build_recovered_descendants(task_id: str, gt: Dict[str, Any]) -> List[Dict[str, Any]]:
    future_paper_count = int(((gt.get("future_half_stats") or {}).get("paper_count") or 0))
    rows = []
    for direction in candidate_directions_from_cluster(gt)[:3]:
        rows.append(
            {
                "node_id": f"recovered_bottleneck/{task_id}/{slugify(direction)}",
                "display_name": direction,
                "created_time_slice": "",
                "future_paper_count": future_paper_count,
                "recovery_source": "historical_future_work_cluster",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill bottleneck future_descendants from historical future-work clusters.")
    parser.add_argument("--base-release", default=str(DEFAULT_BASE))
    parser.add_argument("--output-release", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_release = Path(args.base_release)
    output_release = Path(args.output_release)

    if output_release.exists():
        shutil.rmtree(output_release)
    output_release.mkdir(parents=True, exist_ok=True)

    public_rows = load_jsonl_rows(base_release / "tasks.jsonl")
    hidden_rows = load_jsonl_rows(base_release / "tasks_hidden_eval.jsonl")
    trace_rows = load_jsonl_rows(base_release / "tasks_build_trace.jsonl")
    internal_rows = load_jsonl_rows(base_release / "tasks_internal_full.jsonl")

    hidden_by_id = {str(row["task_id"]): row for row in hidden_rows}
    trace_by_id = {str(row["task_id"]): row for row in trace_rows}
    internal_by_id = {str(row["task_id"]): row for row in internal_rows}

    recovered_reports: List[Dict[str, Any]] = []

    for public_row in public_rows:
        task_id = str(public_row["task_id"])
        hidden_row = hidden_by_id[task_id]
        trace_row = trace_by_id[task_id]
        internal_row = internal_by_id[task_id]
        if str(public_row.get("family") or "") != "bottleneck_opportunity_discovery":
            continue

        recovered_any = False
        for row in (hidden_row, trace_row, internal_row):
            gt = row.get("ground_truth") or {}
            if gt.get("future_descendants"):
                continue
            candidate_dirs = candidate_directions_from_cluster(gt)
            if not candidate_dirs:
                continue
            recovered = build_recovered_descendants(task_id, gt)
            gt["future_descendants"] = recovered
            if not gt.get("realized_opportunity_directions"):
                gt["realized_opportunity_directions"] = json.loads(json.dumps(recovered, ensure_ascii=False))
            row["ground_truth"] = gt
            public_meta = row.get("public_metadata") or {}
            if not public_meta.get("future_themes"):
                public_meta["future_themes"] = [item["display_name"] for item in recovered]
            row["public_metadata"] = public_meta
            recovered_any = True

        if recovered_any:
            recovered_reports.append(
                {
                    "task_id": task_id,
                    "family": public_row.get("family"),
                    "domain": public_row.get("domain"),
                    "title": public_row.get("title"),
                    "recovered_future_descendants": [item["display_name"] for item in (hidden_by_id[task_id].get("ground_truth") or {}).get("future_descendants", [])],
                }
            )

    dump_jsonl(output_release / "tasks.jsonl", public_rows)
    dump_jsonl(output_release / "tasks_hidden_eval.jsonl", hidden_rows)
    dump_jsonl(output_release / "tasks_build_trace.jsonl", trace_rows)
    dump_jsonl(output_release / "tasks_internal_full.jsonl", internal_rows)
    (output_release / "task_ids.txt").write_text(
        "\n".join(str(row["task_id"]) for row in public_rows) + "\n",
        encoding="utf-8",
    )

    for name in [
        "dropped_tasks.json",
        "venue_repairs.json",
        "recovered_tasks.json",
        "language_polish_audit.json",
        "tasks_hidden_eval_v3.jsonl",
        "tasks_hidden_eval_v3_manifest.json",
        "tasks_hidden_eval_v3_1.jsonl",
        "tasks_hidden_eval_v3_1_manifest.json",
    ]:
        src = base_release / name
        if src.exists() and name not in {
            "tasks_hidden_eval_v3.jsonl",
            "tasks_hidden_eval_v3_manifest.json",
            "tasks_hidden_eval_v3_1.jsonl",
            "tasks_hidden_eval_v3_1_manifest.json",
        }:
            shutil.copy2(src, output_release / name)

    dump_json(output_release / "recovered_bottleneck_future_descendants.json", recovered_reports)

    family_counts = Counter(row["family"] for row in public_rows)
    subtype_counts = Counter((row["family"], row.get("subtype") or "") for row in public_rows)
    manifest = json.loads((base_release / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        {
            "release_name": output_release.name,
            "base_release": str(base_release),
            "task_count": len(public_rows),
            "family_counts": dict(family_counts),
            "subtype_counts": {f"{family}::{subtype}": count for (family, subtype), count in sorted(subtype_counts.items())},
            "bottleneck_future_descendants_recovered_count": len(recovered_reports),
        }
    )
    files = dict(manifest.get("files") or {})
    files["recovered_bottleneck_future_descendants"] = "recovered_bottleneck_future_descendants.json"
    manifest["files"] = files
    recovery_policy = dict(manifest.get("recovery_policy") or {})
    notes = list(recovery_policy.get("notes") or [])
    notes.append("Backfill bottleneck future_descendants from historical_future_work_cluster when future_descendants were empty.")
    recovery_policy["notes"] = notes
    manifest["recovery_policy"] = recovery_policy
    dump_json(output_release / "manifest.json", manifest)

    readme = f"""# {output_release.name}

## Summary
- tasks: {len(public_rows)}
- base release: {base_release.name}
- bottleneck future-descendant recoveries: {len(recovered_reports)}

## Recovery notes
- only bottleneck tasks with empty `future_descendants` and non-empty `historical_future_work_cluster` were modified
- recovered labels are human-readable direction strings copied from `historical_future_work_cluster.direction`
- `realized_opportunity_directions` and `public_metadata.future_themes` were backfilled together to keep eval artifacts aligned
"""
    (output_release / "README.md").write_text(readme, encoding="utf-8")

    print(
        json.dumps(
            {
                "release_name": output_release.name,
                "task_count": len(public_rows),
                "bottleneck_future_descendants_recovered_count": len(recovered_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
