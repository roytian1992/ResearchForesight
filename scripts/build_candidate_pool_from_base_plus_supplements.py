from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]


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


def load_release_rows(release_dir: Path, filename: str) -> Dict[str, Dict[str, Any]]:
    return {row["task_id"]: row for row in iter_jsonl(release_dir / filename)}


def maybe_copy_or_link(src_dir: Path, dst_dir: Path, name: str) -> None:
    src = src_dir / name
    dst = dst_dir / name
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.is_dir():
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copy2(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a candidate release using base rows plus new supplemental task IDs.")
    parser.add_argument(
        "--base-release",
        default=str(ROOT / "data" / "releases" / "benchmark_full_curated_polished"),
    )
    parser.add_argument(
        "--supplement-release",
        default=str(ROOT / "data" / "releases" / "benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover"),
    )
    parser.add_argument(
        "--output-release",
        default=str(ROOT / "data" / "releases" / "benchmark_full_curated422_plus_supplements106_candidate_v1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_release = Path(args.base_release)
    supplement_release = Path(args.supplement_release)
    output_release = Path(args.output_release)
    output_release.mkdir(parents=True, exist_ok=True)

    base_public = list(iter_jsonl(base_release / "tasks.jsonl"))
    base_ids = [row["task_id"] for row in base_public]
    base_id_set = set(base_ids)

    supp_public = load_release_rows(supplement_release, "tasks.jsonl")
    supp_hidden = load_release_rows(supplement_release, "tasks_hidden_eval.jsonl")
    supp_trace = load_release_rows(supplement_release, "tasks_build_trace.jsonl")
    supp_internal = load_release_rows(supplement_release, "tasks_internal_full.jsonl")

    base_hidden = load_release_rows(base_release, "tasks_hidden_eval.jsonl")
    base_trace = load_release_rows(base_release, "tasks_build_trace.jsonl")
    base_internal = load_release_rows(base_release, "tasks_internal_full.jsonl")

    extra_ids = sorted(set(supp_public) - base_id_set)
    ordered_ids = list(base_ids) + extra_ids

    merged_public = [base_public[i] for i in range(len(base_public))]
    merged_hidden = [base_hidden[task_id] for task_id in base_ids]
    merged_trace = [base_trace[task_id] for task_id in base_ids]
    merged_internal = [base_internal[task_id] for task_id in base_ids]

    for task_id in extra_ids:
        merged_public.append(supp_public[task_id])
        merged_hidden.append(supp_hidden[task_id])
        merged_trace.append(supp_trace[task_id])
        merged_internal.append(supp_internal[task_id])

    dump_jsonl(output_release / "tasks.jsonl", merged_public)
    dump_jsonl(output_release / "tasks_hidden_eval.jsonl", merged_hidden)
    dump_jsonl(output_release / "tasks_build_trace.jsonl", merged_trace)
    dump_jsonl(output_release / "tasks_internal_full.jsonl", merged_internal)
    (output_release / "task_ids.txt").write_text("\n".join(ordered_ids) + "\n", encoding="utf-8")

    for name in ("kb", "future_kb"):
        maybe_copy_or_link(supplement_release, output_release, name)

    family_counts = Counter(row["family"] for row in merged_public)
    domain_counts = Counter(row["domain"] for row in merged_public)
    manifest = {
        "release_name": output_release.name,
        "source_release": str(base_release),
        "supplement_release": str(supplement_release),
        "task_count": len(merged_public),
        "base_task_count": len(base_ids),
        "supplement_added_task_count": len(extra_ids),
        "supplement_added_task_ids": extra_ids,
        "selection_policy": {
            "base_policy": "keep all rows from base release exactly as-is",
            "supplement_policy": "append only task IDs absent from base release, taking rows from supplement release",
        },
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "task_ids": "task_ids.txt",
        },
    }
    dump_json(output_release / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
