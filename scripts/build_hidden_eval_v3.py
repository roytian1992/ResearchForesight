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

from researchworld.eval_v3 import build_hidden_eval_v3_row


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark hidden-eval v3 with slot/claim/judge layers.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--manifest", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    hidden_path = release_dir / "tasks_hidden_eval.jsonl"
    trace_path = release_dir / "tasks_build_trace.jsonl"
    output_path = Path(args.output) if args.output else release_dir / "tasks_hidden_eval_v3.jsonl"
    manifest_path = Path(args.manifest) if args.manifest else release_dir / "tasks_hidden_eval_v3_manifest.json"

    trace_by_id = {row["task_id"]: row for row in iter_jsonl(trace_path)}
    rows = []
    family_counts = Counter()
    domain_counts = Counter()
    claim_type_counts = Counter()
    slot_field_counts = defaultdict(Counter)

    for hidden_row in iter_jsonl(hidden_path):
        task_id = hidden_row["task_id"]
        trace_row = trace_by_id[task_id]
        row = build_hidden_eval_v3_row(hidden_row, trace_row)
        rows.append(row)
        family_counts[row["family"]] += 1
        domain_counts[row["domain"]] += 1
        for claim in row.get("claim_bank") or []:
            claim_type_counts[claim.get("claim_type")] += 1
        for key in (row.get("slot_targets") or {}).keys():
            slot_field_counts[row["family"]][key] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "release_dir": str(release_dir),
        "output": str(output_path),
        "task_count": len(rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "claim_type_counts": dict(claim_type_counts),
        "slot_field_coverage": {family: dict(counter) for family, counter in slot_field_counts.items()},
        "notes": [
            "v3 hidden eval adds slot_targets, claim_bank, and judge_profile on top of the existing hidden eval.",
            "Current claim_bank is rule-bootstrapped from benchmark construction traces and should be refined in the next pass with evidence-linked canonical claims.",
            "This artifact is intended to support benchmark-aware FactScore and benchmark-aware LLM-as-judge.",
        ],
    }
    dump_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
