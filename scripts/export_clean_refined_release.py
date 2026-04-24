from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


KEEP_TOP_LEVEL = [
    "schema_version",
    "task_id",
    "family",
    "subtype",
    "domain",
    "horizon",
    "title",
    "question",
    "time_cutoff",
    "deliverable_spec",
    "gold_answer",
    "expected_answer_points",
    "evaluation_rubric",
    "eval_targets",
    "trace",
    "answer_contract",
]

KEEP_TRACE = [
    "time_context",
    "history_evidence",
    "future_evidence",
    "notes",
]

KEEP_TIME_CONTEXT = [
    "history_end",
    "history_structure_slice",
    "future_window",
]


def _clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {key: row[key] for key in KEEP_TOP_LEVEL if key in row}

    trace = dict(row.get("trace") or {})
    cleaned_trace = {key: trace[key] for key in KEEP_TRACE if key in trace}

    time_context = dict(cleaned_trace.get("time_context") or {})
    if time_context:
        cleaned_trace["time_context"] = {
            key: time_context[key] for key in KEEP_TIME_CONTEXT if key in time_context
        }

    cleaned["trace"] = cleaned_trace
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a clean public refined release jsonl.")
    parser.add_argument("--input", required=True, help="Input jsonl path.")
    parser.add_argument("--output", required=True, help="Output jsonl path.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            cleaned = _clean_row(row)
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
