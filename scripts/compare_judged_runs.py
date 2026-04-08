from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_task_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = [float((row.get("judge") or {}).get("overall_score") or 0.0) for row in rows]
    family: Dict[str, List[float]] = {}
    domain: Dict[str, List[float]] = {}
    for row, score in zip(rows, overall):
        family.setdefault(str(row.get("family") or "unknown"), []).append(score)
        domain.setdefault(str(row.get("domain_id") or row.get("domain") or "unknown"), []).append(score)
    return {
        "count": len(rows),
        "mean_overall_score": round(sum(overall) / len(overall), 4) if overall else 0.0,
        "family_scores": {k: round(sum(v) / len(v), 4) for k, v in sorted(family.items())},
        "domain_scores": {k: round(sum(v) / len(v), 4) for k, v in sorted(domain.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare judged result files on an optional task subset.")
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--task-ids-file", default=None)
    args = parser.parse_args()

    task_ids = load_task_ids(Path(args.task_ids_file)) if args.task_ids_file else None
    output = {}
    for name, raw_path in args.run:
        rows = load_jsonl(Path(raw_path))
        if task_ids is not None:
            rows = [row for row in rows if str(row.get("task_id") or "") in task_ids]
        output[name] = summarize(rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
