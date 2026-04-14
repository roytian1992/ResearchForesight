from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def normalized(text: str) -> str:
    text = normalize_space(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def contains_any(text: str, phrases: List[str]) -> bool:
    return any(normalized(phrase) in text for phrase in phrases)


RANK_APPEND = "Rank only the listed options; do not introduce new candidate directions."
DIRECTION_APPEND = "Identify exactly one concrete next-step research direction."
VENUE_APPEND = "State one most likely top-tier venue bucket for that direction."


def tidy_question(question: str) -> str:
    q = normalize_space(question)

    if RANK_APPEND in q:
        prefix = normalize_space(q.replace(RANK_APPEND, "").strip())
        if contains_any(
            normalized(prefix),
            [
                "only the listed directions should be ranked",
                "only the listed directions may be ranked",
                "do not introduce new candidate directions",
                "do not introduce new directions",
                "do not add new directions",
                "do not add new candidate directions",
                "no additional directions should be introduced",
                "limit analysis to the options listed",
            ],
        ):
            q = prefix

    if DIRECTION_APPEND in q:
        prefix = normalize_space(q.replace(DIRECTION_APPEND, "").strip())
        if contains_any(
            normalized(prefix),
            [
                "single most concrete next step research direction",
                "single concrete next step research direction",
                "single most plausible concrete research direction",
                "single most viable concrete research direction",
                "single concrete research direction",
                "one concrete next step research direction",
                "one concrete research direction",
            ],
        ):
            q = prefix

    if VENUE_APPEND in q:
        prefix = normalize_space(q.replace(VENUE_APPEND, "").strip())
        if contains_any(
            normalized(prefix),
            [
                "one top tier venue bucket",
                "single most likely top tier venue bucket",
                "single most likely venue bucket",
                "one most fitting top tier venue bucket",
                "best fitting top tier venue bucket",
                "most likely venue bucket",
                "one likely top tier venue bucket",
            ],
        ):
            q = prefix

    q = q.replace(" .", ".")
    return normalize_space(q)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove redundant post-polish contract sentences from public benchmark questions.")
    parser.add_argument("--release-dir", required=True)
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    public_path = release_dir / "tasks.jsonl"
    internal_path = release_dir / "tasks_internal_full.jsonl"
    trace_path = release_dir / "tasks_build_trace.jsonl"

    public_rows = iter_jsonl(public_path)
    internal_rows = iter_jsonl(internal_path) if internal_path.exists() else []
    trace_rows = iter_jsonl(trace_path) if trace_path.exists() else []

    public_by_id = {row["task_id"]: row for row in public_rows}
    internal_by_id = {row["task_id"]: row for row in internal_rows}
    trace_by_id = {row["task_id"]: row for row in trace_rows}

    changed = 0
    for task_id, row in public_by_id.items():
        old_question = str(row.get("question") or "")
        new_question = tidy_question(old_question)
        if new_question == old_question:
            continue
        row["question"] = new_question
        if task_id in internal_by_id:
            internal_by_id[task_id]["question"] = new_question
            if "draft_question" in internal_by_id[task_id]:
                internal_by_id[task_id]["draft_question"] = new_question
        if task_id in trace_by_id:
            polish = trace_by_id[task_id].get("language_polish") or {}
            if polish.get("final_question") == old_question:
                polish["final_question"] = new_question
            trace_by_id[task_id]["language_polish"] = polish
        changed += 1

    dump_jsonl(public_path, public_rows)
    if internal_rows:
        dump_jsonl(internal_path, internal_rows)
    if trace_rows:
        dump_jsonl(trace_path, trace_rows)

    print(json.dumps({"release_dir": str(release_dir), "changed_questions": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
