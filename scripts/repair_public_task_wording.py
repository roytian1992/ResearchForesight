from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def canonicalize_topic(text: str) -> str:
    topic = normalize_space(text)
    patterns = [
        r"^Planning Near-Term Research Directions in (.+)$",
        r"^Forecasting the Next Research Direction in (.+)$",
        r"^Using only literature available by August 31, 2025, propose and rank up to \d+ concrete next-step research directions in (.+)$",
        r"^Using only literature available by August 31, 2025, identify the single concrete next-step research direction in (.+)$",
        r"^As of August 31, 2025, you are tasked with planning a six-month research agenda on (.+?)\..+$",
        r"^As of August 31, 2025, consider the state of research on (.+?)\..+$",
        r"^Given the research landscape on (.+?) as of August 31, 2025, .+$",
    ]
    changed = True
    while changed:
        changed = False
        if topic.endswith(" and Its Likely Venue Fit"):
            topic = topic[: -len(" and Its Likely Venue Fit")].strip()
            changed = True
        for pattern in patterns:
            match = re.match(pattern, topic)
            if match:
                topic = match.group(1).strip()
                changed = True
    return topic


def display_topic(topic: str) -> str:
    cleaned = canonicalize_topic(topic)
    if not cleaned:
        return cleaned
    return cleaned[:1].upper() + cleaned[1:]


def extract_limit(question: str) -> int:
    text = str(question or "").lower()
    match = re.search(r"no more than (\w+)", text)
    if match:
        token = match.group(1)
        if token.isdigit():
            return int(token)
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    match = re.search(r"which (\w+) research directions should be prioritized", text)
    if match:
        token = match.group(1)
        if token.isdigit():
            return int(token)
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    match = re.search(r"identify and rank a small number of promising research directions", text)
    if match:
        return 4
    return 4


def strip_prefix(text: str, prefixes: list[str]) -> str:
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return text.strip()


def extract_topic_from_title(title: str) -> str:
    normalized = normalize_space(title)
    if normalized == "Prioritization of Instruction Tuning Research Directions Based on Pre-Cutoff Evidence":
        return "Instruction Tuning"
    stripped = strip_prefix(
        normalized,
        [
            "Planning Near-Term Research Directions in ",
            "Prioritization of Research Directions in ",
            "Prioritizing Research Directions in ",
            "Strategic Research Prioritization for ",
            "Forecasting the Next Research Direction in ",
            "Forecasting Top-Venue Traction in ",
            "Forecasting a High-Impact, Venue-Aware Research Direction in ",
            "Forecasting Research Directions in ",
            "Forecasting Emerging Research Directions in ",
            "Forecasting Next-Step Research Directions in ",
            "Forecasting Near-Term Research Directions in ",
            "Predicting Emerging Research Directions in ",
        ],
    )
    if stripped.endswith(" and Its Likely Venue Fit"):
        stripped = stripped[: -len(" and Its Likely Venue Fit")].strip()
    return canonicalize_topic(stripped)


def extract_topic_from_question(question: str) -> str:
    text = normalize_space(question)
    patterns = [
        r"research directions in (.+?) for the period from September 2025 to February 2026",
        r"state of research on (.+?)\.",
        r"planning a six-month research agenda on (.+?)\.",
        r"planning the next six months of research on (.+?)\.",
        r"research direction in (.+?) that is most likely",
        r"direction within (.+?) that is most likely",
        r"within (.+?) that is most likely",
        r"research landscape on (.+?) as of",
        r"state of research in (.+?)\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return canonicalize_topic(match.group(1).strip())
    return canonicalize_topic(text)


def topic_for_row(row: dict[str, Any]) -> str:
    question_topic = extract_topic_from_question(str(row.get("question") or ""))
    if question_topic and question_topic != normalize_space(str(row.get("question") or "")):
        return question_topic
    title_topic = extract_topic_from_title(str(row.get("title") or ""))
    if title_topic and title_topic != normalize_space(str(row.get("title") or "")):
        return title_topic
    return question_topic


def topic_for_row_with_trace(row: dict[str, Any], trace_row: dict[str, Any]) -> str:
    public_meta = trace_row.get("public_metadata") or {}
    trace_topic = normalize_space(str(public_meta.get("topic_title") or public_meta.get("topic") or ""))
    if trace_topic:
        return canonicalize_topic(trace_topic)
    return topic_for_row(row)


def strategic_title(topic: str) -> str:
    return f"Planning Near-Term Research Directions in {display_topic(topic)}"


def format_candidate_directions(candidate_directions: list[str]) -> str:
    parts = []
    for idx, direction in enumerate(candidate_directions, start=1):
        cleaned = normalize_space(str(direction or "")).replace("_", " ")
        if cleaned:
            parts.append(f"({idx}) {cleaned}")
    return "; ".join(parts)


def strategic_question(topic: str, limit: int, candidate_directions: list[str] | None = None) -> str:
    candidate_directions = [normalize_space(str(x or "")).replace("_", " ") for x in (candidate_directions or []) if normalize_space(str(x or ""))]
    if candidate_directions:
        return normalize_space(
            f"Using only literature available by August 31, 2025, rank the following candidate next-step research directions in {topic} "
            f"for the period from September 2025 to February 2026: {format_candidate_directions(candidate_directions)}. "
            f"Return a complete ordering of the listed options from highest to lowest priority and justify the ranking using evidence from publication trends, "
            f"venue signals, technical dependencies, and unresolved limitations visible in the pre-cutoff literature. Do not introduce new candidate directions."
        )
    return normalize_space(
        f"Using only literature available by August 31, 2025, identify and rank up to {limit} concrete next-step research directions in {topic} "
        f"for the period from September 2025 to February 2026. Each ranked direction should target a distinct technical bottleneck, enabling dependency, "
        f"or underdeveloped mechanism visible in the historical literature. For each direction, explain why it should be pursued earlier rather than later, "
        f"using evidence from publication trends, venue signals, citation uptake, and unresolved technical limitations."
    )


def strategic_requirements(candidate_directions: list[str] | None = None) -> list[str]:
    base = [
        "Use only evidence available up to the cutoff unless the benchmark setting explicitly provides an offline knowledge base.",
        "State a concrete conclusion rather than vague trend language.",
        "Support the conclusion with literature-based reasoning.",
    ]
    if candidate_directions:
        base.append("Rank the listed candidate directions from highest to lowest priority and do not introduce new directions.")
    else:
        base.append("Identify and justify a small ranked set of next-step directions.")
    return base


def venue_title(topic: str) -> str:
    return f"Forecasting the Next Research Direction in {display_topic(topic)} and Its Likely Venue Fit"


def venue_question(topic: str) -> str:
    return normalize_space(
        f"Using only literature available by August 31, 2025, identify the single concrete next-step research direction in {topic} "
        f"that is most likely to gain top-tier traction during the following six months, from September 2025 to February 2026. "
        f"State the one most likely top-tier venue bucket for that direction, for example AAAI-like, EMNLP-like, or ICLR-like, "
        f"and explain what technical framing or contribution profile makes that venue fit plausible. Do not rely on developments after August 31, 2025."
    )


def venue_planning_question(topic: str, bucket: str, candidate_directions: list[str] | None = None) -> str:
    bucket_label = f"{bucket}-like" if bucket else "top-tier"
    candidate_directions = [normalize_space(str(x or "")).replace("_", " ") for x in (candidate_directions or []) if normalize_space(str(x or ""))]
    if candidate_directions:
        return normalize_space(
            f"A research team wants to maximize its relevance for {bucket_label} venues in the next submission cycle. "
            f"Based only on literature available before September 1, 2025, rank the following candidate next-step research directions in {topic}: "
            f"{format_candidate_directions(candidate_directions)}. Return a complete ordering of the listed options and explain why each direction is more or less compatible "
            f"with {bucket_label} contribution style, evaluation preferences, and near-term momentum. Do not introduce new candidate directions."
        )
    return normalize_space(
        f"A research team wants to maximize its relevance for {bucket_label} venues in the next submission cycle. "
        f"Based only on literature available before September 1, 2025, which one or two next-step research directions in {topic} should be prioritized, "
        f"and what evidence-based rationale supports that ranking?"
    )


def venue_requirements() -> list[str]:
    return [
        "Use only evidence available up to the cutoff unless the benchmark setting explicitly provides an offline knowledge base.",
        "State a concrete conclusion rather than vague trend language.",
        "Support the conclusion with literature-based reasoning.",
        "Name one specific next-step direction and characterize the trajectory.",
        "Identify one likely top-tier venue bucket for that direction and explain why the venue fit is plausible.",
    ]


def venue_planning_requirements(candidate_directions: list[str] | None = None) -> list[str]:
    base = [
        "Use only evidence available up to the cutoff unless the benchmark setting explicitly provides an offline knowledge base.",
        "State a concrete conclusion rather than vague trend language.",
        "Support the conclusion with literature-based reasoning.",
    ]
    if candidate_directions:
        base.append("Rank the listed candidate directions for the named venue bucket and do not introduce new directions.")
    else:
        base.append("Return a small ranked set of venue-positioned directions rather than an unstructured list.")
    base.append("Explain why the ordering fits the target venue bucket.")
    return base


def load_trace_by_task(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    return {row["task_id"]: row for row in iter_jsonl(path)}


def repair_tasks(tasks_path: Path, trace_by_task: dict[str, dict[str, Any]]) -> tuple[int, int]:
    rows = iter_jsonl(tasks_path)
    strategic_count = 0
    venue_count = 0
    for row in rows:
        family = str(row.get("family") or "")
        subtype = str(row.get("subtype") or "")
        trace_row = trace_by_task.get(str(row.get("task_id") or "")) or {}
        topic = topic_for_row_with_trace(row, trace_row)
        if family == "strategic_research_planning" and subtype == "agenda_priority_selection":
            candidate_directions = ((trace_row.get("support_context") or {}).get("candidate_directions") or [])
            row["title"] = strategic_title(topic)
            row["question"] = strategic_question(topic, extract_limit(str(row.get("question") or "")), candidate_directions)
            deliverable = dict(row.get("deliverable_spec") or {})
            deliverable["requirements"] = strategic_requirements(candidate_directions)
            row["deliverable_spec"] = deliverable
            strategic_count += 1
        elif subtype == "venue_aware_direction_forecast":
            row["title"] = venue_title(topic)
            row["question"] = venue_question(topic)
            deliverable = dict(row.get("deliverable_spec") or {})
            deliverable["requirements"] = venue_requirements()
            row["deliverable_spec"] = deliverable
            venue_count += 1
        elif family == "venue_aware_research_positioning" and subtype == "venue_targeted_planning":
            support = trace_row.get("support_context") or {}
            candidate_directions = support.get("candidate_directions") or []
            bucket = normalize_space(str(support.get("target_venue_bucket") or ""))
            row["question"] = venue_planning_question(topic, bucket, candidate_directions)
            deliverable = dict(row.get("deliverable_spec") or {})
            deliverable["requirements"] = venue_planning_requirements(candidate_directions)
            row["deliverable_spec"] = deliverable
            venue_count += 1
    dump_jsonl(tasks_path, rows)
    return strategic_count, venue_count


def repair_venue_summary(path: Path, tasks_by_id: dict[str, dict[str, Any]]) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0].keys()) if rows else []
    touched = 0
    for row in rows:
        task_id = str(row.get("task_id") or "")
        public_row = tasks_by_id.get(task_id) or {}
        if str(row.get("subtype") or "") != "venue_aware_direction_forecast" or not public_row:
            continue
        row["title"] = str(public_row.get("title") or row.get("title") or "")
        touched += 1
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return touched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair unclear public wording for selected ResearchForesight task types.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--trace-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    tasks_path = release_dir / "tasks.jsonl"
    summary_path = release_dir / "venue_task_summary.csv"
    trace_path = Path(args.trace_path) if args.trace_path else release_dir / "tasks_build_trace.jsonl"
    trace_by_task = load_trace_by_task(trace_path)
    strategic_count, venue_count = repair_tasks(tasks_path, trace_by_task)
    tasks_by_id = {row["task_id"]: row for row in iter_jsonl(tasks_path)}
    summary_count = repair_venue_summary(summary_path, tasks_by_id)
    print(
        json.dumps(
            {
                "release_dir": str(release_dir),
                "strategic_tasks_rewritten": strategic_count,
                "venue_tasks_rewritten": venue_count,
                "venue_summary_rows_rewritten": summary_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
