from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.llm import FallbackOpenAICompatChatClient, OpenAICompatChatClient, complete_json_object, load_openai_compat_config


VENUE_SUBTYPES = {"venue_aware_direction_forecast", "venue_targeted_planning"}


def iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polish venue-aware task titles/questions with LLM.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--fallback-llm-config", default="configs/llm/qwen_235b.local.yaml")
    return parser.parse_args()


def sanitize_cutoff_language(text: str, *, date_text: str) -> str:
    out = str(text or "")
    replacements = [
        ("pre-cutoff literature", f"historical literature available before {date_text}"),
        ("pre-cutoff record", f"historical record available before {date_text}"),
        ("pre-cutoff indicators", f"historical indicators visible before {date_text}"),
        ("pre-cutoff evidence", f"historical evidence available before {date_text}"),
        ("pre-cutoff signals", f"historical signals available before {date_text}"),
        ("pre-cutoff", f"before {date_text}"),
        ("post-cutoff", "subsequent"),
        ("cutoff", date_text),
    ]
    for src, tgt in replacements:
        out = out.replace(src, tgt)
        out = out.replace(src.capitalize(), tgt[:1].upper() + tgt[1:])
    return " ".join(out.split())


def rewrite_prompt(*, public_row: Dict[str, Any], hidden_row: Dict[str, Any]) -> str:
    family = str(public_row.get("family") or "")
    subtype = str(public_row.get("subtype") or "")
    topic = str((hidden_row.get("public_metadata") or {}).get("topic_title") or (hidden_row.get("public_metadata") or {}).get("topic") or "")
    likely_bucket = str((hidden_row.get("ground_truth") or {}).get("target_venue_bucket") or ((hidden_row.get("ground_truth") or {}).get("venue_forecast") or {}).get("likely_bucket") or "")
    gt = hidden_row.get("ground_truth") or {}
    predicted_direction = str((gt.get("future_terminal") or {}).get("display_name") or "")
    if not predicted_direction:
        candidate_dirs = gt.get("candidate_directions") or []
        predicted_direction = ", ".join(str(x) for x in candidate_dirs[:2] if str(x).strip())
    return f"""You are polishing public benchmark tasks for a research benchmark.

Goal:
- Rewrite the title, question, and hidden reference answer into natural, formal, technical English.
- Keep the original intent unchanged.
- Do not leak any hidden ground truth.
- Do not mention internal metadata, post-cutoff papers, realized outcomes, or exact future statistics.
- Avoid awkward capitalization and repetitive phrasing.
- Keep the question benchmark-ready: clear, concise, and executable.
- Do not use the word "cutoff" anywhere.

Task metadata:
- Family: {family}
- Subtype: {subtype}
- Topic: {topic}
- Time cutoff: {public_row.get('time_cutoff')}
- Current title: {public_row.get('title')}
- Current question: {public_row.get('question')}
- Current hidden gold answer: {hidden_row.get('gold_answer')}
- Hidden venue bucket (for internal consistency only, do not reveal unless the current question already explicitly targets a venue bucket): {likely_bucket}
- Hidden target direction(s) for internal consistency only: {predicted_direction}

Family-specific guidance:
- If family = direction_forecasting, ask for one concrete next-step direction plus one likely venue bucket, but keep the venue categories generic.
- If family = strategic_research_planning, keep the venue-targeted framing explicit because the task is intentionally venue-conditioned.
- The hidden gold answer should read like a strong reference answer: directly state the conclusion, then explain the main reasons and technical rationale. Avoid meta phrasing such as "this forecast is supported by..." or "the post-cutoff window validates...".

Output JSON only:
{{
  "title": "...",
  "question": "...",
  "gold_answer": "..."
}}
"""


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    public_path = release_dir / "tasks.jsonl"
    hidden_path = release_dir / "tasks_hidden_eval.jsonl"
    internal_path = release_dir / "tasks_internal_full.jsonl"

    public_rows = iter_jsonl(public_path)
    hidden_rows = iter_jsonl(hidden_path)
    internal_rows = iter_jsonl(internal_path)

    hidden_by_id = {row["task_id"]: row for row in hidden_rows}
    internal_by_id = {row["task_id"]: row for row in internal_rows}

    fallback_client = None
    fallback_path = Path(args.fallback_llm_config) if args.fallback_llm_config else None
    if fallback_path and fallback_path.exists():
        fallback_client = OpenAICompatChatClient(load_openai_compat_config(fallback_path))
    client = FallbackOpenAICompatChatClient(
        OpenAICompatChatClient(load_openai_compat_config(Path(args.llm_config))),
        fallback_client,
    )

    touched = 0
    for public_row in public_rows:
        if str(public_row.get("subtype") or "") not in VENUE_SUBTYPES:
            continue
        hidden_row = hidden_by_id[public_row["task_id"]]
        obj = complete_json_object(
            client,
            [{"role": "user", "content": rewrite_prompt(public_row=public_row, hidden_row=hidden_row)}],
            temperature=0.0,
            timeout=180,
            max_tokens=900,
            transport_retries=2,
            max_parse_attempts=3,
        )
        date_text = "September 1, 2025"
        title = sanitize_cutoff_language(str(obj.get("title") or public_row.get("title") or "").strip(), date_text=date_text)
        question = sanitize_cutoff_language(str(obj.get("question") or public_row.get("question") or "").strip(), date_text=date_text)
        gold_answer = sanitize_cutoff_language(str(obj.get("gold_answer") or hidden_row.get("gold_answer") or "").strip(), date_text=date_text)
        public_row["title"] = title
        public_row["question"] = question
        hidden_row["title"] = title
        hidden_row["gold_answer"] = gold_answer
        internal_row = internal_by_id.get(hidden_row["internal_task_id"])
        if internal_row is not None:
            internal_row["title"] = title
            internal_row["question"] = question
            internal_row["draft_question"] = question
            internal_row["gold_answer"] = gold_answer
            internal_row["draft_reference_answer"] = gold_answer
        touched += 1
        print(f"polished {touched}: {public_row['task_id']}", flush=True)

    dump_jsonl(public_path, public_rows)
    dump_jsonl(hidden_path, hidden_rows)
    dump_jsonl(internal_path, internal_rows)
    print(json.dumps({"release_dir": str(release_dir), "polished_tasks": touched}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
