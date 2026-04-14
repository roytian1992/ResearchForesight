from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.llm import FallbackOpenAICompatChatClient, OpenAICompatChatClient, complete_json_object, load_openai_compat_config


STYLE_BANK = {
    "bottleneck_opportunity_discovery": [
        "evidence_first",
        "mechanism_driven",
        "problem_to_opportunity",
        "historical_synthesis",
    ],
    "direction_forecasting": [
        "trajectory_focused",
        "forward_looking",
        "historical_signal_first",
        "research_dynamics",
    ],
    "strategic_research_planning": [
        "research_manager",
        "portfolio_selection",
        "priority_setting",
        "decision_memo",
    ],
    "venue_aware_research_positioning": [
        "submission_strategy",
        "positioning_memo",
        "venue_fit",
        "program_committee_view",
    ],
}


def iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-polish benchmark release language while preserving task contracts.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--fallback-llm-config", default="")
    parser.add_argument("--families", default="")
    parser.add_argument("--task-ids-file", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--max-parse-attempts", type=int, default=3)
    parser.add_argument("--audit-path", default="")
    parser.add_argument("--rewrite-gold-answer", action="store_true")
    return parser.parse_args()


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def family_filter_arg(text: str) -> set[str]:
    return {x.strip() for x in str(text or "").split(",") if x.strip()}


def load_task_id_filter(path_text: str) -> Optional[set[str]]:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def style_hint(task_id: str, family: str) -> str:
    bank = STYLE_BANK.get(family) or ["neutral"]
    digest = hashlib.md5(task_id.encode("utf-8")).digest()[0]
    return bank[digest % len(bank)]


def normalized(text: Any) -> str:
    text = normalize_space(str(text or "")).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_space(text)


def contains_any(text: str, phrases: List[str]) -> bool:
    return any(normalized(phrase) in text for phrase in phrases)


def render_candidate_directions(candidate_directions: List[str]) -> str:
    return "; ".join(f"({idx}) {item}" for idx, item in enumerate(candidate_directions, start=1))


def enforce_contract(payload: Dict[str, Any], title: str, question: str) -> tuple[str, str]:
    family = str(payload.get("family") or "")
    subtype = str(payload.get("subtype") or "")
    candidate_directions = [normalize_space(x) for x in (payload.get("candidate_directions") or []) if normalize_space(x)]
    target_venue_bucket = normalize_space(payload.get("target_venue_bucket") or "")

    out_title = normalize_space(title)
    out_question = normalize_space(question)
    out_question_norm = normalized(out_question)

    if candidate_directions:
        missing = [x for x in candidate_directions if normalized(x) not in out_question_norm]
        if missing:
            out_question = normalize_space(
                f"{out_question} Consider only the following candidate directions: {render_candidate_directions(candidate_directions)}."
            )
            out_question_norm = normalized(out_question)

    if family in {"strategic_research_planning"} or (family == "venue_aware_research_positioning" and subtype == "venue_targeted_planning"):
        if not contains_any(
            out_question_norm,
            [
                "do not introduce new candidate directions",
                "do not add new candidate directions",
                "do not propose new candidate directions",
                "rank only the listed options",
                "consider only the listed options",
                "only the listed directions should be ranked",
                "only the listed directions may be ranked",
                "do not introduce new directions",
                "no additional directions should be introduced",
            ],
        ):
            out_question = normalize_space(f"{out_question} Rank only the listed options; do not introduce new candidate directions.")
            out_question_norm = normalized(out_question)

    if family == "venue_aware_research_positioning" and target_venue_bucket and normalized(target_venue_bucket) not in out_question_norm:
        out_question = normalize_space(f"{out_question} The target venue bucket is {target_venue_bucket}.")
        out_question_norm = normalized(out_question)

    if family == "direction_forecasting" and "accelerating, fragmenting, steady, or cooling" not in out_question_norm:
        out_question = normalize_space(
            f"{out_question} Use exactly one trajectory label from: accelerating, fragmenting, steady, or cooling."
        )
        out_question_norm = normalized(out_question)

    if family == "bottleneck_opportunity_discovery":
        if "bottleneck" not in out_question_norm:
            out_question = normalize_space(f"{out_question} Identify one unresolved technical bottleneck.")
            out_question_norm = normalized(out_question)
        if "opportunity" not in out_question_norm:
            out_question = normalize_space(
                f"{out_question} Then describe one concrete downstream research opportunity that would become viable if the bottleneck were addressed."
            )
            out_question_norm = normalized(out_question)

    if family == "venue_aware_research_positioning" and subtype == "venue_aware_direction_forecast":
        if not contains_any(
            out_question_norm,
            [
                "single concrete next-step research direction",
                "one concrete next-step research direction",
                "single next-step research direction",
                "one next-step research direction",
            ],
        ):
            out_question = normalize_space(
                f"{out_question} Identify exactly one concrete next-step research direction."
            )
            out_question_norm = normalized(out_question)
        if not contains_any(
            out_question_norm,
            [
                "one most likely top-tier venue bucket",
                "single most likely top-tier venue bucket",
                "one likely top-tier venue bucket",
                "single likely top-tier venue bucket",
            ],
        ):
            out_question = normalize_space(
                f"{out_question} State one most likely top-tier venue bucket for that direction."
            )
    return out_title, out_question


def validate_rewrite(payload: Dict[str, Any], title: str, question: str) -> List[str]:
    errors: List[str] = []
    family = str(payload.get("family") or "")
    subtype = str(payload.get("subtype") or "")
    question_norm = normalized(question)
    title_norm = normalized(title)
    candidate_directions = [normalize_space(x) for x in (payload.get("candidate_directions") or []) if normalize_space(x)]
    target_venue_bucket = normalize_space(payload.get("target_venue_bucket") or "")
    trajectory_label = normalize_space(payload.get("trajectory_label") or "")

    if not title_norm:
        errors.append("empty_title")
    if not question_norm:
        errors.append("empty_question")

    if candidate_directions:
        missing = [x for x in candidate_directions if normalized(x) not in question_norm]
        if missing:
            errors.append(f"missing_candidates:{missing}")
        if family == "strategic_research_planning" and not contains_any(
            question_norm,
            [
                "do not introduce new candidate directions",
                "do not add new candidate directions",
                "do not propose new candidate directions",
                "rank only the listed options",
                "consider only the listed options",
                "only the listed directions should be ranked",
                "only the listed directions may be ranked",
                "do not introduce new directions",
                "no additional directions should be introduced",
            ],
        ):
            errors.append("missing_candidate_constraint")

    if target_venue_bucket and family == "venue_aware_research_positioning" and normalized(target_venue_bucket) not in question_norm:
        errors.append(f"missing_venue_bucket:{target_venue_bucket}")

    if family == "direction_forecasting":
        label_set = "accelerating, fragmenting, steady, or cooling"
        if label_set not in question_norm:
            errors.append("missing_trajectory_label_set")
        if trajectory_label and trajectory_label not in {"accelerating", "fragmenting", "steady", "cooling"}:
            errors.append(f"unexpected_trajectory_label:{trajectory_label}")

    if family == "bottleneck_opportunity_discovery":
        if "bottleneck" not in question_norm:
            errors.append("missing_bottleneck_requirement")
        if "opportunity" not in question_norm:
            errors.append("missing_opportunity_requirement")

    if family == "venue_aware_research_positioning" and subtype == "venue_aware_direction_forecast":
        if not contains_any(
            question_norm,
            [
                "single concrete next-step research direction",
                "one concrete next-step research direction",
                "single next-step research direction",
                "one next-step research direction",
            ],
        ):
            errors.append("missing_single_direction_requirement")
        if not contains_any(
            question_norm,
            [
                "one most likely top-tier venue bucket",
                "single most likely top-tier venue bucket",
                "one likely top-tier venue bucket",
                "single likely top-tier venue bucket",
            ],
        ):
            errors.append("missing_single_venue_bucket_requirement")

    return errors


def rewrite_payload(public_row: Dict[str, Any], hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> Dict[str, Any]:
    family = str(public_row.get("family") or "")
    subtype = str(public_row.get("subtype") or "")
    support = trace_row.get("support_context") or {}
    gt = hidden_row.get("ground_truth") or trace_row.get("ground_truth") or {}
    public_meta = hidden_row.get("public_metadata") or trace_row.get("public_metadata") or {}
    return {
        "task_id": public_row.get("task_id"),
        "family": family,
        "subtype": subtype,
        "domain": public_row.get("domain"),
        "horizon": public_row.get("horizon"),
        "time_cutoff": public_row.get("time_cutoff"),
        "title": public_row.get("title"),
        "question": public_row.get("question"),
        "gold_answer": hidden_row.get("gold_answer"),
        "expected_answer_points": hidden_row.get("expected_answer_points") or [],
        "topic_title": public_meta.get("topic_title") or public_meta.get("topic") or "",
        "future_themes": public_meta.get("future_themes") or [],
        "candidate_directions": support.get("candidate_directions") or gt.get("candidate_directions") or [],
        "target_venue_bucket": support.get("target_venue_bucket") or gt.get("target_venue_bucket") or (gt.get("venue_forecast") or {}).get("likely_bucket") or "",
        "target_venue_name": support.get("target_venue_name") or gt.get("target_venue_name") or (gt.get("venue_forecast") or {}).get("likely_venue") or "",
        "trajectory_label": ((gt.get("trajectory") or {}).get("trajectory_label") or ""),
    }


def rewrite_prompt(payload: Dict[str, Any], style: str) -> str:
    family = str(payload.get("family") or "")
    subtype = str(payload.get("subtype") or "")
    return f"""You are polishing benchmark task language for a research benchmark.

Goal:
- Rewrite the public title and public question so they sound like carefully edited benchmark tasks rather than repetitive templates.
- Optionally rewrite the hidden gold answer so it reads like a strong reference answer rather than a canned explanation.
- Preserve the exact task contract.

Hard constraints:
1. Do NOT change the family, subtype, topic, time boundary, or evaluation intent.
2. Do NOT leak hidden future outcomes into the public title or public question.
3. If candidate_directions are provided, the rewritten public question must preserve all of them explicitly and must still instruct the model to rank only those listed options.
4. If a target_venue_bucket is provided for a venue-targeted planning task, the rewritten public question must explicitly mention that venue bucket.
5. For direction forecasting tasks, keep the trajectory label set explicit: accelerating, fragmenting, steady, or cooling.
6. For bottleneck tasks, the question must still require one bottleneck plus one downstream opportunity.
7. For venue-aware direction forecasting, the question must still require exactly one concrete direction plus one likely venue bucket.
8. Keep the public question concise enough to feel natural. Remove boilerplate repetition when possible.
9. Vary surface style according to the style hint, but never at the cost of clarity.
10. Return JSON only.

Style hint:
- Use the style "{style}".
- This should influence tone and sentence shape, not task semantics.
- Avoid generic openings like "Using only literature available by..." unless they are needed for clarity; feel free to vary phrasing while keeping the same temporal restriction.

Task payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Family-specific guidance:
- strategic_research_planning:
  Preserve ranking and candidate-list constraints. The wording should feel like a real agenda-setting decision problem.
- venue_aware_research_positioning:
  Make venue fit sound concrete and methodological, not like prestige-chasing.
- direction_forecasting:
  Make the future call sound discriminative and technically anchored.
- bottleneck_opportunity_discovery:
  Emphasize mechanism and causal linkage between the bottleneck and the opportunity.

Output schema:
{{
  "title": "...",
  "question": "...",
  "gold_answer": "...",
  "notes": {{
    "style_applied": "{style}",
    "preserved_candidate_count": 0,
    "preserved_venue_bucket": true
  }}
}}
"""


def build_client(args: argparse.Namespace):
    primary = OpenAICompatChatClient(load_openai_compat_config(Path(args.llm_config)))
    fallback = None
    if args.fallback_llm_config:
        fallback_path = Path(args.fallback_llm_config)
        if fallback_path.exists():
            fallback = OpenAICompatChatClient(load_openai_compat_config(fallback_path))
    return FallbackOpenAICompatChatClient(primary, fallback)


def apply_polish(
    *,
    public_row: Dict[str, Any],
    hidden_row: Dict[str, Any],
    trace_row: Dict[str, Any],
    internal_row: Optional[Dict[str, Any]],
    client,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    payload = rewrite_payload(public_row, hidden_row, trace_row)
    style = style_hint(str(public_row.get("task_id") or ""), str(public_row.get("family") or ""))
    original_title = str(public_row.get("title") or "")
    original_question = str(public_row.get("question") or "")
    original_gold_answer = str(hidden_row.get("gold_answer") or "")
    obj = complete_json_object(
        client,
        [{"role": "user", "content": rewrite_prompt(payload, style)}],
        temperature=args.temperature,
        timeout=args.timeout,
        max_tokens=1400,
        transport_retries=args.transport_retries,
        max_parse_attempts=args.max_parse_attempts,
    )
    proposed_title = normalize_space(str(obj.get("title") or original_title))
    proposed_question = normalize_space(str(obj.get("question") or original_question))
    title, question = enforce_contract(payload, proposed_title, proposed_question)
    gold_answer = normalize_space(str(obj.get("gold_answer") or original_gold_answer))
    notes = obj.get("notes") or {}
    validation_errors = validate_rewrite(payload, title, question)
    accepted = not validation_errors

    if not accepted:
        title = original_title
        question = original_question
        gold_answer = original_gold_answer

    public_row["title"] = title
    public_row["question"] = question
    hidden_row["title"] = title
    if args.rewrite_gold_answer and accepted:
        hidden_row["gold_answer"] = gold_answer
    trace_row["language_polish"] = {
        "style_hint": style,
        "notes": notes,
        "accepted": accepted,
        "validation_errors": validation_errors,
        "proposed_title": proposed_title,
        "proposed_question": proposed_question,
        "final_title": title,
        "final_question": question,
    }
    if internal_row is not None:
        internal_row["title"] = title
        internal_row["question"] = question
        if "draft_question" in internal_row:
            internal_row["draft_question"] = question
        if args.rewrite_gold_answer and accepted:
            internal_row["gold_answer"] = gold_answer
            if "draft_reference_answer" in internal_row:
                internal_row["draft_reference_answer"] = gold_answer
    return {
        "task_id": public_row.get("task_id"),
        "family": public_row.get("family"),
        "subtype": public_row.get("subtype"),
        "style_hint": style,
        "accepted": accepted,
        "validation_errors": validation_errors,
        "original_title": original_title,
        "proposed_title": proposed_title,
        "title": title,
        "original_question": original_question,
        "proposed_question": proposed_question,
        "question": question,
        "original_gold_answer_changed": bool(args.rewrite_gold_answer and accepted and gold_answer != original_gold_answer),
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    public_path = release_dir / "tasks.jsonl"
    hidden_path = release_dir / "tasks_hidden_eval.jsonl"
    trace_path = release_dir / "tasks_build_trace.jsonl"
    internal_path = release_dir / "tasks_internal_full.jsonl"

    public_rows = iter_jsonl(public_path)
    hidden_rows = iter_jsonl(hidden_path)
    trace_rows = iter_jsonl(trace_path)
    internal_rows = iter_jsonl(internal_path) if internal_path.exists() else []

    hidden_by_id = {row["task_id"]: row for row in hidden_rows}
    trace_by_id = {row["task_id"]: row for row in trace_rows}
    internal_by_id = {row["task_id"]: row for row in internal_rows}

    families = family_filter_arg(args.families)
    task_id_filter = load_task_id_filter(args.task_ids_file)
    client = build_client(args)

    audits: List[Dict[str, Any]] = []
    touched = 0
    for public_row in public_rows:
        task_id = str(public_row.get("task_id") or "")
        if families and str(public_row.get("family") or "") not in families:
            continue
        if task_id_filter is not None and task_id not in task_id_filter:
            continue
        if args.limit and touched >= args.limit:
            break
        hidden_row = hidden_by_id.get(task_id)
        trace_row = trace_by_id.get(task_id)
        if not hidden_row or not trace_row:
            continue
        audit = apply_polish(
            public_row=public_row,
            hidden_row=hidden_row,
            trace_row=trace_row,
            internal_row=internal_by_id.get(task_id),
            client=client,
            args=args,
        )
        audits.append(audit)
        touched += 1
        print(f"polished {touched}: {task_id}", flush=True)

    dump_jsonl(public_path, public_rows)
    dump_jsonl(hidden_path, hidden_rows)
    dump_jsonl(trace_path, trace_rows)
    if internal_rows:
        dump_jsonl(internal_path, internal_rows)

    audit_path = Path(args.audit_path) if args.audit_path else (release_dir / "language_polish_audit.json")
    dump_json(audit_path, {"release_dir": str(release_dir), "polished_tasks": touched, "audits": audits})
    print(json.dumps({"release_dir": str(release_dir), "polished_tasks": touched, "audit_path": str(audit_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
