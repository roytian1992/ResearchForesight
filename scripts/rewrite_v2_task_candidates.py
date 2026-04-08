from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.llm import OpenAICompatChatClient, complete_json_object, load_openai_compat_config
from researchworld.prompting import YAMLPromptLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite benchmark v2 task candidates with an OpenAI-compatible LLM.")
    parser.add_argument("--input", default=str(ROOT / "data" / "task_candidates" / "all_candidates.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "task_candidates" / "all_candidates.rewritten.jsonl"))
    parser.add_argument("--errors", default=str(ROOT / "data" / "task_candidates" / "all_candidates.rewrite_errors.jsonl"))
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    return parser.parse_args()


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("task_id")) for row in iter_jsonl(path) if row.get("task_id")}


def candidate_brief(row: Dict[str, Any]) -> Dict[str, Any]:
    support = row.get("support_context") or {}
    gt = row.get("ground_truth") or {}
    public = row.get("public_metadata") or {}
    return {
        "task_id": row.get("task_id"),
        "family": row.get("family"),
        "subtype": row.get("subtype"),
        "horizon": row.get("horizon"),
        "domain": row.get("domain"),
        "public_topic": public.get("topic"),
        "public_topic_title": public.get("topic_title"),
        "public_future_themes": public.get("future_themes") or [],
        "time_context": row.get("time_context"),
        "node_description": support.get("node_description"),
        "historical_stats": support.get("historical_stats"),
        "top_limitations": (support.get("top_limitations") or [])[:3],
        "top_future_work": (support.get("top_future_work") or [])[:3],
        "history_representative_papers": (support.get("history_representative_papers") or [])[:3],
        "future_q4_representative_papers": (support.get("future_q4_representative_papers") or [])[:2],
        "future_q1_representative_papers": (support.get("future_q1_representative_papers") or [])[:2],
        "draft_question": row.get("draft_question"),
        "draft_reference_answer": row.get("draft_reference_answer"),
        "ground_truth_core": {
            "trajectory": gt.get("trajectory"),
            "future_half_stats": gt.get("future_half_stats"),
            "future_descendants": public.get("future_themes") or [],
            "target_window_stats": gt.get("target_window_stats"),
            "structure_coverage": gt.get("structure_coverage"),
        },
        "evaluation_rubric": row.get("evaluation_rubric"),
    }


def rewrite_one(row: Dict[str, Any], prompt_loader: YAMLPromptLoader, llm: OpenAICompatChatClient, args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    prompt = prompt_loader.render("benchmark_v2/rewrite_candidate_task", task_values={"candidate_json": candidate_brief(row)})
    raw_text = ""
    last_error = ""
    for _ in range(args.max_retries + 1):
        try:
            obj = complete_json_object(
                llm,
                [{"role": "user", "content": prompt}],
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=1800,
                max_parse_attempts=3,
            )
            row = dict(row)
            row["rewrite"] = obj
            row["title"] = str(obj.get("title") or row.get("title") or "")
            row["question"] = str(obj.get("question") or row.get("draft_question") or "")
            row["gold_answer"] = str(obj.get("gold_answer") or row.get("draft_reference_answer") or "")
            row["expected_answer_points"] = [x for x in (obj.get("expected_answer_points") or []) if isinstance(x, str)]
            row["rewrite_leakage_check"] = obj.get("leakage_check") or {}
            return True, row
        except Exception as exc:
            last_error = str(exc)
    return False, {"task_id": row.get("task_id"), "error": last_error, "raw_text": raw_text}


def rows_to_process(path: Path, seen: set[str], limit: int) -> list[Dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id in seen:
            continue
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = Path(args.errors)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    seen = load_seen(output_path) if args.resume else set()
    rows = rows_to_process(input_path, seen, args.limit)
    llm = OpenAICompatChatClient(load_openai_compat_config(args.llm_config))
    prompt_loader = YAMLPromptLoader(ROOT / "prompts")

    with open(output_path, "a" if args.resume else "w", encoding="utf-8") as out_handle, open(
        error_path, "a" if args.resume else "w", encoding="utf-8"
    ) as err_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(rewrite_one, row, prompt_loader, llm, args): row["task_id"] for row in rows}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                ok, row = future.result()
                handle = out_handle if ok else err_handle
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done += 1
                print("rewritten", done, "/", len(rows), row.get("task_id"))


if __name__ == "__main__":
    main()
