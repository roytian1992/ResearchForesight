from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.llm import OpenAICompatChatClient, complete_json_object, load_openai_compat_config
from researchworld.prompting import YAMLPromptLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge screening for benchmark v2 candidates.")
    parser.add_argument("--input", default=str(ROOT / "data" / "task_candidates" / "all_candidates.rewritten.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "task_candidates" / "all_candidates.judged.jsonl"))
    parser.add_argument("--errors", default=str(ROOT / "data" / "task_candidates" / "all_candidates.judge_errors.jsonl"))
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--accept-threshold", type=float, default=0.78)
    parser.add_argument("--borderline-threshold", type=float, default=0.66)
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
        "title": row.get("title") or row.get("task_id"),
        "question": row.get("question") or row.get("draft_question"),
        "gold_answer": row.get("gold_answer") or row.get("draft_reference_answer"),
        "expected_answer_points": row.get("expected_answer_points") or [],
        "historical_stats": support.get("historical_stats"),
        "history_structure_coverage": support.get("history_structure_coverage"),
        "top_limitations": (support.get("top_limitations") or [])[:3],
        "ground_truth_core": {
            "trajectory": gt.get("trajectory"),
            "future_half_stats": gt.get("future_half_stats"),
            "target_window_stats": gt.get("target_window_stats"),
            "future_descendants": public.get("future_themes") or [],
            "structure_coverage": gt.get("structure_coverage"),
        },
        "rewrite_leakage_check": row.get("rewrite_leakage_check") or {},
        "heuristic_score": (row.get("quality_signals") or {}).get("heuristic_score"),
    }


def judge_one(row: Dict[str, Any], prompt_loader: YAMLPromptLoader, llm: OpenAICompatChatClient, args: argparse.Namespace) -> Tuple[bool, Dict[str, Any]]:
    prompt = prompt_loader.render("benchmark_v2/judge_candidate_task", task_values={"candidate_json": candidate_brief(row)})
    raw_text = ""
    last_error = ""
    for _ in range(args.max_retries + 1):
        try:
            obj = complete_json_object(
                llm,
                [{"role": "user", "content": prompt}],
                temperature=args.temperature,
                timeout=args.timeout,
                max_tokens=1400,
                max_parse_attempts=3,
            )
            row = dict(row)
            judge = obj
            overall = float(judge.get("overall_score") or 0.0)
            decision = str(judge.get("decision") or "").strip().lower()
            if not decision:
                if overall >= args.accept_threshold:
                    decision = "accept"
                elif overall >= args.borderline_threshold:
                    decision = "borderline"
                else:
                    decision = "reject"
            row["judge"] = judge
            row["judge"]["decision"] = decision
            row["judge"]["overall_score"] = overall
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
            futures = {executor.submit(judge_one, row, prompt_loader, llm, args): row["task_id"] for row in rows}
            done = 0
            for future in concurrent.futures.as_completed(futures):
                ok, row = future.result()
                handle = out_handle if ok else err_handle
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                done += 1
                print("judged", done, "/", len(rows), row.get("task_id"))


if __name__ == "__main__":
    main()
