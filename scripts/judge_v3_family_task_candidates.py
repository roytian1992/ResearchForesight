from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
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


PROMPT_BY_FAMILY = {
    "direction_forecasting": "benchmark_v3/judge_direction_forecasting",
    "bottleneck_opportunity_discovery": "benchmark_v3/judge_bottleneck_opportunity_discovery",
    "strategic_research_planning": "benchmark_v3/judge_strategic_research_planning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge family-specific benchmark v3 task candidates.")
    parser.add_argument("--input", default=str(ROOT / "data" / "task_candidates_v3" / "all_candidates.rewritten.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "task_candidates_v3" / "all_candidates.judged.jsonl"))
    parser.add_argument("--errors", default=str(ROOT / "data" / "task_candidates_v3" / "all_candidates.judge_errors.jsonl"))
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--fallback-llm-config", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--accept-threshold", type=float, default=0.78)
    parser.add_argument("--borderline-threshold", type=float, default=0.66)
    parser.add_argument("--transport-retries", type=int, default=2)
    return parser.parse_args()


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("task_id")) for row in iter_jsonl(path) if row.get("task_id")}


def candidate_brief(row: Dict[str, Any]) -> Dict[str, Any]:
    family = str(row.get("family") or "")
    support = row.get("support_context") or {}
    gt = row.get("ground_truth") or {}
    public = row.get("public_metadata") or {}
    brief = {
        "task_id": row.get("task_id"),
        "family": family,
        "subtype": row.get("subtype"),
        "domain": row.get("domain"),
        "title": row.get("title") or row.get("task_id"),
        "question": row.get("question") or row.get("draft_question"),
        "gold_answer": row.get("gold_answer") or row.get("draft_reference_answer"),
        "expected_answer_points": row.get("expected_answer_points") or [],
        "topic": public.get("topic"),
        "topic_title": public.get("topic_title"),
        "future_themes": public.get("future_themes") or [],
        "time_context": row.get("time_context"),
        "historical_stats": support.get("historical_stats"),
        "heuristic_score": (row.get("quality_signals") or {}).get("heuristic_score"),
        "rewrite_leakage_check": row.get("rewrite_leakage_check") or {},
        "rewrite_surface_check": row.get("rewrite_surface_check") or {},
    }
    if family == "direction_forecasting":
        brief["ground_truth_core"] = {
            "trajectory": gt.get("trajectory"),
            "future_half_stats": gt.get("future_half_stats"),
            "emergent_descendants": public.get("future_themes") or [],
        }
    elif family == "bottleneck_opportunity_discovery":
        brief["top_limitations"] = [x.get("name") for x in (support.get("top_limitations") or [])[:4] if x.get("name")]
        brief["top_future_work"] = [x.get("direction") for x in (support.get("top_future_work") or [])[:4] if x.get("direction")]
        brief["history_structure_coverage"] = support.get("history_structure_coverage")
        brief["ground_truth_core"] = {
            "historical_limitation_signals": brief["top_limitations"],
            "future_themes": public.get("future_themes") or [],
            "future_half_stats": gt.get("future_half_stats"),
        }
    elif family == "strategic_research_planning":
        brief["candidate_directions"] = support.get("candidate_directions") or []
        brief["ranking_axes"] = support.get("ranking_axes")
        brief["ground_truth_core"] = {
            "future_themes": public.get("future_themes") or [],
            "target_window_stats": gt.get("target_window_stats"),
            "planning_priority_score": gt.get("planning_priority_score"),
        }
    return brief


def judge_one(
    row: Dict[str, Any],
    prompt_loader: YAMLPromptLoader,
    llm: OpenAICompatChatClient,
    fallback_llm: OpenAICompatChatClient | None,
    args: argparse.Namespace,
) -> Tuple[bool, Dict[str, Any]]:
    family = str(row.get("family") or "")
    prompt_id = PROMPT_BY_FAMILY[family]
    prompt = prompt_loader.render(prompt_id, task_values={"candidate_json": candidate_brief(row)})
    raw_text = ""
    last_error = ""
    clients = [("primary", llm)]
    if fallback_llm is not None:
        clients.append(("fallback", fallback_llm))
    for client_name, client in clients:
        for attempt in range(args.max_retries + 1):
            try:
                obj = complete_json_object(
                    client,
                    [{"role": "user", "content": prompt}],
                    temperature=args.temperature,
                    timeout=args.timeout,
                    max_tokens=1400,
                    transport_retries=args.transport_retries,
                    max_parse_attempts=3,
                )
                out = dict(row)
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
                out["judge"] = judge
                out["judge"]["decision"] = decision
                out["judge"]["overall_score"] = overall
                out["judge_model_source"] = client_name
                return True, out
            except Exception as exc:
                last_error = f"{client_name}: {exc}"
                time.sleep(min(4.0, 0.8 * (attempt + 1)))
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
    fallback_llm = OpenAICompatChatClient(load_openai_compat_config(args.fallback_llm_config)) if args.fallback_llm_config else None
    prompt_loader = YAMLPromptLoader(ROOT / "prompts")

    with open(output_path, "a" if args.resume else "w", encoding="utf-8") as out_handle, open(
        error_path, "a" if args.resume else "w", encoding="utf-8"
    ) as err_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(judge_one, row, prompt_loader, llm, fallback_llm, args): row["task_id"] for row in rows
            }
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
