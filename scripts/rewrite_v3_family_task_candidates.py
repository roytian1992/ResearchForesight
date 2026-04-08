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
    "direction_forecasting": "benchmark_v3/rewrite_direction_forecasting",
    "bottleneck_opportunity_discovery": "benchmark_v3/rewrite_bottleneck_opportunity_discovery",
    "strategic_research_planning": "benchmark_v3/rewrite_strategic_research_planning",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite family-specific benchmark v3 task candidates.")
    parser.add_argument("--input", default=str(ROOT / "data" / "task_candidates_v3" / "all_candidates.jsonl"))
    parser.add_argument("--output", default=str(ROOT / "data" / "task_candidates_v3" / "all_candidates.rewritten.jsonl"))
    parser.add_argument("--errors", default=str(ROOT / "data" / "task_candidates_v3" / "all_candidates.rewrite_errors.jsonl"))
    parser.add_argument("--llm-config", required=True)
    parser.add_argument("--fallback-llm-config", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=180)
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
        "horizon": row.get("horizon"),
        "topic": public.get("topic"),
        "topic_title": public.get("topic_title"),
        "time_context": row.get("time_context"),
        "draft_question": row.get("draft_question"),
        "draft_reference_answer": row.get("draft_reference_answer"),
        "historical_stats": support.get("historical_stats"),
        "heuristic_score": (row.get("quality_signals") or {}).get("heuristic_score"),
    }
    if family == "direction_forecasting":
        brief["history_chain"] = [
            x.get("display_name") for x in (support.get("history_chain") or []) if x.get("display_name")
        ]
        brief["history_representative_papers"] = [
            x.get("title") for x in (support.get("history_representative_papers") or [])[:4] if x.get("title")
        ]
        brief["ground_truth_core"] = {
            "future_themes": public.get("future_themes") or [],
            "trajectory": gt.get("trajectory"),
            "future_half_stats": gt.get("future_half_stats"),
        }
    elif family == "bottleneck_opportunity_discovery":
        brief["history_reading_set"] = [
            x.get("title") for x in (support.get("history_reading_set") or [])[:6] if x.get("title")
        ]
        brief["top_limitations"] = [x.get("name") for x in (support.get("top_limitations") or [])[:4] if x.get("name")]
        brief["top_future_work"] = [
            x.get("direction") for x in (support.get("top_future_work") or [])[:4] if x.get("direction")
        ]
        brief["ground_truth_core"] = {
            "future_themes": public.get("future_themes") or [],
            "future_half_stats": gt.get("future_half_stats"),
            "history_structure_coverage": support.get("history_structure_coverage"),
        }
    elif family == "strategic_research_planning":
        brief["candidate_directions"] = support.get("candidate_directions") or []
        brief["history_representative_papers"] = [
            x.get("title") for x in (support.get("history_representative_papers") or [])[:5] if x.get("title")
        ]
        brief["ranking_axes"] = support.get("ranking_axes")
        brief["ground_truth_core"] = {
            "future_themes": public.get("future_themes") or [],
            "target_window_stats": gt.get("target_window_stats"),
            "planning_priority_score": gt.get("planning_priority_score"),
        }
    return brief


def rewrite_one(
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
                    max_tokens=1800,
                    transport_retries=args.transport_retries,
                    max_parse_attempts=3,
                )
                out = dict(row)
                out["rewrite"] = obj
                out["title"] = str(obj.get("title") or row.get("title") or "")
                out["question"] = str(obj.get("question") or row.get("draft_question") or "")
                out["gold_answer"] = str(obj.get("gold_answer") or row.get("draft_reference_answer") or "")
                out["expected_answer_points"] = [x for x in (obj.get("expected_answer_points") or []) if isinstance(x, str)]
                out["rewrite_leakage_check"] = obj.get("leakage_check") or {}
                out["rewrite_surface_check"] = obj.get("surface_check") or {}
                out["rewrite_model_source"] = client_name
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
                executor.submit(rewrite_one, row, prompt_loader, llm, fallback_llm, args): row["task_id"] for row in rows
            }
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
