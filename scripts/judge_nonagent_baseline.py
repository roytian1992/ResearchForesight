from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from researchworld.baseline_runner import aggregate_scores
from researchworld.llm import (
    OpenAICompatChatClient,
    complete_json_object,
    extract_json_object,
    load_openai_compat_config,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_hidden_map(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = load_jsonl(path)
    return {str(row["task_id"]): row for row in rows}


def judge_single(
    client: OpenAICompatChatClient,
    *,
    public_result: Dict[str, Any],
    hidden_task: Dict[str, Any],
) -> Dict[str, Any]:
    public_task = {
        "task_id": public_result.get("task_id"),
        "family": public_result.get("family"),
        "domain": public_result.get("domain"),
        "title": public_result.get("title"),
        "question": public_result.get("question"),
        "time_cutoff": public_result.get("time_cutoff"),
    }
    prompt = f"""Evaluate a baseline answer for the research benchmark.

Public task:
{json.dumps(public_task, ensure_ascii=False, indent=2)}

Reference signals:
- gold_answer: {hidden_task.get('gold_answer')}
- expected_answer_points: {json.dumps(hidden_task.get('expected_answer_points') or [], ensure_ascii=False)}
- rubric: {json.dumps(hidden_task.get('evaluation_rubric') or {}, ensure_ascii=False)}

Candidate answer:
{public_result.get('answer') or ''}

Return JSON only:
{{
  "overall_score": float,
  "dimension_scores": {{"dimension_name": float}},
  "verdict": "strong" | "acceptable" | "weak",
  "reasoning": "brief explanation"
}}

Scores must be in [0, 1]."""
    return complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict benchmark judge. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        transport_retries=2,
        max_parse_attempts=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge completed non-agent baseline answers.")
    parser.add_argument("--input-results", required=True)
    parser.add_argument("--hidden-eval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    args = parser.parse_args()

    results = load_jsonl(Path(args.input_results))
    hidden_by_id = build_hidden_map(Path(args.hidden_eval))
    client = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))

    judged_rows: List[Dict[str, Any]] = []
    total = len(results)
    for idx, row in enumerate(results, start=1):
        task_id = str(row.get("task_id") or "")
        hidden = hidden_by_id.get(task_id)
        if hidden is None:
            continue
        print(
            f"[judge] {idx}/{total} task={task_id} baseline={row.get('baseline')} family={row.get('family')}",
            flush=True,
        )
        judged = dict(row)
        judged["judge"] = judge_single(client, public_result=row, hidden_task=hidden)
        judged_rows.append(judged)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results_judged.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in judged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_results": str(args.input_results),
        "task_count": len(judged_rows),
        "score_summary": aggregate_scores(judged_rows),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
