from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.llm import OpenAICompatChatClient, complete_json_object, load_openai_compat_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct best-of-k pairwise judging on benchmark v3 answers.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--fallback-judge-llm-config", default="configs/llm/qwen_235b.local.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-rounds", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--escalate-conf-threshold", type=float, default=0.65)
    parser.add_argument("--job-retries", type=int, default=3)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Method spec in form method_id=path/to/results.jsonl",
    )
    return parser.parse_args()


def load_public_tasks(release_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["task_id"]): row for row in iter_jsonl(release_dir / "tasks.jsonl")}


def load_method_results(specs: Iterable[str]) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]]]:
    methods: List[str] = []
    results_by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --input spec: {spec}")
        method, path_str = spec.split("=", 1)
        method = method.strip()
        path = Path(path_str.strip())
        if not method or not path.exists():
            raise ValueError(f"Invalid method/path: {spec}")
        methods.append(method)
        results_by_method[method] = {str(row["task_id"]): row for row in iter_jsonl(path)}
    return methods, results_by_method


def family_dimensions(family: str) -> List[str]:
    dims = ["task_fit", "insightfulness", "specificity", "clarity", "strategic_value"]
    if family == "bottleneck_opportunity_discovery":
        dims.append("bottleneck_opportunity_linkage")
    elif family == "direction_forecasting":
        dims.append("trajectory_plausibility")
    elif family == "strategic_research_planning":
        dims.append("agenda_quality")
    return dims


def base_orientation(seed: int, task_id: str, method_x: str, method_y: str) -> Tuple[str, str]:
    key = f"{seed}:{task_id}:{method_x}:{method_y}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (method_x, method_y) if int(digest, 16) % 2 == 0 else (method_y, method_x)


def round_orientation(seed: int, task_id: str, method_x: str, method_y: str, round_index: int) -> Tuple[str, str]:
    first, second = base_orientation(seed, task_id, method_x, method_y)
    if round_index % 2 == 1:
        return first, second
    return second, first


def judge_round(
    client: OpenAICompatChatClient,
    fallback_client: OpenAICompatChatClient | None,
    *,
    public_task: Dict[str, Any],
    family: str,
    method_a: str,
    method_b: str,
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
) -> Dict[str, Any]:
    dims = family_dimensions(family)
    prompt = f"""# Role
You are a Lead Research Auditor conducting a blind peer review of two competing technical responses. Your goal is to identify which answer provides superior "Research Intelligence" and "Strategic Utility."

# Core Evaluation Philosophy
- Substance over Surface: Prefer a concise, high-signal answer over a verbose, generic one.
- The "So What?" Test: Does the answer identify critical bottlenecks or non-obvious trade-offs?
- Temporal Integrity: Strictly penalize any future leakage (knowledge beyond the {public_task.get("time_cutoff")}) or hindsight bias.
- Task Alignment: Evaluate against the specific requirements of the "{family}" task family.

# Input Context
- Task ID: {public_task.get("task_id")}
- Domain: {public_task.get("domain")}
- Time Cutoff: {public_task.get("time_cutoff")}
- Strategic Question: {public_task.get("question")}

# Dimensions to Judge
{json.dumps(dims, ensure_ascii=False)}

# Rules of Engagement
1. Strictly Evidence-Based: Compare Answer A and Answer B only on their internal logic and specificity.
2. Anti-Generic Bias: Heavily penalize consultant-speak (vague, universally true statements that lack domain-specific friction).
3. Winner Selection: Avoid tie unless the answers are structurally and qualitatively indistinguishable. A tie is a failure of discrimination.

# Evaluation Data
- Answer A: {str(row_a.get('answer') or '')}
- Answer B: {str(row_b.get('answer') or '')}

# Output Format (Strict JSON)
{{
  "winner": "A | B | tie",
  "confidence": 0.0,
  "reason": "Identify the inflection point that separated the winner from the loser. Be specific about logic, temporal discipline, or strategic depth.",
  "dimension_votes": {{
    "dimension_name": "A | B | tie"
  }}
}}
"""
    messages = [
        {"role": "system", "content": "You are a strict pairwise benchmark judge. Return JSON only."},
        {"role": "user", "content": prompt},
    ]
    try:
        obj = complete_json_object(
            client,
            messages,
            response_format={"type": "json_object"},
            max_tokens=900,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
    except Exception as exc:
        if fallback_client is None:
            raise
        err = str(exc).lower()
        fallback_triggers = ["httperror", "urlerror", "timeout", "rate limit", "429", "connection reset", "temporarily unavailable"]
        if not any(token in err for token in fallback_triggers):
            raise
        obj = complete_json_object(
            fallback_client,
            messages,
            response_format={"type": "json_object"},
            max_tokens=900,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
    winner = str(obj.get("winner") or "tie").strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    winner_method = "tie"
    if winner == "A":
        winner_method = method_a
    elif winner == "B":
        winner_method = method_b
    dim_votes_raw = obj.get("dimension_votes") or {}
    dim_votes = {}
    for dim in dims:
        value = str(dim_votes_raw.get(dim) or "tie").strip()
        dim_votes[dim] = value if value in {"A", "B", "tie"} else "tie"
    return {
        "winner_label": winner,
        "winner_method": winner_method,
        "confidence": round(float(obj.get("confidence") or 0.0), 4),
        "reason": str(obj.get("reason") or "").strip(),
        "dimension_votes": dim_votes,
    }


def needs_escalation(round_rows: List[Dict[str, Any]], *, min_rounds: int, conf_threshold: float) -> bool:
    if len(round_rows) < min_rounds:
        return True
    winners = [str(row.get("winner_method") or "tie") for row in round_rows]
    non_ties = [w for w in winners if w != "tie"]
    counts = {w: non_ties.count(w) for w in set(non_ties)}
    majority = len(round_rows) // 2 + 1
    majority_found = any(v >= majority for v in counts.values())
    mean_conf = sum(float(row.get("confidence") or 0.0) for row in round_rows) / max(len(round_rows), 1)
    mirror_non_ties = [winners[0], winners[1]] if len(winners) >= 2 else winners
    mirror_non_ties = [w for w in mirror_non_ties if w != "tie"]
    mirror_consistent = len(set(mirror_non_ties)) <= 1
    tie_count = winners.count("tie")
    return (not majority_found) or (mean_conf < conf_threshold) or (not mirror_consistent) or (tie_count >= 1)


def decide_final(round_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    winners = [str(row.get("winner_method") or "tie") for row in round_rows]
    non_ties = [w for w in winners if w != "tie"]
    counts = {w: non_ties.count(w) for w in set(non_ties)}
    majority = len(round_rows) // 2 + 1
    final_winner = "tie"
    for method, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if count >= majority:
            final_winner = method
            break
    mean_conf = sum(float(row.get("confidence") or 0.0) for row in round_rows) / max(len(round_rows), 1)
    first_two = winners[:2]
    mirror_consistent = len(set([w for w in first_two if w != "tie"])) <= 1
    return {
        "winner_method": final_winner,
        "wins": {k: int(v) for k, v in counts.items()} | {"tie": int(winners.count("tie"))},
        "rounds_run": len(round_rows),
        "mean_confidence": round(mean_conf, 4),
        "mirror_consistent": mirror_consistent,
        "unstable": final_winner == "tie" or not mirror_consistent,
    }


def run_comparison(
    client: OpenAICompatChatClient,
    fallback_client: OpenAICompatChatClient | None,
    *,
    seed: int,
    public_task: Dict[str, Any],
    method_x: str,
    method_y: str,
    row_x: Dict[str, Any],
    row_y: Dict[str, Any],
    min_rounds: int,
    max_rounds: int,
    escalate_conf_threshold: float,
) -> Dict[str, Any]:
    task_id = str(public_task["task_id"])
    family = str(public_task.get("family") or "")
    comparison_key = f"{task_id}::{min(method_x, method_y)}__vs__{max(method_x, method_y)}"
    raw_rounds = []
    for round_index in range(1, max_rounds + 1):
        method_a, method_b = round_orientation(seed, task_id, method_x, method_y, round_index)
        row_a = row_x if method_a == method_x else row_y
        row_b = row_y if method_b == method_y else row_x
        judged = judge_round(
            client,
            fallback_client,
            public_task=public_task,
            family=family,
            method_a=method_a,
            method_b=method_b,
            row_a=row_a,
            row_b=row_b,
        )
        raw_rounds.append(
            {
                "round_index": round_index,
                "orientation": f"round_{round_index}",
                "method_a": method_a,
                "method_b": method_b,
                **judged,
            }
        )
        if round_index >= min_rounds and not needs_escalation(
            raw_rounds,
            min_rounds=min_rounds,
            conf_threshold=escalate_conf_threshold,
        ):
            break
    final = decide_final(raw_rounds)
    return {
        "comparison_key": comparison_key,
        "task_id": task_id,
        "family": family,
        "domain": str(public_task.get("domain") or ""),
        "methods": [method_x, method_y],
        "rounds": raw_rounds,
        **final,
    }


def run_comparison_with_retries(
    client: OpenAICompatChatClient,
    fallback_client: OpenAICompatChatClient | None,
    *,
    seed: int,
    public_task: Dict[str, Any],
    method_x: str,
    method_y: str,
    row_x: Dict[str, Any],
    row_y: Dict[str, Any],
    min_rounds: int,
    max_rounds: int,
    escalate_conf_threshold: float,
    job_retries: int,
) -> Dict[str, Any]:
    last_error = ""
    for attempt in range(job_retries + 1):
        try:
            return run_comparison(
                client,
                fallback_client,
                seed=seed,
                public_task=public_task,
                method_x=method_x,
                method_y=method_y,
                row_x=row_x,
                row_y=row_y,
                min_rounds=min_rounds,
                max_rounds=max_rounds,
                escalate_conf_threshold=escalate_conf_threshold,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < job_retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"run_comparison failed after retries: task={public_task.get('task_id')} pair={method_x}__vs__{method_y}; error={last_error}"
    )


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_jsonl = output_dir / "pairwise_bestofk_results.jsonl"
    raw_jsonl = output_dir / "pairwise_bestofk_rounds.jsonl"
    error_jsonl = output_dir / "pairwise_bestofk_errors.jsonl"

    public_by_id = load_public_tasks(release_dir)
    methods, results_by_method = load_method_results(args.input)
    allowed_task_ids = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    task_ids = sorted(set.intersection(*(set(results_by_method[m].keys()) for m in methods)))
    if allowed_task_ids is not None:
        task_ids = [task_id for task_id in task_ids if task_id in allowed_task_ids]
    if args.task_limit is not None:
        task_ids = task_ids[: args.task_limit]

    jobs = []
    for task_id in task_ids:
        public_task = public_by_id.get(task_id)
        if not public_task:
            continue
        for method_x, method_y in combinations(methods, 2):
            jobs.append(
                {
                    "comparison_key": f"{task_id}::{min(method_x, method_y)}__vs__{max(method_x, method_y)}",
                    "public_task": public_task,
                    "method_x": method_x,
                    "method_y": method_y,
                    "row_x": results_by_method[method_x][task_id],
                    "row_y": results_by_method[method_y][task_id],
                }
            )

    completed = set()
    if args.resume and final_jsonl.exists():
        completed = {str(row.get("comparison_key") or "") for row in iter_jsonl(final_jsonl)}
    if completed:
        jobs = [job for job in jobs if job["comparison_key"] not in completed]

    judge_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))
    fallback_client = None
    fallback_path = Path(args.fallback_judge_llm_config)
    if fallback_path.exists():
        fallback_client = OpenAICompatChatClient(load_openai_compat_config(fallback_path))
    final_mode = "a" if args.resume and final_jsonl.exists() else "w"
    raw_mode = "a" if args.resume and raw_jsonl.exists() else "w"
    err_mode = "a" if args.resume and error_jsonl.exists() else "w"
    total = len(jobs)
    with (
        final_jsonl.open(final_mode, encoding="utf-8") as final_handle,
        raw_jsonl.open(raw_mode, encoding="utf-8") as raw_handle,
        error_jsonl.open(err_mode, encoding="utf-8") as err_handle,
    ):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    run_comparison_with_retries,
                    judge_client,
                    fallback_client,
                    seed=args.seed,
                    public_task=job["public_task"],
                    method_x=job["method_x"],
                    method_y=job["method_y"],
                    row_x=job["row_x"],
                    row_y=job["row_y"],
                    min_rounds=args.min_rounds,
                    max_rounds=args.max_rounds,
                    escalate_conf_threshold=args.escalate_conf_threshold,
                    job_retries=args.job_retries,
                ): job
                for job in jobs
            }
            for idx, future in enumerate(as_completed(future_map), start=1):
                job = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:
                    err_row = {
                        "comparison_key": job["comparison_key"],
                        "task_id": job["public_task"].get("task_id"),
                        "family": job["public_task"].get("family"),
                        "domain": job["public_task"].get("domain"),
                        "methods": [job["method_x"], job["method_y"]],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    err_handle.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    err_handle.flush()
                    print(f"[pairwise-bestofk][error] {idx}/{total} {job['comparison_key']} {err_row['error']}", flush=True)
                    continue
                for round_row in row["rounds"]:
                    raw_handle.write(
                        json.dumps(
                            {
                                "comparison_key": row["comparison_key"],
                                "task_id": row["task_id"],
                                "family": row["family"],
                                "domain": row["domain"],
                                "methods": row["methods"],
                                **round_row,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                raw_handle.flush()
                final_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                final_handle.flush()
                print(
                    f"[pairwise-bestofk] {idx}/{total} {row['comparison_key']} winner={row['winner_method']} rounds={row['rounds_run']}",
                    flush=True,
                )

    summary = {
        "task_count": len(task_ids),
        "method_count": len(methods),
        "methods": methods,
        "comparison_count": sum(1 for _ in iter_jsonl(final_jsonl)),
        "rounds_path": str(raw_jsonl),
        "results_path": str(final_jsonl),
        "errors_path": str(error_jsonl),
        "min_rounds": args.min_rounds,
        "max_rounds": args.max_rounds,
        "job_retries": args.job_retries,
        "fallback_judge_llm_config": str(fallback_path) if fallback_client is not None else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
