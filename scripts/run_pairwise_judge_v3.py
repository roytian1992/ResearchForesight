from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
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
    parser = argparse.ArgumentParser(description="Run round-robin pairwise judging on benchmark v3 answers.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Method spec in form method_id=path/to/results.jsonl",
    )
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--task-ids-file", default=None)
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
        rows = {str(row["task_id"]): row for row in iter_jsonl(path)}
        methods.append(method)
        results_by_method[method] = rows
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


def stable_order(seed: int, task_id: str, method_x: str, method_y: str) -> Tuple[str, str]:
    key = f"{seed}:{task_id}:{method_x}:{method_y}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (method_x, method_y) if int(digest, 16) % 2 == 0 else (method_y, method_x)


def build_jobs(
    *,
    public_by_id: Dict[str, Dict[str, Any]],
    methods: List[str],
    results_by_method: Dict[str, Dict[str, Dict[str, Any]]],
    seed: int,
    mirror: bool,
    allowed_task_ids: set[str] | None,
    task_limit: int | None,
) -> List[Dict[str, Any]]:
    task_ids = sorted(set.intersection(*(set(results_by_method[m].keys()) for m in methods)))
    if allowed_task_ids is not None:
        task_ids = [task_id for task_id in task_ids if task_id in allowed_task_ids]
    if task_limit is not None:
        task_ids = task_ids[:task_limit]

    jobs: List[Dict[str, Any]] = []
    for task_id in task_ids:
        public_task = public_by_id.get(task_id)
        if not public_task:
            continue
        family = str(public_task.get("family") or "")
        domain = str(public_task.get("domain") or "")
        for method_x, method_y in combinations(methods, 2):
            first, second = stable_order(seed, task_id, method_x, method_y)
            for order_tag, pair in [("forward", (first, second))]:
                a_method, b_method = pair
                jobs.append(
                    {
                        "instance_id": f"{task_id}::{method_x}__vs__{method_y}::{order_tag}",
                        "comparison_key": f"{task_id}::{min(method_x, method_y)}__vs__{max(method_x, method_y)}",
                        "task_id": task_id,
                        "family": family,
                        "domain": domain,
                        "orientation": order_tag,
                        "method_a": a_method,
                        "method_b": b_method,
                        "row_a": results_by_method[a_method][task_id],
                        "row_b": results_by_method[b_method][task_id],
                        "public_task": public_task,
                    }
                )
            if mirror:
                jobs.append(
                    {
                        "instance_id": f"{task_id}::{method_x}__vs__{method_y}::mirror",
                        "comparison_key": f"{task_id}::{min(method_x, method_y)}__vs__{max(method_x, method_y)}",
                        "task_id": task_id,
                        "family": family,
                        "domain": domain,
                        "orientation": "mirror",
                        "method_a": second,
                        "method_b": first,
                        "row_a": results_by_method[second][task_id],
                        "row_b": results_by_method[first][task_id],
                        "public_task": public_task,
                    }
                )
    return jobs


def judge_pair(
    client: OpenAICompatChatClient,
    job: Dict[str, Any],
) -> Dict[str, Any]:
    public_task = job["public_task"]
    family = str(job["family"] or "")
    dims = family_dimensions(family)
    prompt = f"""You are judging two benchmark answers in a blind head-to-head comparison.

Task:
{json.dumps({
    "task_id": public_task.get("task_id"),
    "family": public_task.get("family"),
    "domain": public_task.get("domain"),
    "time_cutoff": public_task.get("time_cutoff"),
    "title": public_task.get("title"),
    "question": public_task.get("question"),
}, ensure_ascii=False, indent=2)}

Evaluation rules:
- Compare only the quality of Answer A vs Answer B for this task.
- Prefer answers that are more faithful to the task, more insightful, more specific, and more strategically useful.
- Penalize visible hallucination, future leakage beyond the cutoff, vague generic language, and poor task-family fit.
- Use "tie" only when the two answers are genuinely close.
- Do not infer or reward any hidden method identity.

Dimension labels to vote on:
{json.dumps(dims, ensure_ascii=False)}

Answer A:
{str(job['row_a'].get('answer') or '')}

Answer B:
{str(job['row_b'].get('answer') or '')}

Return JSON only:
{{
  "winner": "A" | "B" | "tie",
  "confidence": 0.0,
  "reason": "brief justification",
  "dimension_votes": {{
    "task_fit": "A" | "B" | "tie"
  }}
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict pairwise benchmark judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=900,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
    )
    winner = str(obj.get("winner") or "tie").strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    dim_votes_raw = obj.get("dimension_votes") or {}
    dim_votes = {}
    for dim in dims:
        value = str(dim_votes_raw.get(dim) or "tie").strip()
        dim_votes[dim] = value if value in {"A", "B", "tie"} else "tie"
    canonical_winner = "tie"
    if winner == "A":
        canonical_winner = str(job["method_a"])
    elif winner == "B":
        canonical_winner = str(job["method_b"])
    return {
        "instance_id": job["instance_id"],
        "comparison_key": job["comparison_key"],
        "task_id": job["task_id"],
        "family": job["family"],
        "domain": job["domain"],
        "orientation": job["orientation"],
        "method_a": job["method_a"],
        "method_b": job["method_b"],
        "winner_label": winner,
        "winner_method": canonical_winner,
        "confidence": round(float(obj.get("confidence") or 0.0), 4),
        "reason": str(obj.get("reason") or "").strip(),
        "dimension_votes": dim_votes,
    }


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / "pairwise_results.jsonl"

    public_by_id = load_public_tasks(release_dir)
    methods, results_by_method = load_method_results(args.input)
    allowed_task_ids = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    jobs = build_jobs(
        public_by_id=public_by_id,
        methods=methods,
        results_by_method=results_by_method,
        seed=args.seed,
        mirror=args.mirror,
        allowed_task_ids=allowed_task_ids,
        task_limit=args.task_limit,
    )

    completed_instance_ids = set()
    if args.resume and out_jsonl.exists():
        for row in iter_jsonl(out_jsonl):
            completed_instance_ids.add(str(row.get("instance_id") or ""))
    if completed_instance_ids:
        jobs = [job for job in jobs if job["instance_id"] not in completed_instance_ids]

    judge_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))
    write_mode = "a" if args.resume and out_jsonl.exists() else "w"
    total = len(jobs)
    with out_jsonl.open(write_mode, encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(judge_pair, judge_client, job): job for job in jobs}
            for idx, future in enumerate(as_completed(future_map), start=1):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(
                    f"[pairwise] {idx}/{total} task={row['task_id']} {row['method_a']} vs {row['method_b']} winner={row['winner_method']}",
                    flush=True,
                )

    summary = {
        "task_count": len({job["task_id"] for job in build_jobs(
            public_by_id=public_by_id,
            methods=methods,
            results_by_method=results_by_method,
            seed=args.seed,
            mirror=args.mirror,
            allowed_task_ids=allowed_task_ids,
            task_limit=args.task_limit,
        )}),
        "method_count": len(methods),
        "methods": methods,
        "comparison_count": sum(1 for _ in iter_jsonl(out_jsonl)),
        "mirror": bool(args.mirror),
        "results_path": str(out_jsonl),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
