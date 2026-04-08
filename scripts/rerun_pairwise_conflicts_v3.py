from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.llm import OpenAICompatChatClient, complete_json_object, load_openai_compat_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun unstable pairwise benchmark comparisons to best-of-3 / best-of-5.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--base-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bestof3-conf-threshold", type=float, default=0.75)
    parser.add_argument("--bestof5-conf-threshold", type=float, default=0.65)
    parser.add_argument("--bestof5-tie-threshold", type=float, default=0.34)
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


def load_method_results(specs: Iterable[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    results_by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in specs:
        method, path_str = spec.split("=", 1)
        results_by_method[method.strip()] = {str(row["task_id"]): row for row in iter_jsonl(Path(path_str.strip()))}
    return results_by_method


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


def choose_orientation(seed: int, task_id: str, method_x: str, method_y: str, round_index: int) -> Tuple[str, str]:
    first, second = stable_order(seed + round_index, task_id, method_x, method_y)
    return first, second


def judge_pair(client: OpenAICompatChatClient, job: Dict[str, Any]) -> Dict[str, Any]:
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
        "round_source": "rerun",
        "round_index": job["round_index"],
        "target_k": job["target_k"],
        "trigger": job["trigger"],
    }


def analyze_group(
    rows: List[Dict[str, Any]],
    *,
    bestof3_conf_threshold: float,
    bestof5_conf_threshold: float,
    bestof5_tie_threshold: float,
) -> Tuple[int, str]:
    winners = [str(row.get("winner_method") or "tie") for row in rows]
    non_ties = [w for w in winners if w != "tie"]
    mean_conf = sum(float(row.get("confidence") or 0.0) for row in rows) / max(len(rows), 1)
    tie_share = winners.count("tie") / max(len(rows), 1)
    mixed_non_tie = len(set(non_ties)) > 1
    has_tie = "tie" in winners
    current = len(rows)

    if current < 3 and (has_tie or mixed_non_tie or mean_conf < bestof3_conf_threshold):
        return 3, "unstable_or_low_conf_to_bestof3"

    counts = Counter(non_ties)
    max_votes = max(counts.values(), default=0)
    no_majority = current >= 3 and max_votes < 2
    if current < 5 and current >= 3 and (has_tie or mixed_non_tie or mean_conf < bestof5_conf_threshold or tie_share >= bestof5_tie_threshold or no_majority):
        return 5, "persistent_conflict_to_bestof5"

    return current, "stable"


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / "pairwise_rerun_results.jsonl"

    public_by_id = load_public_tasks(release_dir)
    results_by_method = load_method_results(args.input)
    allowed_task_ids = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    all_rows = list(iter_jsonl(Path(args.base_jsonl)))
    if args.resume and out_jsonl.exists():
        all_rows.extend(list(iter_jsonl(out_jsonl)))

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        task_id = str(row.get("task_id") or "")
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        grouped[str(row["comparison_key"])].append(row)

    jobs: List[Dict[str, Any]] = []
    completed_instance_ids = set()
    if args.resume and out_jsonl.exists():
        completed_instance_ids = {str(row.get("instance_id") or "") for row in iter_jsonl(out_jsonl)}

    for comparison_key, rows in sorted(grouped.items()):
        task_id = str(rows[0]["task_id"])
        if args.task_limit is not None:
            prefix_task_num = int(task_id.split("-")[-1])
            if prefix_task_num > args.task_limit:
                continue
        target_k, trigger = analyze_group(
            rows,
            bestof3_conf_threshold=args.bestof3_conf_threshold,
            bestof5_conf_threshold=args.bestof5_conf_threshold,
            bestof5_tie_threshold=args.bestof5_tie_threshold,
        )
        current = len(rows)
        if target_k <= current:
            continue
        methods = sorted({str(rows[0]["method_a"]), str(rows[0]["method_b"])})
        method_x, method_y = methods[0], methods[1]
        public_task = public_by_id[task_id]
        family = str(public_task.get("family") or "")
        domain = str(public_task.get("domain") or "")
        for round_index in range(current + 1, target_k + 1):
            a_method, b_method = choose_orientation(args.seed, task_id, method_x, method_y, round_index)
            instance_id = f"{comparison_key}::reround_{round_index:02d}"
            if instance_id in completed_instance_ids:
                continue
            jobs.append(
                {
                    "instance_id": instance_id,
                    "comparison_key": comparison_key,
                    "task_id": task_id,
                    "family": family,
                    "domain": domain,
                    "orientation": f"reround_{round_index:02d}",
                    "method_a": a_method,
                    "method_b": b_method,
                    "row_a": results_by_method[a_method][task_id],
                    "row_b": results_by_method[b_method][task_id],
                    "public_task": public_task,
                    "round_index": round_index,
                    "target_k": target_k,
                    "trigger": trigger,
                }
            )

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
                    f"[pairwise-rerun] {idx}/{total} {row['comparison_key']} round={row['round_index']} winner={row['winner_method']}",
                    flush=True,
                )

    summary = {
        "base_jsonl": str(args.base_jsonl),
        "generated_reruns": total,
        "results_path": str(out_jsonl),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
