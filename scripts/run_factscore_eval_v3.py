from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.factscore_eval_v3 import FactScoreV3Config, evaluate_answer_factscore_v3
from researchworld.llm import OpenAICompatChatClient, load_openai_compat_config
from researchworld.offline_kb import OfflineKnowledgeBase, PUBLIC_DOMAIN_TO_ID


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def infer_domain_id(row: dict) -> str:
    domain_id = str(row.get("domain_id") or "").strip()
    if domain_id:
        return domain_id
    domain = str(row.get("domain") or "").strip()
    return PUBLIC_DOMAIN_TO_ID.get(domain, domain)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark-aware FactScore v3.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--history-kb-dir", required=True)
    parser.add_argument("--future-kb-dir", required=True)
    parser.add_argument("--hidden-eval-v3", required=True)
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-claims", type=int, default=8)
    parser.add_argument("--task-limit", type=int, default=None)
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(results_path))
    if args.task_limit is not None:
        rows = rows[: args.task_limit]

    gt_by_task = {row["task_id"]: row for row in iter_jsonl(Path(args.hidden_eval_v3))}
    history_kb = OfflineKnowledgeBase(Path(args.history_kb_dir))
    future_kb = OfflineKnowledgeBase(Path(args.future_kb_dir))
    judge_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))
    cfg = FactScoreV3Config(max_claims=args.max_claims)

    outputs = []
    for idx, row in enumerate(rows, start=1):
        task_id = str(row.get("task_id") or "")
        gt_row = gt_by_task.get(task_id)
        if not gt_row:
            continue
        row = dict(row)
        row["domain_id"] = infer_domain_id(row)
        print(f"[factscore_v3] {idx}/{len(rows)} {task_id} domain={row.get('domain_id')}", flush=True)
        fact = evaluate_answer_factscore_v3(
            history_kb=history_kb,
            future_kb=future_kb,
            judge_client=judge_client,
            result_row=row,
            gt_row=gt_row,
            cfg=cfg,
        )
        outputs.append(
            {
                "task_id": task_id,
                "domain_id": row.get("domain_id"),
                "family": row.get("family"),
                "factscore_v3": fact,
            }
        )

    out_jsonl = output_dir / "factscore_v3_results.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_tasks = len(outputs)
    benchmark_scores = [float(row["factscore_v3"]["benchmark_factscore"]) for row in outputs]
    precision_scores = [float(row["factscore_v3"]["precision_score"]) for row in outputs]
    coverage_scores = [float(row["factscore_v3"]["coverage_score"]) for row in outputs]
    summary = {
        "results_jsonl": str(results_path),
        "task_count": total_tasks,
        "mean_benchmark_factscore": round(sum(benchmark_scores) / total_tasks, 4) if total_tasks else 0.0,
        "mean_precision_score": round(sum(precision_scores) / total_tasks, 4) if total_tasks else 0.0,
        "mean_coverage_score": round(sum(coverage_scores) / total_tasks, 4) if total_tasks else 0.0,
        "output_jsonl": str(out_jsonl),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
