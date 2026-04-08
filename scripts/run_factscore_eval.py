from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.factscore_eval import FactScoreConfig, evaluate_answer_factscore
from researchworld.llm import OpenAICompatChatClient, load_openai_compat_config
from researchworld.offline_kb import OfflineKnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FactScore-style factual-grounding evaluation on benchmark outputs.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--kb-dir", required=True)
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-claims", type=int, default=8)
    parser.add_argument("--task-limit", type=int, default=None)
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in results_path.open("r", encoding="utf-8") if line.strip()]
    if args.task_limit is not None:
        rows = rows[: args.task_limit]

    kb = OfflineKnowledgeBase(Path(args.kb_dir))
    judge_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))
    cfg = FactScoreConfig(max_claims=args.max_claims)

    outputs = []
    for idx, row in enumerate(rows, start=1):
        print(f"[factscore] {idx}/{len(rows)} {row.get('task_id')} domain={row.get('domain_id')}", flush=True)
        fact = evaluate_answer_factscore(kb=kb, judge_client=judge_client, row=row, cfg=cfg)
        outputs.append(
            {
                "task_id": row.get("task_id"),
                "domain_id": row.get("domain_id"),
                "family": row.get("family"),
                "factscore": fact,
            }
        )

    out_jsonl = output_dir / "factscore_results.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_tasks = len(outputs)
    task_scores = [float(row["factscore"]["factscore"]) for row in outputs]
    claim_counts = [int(row["factscore"]["claim_count"]) for row in outputs]
    summary = {
        "results_jsonl": str(results_path),
        "task_count": total_tasks,
        "mean_factscore": round(sum(task_scores) / total_tasks, 4) if total_tasks else 0.0,
        "mean_claim_count": round(sum(claim_counts) / total_tasks, 4) if total_tasks else 0.0,
        "output_jsonl": str(out_jsonl),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
