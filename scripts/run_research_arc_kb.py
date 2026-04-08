from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.baseline_runner import PUBLIC_DOMAIN_TO_ID, aggregate_scores, judge_answer, load_hidden_eval, load_release_tasks
from researchworld.llm import OpenAICompatChatClient, load_openai_compat_config
from researchworld.offline_kb import OfflineKnowledgeBase
from researchworld.research_arc_kb import ResearchArcKB


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KB-grounded ResearchArc on RTL benchmark tasks.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--kb-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--answer-llm-config", default="configs/llm/mimo_flash.local.yaml")
    parser.add_argument("--critic-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--task-ids-file", default=None)
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    kb_dir = Path(args.kb_dir) if args.kb_dir else (release_dir / "kb")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    answer_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.answer_llm_config)))
    critic_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.critic_llm_config)))
    kb = OfflineKnowledgeBase(kb_dir)
    agent = ResearchArcKB(kb=kb, answer_client=answer_client, critic_client=critic_client)
    hidden_by_id = load_hidden_eval(release_dir) if not args.skip_judge else {}

    tasks = load_release_tasks(release_dir)
    domain_filter = set(args.domains) if args.domains else None
    family_filter = set(args.families) if args.families else None
    allowed_task_ids = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    rows = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        domain_id = PUBLIC_DOMAIN_TO_ID.get(str(task.get("domain") or "").strip())
        if not domain_id:
            continue
        if domain_filter and domain_id not in domain_filter:
            continue
        if family_filter and str(task.get("family") or "") not in family_filter:
            continue
        rows.append((task, domain_id))
    if args.task_limit is not None:
        rows = rows[: args.task_limit]

    outputs = []
    for idx, (task, domain_id) in enumerate(rows, start=1):
        print(f"[ResearchArcKB] {idx}/{len(rows)} {task['task_id']} domain={domain_id} family={task['family']}", flush=True)
        result = agent.run_task(task=task, domain_id=domain_id)
        row = {
            "task_id": task["task_id"],
            "family": task["family"],
            "domain": task["domain"],
            "domain_id": domain_id,
            "agent": "ResearchArcKB",
            "title": task["title"],
            "question": task["question"],
            "time_cutoff": task["time_cutoff"],
            "answer": result["answer"],
            "trace": result,
        }
        if not args.skip_judge:
            hidden_task = hidden_by_id.get(str(task["task_id"]))
            if hidden_task is not None:
                row["judge"] = judge_answer(critic_client, public_task=task, hidden_task=hidden_task, candidate_answer=result["answer"])
        outputs.append(row)

    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "agent": "ResearchArcKB",
        "release_dir": str(release_dir),
        "kb_dir": str(kb_dir),
        "task_count": len(outputs),
        "results_path": str(results_path),
        "score_summary": aggregate_scores(outputs),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
