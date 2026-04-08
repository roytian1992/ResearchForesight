from __future__ import annotations

import argparse
import json
from pathlib import Path

from researchworld.baseline_runner import PUBLIC_DOMAIN_TO_ID, load_release_tasks
from researchworld.llm import OpenAICompatChatClient, load_openai_compat_config
from researchworld.research_arc_v2 import ResearchArcV2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ResearchArcV2 on RTL benchmark tasks.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--answer-llm-config", default="configs/llm/mimo_flash.local.yaml")
    parser.add_argument("--critic-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--task-ids-file", default=None)
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    answer_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.answer_llm_config)))
    critic_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.critic_llm_config)))
    agent = ResearchArcV2(answer_client=answer_client, critic_client=critic_client)

    tasks = load_release_tasks(release_dir)
    allowed_task_ids = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    domain_filter = set(args.domains) if args.domains else None
    family_filter = set(args.families) if args.families else None

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
        print(
            f"[ResearchArcV2] {idx}/{len(rows)} {task['task_id']} "
            f"domain={domain_id} family={task['family']}",
            flush=True,
        )
        result = agent.run_task(task=task, domain_id=domain_id)
        outputs.append(
            {
                "task_id": task["task_id"],
                "family": task["family"],
                "domain": task["domain"],
                "domain_id": domain_id,
                "agent": "ResearchArcV2",
                "title": task["title"],
                "question": task["question"],
                "time_cutoff": task["time_cutoff"],
                "answer": result["answer"],
                "trace": result,
            }
        )

    results_path = out_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "agent": "ResearchArcV2",
        "task_count": len(outputs),
        "results_path": str(results_path),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
