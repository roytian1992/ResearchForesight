from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.baseline_runner import PUBLIC_DOMAIN_TO_ID, load_release_tasks
from researchworld.coi_agent_offline import CoIAgentOffline
from researchworld.fulltext_cache import LocalFulltextCache
from researchworld.llm import FallbackOpenAICompatChatClient, OpenAICompatChatClient, load_openai_compat_config
from researchworld.offline_kb import OfflineKnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoI-Agent-Offline on RTL benchmark tasks.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--kb-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--main-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--cheap-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--fallback-llm-config", default="configs/llm/qwen_235b.local.yaml")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument("--fulltext-cache-root", default="", help="Optional local fulltext cache root.")
    parser.add_argument("--allow-fulltext-fetch", action="store_true", help="Allow fetching versioned PDFs into fulltext cache when missing.")
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    kb_dir = Path(args.kb_dir) if args.kb_dir else (release_dir / "kb")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fallback_path = Path(args.fallback_llm_config) if args.fallback_llm_config else None
    fallback_client = OpenAICompatChatClient(load_openai_compat_config(fallback_path)) if (fallback_path and fallback_path.exists()) else None
    main_client = FallbackOpenAICompatChatClient(
        OpenAICompatChatClient(load_openai_compat_config(Path(args.main_llm_config))),
        fallback_client,
    )
    cheap_client = FallbackOpenAICompatChatClient(
        OpenAICompatChatClient(load_openai_compat_config(Path(args.cheap_llm_config))),
        fallback_client,
    )
    kb = OfflineKnowledgeBase(kb_dir)
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
    results_path = out_dir / "results.jsonl"
    summary_path = out_dir / "summary.json"
    completed_task_ids = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            task_id = str(row.get("task_id") or "")
            if task_id:
                completed_task_ids.add(task_id)
                outputs.append(row)
    if completed_task_ids:
        print(
            f"[CoI-Agent-Offline] resume mode: found {len(completed_task_ids)} completed tasks in {results_path}",
            flush=True,
        )
    pending_rows = [(task, domain_id) for task, domain_id in rows if str(task.get("task_id") or "") not in completed_task_ids]
    if not pending_rows:
        summary = {
            "agent": "CoI-Agent-Offline",
            "task_count": len(outputs),
            "total_requested_tasks": len(rows),
            "results_path": str(results_path),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    for idx, (task, domain_id) in enumerate(pending_rows, start=1):
        fulltext_cache = None
        if args.fulltext_cache_root:
            fulltext_cache = LocalFulltextCache(
                domain_id=domain_id,
                papers_jsonl=ROOT / "data" / "domains" / domain_id / "interim" / "papers_merged.publication_enriched.semanticscholar.jsonl",
                cache_root=Path(args.fulltext_cache_root),
            )
        agent = CoIAgentOffline(
            kb=kb,
            main_client=main_client,
            cheap_client=cheap_client,
            fulltext_cache=fulltext_cache,
            allow_fulltext_fetch=bool(args.allow_fulltext_fetch),
        )
        print(
            f"[CoI-Agent-Offline] {idx}/{len(pending_rows)} pending | completed={len(outputs)} total={len(rows)} "
            f"{task['task_id']} domain={domain_id} family={task['family']}",
            flush=True,
        )
        result = agent.run_task(task=task, domain_id=domain_id)
        row = {
            "task_id": task["task_id"],
            "family": task["family"],
            "domain": task["domain"],
            "domain_id": domain_id,
            "agent": "CoI-Agent-Offline",
            "title": task["title"],
            "question": task["question"],
            "time_cutoff": task["time_cutoff"],
            "answer": result["answer"],
            "trace": result,
        }
        outputs.append(row)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = {
            "agent": "CoI-Agent-Offline",
            "task_count": len(outputs),
            "total_requested_tasks": len(rows),
            "results_path": str(results_path),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    summary = {
        "agent": "CoI-Agent-Offline",
        "task_count": len(outputs),
        "total_requested_tasks": len(rows),
        "results_path": str(results_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
