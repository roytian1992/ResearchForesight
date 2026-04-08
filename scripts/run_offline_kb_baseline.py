from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.baseline_runner import (
    aggregate_scores,
    judge_answer,
    load_hidden_eval,
    load_release_tasks,
)
from researchworld.llm import (
    FallbackOpenAICompatChatClient,
    OpenAICompatChatClient,
    load_openai_compat_config,
)
from researchworld.offline_kb import PUBLIC_DOMAIN_TO_ID, OfflineKnowledgeBase, clip_text, merge_multi_query_results
from researchworld.research_arc_kb import extract_focus_text


def render_evidence(rows: List[Dict[str, Any]]) -> str:
    parts = []
    for row in rows:
        head = [f"[{row['evidence_id']}] {row.get('paper_title') or ''}"]
        if row.get("section_title"):
            head.append(f"section={row['section_title']}")
        if row.get("venue"):
            head.append(f"venue={row['venue']}")
        if row.get("citations") is not None:
            head.append(f"citations={row['citations']}")
        parts.append(" | ".join(head))
        parts.append(row.get("snippet") or "")
        parts.append("")
    return "\n".join(parts).strip()


def build_hybrid_evidence(task: Dict[str, Any], domain) -> Dict[str, Any]:
    queries = [task["question"], task["title"], extract_focus_text(task)]
    paper_hits = merge_multi_query_results(
        domain.paper_retriever(cutoff_date=str(task.get("time_cutoff") or "").strip() or None),
        queries,
        top_k_per_query=8,
        limit=12,
    )
    evidence = []
    for idx, (doc, scores) in enumerate(paper_hits, start=1):
        paper = domain.get_paper(doc.paper_id) or {}
        pub = paper.get("publication") or {}
        evidence.append(
            {
                "evidence_id": f"P{idx}",
                "paper_id": doc.paper_id,
                "paper_title": doc.title,
                "venue": pub.get("venue_name"),
                "citations": pub.get("citation_count"),
                "published_date": paper.get("published_date"),
                "snippet": clip_text(doc.text, 1400),
                "scores": scores,
            }
        )
    return {"retrieval_mode": "kb_hybrid_paper", "queries": queries, "retrieved": evidence}


def answer_task(
    client: OpenAICompatChatClient,
    *,
    task: Dict[str, Any],
    mode: str,
    evidence_packet: Optional[Dict[str, Any]] = None,
) -> str:
    if mode == "native":
        prompt = f"""You are answering an offline research benchmark.

Task ID: {task['task_id']}
Family: {task['family']}
Domain: {task['domain']}
Horizon: {task['horizon']}
Time cutoff: {task['time_cutoff']}
Title: {task['title']}
Question:
{task['question']}

Constraints:
- You do not have access to any retrieved reference material.
- Answer only from your own reasoning and parametric knowledge.
- Do not claim access to future papers after the cutoff.
- If uncertain, state the uncertainty explicitly.

Write a concise but substantive research answer."""
    else:
        evidence_block = render_evidence(list(evidence_packet.get("retrieved") or []))
        prompt = f"""You are answering an offline research benchmark.

Task ID: {task['task_id']}
Family: {task['family']}
Domain: {task['domain']}
Horizon: {task['horizon']}
Time cutoff: {task['time_cutoff']}
Title: {task['title']}
Question:
{task['question']}

Constraints:
- Use only the provided evidence.
- Do not claim access to future papers beyond the cutoff.
- Make a concrete conclusion instead of generic trend language.
- Cite evidence inline using the provided evidence ids like [P3].
- If the evidence is insufficient, say what remains uncertain.

Retrieved evidence ({mode}):
{evidence_block}

Write a concise but substantive research answer."""
    return client.complete_text(
        [
            {"role": "system", "content": "You are a precise research assistant."},
            {"role": "user", "content": prompt},
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run native LLM or KB-hybrid baseline on RTL benchmark v2.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--kb-dir", default=None)
    parser.add_argument("--mode", required=True, choices=["native", "hybrid"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--answer-llm-config", default="configs/llm/mimo_flash.local.yaml")
    parser.add_argument("--answer-fallback-llm-config", default="")
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--judge-fallback-llm-config", default="configs/llm/qwen_235b.local.yaml")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument("--resume-seed-results", default="", help="Optional seed results JSONL to initialize a resumable experiment directory.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    kb_dir = Path(args.kb_dir) if args.kb_dir else (release_dir / "kb")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_release_tasks(release_dir)
    hidden_by_id = load_hidden_eval(release_dir) if not args.skip_judge else {}
    answer_primary = OpenAICompatChatClient(load_openai_compat_config(Path(args.answer_llm_config)))
    answer_fallback = None
    if str(args.answer_fallback_llm_config or "").strip():
        answer_fallback = OpenAICompatChatClient(load_openai_compat_config(Path(args.answer_fallback_llm_config)))
    answer_client = FallbackOpenAICompatChatClient(answer_primary, answer_fallback)
    judge_client = None
    if not args.skip_judge:
        judge_primary = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))
        judge_fallback = None
        if str(args.judge_fallback_llm_config or "").strip():
            judge_fallback = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_fallback_llm_config)))
        judge_client = FallbackOpenAICompatChatClient(judge_primary, judge_fallback)

    kb = OfflineKnowledgeBase(kb_dir) if args.mode == "hybrid" else None

    domain_filter = set(args.domains) if args.domains else None
    family_filter = set(args.families) if args.families else None
    allowed_task_ids = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    selected = []
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
        selected.append((task, domain_id))
    if args.task_limit is not None:
        selected = selected[: args.task_limit]

    results_path = output_dir / "results.jsonl"
    rows_out = []
    completed_task_ids = set()
    bootstrapped_from_seed = False
    seed_results_path = Path(args.resume_seed_results) if str(args.resume_seed_results or "").strip() else None
    if args.resume and (not results_path.exists()) and seed_results_path and seed_results_path.exists():
        seed_rows = []
        for line in seed_results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            seed_rows.append(row)
            task_id = str(row.get("task_id") or "").strip()
            if task_id:
                completed_task_ids.add(task_id)
        if seed_rows:
            with results_path.open("w", encoding="utf-8") as handle:
                for row in seed_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_out.extend(seed_rows)
            bootstrapped_from_seed = True
            print(
                f"[kb-baseline:{args.mode}] bootstrapped resumable run from seed={seed_results_path} "
                f"completed={len(completed_task_ids)}",
                flush=True,
            )
    if args.resume and results_path.exists() and not bootstrapped_from_seed:
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows_out.append(row)
                task_id = str(row.get("task_id") or "").strip()
                if task_id:
                    completed_task_ids.add(task_id)
        if completed_task_ids:
            selected = [(task, domain_id) for task, domain_id in selected if str(task.get("task_id") or "") not in completed_task_ids]

    results_handle = results_path.open("a", encoding="utf-8") if args.resume else results_path.open("w", encoding="utf-8")
    for idx, (task, domain_id) in enumerate(selected, start=1):
        print(f"[kb-baseline:{args.mode}] {idx}/{len(selected)} {task['task_id']} domain={domain_id} family={task['family']}", flush=True)
        evidence = None
        if args.mode == "hybrid":
            evidence = build_hybrid_evidence(task, kb.domain(domain_id))
        answer = answer_task(answer_client, task=task, mode=args.mode, evidence_packet=evidence)
        row = {
            "task_id": task["task_id"],
            "family": task["family"],
            "domain": task["domain"],
            "domain_id": domain_id,
            "baseline": f"kb_{args.mode}",
            "title": task["title"],
            "question": task["question"],
            "time_cutoff": task["time_cutoff"],
            "answer": answer,
        }
        if evidence is not None:
            row["evidence"] = evidence
        if judge_client is not None:
            hidden_task = hidden_by_id.get(str(task["task_id"]))
            if hidden_task is not None:
                row["judge"] = judge_answer(judge_client, public_task=task, hidden_task=hidden_task, candidate_answer=answer)
        rows_out.append(row)
        results_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        results_handle.flush()
    results_handle.close()
    summary = {
        "mode": args.mode,
        "release_dir": str(release_dir),
        "kb_dir": str(kb_dir) if kb is not None else None,
        "task_count": len(rows_out),
        "results_path": str(results_path),
        "score_summary": aggregate_scores(rows_out),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
