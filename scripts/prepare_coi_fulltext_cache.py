from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.baseline_runner import PUBLIC_DOMAIN_TO_ID, load_release_tasks
from researchworld.fulltext_cache import LocalFulltextCache
from researchworld.offline_kb import OfflineKnowledgeBase, merge_multi_query_results
from researchworld.research_arc_kb import extract_focus_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local fulltext cache for CoI-Agent-Offline.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--kb-dir", default="")
    parser.add_argument("--fulltext-cache-root", required=True)
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--task-ids-file", default="")
    parser.add_argument("--top-papers-per-task", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    kb_dir = Path(args.kb_dir) if args.kb_dir else (release_dir / "kb")
    kb = OfflineKnowledgeBase(kb_dir)
    tasks = load_release_tasks(release_dir)

    allowed_task_ids: Optional[Set[str]] = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    domain_filter = set(args.domains) if args.domains else None
    family_filter = set(args.families) if args.families else None

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

    cache_by_domain: Dict[str, LocalFulltextCache] = {}
    candidates_by_domain: Dict[str, Set[str]] = {}
    for task, domain_id in selected:
        domain = kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        queries = [str(task.get("question") or ""), str(task.get("title") or ""), extract_focus_text(task)]
        hits = merge_multi_query_results(domain.paper_retriever(cutoff_date=cutoff_date), queries, top_k_per_query=8, limit=args.top_papers_per_task)
        bucket = candidates_by_domain.setdefault(domain_id, set())
        for doc, _ in hits:
            bucket.add(str(doc.paper_id))

    summary = {"task_count": len(selected), "domains": {}}
    for domain_id, paper_ids in sorted(candidates_by_domain.items()):
        cache = cache_by_domain.setdefault(
            domain_id,
            LocalFulltextCache(
                domain_id=domain_id,
                papers_jsonl=ROOT / "data" / "domains" / domain_id / "interim" / "papers_merged.publication_enriched.semanticscholar.jsonl",
                cache_root=Path(args.fulltext_cache_root),
            ),
        )
        ok = 0
        for idx, paper_id in enumerate(sorted(paper_ids), start=1):
            row = cache.ensure_content(paper_id, allow_fetch=True, timeout=args.timeout)
            print(f"[prepare_coi_fulltext_cache] {domain_id} {idx}/{len(paper_ids)} {paper_id} ok={row is not None}", flush=True)
            if row is not None:
                ok += 1
        summary["domains"][domain_id] = {"candidate_paper_count": len(paper_ids), "cached_content_count": ok}

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
