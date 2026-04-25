from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.refined_release import build_public_task_view, load_refined_rows

REQUIRED_TASK_KEYS = [
    "schema_version",
    "task_id",
    "family",
    "subtype",
    "domain",
    "horizon",
    "title",
    "question",
    "time_cutoff",
    "deliverable_spec",
    "answer_contract",
    "gold_answer",
    "expected_answer_points",
    "evaluation_rubric",
    "eval_targets",
    "trace",
]

REQUIRED_EVAL_TARGET_KEYS = [
    "slot_targets",
    "claim_bank",
    "component_targets",
    "future_alignment_targets",
    "temporal_policy",
]

FORBIDDEN_PUBLIC_KEYS = {
    "gold_answer",
    "expected_answer_points",
    "evaluation_rubric",
    "eval_targets",
    "slot_targets",
    "claim_bank",
    "component_targets",
    "future_alignment_targets",
    "temporal_policy",
    "trace",
    "judge_profile",
}

DOMAIN_TO_SLUG = {
    "LLM agents": "llm_agent",
    "LLM fine-tuning and post-training": "llm_finetuning_post_training",
    "LLM finetuning and post-training": "llm_finetuning_post_training",
    "RAG and retrieval structuring": "rag_and_retrieval_structuring",
    "Visual generative modeling and diffusion": "visual_generative_modeling_and_diffusion",
}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_no}: {exc}") from exc


def _date(value: Any) -> str:
    return str(value or "").strip()[:10]


def _present(value: Any) -> bool:
    return value not in (None, "", [], {})


def _add(errors: List[str], task_id: str, message: str) -> None:
    errors.append(f"{task_id}: {message}")


def _check_evidence_window(
    errors: List[str],
    *,
    task_id: str,
    temporal_policy: Dict[str, Any],
    trace: Dict[str, Any],
) -> None:
    history_cutoff = _date(temporal_policy.get("history_cutoff"))
    future_start = _date(temporal_policy.get("future_start"))
    future_end = _date(temporal_policy.get("future_end"))
    for item in trace.get("history_evidence") or []:
        if not isinstance(item, dict):
            _add(errors, task_id, "history_evidence contains a non-object item")
            continue
        published = _date(item.get("published_date"))
        if history_cutoff and published and published > history_cutoff:
            _add(errors, task_id, f"history_evidence paper {item.get('paper_id')} is after cutoff: {published} > {history_cutoff}")
    for item in trace.get("future_evidence") or []:
        if not isinstance(item, dict):
            _add(errors, task_id, "future_evidence contains a non-object item")
            continue
        published = _date(item.get("published_date"))
        if future_start and published and published < future_start:
            _add(errors, task_id, f"future_evidence paper {item.get('paper_id')} is before future_start: {published} < {future_start}")
        if future_end and published and published > future_end:
            _add(errors, task_id, f"future_evidence paper {item.get('paper_id')} is after future_end: {published} > {future_end}")


def _validate_tasks(release_dir: Path, *, expected_count: int) -> tuple[List[Dict[str, Any]], List[str], Dict[str, Any]]:
    errors: List[str] = []
    task_path = release_dir / "task_refined.jsonl"
    if not task_path.exists():
        return [], [f"missing task file: {task_path}"], {}
    rows = load_refined_rows(release_dir)
    if expected_count and len(rows) != expected_count:
        errors.append(f"task_count mismatch: got {len(rows)}, expected {expected_count}")
    ids = [str(row.get("task_id") or "") for row in rows]
    duplicate_ids = [task_id for task_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        errors.append(f"duplicate task_id values: {duplicate_ids[:10]}")
    for idx, row in enumerate(rows, start=1):
        task_id = str(row.get("task_id") or f"<row-{idx}>")
        for key in REQUIRED_TASK_KEYS:
            if not _present(row.get(key)):
                _add(errors, task_id, f"missing or empty required key: {key}")
        public_view = build_public_task_view(row)
        leaked = sorted(FORBIDDEN_PUBLIC_KEYS.intersection(public_view))
        if leaked:
            _add(errors, task_id, f"public view leaks hidden keys: {leaked}")
        if str(row.get("domain") or "") not in DOMAIN_TO_SLUG:
            _add(errors, task_id, f"unknown domain: {row.get('domain')}")
        eval_targets = row.get("eval_targets") or {}
        for key in REQUIRED_EVAL_TARGET_KEYS:
            if not _present(eval_targets.get(key)):
                _add(errors, task_id, f"missing eval_targets.{key}")
        temporal_policy = eval_targets.get("temporal_policy") or row.get("temporal_policy") or {}
        if _date(row.get("time_cutoff")) != _date(temporal_policy.get("history_cutoff")):
            _add(
                errors,
                task_id,
                f"time_cutoff does not match temporal_policy.history_cutoff: {row.get('time_cutoff')} vs {temporal_policy.get('history_cutoff')}",
            )
        _check_evidence_window(errors, task_id=task_id, temporal_policy=temporal_policy, trace=row.get("trace") or {})
    summary = {
        "task_count": len(rows),
        "family_counts": dict(Counter(str(row.get("family") or "") for row in rows)),
        "domain_counts": dict(Counter(str(row.get("domain") or "") for row in rows)),
        "time_cutoff_counts": dict(Counter(str(row.get("time_cutoff") or "") for row in rows)),
    }
    return rows, errors, summary


def _validate_kb(release_dir: Path, rows: List[Dict[str, Any]]) -> tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    kb_dir = release_dir / "kb"
    manifest_path = kb_dir / "manifest.json"
    if not manifest_path.exists():
        return [f"missing KB manifest: {manifest_path}"], {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_cutoffs = [_date(row.get("time_cutoff")) for row in rows if _date(row.get("time_cutoff"))]
    max_task_cutoff = max(task_cutoffs) if task_cutoffs else ""
    max_history_cutoff = _date(manifest.get("max_history_cutoff") or manifest.get("history_cutoff"))
    if max_task_cutoff and max_history_cutoff < max_task_cutoff:
        errors.append(f"KB max cutoff {max_history_cutoff} is earlier than max task cutoff {max_task_cutoff}")
    domain_stats = {}
    for domain_name in sorted({str(row.get("domain") or "") for row in rows}):
        slug = DOMAIN_TO_SLUG.get(domain_name)
        if not slug:
            continue
        papers_path = kb_dir / "domains" / slug / "papers.jsonl"
        if not papers_path.exists():
            errors.append(f"missing KB papers file for domain {domain_name}: {papers_path}")
            continue
        dates = []
        count = 0
        for paper in _iter_jsonl(papers_path):
            count += 1
            published = _date(paper.get("published_date"))
            if published:
                dates.append(published)
            if max_history_cutoff and published and published > max_history_cutoff:
                errors.append(f"KB paper after max_history_cutoff in {slug}: {paper.get('paper_id')} {published} > {max_history_cutoff}")
                if len(errors) > 20:
                    break
        domain_stats[slug] = {
            "paper_count": count,
            "min_date": min(dates) if dates else "",
            "max_date": max(dates) if dates else "",
        }
    return errors, {
        "kb_dir": str(kb_dir),
        "max_task_cutoff": max_task_cutoff,
        "max_history_cutoff": max_history_cutoff,
        "domain_stats": domain_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a unified ResearchForesight task_refined release.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--expected-count", type=int, default=422)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    rows, task_errors, task_summary = _validate_tasks(release_dir, expected_count=args.expected_count)
    kb_errors, kb_summary = _validate_kb(release_dir, rows) if rows else ([], {})
    errors = task_errors + kb_errors
    summary = {
        "release_dir": str(release_dir),
        "error_count": len(errors),
        "errors": errors[:50],
        "task_summary": task_summary,
        "kb_summary": kb_summary,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
