from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.benchmark import load_rows_by_paper_id
from researchworld.config import load_yaml
from researchworld.llm import OpenAICompatChatClient, complete_json_object, load_openai_compat_config
from researchworld.pageindex_compat import select_nodes_for_structure_extraction
from researchworld.prompting import YAMLPromptLoader
from researchworld.technical_vision import load_jsonl, load_jsonl_by_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured technical signals from benchmark papers.")
    parser.add_argument("--cases", required=True, help="Research case JSONL.")
    parser.add_argument(
        "--papers",
        default=str(ROOT / "data" / "interim" / "papers_merged.jsonl"),
        help="Normalized papers JSONL.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Label JSONL.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--error-output", required=True, help="Error JSONL.")
    parser.add_argument("--paper-content", default="", help="Optional normalized paper content JSONL.")
    parser.add_argument("--paper-index", default="", help="Optional PageIndex-compatible JSONL.")
    parser.add_argument(
        "--llm-config",
        default=str(ROOT / ".." / "NarrativeKnowledgeWeaver" / "configs" / "config_openai.yaml"),
        help="OpenAI-compatible YAML config.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers.")
    parser.add_argument("--resume", action="store_true", help="Skip already extracted papers.")
    parser.add_argument("--limit", type=int, default=0, help="Optional paper limit.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retry attempts per paper.")
    parser.add_argument("--timeout", type=int, default=180, help="Request timeout.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    return parser.parse_args()


def load_seen_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen: Set[str] = set()
    for row in load_jsonl(path):
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            seen.add(paper_id)
    return seen


def build_index_lookup(index_row: Dict) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    by_id: Dict[str, Dict] = {}
    by_path: Dict[str, Dict] = {}
    for node in index_row.get("nodes") or []:
        node_id = str(node.get("node_id") or "").strip()
        section_path = str(node.get("section_path") or "").strip()
        if node_id:
            by_id[node_id] = node
        if section_path:
            by_path[section_path] = node
    return by_id, by_path


def infer_evidence_refs(evidence_phrases: List[str], index_row: Dict, *, limit: int = 3) -> List[Dict]:
    if not evidence_phrases or not index_row:
        return []
    candidates = []
    for node in index_row.get("nodes") or []:
        haystack = " ".join(
            [
                str(node.get("title") or ""),
                str(node.get("summary") or ""),
                str(node.get("text") or ""),
            ]
        ).lower()
        score = 0
        matched_quotes = []
        for phrase in evidence_phrases:
            phrase = str(phrase or "").strip()
            if not phrase:
                continue
            lowered = phrase.lower()
            if lowered in haystack:
                score += max(2, min(len(phrase) // 20 + 1, 6))
                matched_quotes.append(phrase)
        if score <= 0:
            continue
        score += max(0, 8 - int(index_row.get("node_count") or 0) // 10)
        candidates.append((score, node, matched_quotes))

    candidates.sort(
        key=lambda item: (
            -item[0],
            len(str(item[1].get("section_path") or "")),
        )
    )
    refs = []
    seen = set()
    for _, node, matched_quotes in candidates:
        node_id = str(node.get("node_id") or "").strip()
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        refs.append(
            {
                "node_id": node_id,
                "section_path": str(node.get("section_path") or ""),
                "kind": str(node.get("kind") or ""),
                "quote": matched_quotes[0] if matched_quotes else "",
                "matched_by": "phrase_match",
            }
        )
        if len(refs) >= limit:
            break
    return refs


def sanitize_evidence_refs(value, evidence_phrases: List[str], index_row: Dict) -> List[Dict]:
    by_id, by_path = build_index_lookup(index_row)
    refs: List[Dict] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            node = None
            node_id = str(item.get("node_id") or "").strip()
            section_path = str(item.get("section_path") or "").strip()
            if node_id and node_id in by_id:
                node = by_id[node_id]
            elif section_path and section_path in by_path:
                node = by_path[section_path]
            elif node_id or section_path:
                refs.append(
                    {
                        "node_id": node_id,
                        "section_path": section_path,
                        "kind": str(item.get("kind") or ""),
                        "quote": str(item.get("quote") or item.get("snippet") or ""),
                        "matched_by": "model_only",
                    }
                )
                continue
            if node is None:
                continue
            refs.append(
                {
                    "node_id": str(node.get("node_id") or ""),
                    "section_path": str(node.get("section_path") or ""),
                    "kind": str(node.get("kind") or ""),
                    "quote": str(item.get("quote") or item.get("snippet") or ""),
                    "matched_by": "model_ref",
                }
            )
    if not refs:
        refs = infer_evidence_refs(evidence_phrases, index_row)
    deduped = []
    seen = set()
    for ref in refs:
        key = (ref.get("node_id"), ref.get("section_path"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped[:3]


def sanitize_items(items, required_keys: List[str], index_row: Dict) -> List[Dict]:
    cleaned = []
    if not isinstance(items, list):
        return cleaned
    for item in items:
        if not isinstance(item, dict):
            continue
        obj = {}
        valid = True
        evidence_phrases: List[str] = []
        for key in required_keys:
            value = item.get(key)
            if key == "evidence_phrases":
                if not isinstance(value, list):
                    value = []
                value = [x for x in value if isinstance(x, str)]
                evidence_phrases = value
            elif key in {"metrics", "baselines", "target_limitations"}:
                if not isinstance(value, list):
                    value = []
                value = [x for x in value if isinstance(x, str)]
            else:
                value = str(value or "")
            if key != "evidence_phrases" and key not in {"metrics", "baselines", "target_limitations"} and not value:
                valid = False
            obj[key] = value
        obj["evidence_refs"] = sanitize_evidence_refs(item.get("evidence_refs"), evidence_phrases, index_row)
        if valid:
            cleaned.append(obj)
    return cleaned


def sanitize_structure(obj: Dict, paper_id: str, paper: Dict, label: Dict, content_row: Dict, index_row: Dict) -> Dict:
    return {
        "paper_id": paper_id,
        "title": paper.get("title"),
        "published": paper.get("published"),
        "taxonomy": label.get("taxonomy") or {},
        "scope_decision": label.get("scope_decision"),
        "source_type": content_row.get("source_type"),
        "source_url": content_row.get("source_url"),
        "problem_statement": str(obj.get("problem_statement") or ""),
        "explicit_limitations": sanitize_items(
            obj.get("explicit_limitations"),
            ["name", "description", "evidence_phrases"],
            index_row,
        ),
        "core_ideas": sanitize_items(
            obj.get("core_ideas"),
            ["name", "mechanism", "target_limitations", "evidence_phrases"],
            index_row,
        ),
        "experiment_signals": sanitize_items(
            obj.get("experiment_signals"),
            ["setting", "metrics", "baselines", "claim", "evidence_phrases"],
            index_row,
        ),
        "falsification_signals": sanitize_items(
            obj.get("falsification_signals"),
            ["hypothesis", "critical_failure_mode", "decisive_test", "evidence_phrases"],
            index_row,
        ),
        "future_work": sanitize_items(
            obj.get("future_work"),
            ["direction", "evidence_phrases"],
            index_row,
        ),
    }


def compact_outline(index_row: Dict, *, max_nodes: int = 12) -> List[Dict]:
    outline = []
    for node in index_row.get("outline") or []:
        outline.append(
            {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "kind": node.get("kind"),
                "section_path": node.get("section_path"),
            }
        )
        if len(outline) >= max_nodes:
            break
    return outline


def annotate_one(
    paper_id: str,
    *,
    papers: Dict[str, Dict],
    labels: Dict[str, Dict],
    content_rows: Dict[str, Dict],
    index_rows: Dict[str, Dict],
    prompt_loader: YAMLPromptLoader,
    llm: OpenAICompatChatClient,
    timeout: int,
    temperature: float,
    max_retries: int,
) -> Tuple[bool, Dict]:
    paper = papers[paper_id]
    label = labels.get(paper_id, {})
    content_row = content_rows.get(paper_id, {})
    index_row = index_rows.get(paper_id, {})
    if content_row and index_row and (content_row.get("source_type") or "") != "abstract_only":
        selected_sections = select_nodes_for_structure_extraction(index_row, max_nodes=6)
        prompt = prompt_loader.render(
            "analysis/extract_paper_structure_from_fulltext",
            task_values={
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "taxonomy": label.get("taxonomy") or {},
                "categories": ", ".join(paper.get("categories") or []),
                "source_type": content_row.get("source_type") or "unknown",
                "index_outline": compact_outline(index_row),
                "selected_sections": selected_sections,
            },
        )
    else:
        prompt = prompt_loader.render(
            "analysis/extract_paper_structure",
            task_values={
                "paper_id": paper_id,
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "taxonomy": label.get("taxonomy") or {},
                "categories": ", ".join(paper.get("categories") or []),
            },
        )
    raw_text = ""
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            obj = complete_json_object(
                llm,
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=timeout,
                max_tokens=1800,
                max_parse_attempts=3,
            )
            structure = sanitize_structure(obj, paper_id, paper, label, content_row, index_row)
            return True, structure
        except Exception as exc:
            last_error = str(exc)
            if attempt == max_retries:
                break
    return False, {"paper_id": paper_id, "error": last_error, "raw_text": raw_text}


def main() -> None:
    args = parse_args()
    papers = load_rows_by_paper_id(args.papers)
    labels = load_rows_by_paper_id(args.labels)
    content_rows = load_jsonl_by_key(args.paper_content, "paper_id") if args.paper_content else {}
    index_rows = load_jsonl_by_key(args.paper_index, "paper_id") if args.paper_index else {}
    cases = load_jsonl(args.cases)
    output_path = Path(args.output)
    error_path = Path(args.error_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = load_seen_ids(output_path) if args.resume else set()
    paper_ids: List[str] = []
    for case in cases:
        for key in ("history_paper_ids", "future_paper_ids"):
            for paper_id in case.get(key) or []:
                if isinstance(paper_id, str) and paper_id and paper_id not in seen_ids:
                    paper_ids.append(paper_id)
    unique_ids = []
    seen = set()
    for paper_id in paper_ids:
        if paper_id in seen:
            continue
        seen.add(paper_id)
        unique_ids.append(paper_id)
        if args.limit and len(unique_ids) >= args.limit:
            break

    llm = OpenAICompatChatClient(load_openai_compat_config(args.llm_config))
    prompt_loader = YAMLPromptLoader(ROOT / "prompts")

    with open(output_path, "a" if args.resume else "w", encoding="utf-8") as out_handle, open(
        error_path,
        "a" if args.resume else "w",
        encoding="utf-8",
    ) as err_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    annotate_one,
                    paper_id,
                    papers=papers,
                    labels=labels,
                    content_rows=content_rows,
                    index_rows=index_rows,
                    prompt_loader=prompt_loader,
                    llm=llm,
                    timeout=args.timeout,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                ): paper_id
                for paper_id in unique_ids
            }
            processed = 0
            for future in concurrent.futures.as_completed(futures):
                ok, row = future.result()
                handle = out_handle if ok else err_handle
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                processed += 1
                print(f"Processed {processed}/{len(unique_ids)}: {row['paper_id']}")

    print(f"Output: {output_path}")
    print(f"Errors: {error_path}")


if __name__ == "__main__":
    main()
