from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


LLM_KEYWORDS = [
    "large language model",
    "language model",
    "llm",
    "gpt",
    "chatgpt",
    "assistant",
]

AGENT_KEYWORDS = [
    "agent",
    "agents",
    "agentic",
    "planning",
    "reasoning",
    "tool use",
    "tool-use",
    "function calling",
    "multi-agent",
    "memory",
    "autonomous",
    "workflow",
    "web",
]


def iter_jsonl(path: str | Path) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_no}") from exc
            if isinstance(obj, dict):
                yield obj


def write_jsonl(path: str | Path, rows: Iterable[Dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def canonical_arxiv_id(raw_id: str) -> Tuple[str, str]:
    value = (raw_id or "").strip()
    match = re.match(r"^(?P<base>[^v]+)(?:v(?P<version>\d+))?$", value)
    if not match:
        return value, value
    return match.group("base"), value


def normalize_paper(record: Dict, source_path: str) -> Dict:
    paper_id = str(record.get("id") or "").strip()
    canonical_id, versioned_id = canonical_arxiv_id(paper_id)
    title = " ".join(str(record.get("title") or "").split())
    abstract = " ".join(str(record.get("abstract") or "").split())
    categories = record.get("categories") or []
    if not isinstance(categories, list):
        categories = [str(categories)]

    return {
        "paper_id": canonical_id,
        "source_paper_id": versioned_id,
        "title": title,
        "abstract": abstract,
        "authors": record.get("authors") or [],
        "categories": categories,
        "published": record.get("published"),
        "updated": record.get("updated"),
        "pdf_url": record.get("pdf_url"),
        "entry_id": record.get("entry_id"),
        "source_path": source_path,
    }


def keyword_hits(text: str, keywords: List[str]) -> List[str]:
    haystack = f" {text.lower()} "
    hits = []
    for keyword in keywords:
        if keyword in haystack:
            hits.append(keyword)
    return hits


def heuristic_profile(paper: Dict) -> Dict:
    return heuristic_profile_from_seed_terms(
        paper,
        positive_terms=AGENT_KEYWORDS,
        caution_terms=[],
        negative_terms=[],
        positive_key="agent_keyword_hits",
    )


def heuristic_profile_from_seed_terms(
    paper: Dict,
    *,
    positive_terms: List[str],
    caution_terms: List[str] | None = None,
    negative_terms: List[str] | None = None,
    llm_terms: List[str] | None = None,
    anchor_terms: List[str] | None = None,
    positive_key: str = "domain_keyword_hits",
) -> Dict:
    text = f"{paper.get('title', '')}\n{paper.get('abstract', '')}"
    effective_anchor_terms = anchor_terms or llm_terms or LLM_KEYWORDS
    anchor_hits = keyword_hits(text, effective_anchor_terms)
    positive_hits = keyword_hits(text, positive_terms)
    caution_hits = keyword_hits(text, caution_terms or [])
    negative_hits = keyword_hits(text, negative_terms or [])

    score = (
        len(anchor_hits)
        + 1.5 * len(positive_hits)
        - 0.25 * len(caution_hits)
        - 0.75 * len(negative_hits)
    )
    if len(anchor_hits) >= 1 and len(positive_hits) >= 2 and len(negative_hits) == 0:
        tier = "core_candidate"
    elif len(anchor_hits) >= 1 and len(positive_hits) >= 1:
        tier = "review_candidate"
    else:
        tier = "out"
    return {
        "anchor_keyword_hits": anchor_hits,
        "llm_keyword_hits": anchor_hits,
        positive_key: positive_hits,
        "caution_keyword_hits": caution_hits,
        "negative_keyword_hits": negative_hits,
        "heuristic_score": score,
        "candidate_tier": tier,
    }


def summarize_candidates(rows: List[Dict]) -> Dict:
    tier_counts = Counter(row.get("candidate_tier", "unknown") for row in rows)
    category_counts = Counter()
    for row in rows:
        for category in row.get("categories", []):
            category_counts[str(category)] += 1
    return {
        "paper_count": len(rows),
        "candidate_tiers": dict(tier_counts),
        "top_categories": category_counts.most_common(20),
    }
