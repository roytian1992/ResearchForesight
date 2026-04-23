from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from researchworld.offline_kb import dedupe


TITLE_PATTERNS = [
    r"^Bottleneck and Opportunity Discovery in\s+",
    r"^Bottleneck and Opportunity Discovery for\s+",
    r"^Forecasting the Trajectory of\s+",
    r"^Forecasting Research Trajectory in\s+",
    r"^Forecasting Research Trajectory for\s+",
    r"^Forecasting Trajectory and Subdirections in\s+",
    r"^Forecast for\s+",
    r"^Strategic Research Planning for\s+",
    r"^Strategic Research Agenda for\s+",
    r"^Identifying Bottlenecks in\s+",
    r"^Identifying a Key Bottleneck in\s+",
    r"^Identifying a Key Bottleneck for\s+",
    r"^Planning Near-Term Research Directions in\s+",
]


def hybrid_focus_text(task: Dict[str, Any]) -> str:
    title = str(task.get("title") or "").strip()
    text = title
    for pattern in TITLE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    if text and text != title:
        return text
    question = str(task.get("question") or "").strip()
    m = re.search(
        r"(?:within|for|on)\s+(.+?)(?:\s+over the next|\s+for the next|\s+considering|\.|$)",
        question,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else (title or question)


def build_hybrid_task_queries(task: Dict[str, Any]) -> List[str]:
    return dedupe(
        [
            str(task.get("question") or ""),
            str(task.get("title") or ""),
            hybrid_focus_text(task),
        ]
    )[:3]


def merge_retrieval_runs(
    named_rows: Sequence[Tuple[str, Iterable[Tuple[Any, Dict[str, Any]]]]],
    *,
    limit: int,
) -> List[Tuple[Any, Dict[str, Any]]]:
    merged: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    for source_name, rows in named_rows:
        for doc, scores in rows:
            doc_id = str(getattr(doc, "doc_id", "") or "")
            if not doc_id:
                continue
            incoming_combined = float(scores.get("combined_score") or scores.get("hybrid_score") or 0.0)
            incoming_queries = [str(x) for x in (scores.get("matched_queries") or []) if str(x).strip()]
            existing = merged.get(doc_id)
            if existing is None:
                merged[doc_id] = (
                    doc,
                    {
                        **scores,
                        "combined_score": incoming_combined,
                        "matched_queries": incoming_queries[:8],
                        "retrieval_sources": [source_name],
                    },
                )
                continue
            _, prev = existing
            prev["combined_score"] = float(prev.get("combined_score") or 0.0) + incoming_combined
            prev["bm25_score"] = max(float(prev.get("bm25_score") or 0.0), float(scores.get("bm25_score") or 0.0))
            prev["tfidf_score"] = max(float(prev.get("tfidf_score") or 0.0), float(scores.get("tfidf_score") or 0.0))
            prev["hybrid_score"] = max(float(prev.get("hybrid_score") or 0.0), float(scores.get("hybrid_score") or 0.0))
            matched_queries = list(prev.get("matched_queries") or [])
            for query in incoming_queries:
                if query not in matched_queries:
                    matched_queries.append(query)
            prev["matched_queries"] = matched_queries[:8]
            sources = list(prev.get("retrieval_sources") or [])
            if source_name not in sources:
                sources.append(source_name)
            prev["retrieval_sources"] = sources
    ranked = sorted(
        merged.values(),
        key=lambda item: float(item[1].get("combined_score") or item[1].get("hybrid_score") or 0.0),
        reverse=True,
    )
    return ranked[:limit]
