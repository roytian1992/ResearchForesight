from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from researchworld.corpus import iter_jsonl


def parse_published_date(value: str | None) -> Optional[date]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def load_rows_by_paper_id(path: str | Path) -> Dict[str, Dict]:
    rows = {}
    for row in iter_jsonl(path):
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            rows[paper_id] = row
    return rows


def merge_papers_with_labels(papers: Dict[str, Dict], labels: Optional[Dict[str, Dict]] = None) -> List[Dict]:
    merged = []
    labels = labels or {}
    for paper_id, paper in papers.items():
        row = dict(paper)
        label = labels.get(paper_id)
        if label:
            row["label"] = label
            row["scope_decision"] = label.get("scope_decision")
            row["taxonomy"] = label.get("taxonomy")
        row["published_date"] = parse_published_date(row.get("published"))
        merged.append(row)
    merged.sort(key=lambda row: (row["published_date"] or date.min, row["paper_id"]))
    return merged


def add_months(anchor: date, months: int) -> date:
    year = anchor.year + (anchor.month - 1 + months) // 12
    month = (anchor.month - 1 + months) % 12 + 1
    return date(year, month, 1)


def summarize_scope(rows: Iterable[Dict]) -> Dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[str(row.get("scope_decision") or "unlabeled")] += 1
    return dict(counts)
