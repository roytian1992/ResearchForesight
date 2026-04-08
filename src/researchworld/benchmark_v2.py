from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from researchworld.corpus import iter_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAINS = [
    "llm_agent",
    "llm_finetuning_post_training",
    "rag_and_retrieval_structuring",
    "visual_generative_modeling_and_diffusion",
]
WINDOW_TO_LABEL = {
    "quarterly_2025q4": "2025-09-01_to_2025-11-30",
    "quarterly_2026q1": "2025-12-01_to_2026-02-28",
    "halfyear_2025q4_2026q1": "2025-09-01_to_2026-02-28",
}


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def dump_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_selected_seed_packets(path: str | Path) -> List[Dict[str, Any]]:
    return list(load_json(path) or [])


def load_all_packets(path: str | Path) -> Dict[str, Dict[str, Any]]:
    rows = load_json(path) or []
    return {str(row["packet_id"]): row for row in rows}


def load_paper_rows(domain: str) -> Dict[str, Dict[str, Any]]:
    clean = ROOT / "data" / "domains" / domain / "clean" / "core_papers.publication_enriched.semanticscholar.jsonl"
    if clean.exists():
        return {str(row["paper_id"]): row for row in iter_jsonl(clean)}
    fallback = ROOT / "data" / "domains" / domain / "interim" / "papers_merged.publication_enriched.semanticscholar.jsonl"
    rows = {str(row["paper_id"]): row for row in iter_jsonl(fallback)}
    labels = {
        str(row["paper_id"])
        for row in iter_jsonl(ROOT / "data" / "domains" / domain / "annotations" / "paper_labels.jsonl")
        if str(row.get("scope_decision") or "") == "core_domain"
    }
    return {paper_id: row for paper_id, row in rows.items() if paper_id in labels}


def load_label_rows(domain: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(row["paper_id"]): row
        for row in iter_jsonl(ROOT / "data" / "domains" / domain / "annotations" / "paper_labels.jsonl")
    }


def domain_papers_path(domain: str) -> Path:
    clean = ROOT / "data" / "domains" / domain / "clean" / "core_papers.publication_enriched.semanticscholar.jsonl"
    if clean.exists():
        return clean
    return ROOT / "data" / "domains" / domain / "interim" / "papers_merged.publication_enriched.semanticscholar.jsonl"


def domain_labels_path(domain: str) -> Path:
    return ROOT / "data" / "domains" / domain / "annotations" / "paper_labels.jsonl"


def compact_paper(row: Dict[str, Any]) -> Dict[str, Any]:
    enrichment = row.get("publication_enrichment") or {}
    taxonomy = row.get("taxonomy") or (row.get("label") or {}).get("taxonomy") or {}
    return {
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "published": row.get("published"),
        "venue": enrichment.get("published_venue_name") or enrichment.get("semantic_scholar_venue") or "unknown",
        "top_venue_bucket": enrichment.get("top_venue_bucket") or "other",
        "is_top_conference": bool(enrichment.get("is_top_ai_venue")),
        "citation": int(enrichment.get("preferred_cited_by_count") or 0),
        "task_settings": list((taxonomy.get("task_settings") or [])[:3]),
        "method_modules": list((taxonomy.get("method_modules") or [])[:4]),
        "evaluation_focus": list((taxonomy.get("evaluation_focus") or [])[:3]),
        "reliability_safety": list((taxonomy.get("reliability_safety") or [])[:3]),
    }


def choose_case_papers(packet: Dict[str, Any], *, history_k: int = 3, future_k: int = 4) -> Dict[str, List[str]]:
    history_ids = [str(row["paper_id"]) for row in (packet.get("historical_representative_papers") or [])[:history_k]]
    future_ids: List[str] = []
    for key in ("quarterly_2025q4", "quarterly_2026q1"):
        rows = ((packet.get("future_windows") or {}).get(key) or {}).get("representative_papers") or []
        future_ids.extend(str(row["paper_id"]) for row in rows[:2])
    if not future_ids:
        rows = ((packet.get("future_windows") or {}).get("halfyear_2025q4_2026q1") or {}).get("representative_papers") or []
        future_ids.extend(str(row["paper_id"]) for row in rows[:future_k])

    def dedupe(values: List[str]) -> List[str]:
        seen = set()
        out = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    return {
        "history_paper_ids": dedupe(history_ids),
        "future_paper_ids": dedupe(future_ids),
    }


def future_stats(packet: Dict[str, Any], horizon: str) -> Dict[str, Any]:
    if horizon == "quarterly":
        q4 = (packet.get("future_windows") or {}).get("quarterly_2025q4") or {}
        q1 = (packet.get("future_windows") or {}).get("quarterly_2026q1") or {}
        return {
            "q4": q4,
            "q1": q1,
            "paper_count": int(q4.get("paper_count") or 0) + int(q1.get("paper_count") or 0),
            "top_conf_count": int(q4.get("top_conf_count") or 0) + int(q1.get("top_conf_count") or 0),
            "top_conf_share": round(
                (
                    int(q4.get("top_conf_count") or 0) + int(q1.get("top_conf_count") or 0)
                )
                / max(1, int(q4.get("paper_count") or 0) + int(q1.get("paper_count") or 0)),
                4,
            ),
        }
    return (packet.get("future_windows") or {}).get("halfyear_2025q4_2026q1") or {}


def compute_trajectory(packet: Dict[str, Any]) -> Dict[str, Any]:
    hist = packet.get("historical_stats") or {}
    half = (packet.get("future_windows") or {}).get("halfyear_2025q4_2026q1") or {}
    hist_count = float(hist.get("paper_count") or 0)
    future_count = float(half.get("paper_count") or 0)
    split_pressure = float(packet.get("split_pressure") or 0)
    future_top_share = float(half.get("top_conf_share") or 0.0)
    hist_top_share = float(hist.get("top_conf_share") or 0.0)
    ratio = future_count / max(1.0, hist_count)
    venue_delta = future_top_share - hist_top_share
    descendants = packet.get("emergent_descendants") or []

    label = "steady"
    if split_pressure >= 14 or len(descendants) >= 4:
        label = "fragmenting"
    elif ratio >= 0.42 or (future_count >= 24 and venue_delta >= 0.03):
        label = "accelerating"
    elif ratio <= 0.15 and future_count <= 8:
        label = "cooling"

    confidence = min(
        0.98,
        0.45
        + min(0.25, abs(ratio - 0.25))
        + min(0.15, abs(venue_delta) * 2.0)
        + min(0.15, split_pressure / 30.0),
    )
    return {
        "trajectory_label": label,
        "future_to_history_ratio": round(ratio, 4),
        "venue_share_delta": round(venue_delta, 4),
        "split_pressure": split_pressure,
        "confidence": round(confidence, 4),
    }


def format_descendants(packet: Dict[str, Any], limit: int = 4) -> List[Dict[str, Any]]:
    rows = []
    for row in (packet.get("emergent_descendants") or [])[:limit]:
        rows.append(
            {
                "node_id": row.get("node_id"),
                "display_name": row.get("display_name"),
                "created_time_slice": row.get("created_time_slice"),
                "future_paper_count": row.get("future_paper_count"),
            }
        )
    return rows


def join_display_names(rows: List[Dict[str, Any]], *, key: str = "display_name", limit: int = 4) -> List[str]:
    out: List[str] = []
    for row in rows[:limit]:
        value = str(row.get(key) or "").strip()
        if value:
            out.append(value)
    return out


def top_limitation_signals(structure_rows: Iterable[Dict[str, Any]], *, top_k: int = 5) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: Dict[str, Dict[str, Any]] = {}
    for row in structure_rows:
        for item in row.get("explicit_limitations") or []:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            counter[name] += 1
            if name not in examples:
                examples[name] = {
                    "name": name,
                    "description": str(item.get("description") or ""),
                    "paper_id": row.get("paper_id"),
                    "title": row.get("title"),
                }
    out = []
    for name, count in counter.most_common(top_k):
        example = dict(examples.get(name) or {})
        example["count"] = count
        out.append(example)
    return out


def top_future_work_signals(structure_rows: Iterable[Dict[str, Any]], *, top_k: int = 5) -> List[Dict[str, Any]]:
    counter: Counter[str] = Counter()
    examples: Dict[str, Dict[str, Any]] = {}
    for row in structure_rows:
        for item in row.get("future_work") or []:
            direction = str(item.get("direction") or "").strip()
            if not direction:
                continue
            counter[direction] += 1
            if direction not in examples:
                examples[direction] = {
                    "direction": direction,
                    "paper_id": row.get("paper_id"),
                    "title": row.get("title"),
                }
    out = []
    for direction, count in counter.most_common(top_k):
        example = dict(examples.get(direction) or {})
        example["count"] = count
        out.append(example)
    return out


def summarize_structure_coverage(structure_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(structure_rows)
    limitation_count = sum(len(row.get("explicit_limitations") or []) for row in rows)
    future_work_count = sum(len(row.get("future_work") or []) for row in rows)
    return {
        "paper_count": len(rows),
        "limitation_count": limitation_count,
        "future_work_count": future_work_count,
    }


def quality_band(score: float) -> str:
    if score >= 0.86:
        return "high"
    if score >= 0.72:
        return "medium"
    return "low"


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def log1p_score(value: float) -> float:
    return round(math.log1p(max(0.0, value)), 4)
