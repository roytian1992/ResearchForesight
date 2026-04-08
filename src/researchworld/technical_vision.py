from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from researchworld.analytics import TAXONOMY_KEYS, normalize_text_tokens, taxonomy_jaccard, text_jaccard
from researchworld.corpus import iter_jsonl


MICRO_TASK_TYPES = [
    "limitation_discovery",
    "idea_improvement",
    "falsification_experiment_design",
]


def load_jsonl(path: str | Path) -> List[Dict]:
    return list(iter_jsonl(path))


def load_jsonl_by_key(path: str | Path, key: str) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    for row in iter_jsonl(path):
        value = row.get(key)
        if isinstance(value, str) and value:
            rows[value] = row
    return rows


def dump_json(path: str | Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def dump_jsonl(path: str | Path, rows: Iterable[Dict]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def taxonomy_overlap_count(row: Dict, signals: Dict) -> int:
    taxonomy = row.get("taxonomy") or {}
    total = 0
    for key in TAXONOMY_KEYS:
        row_values = set(taxonomy.get(key) or [])
        signal_values = set((signals or {}).get(key) or [])
        total += len(row_values & signal_values)
    return total


def select_case_papers(
    rows_by_id: Dict[str, Dict],
    paper_ids: Sequence[str],
    *,
    signals: Dict,
    priority_ids: Sequence[str] | None = None,
    top_k: int = 12,
) -> List[Dict]:
    priority = {paper_id: idx for idx, paper_id in enumerate(priority_ids or [])}
    scored = []
    for paper_id in paper_ids:
        row = rows_by_id.get(paper_id)
        if not row:
            continue
        overlap = taxonomy_overlap_count(row, signals)
        if overlap <= 0 and paper_id not in priority:
            continue
        scored.append(
            (
                0 if paper_id in priority else 1,
                priority.get(paper_id, 9999),
                -overlap,
                str(row.get("published") or ""),
                paper_id,
                row,
            )
        )
    scored.sort(key=lambda item: item[:5])
    return [item[-1] for item in scored[:top_k]]


def compact_case_paper(row: Dict) -> Dict:
    taxonomy = row.get("taxonomy") or {}
    return {
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "published": row.get("published"),
        "task_settings": (taxonomy.get("task_settings") or [])[:3],
        "method_modules": (taxonomy.get("method_modules") or [])[:4],
        "evaluation_focus": (taxonomy.get("evaluation_focus") or [])[:3],
        "reliability_safety": (taxonomy.get("reliability_safety") or [])[:3],
    }


def case_focus_signature(change: Dict) -> str:
    signals = change.get("taxonomy_signals") or {}
    parts: List[str] = []
    for key in ("task_settings", "method_modules", "evaluation_focus", "reliability_safety"):
        values = (signals.get(key) or [])[:2]
        if values:
            parts.extend(values)
    if parts:
        return " / ".join(parts[:4])
    return str(change.get("title") or "case")


def build_case_summary(change: Dict, history_summary: Dict, history_rows: Sequence[Dict], future_rows: Sequence[Dict]) -> str:
    history_titles = "；".join(row.get("title") or "" for row in history_rows[:3])
    future_titles = "；".join(row.get("title") or "" for row in future_rows[:3])
    return (
        f"关注方向：{change.get('title', '')}。"
        f" 历史领域状态：{history_summary.get('evolution_summary') or history_summary.get('field_state_summary') or ''}"
        f" 历史代表论文包括：{history_titles or '无'}。"
        f" 对应未来变化：{change.get('summary', '')}"
        f" 未来代表论文包括：{future_titles or '无'}。"
    ).strip()


def collect_case_history_evidence_ids(case: Dict) -> List[str]:
    ids: List[str] = []
    for row in case.get("history_papers") or []:
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            ids.append(paper_id)
    return ids


def best_list_text_jaccard(pred_values: Sequence[str], gold_values: Sequence[str]) -> float:
    pred_text = " ".join(str(value) for value in pred_values if isinstance(value, str))
    gold_text = " ".join(str(value) for value in gold_values if isinstance(value, str))
    return text_jaccard(pred_text, gold_text)


def paper_id_validity(pred_ids: Sequence[str], allowed_ids: Sequence[str]) -> float:
    allowed = {paper_id for paper_id in allowed_ids if isinstance(paper_id, str)}
    total = 0
    valid = 0
    for paper_id in pred_ids:
        if not isinstance(paper_id, str):
            continue
        total += 1
        if paper_id in allowed:
            valid += 1
    if total == 0:
        return 0.0
    return valid / total


def stringify_taxonomy_signals(signals: Dict) -> str:
    parts: List[str] = []
    for key in TAXONOMY_KEYS:
        values = [str(value) for value in (signals.get(key) or []) if isinstance(value, str)]
        if values:
            parts.append(f"{key}: {', '.join(values[:4])}")
    return " | ".join(parts)


def normalize_text_label(text: str) -> str:
    return " ".join(normalize_text_tokens(text))


def group_rows_by_key(rows: Sequence[Dict], key: str) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            grouped[value].append(row)
    return dict(grouped)


def limitation_match_score(pred: Dict, gold: Dict) -> float:
    return (
        0.45 * taxonomy_jaccard(pred.get("taxonomy_signals") or {}, gold.get("taxonomy_signals") or {})
        + 0.25 * text_jaccard(str(pred.get("limitation_name") or ""), str(gold.get("name") or ""))
        + 0.20 * text_jaccard(str(pred.get("technical_description") or ""), str(gold.get("description") or ""))
        + 0.10 * text_jaccard(str(pred.get("failure_mechanism") or ""), str(gold.get("failure_mechanism") or ""))
    )


def idea_match_score(pred: Dict, gold: Dict) -> float:
    return (
        0.35 * taxonomy_jaccard(pred.get("taxonomy_signals") or {}, gold.get("taxonomy_signals") or {})
        + 0.20 * text_jaccard(str(pred.get("idea_name") or ""), str(gold.get("name") or ""))
        + 0.25 * text_jaccard(str(pred.get("core_mechanism") or ""), str(gold.get("core_mechanism") or ""))
        + 0.10 * text_jaccard(str(pred.get("expected_benefit") or ""), str(gold.get("expected_benefit") or ""))
        + 0.10 * text_jaccard(str(pred.get("tradeoffs") or ""), str(gold.get("tradeoffs") or ""))
    )


def experiment_match_score(pred: Dict, gold: Dict) -> float:
    return (
        0.15 * text_jaccard(str(pred.get("hypothesis") or ""), str(gold.get("hypothesis") or ""))
        + 0.25 * text_jaccard(str(pred.get("critical_experiment") or ""), str(gold.get("critical_experiment") or ""))
        + 0.15 * best_list_text_jaccard(pred.get("controls") or [], gold.get("controls") or [])
        + 0.15 * best_list_text_jaccard(pred.get("metrics") or [], gold.get("metrics") or [])
        + 0.10 * best_list_text_jaccard(pred.get("settings") or [], gold.get("settings") or [])
        + 0.20 * best_list_text_jaccard(pred.get("falsification_criteria") or [], gold.get("failure_criteria") or [])
    )
