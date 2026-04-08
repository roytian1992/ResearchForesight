from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Dict, Iterable, List, Sequence, Tuple


TAXONOMY_KEYS = [
    "task_settings",
    "method_modules",
    "interaction_modes",
    "evaluation_focus",
    "reliability_safety",
]


def count_labels(rows: Iterable[Dict], key: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        taxonomy = row.get("taxonomy") or {}
        values = taxonomy.get(key) or []
        for value in values:
            counter[str(value)] += 1
    return counter


def count_pairs(rows: Iterable[Dict], key_a: str, key_b: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        taxonomy = row.get("taxonomy") or {}
        vals_a = sorted(set(taxonomy.get(key_a) or []))
        vals_b = sorted(set(taxonomy.get(key_b) or []))
        for a in vals_a:
            for b in vals_b:
                counter[(str(a), str(b))] += 1
    return counter


def top_label_entries(rows: Iterable[Dict], key: str, top_k: int = 8) -> List[Dict]:
    return [
        {"label": label, "count": count}
        for label, count in count_labels(rows, key).most_common(top_k)
    ]


def window_split(rows: Sequence[Dict], cutoff: date, recent_days: int = 180) -> Tuple[List[Dict], List[Dict]]:
    recent_start = cutoff - timedelta(days=recent_days)
    previous_start = recent_start - timedelta(days=recent_days)
    previous = []
    recent = []
    for row in rows:
        published = row.get("published_date")
        if published is None:
            continue
        if recent_start <= published < cutoff:
            recent.append(row)
        elif previous_start <= published < recent_start:
            previous.append(row)
    return previous, recent


def rising_labels(rows: Sequence[Dict], cutoff: date, key: str, recent_days: int = 180, top_k: int = 8) -> List[Dict]:
    previous_rows, recent_rows = window_split(rows, cutoff, recent_days=recent_days)
    prev_counts = count_labels(previous_rows, key)
    recent_counts = count_labels(recent_rows, key)
    labels = set(prev_counts) | set(recent_counts)
    scored = []
    for label in labels:
        prev = prev_counts.get(label, 0)
        recent = recent_counts.get(label, 0)
        growth = recent - prev
        ratio = recent / max(prev, 1)
        score = growth + math.log1p(ratio)
        scored.append(
            {
                "label": label,
                "previous_count": prev,
                "recent_count": recent,
                "growth": growth,
                "score": round(score, 4),
            }
        )
    scored.sort(key=lambda item: (-item["score"], -item["recent_count"], item["label"]))
    return scored[:top_k]


def representative_papers(rows: Sequence[Dict], max_papers: int = 20) -> List[Dict]:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row.get("published_date") or date.min,
            float(row.get("confidence", row.get("label", {}).get("confidence", 0.0) or 0.0)),
            row.get("paper_id", ""),
        ),
        reverse=True,
    )
    out = []
    for row in sorted_rows[:max_papers]:
        out.append(
            {
                "paper_id": row.get("paper_id"),
                "title": row.get("title"),
                "published": row.get("published"),
                "taxonomy": row.get("taxonomy") or {},
            }
        )
    return out


def format_label_entries(entries: Sequence[Dict], max_items: int = 3) -> str:
    parts = []
    for item in list(entries)[:max_items]:
        label = str(item.get("label") or "")
        count = item.get("count")
        recent = item.get("recent_count")
        if label and isinstance(count, int):
            parts.append(f"{label} ({count})")
        elif label and isinstance(recent, int):
            parts.append(f"{label} ({recent})")
        elif label:
            parts.append(label)
    return "、".join(parts) if parts else "无明显集中主题"


def format_pair_entries(entries: Sequence[Dict], max_items: int = 3) -> str:
    parts = []
    for item in list(entries)[:max_items]:
        task_setting = str(item.get("task_setting") or "")
        method_module = str(item.get("method_module") or "")
        count = item.get("count")
        if task_setting and method_module and isinstance(count, int):
            parts.append(f"{task_setting} + {method_module} ({count})")
        elif task_setting and method_module:
            parts.append(f"{task_setting} + {method_module}")
    return "、".join(parts) if parts else "无稳定高频组合"


def build_field_state(rows: Sequence[Dict], top_k: int = 8, max_papers: int = 20) -> Dict:
    state = {
        "paper_count": len(rows),
        "top_labels": {},
        "top_pairs": [],
        "representative_papers": representative_papers(rows, max_papers=max_papers),
    }
    for key in TAXONOMY_KEYS:
        state["top_labels"][key] = top_label_entries(rows, key, top_k=top_k)

    pair_counts = count_pairs(rows, "task_settings", "method_modules")
    state["top_pairs"] = [
        {"task_setting": a, "method_module": b, "count": c}
        for (a, b), c in pair_counts.most_common(top_k)
    ]
    state["field_state_summary"] = describe_field_state(state)
    return state


def describe_field_state(summary: Dict) -> str:
    paper_count = int(summary.get("paper_count") or 0)
    top_labels = summary.get("top_labels") or {}
    return (
        f"该时间窗口包含 {paper_count} 篇核心论文。"
        f" 主导任务设置是 {format_label_entries(top_labels.get('task_settings') or [])}。"
        f" 主导方法模块是 {format_label_entries(top_labels.get('method_modules') or [])}。"
        f" 主要交互模式是 {format_label_entries(top_labels.get('interaction_modes') or [])}。"
        f" 高频任务-方法组合包括 {format_pair_entries(summary.get('top_pairs') or [])}。"
    )


def describe_history_state(summary: Dict) -> str:
    rising = summary.get("rising_labels") or {}
    return (
        f"{describe_field_state(summary)}"
        f" 最近升温较快的方法模块是 {format_label_entries(rising.get('method_modules') or [])}。"
        f" 最近升温较快的评测重点是 {format_label_entries(rising.get('evaluation_focus') or [])}。"
        f" 这些信号共同构成预测未来研究走向的历史领域状态。"
    )


def summarize_history(rows: Sequence[Dict], cutoff: date, recent_days: int = 180, top_k: int = 8, max_papers: int = 20) -> Dict:
    summary = build_field_state(rows, top_k=top_k, max_papers=max_papers)
    summary["rising_labels"] = {}
    for key in TAXONOMY_KEYS:
        summary["rising_labels"][key] = rising_labels(rows, cutoff, key, recent_days=recent_days, top_k=top_k)
    summary["evolution_summary"] = describe_history_state(summary)
    return summary


def compact_field_state(summary: Dict, *, top_k: int = 5, max_papers: int = 8) -> Dict:
    compact = {
        "paper_count": summary.get("paper_count", 0),
        "top_labels": {},
        "rising_labels": {},
        "top_pairs": (summary.get("top_pairs") or [])[:top_k],
        "representative_papers": [],
        "field_state_summary": str(summary.get("field_state_summary") or ""),
    }
    for key in TAXONOMY_KEYS:
        compact["top_labels"][key] = (summary.get("top_labels", {}).get(key) or [])[:top_k]

    for paper in (summary.get("representative_papers") or [])[:max_papers]:
        compact["representative_papers"].append(
            {
                "paper_id": paper.get("paper_id"),
                "title": paper.get("title"),
                "published": paper.get("published"),
                "task_settings": (paper.get("taxonomy") or {}).get("task_settings", [])[:3],
                "method_modules": (paper.get("taxonomy") or {}).get("method_modules", [])[:4],
            }
        )
    return compact


def compact_history_summary(summary: Dict, *, top_k: int = 5, max_papers: int = 8) -> Dict:
    compact = compact_field_state(summary, top_k=top_k, max_papers=max_papers)
    compact["rising_labels"] = {}
    for key in TAXONOMY_KEYS:
        compact["rising_labels"][key] = (summary.get("rising_labels", {}).get(key) or [])[:top_k]
    compact["evolution_summary"] = str(summary.get("evolution_summary") or compact.get("field_state_summary") or "")
    return compact


_ASCII_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CJK_SPAN_RE = re.compile(r"[\u4e00-\u9fff]+")


def normalize_text_tokens(text: str) -> List[str]:
    lowered = (text or "").lower()
    tokens: List[str] = []
    tokens.extend(_ASCII_TOKEN_RE.findall(lowered))

    # Keep simple character-level cues for CJK text so Chinese overlap is not
    # silently scored as zero.
    for span in _CJK_SPAN_RE.findall(lowered):
        if not span:
            continue
        tokens.append(span)
        tokens.extend(list(span))
        if len(span) >= 2:
            tokens.extend(span[idx : idx + 2] for idx in range(len(span) - 1))
    return tokens


def text_jaccard(a: str, b: str) -> float:
    a_set = set(normalize_text_tokens(a))
    b_set = set(normalize_text_tokens(b))
    if not a_set and not b_set:
        return 1.0
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set & b_set)
    union = len(a_set | b_set)
    jaccard = inter / union if union else 1.0
    overlap = inter / min(len(a_set), len(b_set))
    # Blend overlap-style recall with Jaccard so detailed-but-correct answers
    # are not over-penalized.
    return 0.4 * jaccard + 0.6 * overlap


def taxonomy_jaccard(a: Dict, b: Dict) -> float:
    scores = []
    for key in TAXONOMY_KEYS:
        a_set = set((a.get(key) or []))
        b_set = set((b.get(key) or []))
        if not a_set and not b_set:
            continue
        if not a_set or not b_set:
            scores.append(0.0)
            continue
        scores.append(len(a_set & b_set) / len(a_set | b_set))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def derive_future_label_targets(rows: Sequence[Dict], top_k: int = 8) -> Dict[str, List[Tuple[str, int]]]:
    targets = {}
    for key in TAXONOMY_KEYS:
        targets[key] = count_labels(rows, key).most_common(top_k)
    return targets
