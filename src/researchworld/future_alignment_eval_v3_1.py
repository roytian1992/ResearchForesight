from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from researchworld.factscore_eval_v3 import FactScoreV3Config, _collect_evidence_from_domain, render_evidence
from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import OfflineKnowledgeBase, clip_text, normalize_ws


@dataclass
class FutureAlignmentV3_1Config:
    max_units: int = 4
    evidence_per_view: int = 3
    max_evidence_rows: int = 8


_WORD_RE = re.compile(r"[a-z0-9]+")


def _token_set(text: Any) -> Set[str]:
    return set(_WORD_RE.findall(normalize_ws(text).lower()))


def _token_overlap(a: Any, b: Any) -> float:
    left = _token_set(a)
    right = _token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _public_task_text(public_task: Dict[str, Any]) -> str:
    fields: List[str] = []
    for key in ["title", "question", "deliverable_spec", "answer_contract"]:
        value = public_task.get(key)
        if isinstance(value, (dict, list)):
            fields.append(json.dumps(value, ensure_ascii=False))
        elif value is not None:
            fields.append(str(value))
    return normalize_ws(" ".join(fields))


def _future_unit_public_leakage(public_task: Dict[str, Any], unit: Dict[str, Any]) -> Dict[str, Any]:
    public_text = _public_task_text(public_task)
    public_lower = public_text.lower()
    aliases = [normalize_ws(unit.get("text"))] + [normalize_ws(x) for x in (unit.get("aliases") or [])]
    aliases = [x for x in aliases if x]
    exact_aliases = [alias for alias in aliases if len(alias) >= 12 and alias.lower() in public_lower]
    max_overlap = max([_token_overlap(alias, public_text) for alias in aliases] or [0.0])
    leaked = bool(exact_aliases) or max_overlap >= 0.72
    return {
        "unit_visible_in_public_task": leaked,
        "max_public_token_overlap": round(max_overlap, 4),
        "exact_public_aliases": exact_aliases[:3],
        "policy": (
            "If visible, mere repetition of this public task phrase is insufficient; the answer must make the requested "
            "forecast, assessment, selection, or ranking call and justify it with task-specific reasoning."
        ),
    }


def _answer_mentions_future_unit(answer: str, unit: Dict[str, Any]) -> bool:
    answer_lower = normalize_ws(answer).lower()
    aliases = [normalize_ws(unit.get("text"))] + [normalize_ws(x) for x in (unit.get("aliases") or [])]
    for alias in aliases:
        if len(alias) >= 12 and alias.lower() in answer_lower:
            return True
        if _token_overlap(alias, answer) >= 0.68:
            return True
    return False


def _answer_makes_task_commitment(answer: str, *, family: str) -> bool:
    text = normalize_ws(answer).lower()
    if not text:
        return False
    generic_commitment = [
        "should be treated",
        "should be prioritized",
        "priority 1",
        "rank 1",
        "ranking",
        "ranks higher",
        "ranked",
        "ranked first",
        "first priority",
        "most promising",
        "most likely",
        "leading near-term",
        "leading direction",
        "forecast",
        "accelerating",
        "fragmenting",
        "steady",
        "cooling",
        "venue fit",
    ]
    if any(cue in text for cue in generic_commitment):
        return True
    if family == "venue_aware_research_positioning" and re.search(r"\b(1|first|top)\b.{0,120}\b(rank|direction|option|bucket|venue)\b", text):
        return True
    if family == "strategic_research_planning" and re.search(r"\b(yes|treat|prioritize|priority|invest|defer|roadmap)\b", text):
        return True
    if family == "direction_forecasting" and re.search(r"\b(next[- ]step|trajectory|likely|emerge|accelerat|fragment|steady|cool)\b", text):
        return True
    return False


def _apply_public_leakage_guard(
    *,
    label: str,
    specificity: float,
    rationale: str,
    public_leakage: Dict[str, Any],
    candidate_answer: str,
    family: str,
    unit: Dict[str, Any],
) -> tuple[str, float, str, Dict[str, Any]]:
    if not public_leakage.get("unit_visible_in_public_task"):
        return label, specificity, rationale, {"applied": False, "reason": "unit_not_visible_in_public_task"}
    mentions_unit = _answer_mentions_future_unit(candidate_answer, unit)
    makes_commitment = _answer_makes_task_commitment(candidate_answer, family=family)
    guard = {
        "applied": False,
        "mentions_unit": mentions_unit,
        "makes_task_commitment": makes_commitment,
    }
    if not makes_commitment and label == "aligned":
        guard.update({"applied": True, "reason": "public_unit_repeated_without_required_task_commitment"})
        return "partial", min(specificity, 0.55), f"{rationale} Public-leakage guard: a visible task phrase without a clear forecast/selection/ranking commitment is capped at partial.", guard
    return label, specificity, rationale, guard


def _future_alignment_json(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    family: str,
    candidate_answer: str,
    unit: Dict[str, Any],
    evidence_rows: List[Dict[str, Any]],
    public_leakage: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = f"""# Role
You are a Future-Alignment Auditor for a time-sliced research benchmark.

# Objective
Judge whether the candidate answer successfully anticipated a future-emergent research direction that was actually realized in the hidden future window.

# Evaluation Philosophy
- You are NOT judging generic answer quality.
- You are judging whether the answer meaningfully anticipated the specific future unit below.
- Reward semantically correct anticipation even if wording differs.
- Do NOT require exact phrase overlap.
- Penalize vague statements that are too broad to count as a meaningful forecast.

# Input Data
- Public Task Definition: {json.dumps(public_task, ensure_ascii=False, indent=2)}
- Task Family: {family}
- Candidate Answer: {candidate_answer}
- Hidden Future Unit: {json.dumps(unit, ensure_ascii=False, indent=2)}
- Public Task Leakage Check: {json.dumps(public_leakage, ensure_ascii=False, indent=2)}
- Future Evidence: {render_evidence(evidence_rows)}

# Labels
- aligned: the answer clearly anticipates this future unit with specific, semantically matching content.
- partial: the answer points in the right direction but is broader, weaker, or only partly specific enough.
- not_aligned: the answer does not meaningfully anticipate this unit.

# Specificity Score
Score from 0.0 to 1.0:
- 0.9-1.0: concrete, discriminative, and close to the realized future unit.
- 0.6-0.8: materially relevant but somewhat broad or underspecified.
- 0.3-0.5: only weakly suggestive.
- 0.0-0.2: generic or effectively unrelated.

# Rules
- Use only the provided candidate answer and future evidence.
- Favor semantic equivalence over literal string overlap.
- If the answer only says something broad like "better evaluation" or "more robust agents", that is usually partial at best.
- If Public Task Leakage Check says the hidden future unit was already visible in the public task, mere repetition is NOT enough for alignment.
- For leaked units, require the answer to perform the requested operation: forecast the unit as the likely next direction, assess it as leading, select/rank it appropriately, or explain why the task-visible unit is the right future-facing choice.
- If a leaked unit is mentioned but the answer ranks/selects a different incompatible direction first, assign partial or not_aligned depending on the explanation.

# Output (Strict JSON)
{{
  "label": "aligned | partial | not_aligned",
  "specificity": 0.0,
  "rationale": "Explain briefly whether and why the answer anticipated this future unit.",
  "cited_evidence_ids": ["id1", "id2"]
}}
"""
    return complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict future-alignment judge for a research benchmark. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=700,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys label, specificity, rationale, cited_evidence_ids.",
    )


def _label_score(label: str) -> float:
    if label == 'aligned':
        return 1.0
    if label == 'partial':
        return 0.5
    return 0.0


def _float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _float_or_default(value)))


def _date_in_window(value: Any, *, start: str = "", end: str = "") -> bool:
    day = normalize_ws(value)[:10]
    if not day:
        return True
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def _filter_evidence_window(rows: List[Dict[str, Any]], *, start: str = "", end: str = "") -> List[Dict[str, Any]]:
    return [row for row in rows if _date_in_window(row.get("published_date"), start=start, end=end)]


def _embedded_future_evidence(
    hidden_row: Dict[str, Any],
    unit: Dict[str, Any],
    *,
    max_rows: int,
    future_start: str = '',
    future_end: str = '',
) -> List[Dict[str, Any]]:
    trace = hidden_row.get('trace') or {}
    future_items = [item for item in (trace.get('future_evidence') or []) if isinstance(item, dict)]
    if not future_items:
        return []
    wanted_titles = {normalize_ws(x).lower() for x in (unit.get('future_paper_titles') or []) if normalize_ws(x)}
    selected: List[Dict[str, Any]] = []
    had_specific_titles = bool(wanted_titles)
    for item in future_items:
        title = normalize_ws(item.get('title'))
        if wanted_titles and title.lower() not in wanted_titles:
            continue
        selected.append(item)
    if not selected and not had_specific_titles:
        selected = future_items
    selected = [
        item
        for item in selected
        if _date_in_window(item.get('published_date'), start=future_start, end=future_end)
    ]
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(selected[:max_rows], start=1):
        rows.append(
            {
                'evidence_id': f'FE{idx}',
                'evidence_source': 'embedded_future',
                'paper_id': item.get('paper_id'),
                'paper_title': item.get('title'),
                'published_date': item.get('published_date'),
                'snippet': clip_text(item.get('why_it_matters') or item.get('title') or '', 1000),
                'scores': {'combined_score': 1.0},
            }
        )
    return rows


def evaluate_future_alignment_v3_1(
    *,
    future_kb: Optional[OfflineKnowledgeBase],
    judge_client: OpenAICompatChatClient,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
    hidden_row: Dict[str, Any],
    cfg: Optional[FutureAlignmentV3_1Config] = None,
) -> Dict[str, Any]:
    cfg = cfg or FutureAlignmentV3_1Config()
    targets = hidden_row.get('future_alignment_targets') or {}
    units = list((targets.get('alignment_units') or []))[: cfg.max_units]
    if not units:
        return {
            'future_alignment_score': 0.0,
            'weighted_unit_alignment': 0.0,
            'alignment_coverage': 0.0,
            'mean_specificity': 0.0,
            'units': [],
        }

    family = str(hidden_row.get('family') or public_task.get('family') or '')
    domain_id = str(result_row.get('domain_id') or result_row.get('domain') or '')
    answer = normalize_ws(result_row.get('answer'))
    temporal_policy = hidden_row.get('temporal_policy') or {}
    future_start = str(temporal_policy.get('future_start') or '')
    cutoff = str(temporal_policy.get('future_end') or '')
    collect_cfg = FactScoreV3Config(evidence_per_view=cfg.evidence_per_view, max_evidence_rows=cfg.max_evidence_rows)

    unit_rows: List[Dict[str, Any]] = []
    total_importance = 0.0
    weighted_alignment = 0.0
    specificity_values: List[float] = []
    covered = 0

    for unit in units:
        raw_queries = [str(unit.get('text') or '')]
        raw_queries.extend(str(x) for x in (unit.get('aliases') or []))
        queries: List[str] = []
        seen = set()
        for q in raw_queries:
            norm = normalize_ws(q)
            key = norm.lower()
            if not norm or key in seen:
                continue
            seen.add(key)
            queries.append(norm)
        evidence_rows: List[Dict[str, Any]] = []
        if future_kb is not None:
            evidence_rows.extend(
                _filter_evidence_window(
                    _collect_evidence_from_domain(
                        future_kb.domain(domain_id),
                        queries,
                        cutoff_date=cutoff,
                        cfg=collect_cfg,
                        source_name='future',
                    ),
                    start=future_start,
                    end=cutoff,
                )
            )
        if not evidence_rows:
            evidence_rows.extend(
                _embedded_future_evidence(
                    hidden_row,
                    unit,
                    max_rows=cfg.max_evidence_rows,
                    future_start=future_start,
                    future_end=cutoff,
                )
            )
        evidence_rows = evidence_rows[: cfg.max_evidence_rows]
        if not evidence_rows:
            label = 'not_aligned'
            specificity = 0.0
            public_leakage = _future_unit_public_leakage(public_task, unit)
            leakage_guard = {'applied': False, 'reason': 'no_future_evidence'}
            obj = {
                'rationale': 'No future evidence rows were available for this alignment unit.',
                'cited_evidence_ids': [],
            }
        else:
            public_leakage = _future_unit_public_leakage(public_task, unit)
            obj = _future_alignment_json(
                judge_client,
                public_task=public_task,
                family=family,
                candidate_answer=answer,
                unit=unit,
                evidence_rows=evidence_rows,
                public_leakage=public_leakage,
            )
            label = str(obj.get('label') or 'not_aligned').strip().lower()
            if label not in {'aligned', 'partial', 'not_aligned'}:
                label = 'not_aligned'
            specificity = _clamp01(obj.get('specificity'))
            label, specificity, guarded_rationale, leakage_guard = _apply_public_leakage_guard(
                label=label,
                specificity=specificity,
                rationale=str(obj.get('rationale') or '').strip(),
                public_leakage=public_leakage,
                candidate_answer=answer,
                family=family,
                unit=unit,
            )
            obj['rationale'] = guarded_rationale
        valid_evidence_ids = {str(row.get('evidence_id') or '') for row in evidence_rows}
        importance = max(0.0, _float_or_default(unit.get('importance'), 1.0))
        base = _label_score(label)
        unit_score = round(base * (0.5 + 0.5 * specificity), 4)
        total_importance += importance
        weighted_alignment += importance * unit_score
        specificity_values.append(specificity)
        if label in {'aligned', 'partial'}:
            covered += 1
        unit_rows.append(
            {
                'unit_id': unit.get('unit_id'),
                'unit_type': unit.get('unit_type'),
                'text': unit.get('text'),
                'importance': importance,
                'label': label,
                'specificity': round(specificity, 4),
                'unit_score': unit_score,
                'rationale': str(obj.get('rationale') or '').strip(),
                'cited_evidence_ids': [
                    str(x)
                    for x in (obj.get('cited_evidence_ids') or [])
                    if str(x).strip() and str(x).strip() in valid_evidence_ids
                ],
                'public_leakage': public_leakage,
                'leakage_guard': leakage_guard,
                'evidence': evidence_rows,
            }
        )

    weighted_unit_alignment = round(weighted_alignment / total_importance, 4) if total_importance else 0.0
    alignment_coverage = round(covered / len(units), 4) if units else 0.0
    mean_specificity = round(sum(specificity_values) / len(specificity_values), 4) if specificity_values else 0.0
    future_alignment_score = round(0.7 * weighted_unit_alignment + 0.2 * alignment_coverage + 0.1 * mean_specificity, 4)
    return {
        'future_alignment_score': future_alignment_score,
        'weighted_unit_alignment': weighted_unit_alignment,
        'alignment_coverage': alignment_coverage,
        'mean_specificity': mean_specificity,
        'unit_count': len(units),
        'units': unit_rows,
    }
