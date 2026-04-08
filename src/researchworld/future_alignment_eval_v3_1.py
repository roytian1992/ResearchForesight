from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from researchworld.factscore_eval_v3 import FactScoreV3Config, _collect_evidence_from_domain, render_evidence
from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import OfflineKnowledgeBase, normalize_ws


@dataclass
class FutureAlignmentV3_1Config:
    max_units: int = 4
    evidence_per_view: int = 3
    max_evidence_rows: int = 8


def _future_alignment_json(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    family: str,
    candidate_answer: str,
    unit: Dict[str, Any],
    evidence_rows: List[Dict[str, Any]],
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


def evaluate_future_alignment_v3_1(
    *,
    future_kb: OfflineKnowledgeBase,
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
    cutoff = str(temporal_policy.get('future_end') or '')
    kb_domain = future_kb.domain(domain_id)
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
        evidence_rows = _collect_evidence_from_domain(
            kb_domain,
            queries,
            cutoff_date=cutoff,
            cfg=collect_cfg,
            source_name='future',
        )[: cfg.max_evidence_rows]
        obj = _future_alignment_json(
            judge_client,
            public_task=public_task,
            family=family,
            candidate_answer=answer,
            unit=unit,
            evidence_rows=evidence_rows,
        )
        label = str(obj.get('label') or 'not_aligned').strip().lower()
        if label not in {'aligned', 'partial', 'not_aligned'}:
            label = 'not_aligned'
        specificity = float(obj.get('specificity') or 0.0)
        specificity = max(0.0, min(1.0, specificity))
        importance = float(unit.get('importance') or 1.0)
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
                'cited_evidence_ids': [str(x) for x in (obj.get('cited_evidence_ids') or []) if str(x).strip()],
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
