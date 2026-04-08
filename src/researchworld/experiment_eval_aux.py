from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from researchworld.experiment_eval_v3 import infer_domain_id
from researchworld.experiment_eval_v4 import _render_support_snapshot
from researchworld.llm import OpenAICompatChatClient, complete_json_object

TEMPORAL_LEAKAGE_DIMENSIONS: List[Dict[str, Any]] = [
    {"name": "time_boundary_compliance", "weight": 0.60},
    {"name": "anti_hindsight_discipline", "weight": 0.40},
]

AUX_DIMENSIONS_BY_FAMILY: Dict[str, List[Dict[str, Any]]] = {
    "bottleneck_opportunity_discovery": [
        {"name": "causal_linkage", "weight": 0.55},
        {"name": "technical_plausibility", "weight": 0.45},
    ],
    "direction_forecasting": [
        {"name": "signal_grounding", "weight": 0.60},
        {"name": "forecast_discipline", "weight": 0.40},
    ],
    "strategic_research_planning": [
        {"name": "dependency_awareness", "weight": 0.55},
        {"name": "priority_rationale", "weight": 0.45},
    ],
}

AUX_NAME_BY_FAMILY: Dict[str, str] = {
    "bottleneck_opportunity_discovery": "opportunity_grounding_score",
    "direction_forecasting": "forecast_grounding_score",
    "strategic_research_planning": "technical_dependency_grounding_score",
}

AUX_LABEL_BY_FAMILY: Dict[str, str] = {
    "bottleneck_opportunity_discovery": "Opportunity Grounding",
    "direction_forecasting": "Forecast Grounding",
    "strategic_research_planning": "Technical Dependency Grounding",
}

_FUTURE_COUNT_PAT = re.compile(r"\b([1-9]\d{2,4}|0\.\d{3,4})\b")


def _weighted(dimensions: List[Dict[str, Any]], raw_scores: Dict[str, Any], fallback_key: str, obj: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
    dim_names = [str(item['name']) for item in dimensions]
    scores = {name: max(0.0, min(1.0, float(raw_scores.get(name) or 0.0))) for name in dim_names}
    if any(name in raw_scores for name in dim_names):
        weighted = 0.0
        total_w = 0.0
        for item in dimensions:
            name = str(item.get('name') or '')
            weight = float(item.get('weight') or 0.0)
            total_w += weight
            weighted += weight * float(scores.get(name) or 0.0)
        overall = weighted / total_w if total_w else sum(scores.values()) / max(1, len(scores))
    else:
        overall = max(0.0, min(1.0, float(obj.get(fallback_key) or 0.0)))
    return round(float(overall), 4), {k: round(float(v), 4) for k, v in scores.items()}


def _compact_public_task(public_task: Dict[str, Any]) -> Dict[str, Any]:
    keep = ['task_id', 'family', 'domain', 'title', 'question', 'time_cutoff']
    return {k: public_task.get(k) for k in keep if k in public_task}


def _future_markers(hidden_row: Dict[str, Any]) -> Dict[str, Any]:
    slot = hidden_row.get('slot_targets') or {}
    fa = hidden_row.get('future_alignment_targets') or {}
    claim_bank = hidden_row.get('claim_bank') or []
    future_titles: List[str] = []
    for x in (fa.get('reference_future_papers') or []):
        s = str(x or '').strip()
        if s:
            future_titles.append(s)
    numeric_markers: List[str] = []
    for claim in claim_bank:
        if not isinstance(claim, dict):
            continue
        if str(claim.get('time_scope') or '') in {'future', 'cross_temporal'}:
            for obj in claim.get('canonical_objects') or []:
                s = str(obj or '').strip()
                if s:
                    numeric_markers.append(s)
    text_markers: List[str] = []
    for x in (slot.get('future_themes') or []):
        s = str(x or '').strip()
        if s:
            text_markers.append(s)
    return {
        'future_titles': future_titles[:12],
        'future_markers': text_markers[:12],
        'sensitive_future_values': numeric_markers[:16],
        'history_cutoff': ((hidden_row.get('temporal_policy') or {}).get('history_cutoff') or ''),
    }


def evaluate_temporal_leakage(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    candidate_answer: str,
) -> Dict[str, Any]:
    dimensions = TEMPORAL_LEAKAGE_DIMENSIONS
    markers = _future_markers(hidden_row)
    answer = str(candidate_answer or '')
    heuristic_flags: List[str] = []
    lower = answer.lower()
    for title in markers['future_titles']:
        if title and title.lower() in lower:
            heuristic_flags.append(f'future_paper_title:{title}')
    for marker in markers['sensitive_future_values']:
        if marker and marker in answer:
            heuristic_flags.append(f'future_value:{marker}')
    if any(phrase in lower for phrase in ['after the cutoff', 'subsequent six months', 'later work showed', 'post-cutoff', 'future papers include']):
        heuristic_flags.append('explicit_hindsight_phrase')
    prompt = f"""# Role
You are a Temporal Leakage Auditor for a forward-looking research benchmark.

# Objective
Judge whether the candidate answer improperly relies on information that would only be knowable after the historical cutoff.

# Rules
- All scores must be between 0.0 and 1.0.
- High score means low leakage risk.
- Penalize direct use of post-cutoff paper titles, realized future counts, hindsight phrasing, or conclusions stated as if the future were already known.
- Do NOT penalize ordinary forecasting language, uncertainty, or conditional reasoning.
- Distinguish legitimate prediction from hindsight contamination.

# Input Data
- Public Task Definition: {json.dumps(_compact_public_task(public_task), ensure_ascii=False, indent=2)}
- Temporal Policy: {json.dumps(hidden_row.get('temporal_policy') or {}, ensure_ascii=False, indent=2)}
- Sensitive Future Markers For Audit Only: {json.dumps(markers, ensure_ascii=False, indent=2)}
- Heuristic Flags: {json.dumps(heuristic_flags, ensure_ascii=False)}

# Candidate Answer
{answer}

# Rubric
1. time_boundary_compliance: Does the answer avoid using post-cutoff knowledge as if it were available before the cutoff?
2. anti_hindsight_discipline: Does the answer avoid hindsight framing and keep future statements in predictive or conditional form?

# Output (Strict JSON)
{{
  "dimension_scores": {{"dimension_name": 0.0}},
  "temporal_leakage_score": 0.0,
  "leakage_flags": ["..."],
  "strengths": ["..."],
  "weaknesses": ["..."]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict temporal-leakage judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=900,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, temporal_leakage_score, leakage_flags, strengths, weaknesses.",
    )
    overall, rubric_scores = _weighted(dimensions, obj.get('dimension_scores') or {}, 'temporal_leakage_score', obj)
    flags = [str(x) for x in (obj.get('leakage_flags') or []) if str(x).strip()]
    for flag in heuristic_flags:
        if flag not in flags:
            flags.append(flag)
    return {
        'temporal_leakage_score': overall,
        'rubric_scores': rubric_scores,
        'leakage_flags': flags,
        'strengths': [str(x) for x in (obj.get('strengths') or []) if str(x).strip()],
        'weaknesses': [str(x) for x in (obj.get('weaknesses') or []) if str(x).strip()],
    }


def evaluate_family_auxiliary(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    result_row: Dict[str, Any],
) -> Dict[str, Any]:
    family = str(public_task.get('family') or hidden_row.get('family') or '')
    answer = str(result_row.get('answer') or '')
    dimensions = AUX_DIMENSIONS_BY_FAMILY.get(family) or []
    score_key = AUX_NAME_BY_FAMILY.get(family) or 'family_aux_score'
    label = AUX_LABEL_BY_FAMILY.get(family) or 'Family Auxiliary'
    if not dimensions:
        return {
            'family_aux_metric_name': label,
            score_key: 0.0,
            'rubric_scores': {},
            'strengths': [],
            'weaknesses': ['Unsupported family for auxiliary evaluation.'],
        }
    support_snapshot = _render_support_snapshot(result_row)
    slot_targets = hidden_row.get('slot_targets') or {}
    component_targets = hidden_row.get('component_targets') or {}

    if family == 'bottleneck_opportunity_discovery':
        rubric_text = (
            '1. causal_linkage: Does the answer explain why the identified opportunity would become viable if the bottleneck were addressed?\n'
            '2. technical_plausibility: Is the bottleneck-opportunity pair technically coherent rather than a loose thematic association?'
        )
        extra_context = {
            'topic': slot_targets.get('topic_title') or slot_targets.get('topic'),
            'historical_bottlenecks': slot_targets.get('core_bottleneck_labels') or [],
            'future_opportunities': slot_targets.get('core_opportunity_labels') or slot_targets.get('future_themes') or [],
        }
    elif family == 'direction_forecasting':
        rubric_text = (
            '1. signal_grounding: Does the forecast clearly arise from observable pre-cutoff signals, frictions, or momentum cues?\n'
            '2. forecast_discipline: Does the answer distinguish grounded expectations from speculation and keep the trajectory call calibrated?'
        )
        extra_context = {
            'topic': slot_targets.get('topic_title') or slot_targets.get('topic'),
            'history_stats': slot_targets.get('historical_stats') or {},
            'trajectory_support': slot_targets.get('trajectory_support') or {},
        }
    else:
        rubric_text = (
            '1. dependency_awareness: Does the plan account for technical prerequisites, enabling infrastructure, bottlenecks, or sequencing constraints?\n'
            '2. priority_rationale: Are the prioritized directions justified by concrete feasibility, timing, or leverage arguments instead of generic popularity?'
        )
        extra_context = {
            'topic': slot_targets.get('topic_title') or slot_targets.get('topic'),
            'priority_candidates': slot_targets.get('priority_direction_labels') or slot_targets.get('future_themes') or [],
            'target_window_stats': slot_targets.get('target_window_stats') or {},
        }

    prompt = f"""# Role
You are a family-specific auxiliary judge for a forward-looking research benchmark.

# Objective
Evaluate the answer on the task-family-specific reasoning quality that is NOT fully captured by generic factuality or traceability metrics.

# Rules
- All scores must be between 0.0 and 1.0.
- Judge the answer on this family-specific criterion only.
- Prefer technical coherence, causal structure, and disciplined reasoning over polished prose.
- Use the support snapshot to understand whether the answer's reasoning is connected to evidence, but do not reduce this to citation counting.
- Do NOT judge based on whether the realized future outcome actually happened; this metric is about family-specific reasoning quality.

# Input Data
- Public Task Definition: {json.dumps(_compact_public_task(public_task), ensure_ascii=False, indent=2)}
- Family: {family}
- Family Context: {json.dumps(extra_context, ensure_ascii=False, indent=2)}
- Hidden Component Map: {json.dumps(component_targets, ensure_ascii=False, indent=2)}
- Support Snapshot: {support_snapshot}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}

# Candidate Answer
{answer}

# Family-Specific Rubric
{rubric_text}

# Output (Strict JSON)
{{
  "dimension_scores": {{"dimension_name": 0.0}},
  "{score_key}": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict family-specific research benchmark judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction=f"Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, {score_key}, strengths, weaknesses.",
    )
    overall, rubric_scores = _weighted(dimensions, obj.get('dimension_scores') or {}, score_key, obj)
    return {
        'family_aux_metric_name': label,
        score_key: overall,
        'rubric_scores': rubric_scores,
        'strengths': [str(x) for x in (obj.get('strengths') or []) if str(x).strip()],
        'weaknesses': [str(x) for x in (obj.get('weaknesses') or []) if str(x).strip()],
    }


def build_aux_result_row(
    *,
    run_id: str,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
    temporal_eval: Dict[str, Any],
    family_aux_eval: Dict[str, Any],
) -> Dict[str, Any]:
    family = str(public_task.get('family') or result_row.get('family') or '')
    score_key = AUX_NAME_BY_FAMILY.get(family) or 'family_aux_score'
    return {
        'run_id': run_id,
        'task_id': public_task.get('task_id'),
        'family': family,
        'domain': infer_domain_id(result_row) or public_task.get('domain'),
        'method': str(result_row.get('agent') or result_row.get('baseline') or 'unknown'),
        'answer': str(result_row.get('answer') or ''),
        'metadata': {
            'task_title': public_task.get('title'),
            'time_cutoff': public_task.get('time_cutoff'),
            'schema_version': 'aux_v1',
        },
        'scores': {
            'temporal_leakage_score': round(float(temporal_eval.get('temporal_leakage_score') or 0.0), 4),
            score_key: round(float(family_aux_eval.get(score_key) or 0.0), 4),
        },
        'temporal_leakage_eval': temporal_eval,
        'family_aux_eval': family_aux_eval,
    }


def summarize_aux_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get('family') or '')].append(row)
        by_domain[str(row.get('domain') or '')].append(row)

    def _group_summary(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {}
        family = str(group[0].get('family') or '')
        score_key = AUX_NAME_BY_FAMILY.get(family) or 'family_aux_score'
        return {
            'count': len(group),
            'mean_temporal_leakage_score': _mean([float((row.get('scores') or {}).get('temporal_leakage_score') or 0.0) for row in group]),
            'mean_family_aux_score': _mean([float((row.get('scores') or {}).get(score_key) or 0.0) for row in group]),
            'family_aux_metric_name': AUX_LABEL_BY_FAMILY.get(family) or score_key,
            'family_aux_score_key': score_key,
        }

    family_summary = {key: _group_summary(group) for key, group in sorted(by_family.items())}
    domain_summary = {}
    for key, group in sorted(by_domain.items()):
        domain_summary[key] = {
            'count': len(group),
            'mean_temporal_leakage_score': _mean([float((row.get('scores') or {}).get('temporal_leakage_score') or 0.0) for row in group]),
        }
    return {
        'task_count': len(rows),
        'mean_temporal_leakage_score': _mean([float((row.get('scores') or {}).get('temporal_leakage_score') or 0.0) for row in rows]),
        'family_summary': family_summary,
        'domain_summary': domain_summary,
    }


def write_aux_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get('method') or '')].append(row)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Method', 'TemporalLeakage', 'BottleneckOpportunityAux', 'DirectionForecastAux', 'StrategicPlanningAux'])
        writer.writeheader()
        for method, group in sorted(by_method.items()):
            fam_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in group:
                fam_groups[str(row.get('family') or '')].append(row)
            writer.writerow({
                'Method': method,
                'TemporalLeakage': round(sum(float((row.get('scores') or {}).get('temporal_leakage_score') or 0.0) for row in group) / max(1, len(group)), 4),
                'BottleneckOpportunityAux': round(sum(float((row.get('scores') or {}).get('opportunity_grounding_score') or 0.0) for row in fam_groups.get('bottleneck_opportunity_discovery', [])) / max(1, len(fam_groups.get('bottleneck_opportunity_discovery', []))), 4) if fam_groups.get('bottleneck_opportunity_discovery') else 0.0,
                'DirectionForecastAux': round(sum(float((row.get('scores') or {}).get('forecast_grounding_score') or 0.0) for row in fam_groups.get('direction_forecasting', [])) / max(1, len(fam_groups.get('direction_forecasting', []))), 4) if fam_groups.get('direction_forecasting') else 0.0,
                'StrategicPlanningAux': round(sum(float((row.get('scores') or {}).get('technical_dependency_grounding_score') or 0.0) for row in fam_groups.get('strategic_research_planning', [])) / max(1, len(fam_groups.get('strategic_research_planning', []))), 4) if fam_groups.get('strategic_research_planning') else 0.0,
            })
