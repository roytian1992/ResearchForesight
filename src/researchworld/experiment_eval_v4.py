from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from researchworld.experiment_eval_v3 import infer_domain_id
from researchworld.llm import OpenAICompatChatClient, complete_json_object


RESEARCH_VALUE_DIMENSIONS_V4: List[Dict[str, Any]] = [
    {"name": "analytical_payoff", "weight": 0.55},
    {"name": "strategic_specificity", "weight": 0.45},
]

EVIDENCE_TRACEABILITY_DIMENSIONS_V4: List[Dict[str, Any]] = [
    {"name": "evidence_linkage", "weight": 0.55},
    {"name": "support_specificity", "weight": 0.45},
]

UNCERTAINTY_CALIBRATION_DIMENSIONS_V4: List[Dict[str, Any]] = [
    {"name": "confidence_calibration", "weight": 0.60},
    {"name": "boundary_awareness", "weight": 0.40},
]


def _weighted(dimensions: List[Dict[str, Any]], raw_scores: Dict[str, Any], fallback_key: str, obj: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
    dim_names = [str(item['name']) for item in dimensions]
    scores = {name: float(raw_scores.get(name) or 0.0) for name in dim_names}
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
        overall = float(obj.get(fallback_key) or 0.0)
    return round(float(overall), 4), {k: round(float(v), 4) for k, v in scores.items()}


def _truncate(text: Any, limit: int = 320) -> str:
    s = str(text or '').strip()
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)].rstrip() + '...'


def _compact_public_task(public_task: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        'task_id',
        'family',
        'domain',
        'title',
        'question',
        'time_cutoff',
        'expected_output',
    ]
    return {k: public_task.get(k) for k in keep if k in public_task}


def _collect_support_items(result_row: Dict[str, Any], max_items: int = 8) -> Dict[str, Any]:
    support: Dict[str, Any] = {
        'method': str(result_row.get('agent') or result_row.get('baseline') or 'unknown'),
        'retrieval_mode': '',
        'queries': [],
        'evidence_items': [],
        'trace_highlights': [],
        'diagnostics': {},
        'has_external_support': False,
    }

    direct_evidence = result_row.get('evidence') or {}
    if isinstance(direct_evidence, dict):
        support['retrieval_mode'] = str(direct_evidence.get('retrieval_mode') or support['retrieval_mode'] or '')
        support['queries'].extend([str(x) for x in (direct_evidence.get('queries') or []) if str(x).strip()])
        for item in direct_evidence.get('retrieved') or []:
            if not isinstance(item, dict):
                continue
            support['evidence_items'].append(
                {
                    'source': 'retrieved',
                    'evidence_id': item.get('evidence_id'),
                    'paper_id': item.get('paper_id'),
                    'paper_title': item.get('paper_title'),
                    'published_date': item.get('published_date'),
                    'venue': item.get('venue'),
                    'citations': item.get('citations'),
                    'snippet': item.get('snippet'),
                }
            )

    trace = result_row.get('trace') or {}
    if isinstance(trace, dict):
        support['retrieval_mode'] = str(trace.get('retrieval_mode') or support['retrieval_mode'] or '')
        support['diagnostics'] = trace.get('diagnostics') or {}
        for key in ['task_parse', 'policy', 'retrieval_plan', 'focus', 'signal_abstraction', 'mechanism_reasoning', 'trend', 'future', 'critique', 'family_head']:
            value = trace.get(key)
            if value is None:
                continue
            if isinstance(value, (dict, list)):
                rendered = _truncate(json.dumps(value, ensure_ascii=False), 420)
            else:
                rendered = _truncate(value, 420)
            if rendered:
                support['trace_highlights'].append({'name': key, 'value': rendered})
        if not support['queries']:
            support['queries'].extend([str(x) for x in (trace.get('queries') or []) if str(x).strip()])
        ev = trace.get('evidence') or {}
        if isinstance(ev, dict):
            for bucket_name in ['papers', 'structures', 'pageindex', 'fulltext', 'candidate_node_evidence']:
                for item in ev.get(bucket_name) or []:
                    if not isinstance(item, dict):
                        continue
                    support['evidence_items'].append(
                        {
                            'source': bucket_name,
                            'evidence_id': item.get('evidence_id') or item.get('paper_id') or item.get('node_id') or item.get('section_id'),
                            'paper_id': item.get('paper_id'),
                            'paper_title': item.get('paper_title') or item.get('title'),
                            'published_date': item.get('published_date'),
                            'venue': item.get('venue'),
                            'citations': item.get('citations'),
                            'snippet': item.get('snippet') or item.get('text') or item.get('summary') or item.get('content'),
                        }
                    )

    deduped_items: List[Dict[str, Any]] = []
    seen = set()
    for item in support['evidence_items']:
        key = (
            str(item.get('source') or ''),
            str(item.get('paper_id') or ''),
            str(item.get('evidence_id') or ''),
            _truncate(item.get('snippet') or '', 120),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped_items.append(item)
        if len(deduped_items) >= max_items:
            break
    support['evidence_items'] = deduped_items
    support['queries'] = support['queries'][:6]
    support['trace_highlights'] = support['trace_highlights'][:6]
    support['has_external_support'] = bool(support['evidence_items'] or support['queries'] or support['trace_highlights'])
    return support


def _render_support_snapshot(result_row: Dict[str, Any], *, max_items: int = 8) -> str:
    support = _collect_support_items(result_row, max_items=max_items)
    lines = [
        f"Method: {support['method']}",
        f"Retrieval mode: {support['retrieval_mode'] or 'none'}",
        f"Has external support artifact: {support['has_external_support']}",
    ]
    if support['queries']:
        lines.append('Queries:')
        for q in support['queries']:
            lines.append(f"- {q}")
    if support['evidence_items']:
        lines.append('Evidence items:')
        for idx, item in enumerate(support['evidence_items'], start=1):
            parts = [
                f"[{idx}] source={item.get('source')}",
                f"title={_truncate(item.get('paper_title') or item.get('paper_id') or item.get('evidence_id') or 'unknown', 120)}",
            ]
            if item.get('published_date'):
                parts.append(f"date={item.get('published_date')}")
            if item.get('venue'):
                parts.append(f"venue={item.get('venue')}")
            if item.get('citations') not in (None, ''):
                parts.append(f"citations={item.get('citations')}")
            snippet = _truncate(item.get('snippet') or '', 260)
            if snippet:
                parts.append(f"snippet={snippet}")
            lines.append(' | '.join(parts))
    if support['trace_highlights']:
        lines.append('Trace highlights:')
        for item in support['trace_highlights']:
            lines.append(f"- {item['name']}: {_truncate(item['value'], 260)}")
    diagnostics = support.get('diagnostics') or {}
    if diagnostics:
        small_diag = {k: diagnostics.get(k) for k in ['tool_calls', 'reflection_steps', 'memory_updates', 'revision_rounds', 'answer_changed_after_revision'] if k in diagnostics}
        if small_diag:
            lines.append(f"Diagnostics: {json.dumps(small_diag, ensure_ascii=False)}")
    if not support['has_external_support']:
        lines.append('No retrieval trace, evidence bundle, or explicit support artifact was attached to this answer.')
    return '\n'.join(lines)


def _blank_metric(score_key: str, dimensions: List[Dict[str, Any]], weakness: str) -> Dict[str, Any]:
    return {
        score_key: 0.0,
        'rubric_scores': {str(item['name']): 0.0 for item in dimensions},
        'strengths': [],
        'weaknesses': [weakness],
    }


def evaluate_research_value_v4(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    family: str,
    candidate_answer: str,
) -> Dict[str, Any]:
    if not str(candidate_answer or '').strip():
        return _blank_metric('research_value_score', RESEARCH_VALUE_DIMENSIONS_V4, 'Empty answer.')
    dimensions = RESEARCH_VALUE_DIMENSIONS_V4
    prompt = f"""# Role
You are a Senior Research Evaluator. Judge the research value of a candidate answer for a forward-looking research benchmark.

# Evaluation Philosophy
- Do NOT check factual correctness against any reference answer.
- Judge whether the answer would be genuinely useful to a research lead who must make decisions.
- Reward analytical compression, non-obvious synthesis, concrete trade-offs, and clear research payoff.
- Penalize generic survey writing, shallow listing, and safe but low-signal commentary.

# Input Data
- Public Task Definition: {json.dumps(_compact_public_task(public_task), ensure_ascii=False, indent=2)}
- Task Family: {family}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}

# Candidate Answer
{candidate_answer}

# Rubric
1. analytical_payoff: Does the answer surface mechanisms, bottlenecks, trade-offs, or decision-relevant causal structure rather than only describing topics?
2. strategic_specificity: Does it make concrete, domain-specific calls that would help prioritize research effort, instead of generic advice?

# Output (Strict JSON)
{{
  "dimension_scores": {{"dimension_name": 0.0}},
  "research_value_score": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict research-value judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=900,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, research_value_score, strengths, weaknesses.",
    )
    overall, rubric_scores = _weighted(dimensions, obj.get('dimension_scores') or {}, 'research_value_score', obj)
    return {
        'research_value_score': overall,
        'rubric_scores': rubric_scores,
        'strengths': [str(x) for x in (obj.get('strengths') or []) if str(x).strip()],
        'weaknesses': [str(x) for x in (obj.get('weaknesses') or []) if str(x).strip()],
    }


def evaluate_evidence_traceability_v4(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    family: str,
    candidate_answer: str,
    result_row: Dict[str, Any],
) -> Dict[str, Any]:
    if not str(candidate_answer or '').strip():
        return _blank_metric('evidence_traceability_score', EVIDENCE_TRACEABILITY_DIMENSIONS_V4, 'Empty answer.')
    dimensions = EVIDENCE_TRACEABILITY_DIMENSIONS_V4
    support_snapshot = _render_support_snapshot(result_row)
    prompt = f"""# Role
You are an Evidence Traceability Auditor for a research benchmark.

# Objective
Judge whether the answer's important conclusions can be traced to explicit support artifacts provided with the method output.

# Core Principles
- Do NOT judge factual truth against outside knowledge.
- Judge traceability: can a reviewer see where the answer came from?
- Reward explicit linkage from claims to papers, snippets, evidence bundles, or reasoning traces.
- Penalize answers that look strong but cannot be connected to identifiable support.
- If no support artifact is attached, score strictly unless the answer itself clearly states concrete evidence bases.

# Input Data
- Public Task Definition: {json.dumps(_compact_public_task(public_task), ensure_ascii=False, indent=2)}
- Task Family: {family}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}

# Candidate Answer
{candidate_answer}

# Attached Support Snapshot
{support_snapshot}

# Rubric
1. evidence_linkage: Are the answer's main claims visibly connected to explicit evidence items, retrieved papers, snippets, or trace steps?
2. support_specificity: Is the support concrete and discriminative enough that a reviewer could audit why the answer made these conclusions instead of generic alternatives?

# Output (Strict JSON)
{{
  "dimension_scores": {{"dimension_name": 0.0}},
  "evidence_traceability_score": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict evidence-traceability judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, evidence_traceability_score, strengths, weaknesses.",
    )
    overall, rubric_scores = _weighted(dimensions, obj.get('dimension_scores') or {}, 'evidence_traceability_score', obj)
    return {
        'evidence_traceability_score': overall,
        'rubric_scores': rubric_scores,
        'strengths': [str(x) for x in (obj.get('strengths') or []) if str(x).strip()],
        'weaknesses': [str(x) for x in (obj.get('weaknesses') or []) if str(x).strip()],
        'support_snapshot': support_snapshot,
    }


def evaluate_uncertainty_calibration_v4(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    family: str,
    candidate_answer: str,
    result_row: Dict[str, Any],
) -> Dict[str, Any]:
    if not str(candidate_answer or '').strip():
        return _blank_metric('uncertainty_calibration_score', UNCERTAINTY_CALIBRATION_DIMENSIONS_V4, 'Empty answer.')
    dimensions = UNCERTAINTY_CALIBRATION_DIMENSIONS_V4
    support_snapshot = _render_support_snapshot(result_row)
    prompt = f"""# Role
You are an Uncertainty Calibration Auditor for a research benchmark.

# Objective
Judge whether the answer expresses confidence in proportion to the support it appears to have.

# Core Principles
- Do NOT fact-check against outside knowledge.
- Judge whether the answer separates well-supported conclusions from hypotheses, forecasts, and open questions.
- Reward disciplined uncertainty, scope control, and acknowledgement of missing evidence or alternative explanations.
- Penalize false precision, over-claiming, and confident language that exceeds the visible support.

# Input Data
- Public Task Definition: {json.dumps(_compact_public_task(public_task), ensure_ascii=False, indent=2)}
- Task Family: {family}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}

# Candidate Answer
{candidate_answer}

# Attached Support Snapshot
{support_snapshot}

# Rubric
1. confidence_calibration: Does the certainty level of the answer match the apparent strength and granularity of the support?
2. boundary_awareness: Does the answer mark assumptions, uncertainty, unresolved questions, or places where evidence is incomplete?

# Output (Strict JSON)
{{
  "dimension_scores": {{"dimension_name": 0.0}},
  "uncertainty_calibration_score": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict uncertainty-calibration judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=900,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, uncertainty_calibration_score, strengths, weaknesses.",
    )
    overall, rubric_scores = _weighted(dimensions, obj.get('dimension_scores') or {}, 'uncertainty_calibration_score', obj)
    return {
        'uncertainty_calibration_score': overall,
        'rubric_scores': rubric_scores,
        'strengths': [str(x) for x in (obj.get('strengths') or []) if str(x).strip()],
        'weaknesses': [str(x) for x in (obj.get('weaknesses') or []) if str(x).strip()],
    }


def build_experiment_result_row_v4(
    *,
    run_id: str,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
    research_value_eval: Dict[str, Any],
    evidence_traceability_eval: Dict[str, Any],
    uncertainty_calibration_eval: Dict[str, Any],
) -> Dict[str, Any]:
    support = _collect_support_items(result_row)
    return {
        'run_id': run_id,
        'task_id': public_task.get('task_id'),
        'family': public_task.get('family') or result_row.get('family'),
        'domain': infer_domain_id(result_row) or public_task.get('domain'),
        'method': str(result_row.get('agent') or result_row.get('baseline') or 'unknown'),
        'answer': str(result_row.get('answer') or ''),
        'metadata': {
            'task_title': public_task.get('title'),
            'time_cutoff': public_task.get('time_cutoff'),
            'schema_version': 'v4',
        },
        'support_profile': {
            'retrieval_mode': support.get('retrieval_mode') or '',
            'has_external_support': bool(support.get('has_external_support')),
            'query_count': len(support.get('queries') or []),
            'evidence_item_count': len(support.get('evidence_items') or []),
            'trace_highlight_count': len(support.get('trace_highlights') or []),
        },
        'scores': {
            'research_value_score': round(float(research_value_eval.get('research_value_score') or 0.0), 4),
            'evidence_traceability_score': round(float(evidence_traceability_eval.get('evidence_traceability_score') or 0.0), 4),
            'uncertainty_calibration_score': round(float(uncertainty_calibration_eval.get('uncertainty_calibration_score') or 0.0), 4),
        },
        'research_value_eval': research_value_eval,
        'evidence_traceability_eval': evidence_traceability_eval,
        'uncertainty_calibration_eval': uncertainty_calibration_eval,
    }


def summarize_results_v4(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    rv_scores = [float((row.get('scores') or {}).get('research_value_score') or 0.0) for row in rows]
    et_scores = [float((row.get('scores') or {}).get('evidence_traceability_score') or 0.0) for row in rows]
    uc_scores = [float((row.get('scores') or {}).get('uncertainty_calibration_score') or 0.0) for row in rows]
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get('family') or '')].append(row)
        by_domain[str(row.get('domain') or '')].append(row)

    def _group_summary(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            'count': len(group),
            'mean_research_value_score': _mean([float((row.get('scores') or {}).get('research_value_score') or 0.0) for row in group]),
            'mean_evidence_traceability_score': _mean([float((row.get('scores') or {}).get('evidence_traceability_score') or 0.0) for row in group]),
            'mean_uncertainty_calibration_score': _mean([float((row.get('scores') or {}).get('uncertainty_calibration_score') or 0.0) for row in group]),
        }

    return {
        'task_count': len(rows),
        'mean_research_value_score': _mean(rv_scores),
        'mean_evidence_traceability_score': _mean(et_scores),
        'mean_uncertainty_calibration_score': _mean(uc_scores),
        'family_summary': {key: _group_summary(group) for key, group in sorted(by_family.items())},
        'domain_summary': {key: _group_summary(group) for key, group in sorted(by_domain.items())},
    }


def write_main_table_csv_v4(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get('method') or '')].append(row)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Method', 'ResearchValue', 'EvidenceTraceability', 'UncertaintyCalibration'])
        writer.writeheader()
        for method, group in sorted(by_method.items()):
            summary = summarize_results_v4(group)
            writer.writerow(
                {
                    'Method': method,
                    'ResearchValue': summary['mean_research_value_score'],
                    'EvidenceTraceability': summary['mean_evidence_traceability_score'],
                    'UncertaintyCalibration': summary['mean_uncertainty_calibration_score'],
                }
            )
