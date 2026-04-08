from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from researchworld.experiment_eval_v3 import infer_domain_id
from researchworld.future_alignment_eval_v3_1 import evaluate_future_alignment_v3_1
from researchworld.llm import OpenAICompatChatClient, complete_json_object


TASK_FULFILLMENT_DIMENSIONS_V3_1: List[Dict[str, Any]] = [
    {"name": "component_coverage", "weight": 0.35},
    {"name": "deliverable_coherence", "weight": 0.25},
    {"name": "conclusion_commitment", "weight": 0.20},
    {"name": "constraint_compliance", "weight": 0.20},
]

STRATEGIC_INTELLIGENCE_DIMENSIONS_V3_1: List[Dict[str, Any]] = [
    {"name": "analytical_depth", "weight": 0.30},
    {"name": "framework_sophistication", "weight": 0.25},
    {"name": "temporal_intelligence", "weight": 0.20},
    {"name": "task_family_fit", "weight": 0.25},
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


def evaluate_component_task_fulfillment_v3_1(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    candidate_answer: str,
) -> Dict[str, Any]:
    dimensions = TASK_FULFILLMENT_DIMENSIONS_V3_1
    components = list(((hidden_row.get('component_targets') or {}).get('components') or []))
    compact_components = [
        {
            'name': c.get('name'),
            'description': c.get('description'),
            'importance': c.get('importance'),
            'canonical_values': c.get('canonical_values'),
            'notes': c.get('notes'),
        }
        for c in components
    ]
    prompt = f"""# Role
You are a Task Fulfillment Auditor for a research benchmark.

# Objective
Judge whether the candidate answer actually completes the requested task, with attention to the hidden component structure of the task.

# Core Principles
- Evaluate task completion quality, not factual correctness.
- Use the component targets as a hidden rubric for what the task substantively requires.
- Reward semantically correct coverage; do not require exact phrase overlap.
- Penalize answers that sound polished but fail to deliver the requested analytical payload.

# Input Data
- Public Task Definition: {json.dumps(public_task, ensure_ascii=False, indent=2)}
- Task Family: {hidden_row.get('family')}
- Hidden Component Targets: {json.dumps(compact_components, ensure_ascii=False, indent=2)}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}

# Candidate Answer under Review
{candidate_answer}

# Rubric
1. component_coverage: Does the answer cover the key hidden components required by the task family?
2. deliverable_coherence: Does the answer connect the requested elements into one coherent analysis rather than a disjoint list?
3. conclusion_commitment: Does the answer make concrete commitments, priorities, trajectory calls, bottlenecks, opportunities, or plans?
4. constraint_compliance: Does the answer respect time framing, output constraints, and any required linkage between requested elements?

# Output (Strict JSON)
{{
  "dimension_scores": {{"dimension_name": 0.0}},
  "task_fulfillment_score": 0.0,
  "component_scores": {{"component_name": 0.0}},
  "strengths": ["..."],
  "weaknesses": ["..."]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict component-aware task-fulfillment judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1100,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, task_fulfillment_score, component_scores, strengths, weaknesses.",
    )
    overall, rubric_scores = _weighted(dimensions, obj.get('dimension_scores') or {}, 'task_fulfillment_score', obj)
    component_raw = obj.get('component_scores') or {}
    component_scores = {}
    for c in components:
        name = str(c.get('name') or '')
        if not name:
            continue
        component_scores[name] = round(float(component_raw.get(name) or 0.0), 4)
    return {
        'task_fulfillment_score': overall,
        'rubric_scores': rubric_scores,
        'component_scores': component_scores,
        'strengths': [str(x) for x in (obj.get('strengths') or []) if str(x).strip()],
        'weaknesses': [str(x) for x in (obj.get('weaknesses') or []) if str(x).strip()],
    }


def evaluate_strategic_intelligence_v3_1(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    candidate_answer: str,
) -> Dict[str, Any]:
    dimensions = STRATEGIC_INTELLIGENCE_DIMENSIONS_V3_1
    prompt = f"""# Role
You are an Advanced Research Auditor and Strategy Consultant. Your mission is to evaluate the intellectual grade of a candidate's research output without using any reference answer.

# Evaluation Philosophy
- Judge latent reasoning quality, structural clarity, and strategic usefulness.
- Do NOT evaluate factual correctness.
- Do NOT compare against hidden ground truth.
- Prefer concise, high-signal reasoning over long but generic prose.

# Input Data
- Public Task Definition: {json.dumps(public_task, ensure_ascii=False, indent=2)}
- Task Family: {hidden_row.get('family')}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}

# Candidate Answer under Review
{candidate_answer}

# Evaluation Pillars
1. analytical_depth: Does the answer move beyond description into mechanisms, trade-offs, or causal structure?
2. framework_sophistication: Does it organize ideas into a useful structure instead of a flat list?
3. temporal_intelligence: Does it separate historical evidence, present assessment, and future projection in a disciplined way?
4. task_family_fit: Does it satisfy the specific demands of this task family?

# Output (Strict JSON)
{{
  "dimension_scores": {{"dimension_name": 0.0}},
  "strategic_intelligence_score": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict strategic-intelligence judge for a research benchmark. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, strategic_intelligence_score, strengths, weaknesses.",
    )
    overall, rubric_scores = _weighted(dimensions, obj.get('dimension_scores') or {}, 'strategic_intelligence_score', obj)
    return {
        'strategic_intelligence_score': overall,
        'rubric_scores': rubric_scores,
        'strengths': [str(x) for x in (obj.get('strengths') or []) if str(x).strip()],
        'weaknesses': [str(x) for x in (obj.get('weaknesses') or []) if str(x).strip()],
    }


def build_experiment_result_row_v3_1(
    *,
    run_id: str,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    result_row: Dict[str, Any],
    fact_eval: Dict[str, Any],
    task_fulfillment_eval: Dict[str, Any],
    strategic_eval: Dict[str, Any],
    future_alignment_eval: Dict[str, Any],
) -> Dict[str, Any]:
    trace = result_row.get('trace') or {}
    diagnostics = trace.get('diagnostics') or {}
    evidence = trace.get('evidence') or {}
    retrieval_diag = {
        'retrieval_mode': trace.get('retrieval_mode') or ((result_row.get('evidence') or {}).get('retrieval_mode') if isinstance(result_row.get('evidence'), dict) else ''),
        'retrieved_doc_count': len((evidence.get('papers') or [])) + len((evidence.get('structures') or [])) + len((evidence.get('pageindex') or [])) + len((evidence.get('fulltext') or [])),
        'unique_paper_count': len({str(x.get('paper_id') or '') for x in (evidence.get('papers') or []) if str(x.get('paper_id') or '').strip()}),
        'pageindex_hit_count': len(evidence.get('pageindex') or []),
        'fulltext_hit_count': len(evidence.get('fulltext') or []),
        'tool_call_count': int(diagnostics.get('tool_calls') or 0),
    }
    agent_diag = {
        'reflection_steps': int(diagnostics.get('reflection_steps') or 0),
        'memory_updates': int(diagnostics.get('memory_updates') or 0),
        'revision_rounds': int(diagnostics.get('revision_rounds') or 0),
        'answer_changed_after_revision': bool(diagnostics.get('answer_changed_after_revision') or False),
    }
    return {
        'run_id': run_id,
        'task_id': public_task.get('task_id'),
        'family': public_task.get('family'),
        'domain': infer_domain_id(result_row),
        'method': str(result_row.get('agent') or result_row.get('baseline') or 'unknown'),
        'answer': str(result_row.get('answer') or ''),
        'metadata': {
            'task_title': public_task.get('title'),
            'time_cutoff': public_task.get('time_cutoff'),
            'schema_version': 'v3.1',
        },
        'retrieval_diagnostics': retrieval_diag,
        'agent_diagnostics': agent_diag,
        'scores': {
            'fact_precision_score': round(float(fact_eval.get('benchmark_factscore') or 0.0), 4),
            'task_fulfillment_score': round(float(task_fulfillment_eval.get('task_fulfillment_score') or 0.0), 4),
            'strategic_intelligence_score': round(float(strategic_eval.get('strategic_intelligence_score') or 0.0), 4),
            'future_alignment_score': round(float(future_alignment_eval.get('future_alignment_score') or 0.0), 4),
        },
        'fact_eval': fact_eval,
        'task_fulfillment_eval': task_fulfillment_eval,
        'strategic_eval': strategic_eval,
        'future_alignment_eval': future_alignment_eval,
    }


def summarize_results_v3_1(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    fact_scores = [float((row.get('scores') or {}).get('fact_precision_score') or 0.0) for row in rows]
    tf_scores = [float((row.get('scores') or {}).get('task_fulfillment_score') or 0.0) for row in rows]
    si_scores = [float((row.get('scores') or {}).get('strategic_intelligence_score') or 0.0) for row in rows]
    fa_scores = [float((row.get('scores') or {}).get('future_alignment_score') or 0.0) for row in rows]
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get('family') or '')].append(row)
        by_domain[str(row.get('domain') or '')].append(row)

    def _group_summary(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            'count': len(group),
            'mean_fact_precision_score': _mean([float((row.get('scores') or {}).get('fact_precision_score') or 0.0) for row in group]),
            'mean_task_fulfillment_score': _mean([float((row.get('scores') or {}).get('task_fulfillment_score') or 0.0) for row in group]),
            'mean_strategic_intelligence_score': _mean([float((row.get('scores') or {}).get('strategic_intelligence_score') or 0.0) for row in group]),
            'mean_future_alignment_score': _mean([float((row.get('scores') or {}).get('future_alignment_score') or 0.0) for row in group]),
        }

    return {
        'task_count': len(rows),
        'mean_fact_precision_score': _mean(fact_scores),
        'mean_task_fulfillment_score': _mean(tf_scores),
        'mean_strategic_intelligence_score': _mean(si_scores),
        'mean_future_alignment_score': _mean(fa_scores),
        'family_summary': {key: _group_summary(group) for key, group in sorted(by_family.items())},
        'domain_summary': {key: _group_summary(group) for key, group in sorted(by_domain.items())},
    }


def write_main_table_csv_v3_1(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get('method') or '')].append(row)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Method', 'FactScore', 'TaskFulfillment', 'StrategicIntelligence', 'FutureAlignment'])
        writer.writeheader()
        for method, group in sorted(by_method.items()):
            summary = summarize_results_v3_1(group)
            writer.writerow(
                {
                    'Method': method,
                    'FactScore': summary['mean_fact_precision_score'],
                    'TaskFulfillment': summary['mean_task_fulfillment_score'],
                    'StrategicIntelligence': summary['mean_strategic_intelligence_score'],
                    'FutureAlignment': summary['mean_future_alignment_score'],
                }
            )
