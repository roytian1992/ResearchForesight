from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from researchworld.experiment_eval_v3 import infer_domain_id


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _fact_component_summary(group: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        'precision_score': _mean([float(((row.get('fact_eval') or {}).get('precision_score') or 0.0)) for row in group]),
        'coverage_score': _mean([float(((row.get('fact_eval') or {}).get('coverage_score') or 0.0)) for row in group]),
    }


def _future_alignment_component_summary(group: List[Dict[str, Any]]) -> Dict[str, float]:
    return {
        'weighted_unit_alignment': _mean([float(((row.get('future_alignment_eval') or {}).get('weighted_unit_alignment') or 0.0)) for row in group]),
        'alignment_coverage': _mean([float(((row.get('future_alignment_eval') or {}).get('alignment_coverage') or 0.0)) for row in group]),
        'mean_specificity': _mean([float(((row.get('future_alignment_eval') or {}).get('mean_specificity') or 0.0)) for row in group]),
    }


def build_experiment_result_row_v3_1(
    *,
    run_id: str,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    result_row: Dict[str, Any],
    fact_eval: Dict[str, Any],
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
            'future_alignment_score': round(float(future_alignment_eval.get('future_alignment_score') or 0.0), 4),
        },
        'fact_eval': fact_eval,
        'future_alignment_eval': future_alignment_eval,
    }


def summarize_results_v3_1(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fact_scores = [float((row.get('scores') or {}).get('fact_precision_score') or 0.0) for row in rows]
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
            'mean_future_alignment_score': _mean([float((row.get('scores') or {}).get('future_alignment_score') or 0.0) for row in group]),
            'fact_component_summary': _fact_component_summary(group),
            'future_alignment_component_summary': _future_alignment_component_summary(group),
        }

    return {
        'task_count': len(rows),
        'mean_fact_precision_score': _mean(fact_scores),
        'mean_future_alignment_score': _mean(fa_scores),
        'fact_component_summary': _fact_component_summary(rows),
        'future_alignment_component_summary': _future_alignment_component_summary(rows),
        'family_summary': {key: _group_summary(group) for key, group in sorted(by_family.items())},
        'domain_summary': {key: _group_summary(group) for key, group in sorted(by_domain.items())},
    }


def write_main_table_csv_v3_1(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get('method') or '')].append(row)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Method', 'FactScore', 'FutureAlignment'])
        writer.writeheader()
        for method, group in sorted(by_method.items()):
            summary = summarize_results_v3_1(group)
            writer.writerow(
                {
                    'Method': method,
                    'FactScore': summary['mean_fact_precision_score'],
                    'FutureAlignment': summary['mean_future_alignment_score'],
                }
            )


def write_breakdown_csv_v3_1(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_results_v3_1(rows)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scope_type', 'scope_value', 'metric', 'component', 'mean_score', 'count'])
        writer.writeheader()

        def _emit(scope_type: str, scope_value: str, count: int, metric: str, components: Dict[str, float]) -> None:
            for component, score in components.items():
                writer.writerow(
                    {
                        'scope_type': scope_type,
                        'scope_value': scope_value,
                        'metric': metric,
                        'component': component,
                        'mean_score': score,
                        'count': count,
                    }
                )

        _emit('overall', 'all', int(summary.get('task_count') or 0), 'fact_precision', summary.get('fact_component_summary') or {})
        _emit('overall', 'all', int(summary.get('task_count') or 0), 'future_alignment', summary.get('future_alignment_component_summary') or {})
        for family, group in sorted((summary.get('family_summary') or {}).items()):
            _emit('family', family, int(group.get('count') or 0), 'fact_precision', group.get('fact_component_summary') or {})
            _emit('family', family, int(group.get('count') or 0), 'future_alignment', group.get('future_alignment_component_summary') or {})
        for domain, group in sorted((summary.get('domain_summary') or {}).items()):
            _emit('domain', domain, int(group.get('count') or 0), 'fact_precision', group.get('fact_component_summary') or {})
            _emit('domain', domain, int(group.get('count') or 0), 'future_alignment', group.get('future_alignment_component_summary') or {})
