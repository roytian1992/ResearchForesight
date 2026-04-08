from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _mean(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--v31-dir', action='append', required=True, help='method=path')
    ap.add_argument('--v4-dir', action='append', required=True, help='method=path')
    ap.add_argument('--aux-dir', action='append', required=True, help='method=path')
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()

    def parse_map(items: List[str]) -> Dict[str, Path]:
        out = {}
        for item in items:
            method, path = item.split('=', 1)
            out[method] = Path(path)
        return out

    v31_map = parse_map(args.v31_dir)
    v4_map = parse_map(args.v4_dir)
    aux_map = parse_map(args.aux_dir)
    methods = sorted(set(v31_map) & set(v4_map) & set(aux_map))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    family_rows = []
    for method in methods:
        v31 = _load_jsonl(v31_map[method] / 'results_eval_v3_1.jsonl')
        v4 = _load_jsonl(v4_map[method] / 'results_eval_v4.jsonl')
        aux = _load_jsonl(aux_map[method] / 'results_eval_aux.jsonl')
        by_task: Dict[str, Dict[str, Any]] = {}
        for row in v31:
            by_task.setdefault(str(row['task_id']), {})['v31'] = row
        for row in v4:
            by_task.setdefault(str(row['task_id']), {})['v4'] = row
        for row in aux:
            by_task.setdefault(str(row['task_id']), {})['aux'] = row
        merged = []
        for task_id, pack in by_task.items():
            if set(pack) == {'v31', 'v4', 'aux'}:
                merged.append({
                    'task_id': task_id,
                    'family': pack['v31']['family'],
                    'domain': pack['v31']['domain'],
                    'fact': float(pack['v31']['scores']['fact_precision_score']),
                    'future_alignment': float(pack['v31']['scores']['future_alignment_score']),
                    'task_fulfillment': float(pack['v31']['scores']['task_fulfillment_score']),
                    'evidence_traceability': float(pack['v4']['scores']['evidence_traceability_score']),
                    'temporal_leakage': float(pack['aux']['scores']['temporal_leakage_score']),
                    'opportunity_grounding': float(pack['aux']['scores'].get('opportunity_grounding_score') or 0.0),
                    'forecast_grounding': float(pack['aux']['scores'].get('forecast_grounding_score') or 0.0),
                    'technical_dependency_grounding': float(pack['aux']['scores'].get('technical_dependency_grounding_score') or 0.0),
                })
        if not merged:
            continue
        rows.append({
            'Method': method,
            'Fact': _mean([r['fact'] for r in merged]),
            'FutureAlignment': _mean([r['future_alignment'] for r in merged]),
            'EvidenceTraceability': _mean([r['evidence_traceability'] for r in merged]),
            'TemporalLeakage': _mean([r['temporal_leakage'] for r in merged]),
            'TaskFulfillment': _mean([r['task_fulfillment'] for r in merged]),
            'OpportunityGrounding': _mean([r['opportunity_grounding'] for r in merged if r['family']=='bottleneck_opportunity_discovery']),
            'ForecastGrounding': _mean([r['forecast_grounding'] for r in merged if r['family']=='direction_forecasting']),
            'TechnicalDependencyGrounding': _mean([r['technical_dependency_grounding'] for r in merged if r['family']=='strategic_research_planning']),
            'TaskCount': len(merged),
        })
        fam_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in merged:
            fam_groups[r['family']].append(r)
        for fam, grp in sorted(fam_groups.items()):
            family_rows.append({
                'Method': method,
                'Family': fam,
                'Fact': _mean([r['fact'] for r in grp]),
                'FutureAlignment': _mean([r['future_alignment'] for r in grp]),
                'EvidenceTraceability': _mean([r['evidence_traceability'] for r in grp]),
                'TemporalLeakage': _mean([r['temporal_leakage'] for r in grp]),
                'TaskFulfillment': _mean([r['task_fulfillment'] for r in grp]),
                'FamilyAux': _mean([
                    r['opportunity_grounding'] if fam=='bottleneck_opportunity_discovery' else
                    r['forecast_grounding'] if fam=='direction_forecasting' else
                    r['technical_dependency_grounding']
                    for r in grp
                ]),
                'TaskCount': len(grp),
            })

    with (out_dir / 'final_overall.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / 'final_by_family.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        writer.writeheader()
        writer.writerows(family_rows)
    (out_dir / 'final_overall.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'final_by_family.json').write_text(json.dumps(family_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'overall': rows, 'by_family_count': len(family_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
