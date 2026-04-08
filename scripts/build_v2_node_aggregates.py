from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl

DOMAINS = [
    'llm_agent',
    'llm_finetuning_post_training',
    'rag_and_retrieval_structuring',
    'visual_generative_modeling_and_diffusion',
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build v2 node-level quarterly aggregate tables.')
    p.add_argument('--domains', default=','.join(DOMAINS), help='Comma-separated domain ids.')
    p.add_argument('--out-dir', default=str(ROOT / 'data' / 'aggregates'), help='Output directory.')
    return p.parse_args()


def load_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_core_scope_ids(labels_path: Path) -> set[str]:
    ids: set[str] = set()
    if not labels_path.exists():
        return ids
    for row in iter_jsonl(labels_path):
        if str(row.get('scope_decision') or '') == 'core_domain':
            pid = str(row.get('paper_id') or '').strip()
            if pid:
                ids.add(pid)
    return ids


def load_domain_papers(domain: str) -> dict[str, dict[str, Any]]:
    clean_enriched = ROOT / 'data' / 'domains' / domain / 'clean' / 'core_papers.publication_enriched.semanticscholar.jsonl'
    if clean_enriched.exists():
        return {str(r['paper_id']): r for r in iter_jsonl(clean_enriched)}

    interim_enriched = ROOT / 'data' / 'domains' / domain / 'interim' / 'papers_merged.publication_enriched.semanticscholar.jsonl'
    labels_path = ROOT / 'data' / 'domains' / domain / 'annotations' / 'paper_labels.jsonl'
    core_ids = load_core_scope_ids(labels_path)
    papers: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(interim_enriched):
        pid = str(row.get('paper_id') or '').strip()
        if pid and pid in core_ids:
            papers[pid] = row
    return papers


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            ent -= p * math.log(p, 2)
    return ent


def build_assignment_index(assignments_path: Path, nodes_path: Path) -> dict[str, set[str]]:
    nodes = load_json(nodes_path)
    parent_by_id = {row['node_id']: row.get('parent_id') for row in nodes}
    per_node: dict[str, set[str]] = defaultdict(set)

    for row in iter_jsonl(assignments_path):
        pid = str(row.get('paper_id') or '').strip()
        if not pid:
            continue
        dim_assignments = row.get('dimension_assignments') or {}
        seen_nodes: set[str] = set()
        for _dim, items in dim_assignments.items():
            for item in items or []:
                node_id = str(item.get('node_id') or '').strip()
                current = node_id
                while current:
                    if current not in seen_nodes:
                        per_node[current].add(pid)
                        seen_nodes.add(current)
                    current = parent_by_id.get(current)
    return per_node


def node_row_stats(node_meta: dict[str, Any], paper_ids: set[str], paper_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    venue_counter: Counter[str] = Counter()
    venue_bucket_counter: Counter[str] = Counter()
    venue_type_counter: Counter[str] = Counter()
    citations: list[int] = []
    top_conf_count = 0

    for pid in paper_ids:
        row = paper_map.get(pid)
        if not row:
            continue
        enrich = row.get('publication_enrichment') or {}
        citations.append(int(enrich.get('preferred_cited_by_count') or 0))
        if bool(enrich.get('is_top_ai_venue')):
            top_conf_count += 1
        venue_name = str(enrich.get('published_venue_name') or enrich.get('semantic_scholar_venue') or 'unknown').strip() or 'unknown'
        venue_counter[venue_name] += 1
        venue_bucket = str(enrich.get('top_venue_bucket') or 'other').strip() or 'other'
        venue_bucket_counter[venue_bucket] += 1
        venue_type = str(enrich.get('published_venue_type') or 'unknown').strip() or 'unknown'
        venue_type_counter[venue_type] += 1

    paper_count = len(paper_ids)
    return {
        'node_id': node_meta['node_id'],
        'dimension_id': node_meta['dimension_id'],
        'label': node_meta['label'],
        'display_name': node_meta['display_name'],
        'level': node_meta['level'],
        'created_time_slice': node_meta.get('created_time_slice'),
        'paper_count': paper_count,
        'top_conf_count': top_conf_count,
        'top_conf_share': round(top_conf_count / paper_count, 4) if paper_count else 0.0,
        'citation_median': round(float(median(citations)), 4) if citations else 0.0,
        'citation_p75': round(percentile(citations, 0.75), 4),
        'citation_p90': round(percentile(citations, 0.90), 4),
        'venue_entropy': round(entropy(venue_counter), 4),
        'top_venue_buckets': dict(venue_bucket_counter.most_common(10)),
        'venue_types': dict(venue_type_counter),
        'top_venues': dict(venue_counter.most_common(10)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = [x.strip() for x in args.domains.split(',') if x.strip()]

    all_rows: list[dict[str, Any]] = []
    growth_rows: list[dict[str, Any]] = []
    domain_overview: list[dict[str, Any]] = []

    for domain in domains:
        paper_map = load_domain_papers(domain)
        base = ROOT / 'data' / 'domains' / domain / 'taxoadapt_quarterly' / 'quarterly_v1'
        summaries = sorted(base.glob('*/summary.json'))
        prior_counts: dict[str, int] = {}
        last_summary = None
        for summary_path in summaries:
            slice_id = summary_path.parent.name
            nodes_path = summary_path.parent / 'taxonomy_nodes.json'
            assignments_path = summary_path.parent / 'paper_assignments.jsonl'
            nodes = load_json(nodes_path)
            assigned = build_assignment_index(assignments_path, nodes_path)
            current_counts: dict[str, int] = {}
            for node in nodes:
                row = node_row_stats(node, assigned.get(node['node_id'], set()), paper_map)
                row['domain'] = domain
                row['time_slice'] = slice_id
                row['new_paper_count'] = load_json(summary_path).get('new_paper_count')
                row['cumulative_paper_count'] = load_json(summary_path).get('cumulative_paper_count')
                all_rows.append(row)
                current_counts[node['node_id']] = int(row['paper_count'])
                growth_rows.append({
                    'domain': domain,
                    'time_slice': slice_id,
                    'node_id': node['node_id'],
                    'display_name': node['display_name'],
                    'dimension_id': node['dimension_id'],
                    'paper_count': int(row['paper_count']),
                    'previous_paper_count': int(prior_counts.get(node['node_id'], 0)),
                    'paper_growth': int(row['paper_count']) - int(prior_counts.get(node['node_id'], 0)),
                    'top_conf_share': row['top_conf_share'],
                    'citation_median': row['citation_median'],
                })
            prior_counts = current_counts
            last_summary = load_json(summary_path)

        if last_summary is not None:
            final_summary = load_json(base / 'final_summary.json')
            domain_overview.append({
                'domain': domain,
                'slice_count': len(summaries),
                'final_time_slice': summaries[-1].parent.name,
                'final_node_count': final_summary.get('node_count'),
                'final_cumulative_papers': last_summary.get('cumulative_paper_count'),
                'final_new_papers': last_summary.get('new_paper_count'),
                'core_paper_map_size': len(paper_map),
            })

    (out_dir / 'quarterly_node_aggregates.json').write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'quarterly_node_growth.json').write_text(json.dumps(growth_rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'domain_overview.json').write_text(json.dumps(domain_overview, ensure_ascii=False, indent=2), encoding='utf-8')

    def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                flat = dict(row)
                for k, v in list(flat.items()):
                    if isinstance(v, (dict, list)):
                        flat[k] = json.dumps(v, ensure_ascii=False)
                writer.writerow({k: flat.get(k) for k in fields})

    write_csv(
        out_dir / 'quarterly_node_aggregates.csv',
        all_rows,
        ['domain','time_slice','dimension_id','node_id','display_name','level','created_time_slice','paper_count','top_conf_count','top_conf_share','citation_median','citation_p75','citation_p90','venue_entropy','new_paper_count','cumulative_paper_count','top_venue_buckets','venue_types','top_venues'],
    )
    write_csv(
        out_dir / 'quarterly_node_growth.csv',
        growth_rows,
        ['domain','time_slice','dimension_id','node_id','display_name','previous_paper_count','paper_count','paper_growth','top_conf_share','citation_median'],
    )
    write_csv(
        out_dir / 'domain_overview.csv',
        domain_overview,
        ['domain','slice_count','final_time_slice','final_node_count','final_cumulative_papers','final_new_papers','core_paper_map_size'],
    )

    print('wrote', out_dir)
    print('aggregate_rows', len(all_rows))
    print('growth_rows', len(growth_rows))
    print('domains', len(domain_overview))


if __name__ == '__main__':
    main()
