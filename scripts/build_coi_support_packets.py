from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl

DEFAULT_DOMAINS = [
    'llm_agent',
    'llm_finetuning_post_training',
    'rag_and_retrieval_structuring',
    'visual_generative_modeling_and_diffusion',
]
DEFAULT_HISTORY_STRUCTURE_SLICE = '2025Q2'
DEFAULT_HISTORY_END = '2025-08-31'
WINDOWS = {
    'quarterly_2025q4': ('2025-09-01', '2025-12-01'),
    'quarterly_2026q1': ('2025-12-01', '2026-03-01'),
    'halfyear_2025q4_2026q1': ('2025-09-01', '2026-03-01'),
}
ALLOWED_DIMENSIONS = {'tasks', 'methodologies', 'evaluation_methods'}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build CoI-style support packets from taxonomy and enriched papers.')
    p.add_argument('--domains', default=','.join(DEFAULT_DOMAINS))
    p.add_argument('--history-structure-slice', default=DEFAULT_HISTORY_STRUCTURE_SLICE)
    p.add_argument('--history-end', default=DEFAULT_HISTORY_END)
    p.add_argument('--out-dir', default=str(ROOT / 'data' / 'support_packets'))
    p.add_argument('--top-k-per-domain', type=int, default=36)
    return p.parse_args()


def slice_order(domain: str) -> list[str]:
    fs = json.loads((ROOT / 'data' / 'domains' / domain / 'taxoadapt_quarterly' / 'quarterly_v1' / 'final_summary.json').read_text())
    return [row['time_slice'] for row in fs['years']]


def to_date(s: str) -> date:
    return date.fromisoformat(s[:10])


def load_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_core_papers(domain: str) -> dict[str, dict[str, Any]]:
    clean = ROOT / 'data' / 'domains' / domain / 'clean' / 'core_papers.publication_enriched.semanticscholar.jsonl'
    if clean.exists():
        return {str(r['paper_id']): r for r in iter_jsonl(clean)}
    labels = {str(r['paper_id']) for r in iter_jsonl(ROOT / 'data' / 'domains' / domain / 'annotations' / 'paper_labels.jsonl') if str(r.get('scope_decision') or '') == 'core_domain'}
    interim = ROOT / 'data' / 'domains' / domain / 'interim' / 'papers_merged.publication_enriched.semanticscholar.jsonl'
    return {str(r['paper_id']): r for r in iter_jsonl(interim) if str(r.get('paper_id')) in labels}


def enrich_fields(row: dict[str, Any]) -> dict[str, Any]:
    enrich = row.get('publication_enrichment') or {}
    return {
        'top_conf': bool(enrich.get('is_top_ai_venue')),
        'top_bucket': str(enrich.get('top_venue_bucket') or 'other'),
        'venue': str(enrich.get('published_venue_name') or enrich.get('semantic_scholar_venue') or 'unknown'),
        'venue_type': str(enrich.get('published_venue_type') or 'unknown'),
        'citation': int(enrich.get('preferred_cited_by_count') or 0),
    }


def rank_papers(paper_ids: list[str], paper_map: dict[str, dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    for pid in paper_ids:
        row = paper_map.get(pid)
        if not row:
            continue
        ef = enrich_fields(row)
        rows.append((
            1 if ef['top_conf'] else 0,
            ef['citation'],
            str(row.get('published') or row.get('published_date') or ''),
            {
                'paper_id': pid,
                'title': row.get('title'),
                'published': row.get('published') or row.get('published_date'),
                'venue': ef['venue'],
                'top_venue_bucket': ef['top_bucket'],
                'is_top_conference': ef['top_conf'],
                'citation': ef['citation'],
            }
        ))
    rows.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [x[3] for x in rows[:limit]]


def aggregate_paper_stats(paper_ids: set[str], paper_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not paper_ids:
        return {
            'paper_count': 0,
            'top_conf_count': 0,
            'top_conf_share': 0.0,
            'citation_median': 0.0,
            'top_venue_buckets': {},
            'top_venues': {},
        }
    citations = []
    top = 0
    buckets = Counter()
    venues = Counter()
    for pid in paper_ids:
        row = paper_map.get(pid)
        if not row:
            continue
        ef = enrich_fields(row)
        citations.append(ef['citation'])
        if ef['top_conf']:
            top += 1
        buckets[ef['top_bucket']] += 1
        venues[ef['venue']] += 1
    citations.sort()
    mid = citations[len(citations)//2] if citations else 0
    return {
        'paper_count': len(paper_ids),
        'top_conf_count': top,
        'top_conf_share': round(top / len(paper_ids), 4) if paper_ids else 0.0,
        'citation_median': float(mid),
        'top_venue_buckets': dict(buckets.most_common(5)),
        'top_venues': dict(venues.most_common(5)),
    }


def build_parent_maps(nodes: list[dict[str, Any]]) -> tuple[dict[str, str | None], dict[str, dict[str, Any]], dict[str, list[str]]]:
    parent = {}
    meta = {}
    children = defaultdict(list)
    for row in nodes:
        nid = row['node_id']
        parent[nid] = row.get('parent_id')
        meta[nid] = row
        if row.get('parent_id'):
            children[row['parent_id']].append(nid)
    return parent, meta, children


def created_slice(meta: dict[str, Any]) -> str:
    val = meta.get('created_time_slice') or meta.get('created_year') or '2023Q1'
    return str(val)


def climb_to_history_node(leaf_id: str, hist_slice: str, order_index: dict[str, int], parent_by_id: dict[str, str | None], meta_by_id: dict[str, dict[str, Any]]) -> str | None:
    current = leaf_id
    hist_idx = order_index[hist_slice]
    while current:
        meta = meta_by_id[current]
        created = created_slice(meta)
        if order_index.get(created, hist_idx) <= hist_idx:
            return current
        current = parent_by_id.get(current)
    return None


def descendants_after_hist(node_id: str, hist_slice: str, order_index: dict[str, int], children_by_id: dict[str, list[str]], meta_by_id: dict[str, dict[str, Any]]) -> list[str]:
    hist_idx = order_index[hist_slice]
    out = []
    stack = list(children_by_id.get(node_id, []))
    while stack:
        nid = stack.pop()
        if order_index.get(created_slice(meta_by_id[nid]), hist_idx) > hist_idx:
            out.append(nid)
        stack.extend(children_by_id.get(nid, []))
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = [d.strip() for d in args.domains.split(',') if d.strip()]

    all_packets = []
    all_seeds = []

    history_end = to_date(args.history_end)
    window_dates = {k: (to_date(v[0]), to_date(v[1])) for k, v in WINDOWS.items()}

    for domain in domains:
        order = slice_order(domain)
        order_index = {s: i for i, s in enumerate(order)}
        final_dir = ROOT / 'data' / 'domains' / domain / 'taxoadapt_quarterly' / 'quarterly_v1' / '2026Q1'
        nodes = load_json(final_dir / 'taxonomy_nodes.json')
        assignments = list(iter_jsonl(final_dir / 'paper_assignments.jsonl'))
        parent_by_id, meta_by_id, children_by_id = build_parent_maps(nodes)
        paper_map = load_core_papers(domain)

        hist_nodes = {
            row['node_id'] for row in nodes
            if row['dimension_id'] in ALLOWED_DIMENSIONS and row['level'] >= 1 and order_index.get(created_slice(row), 0) <= order_index[args.history_structure_slice]
        }

        hist_papers_by_node: dict[str, set[str]] = defaultdict(set)
        window_papers_by_node: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        future_desc_by_node: dict[str, Counter[str]] = defaultdict(Counter)

        for row in assignments:
            pid = str(row.get('paper_id') or '').strip()
            paper = paper_map.get(pid)
            if not paper:
                continue
            pub = str(paper.get('published') or row.get('published_date') or '')
            if not pub:
                continue
            pub_date = to_date(pub)
            dim_assignments = row.get('dimension_assignments') or {}
            for dim, items in dim_assignments.items():
                if dim not in ALLOWED_DIMENSIONS:
                    continue
                for item in items or []:
                    leaf_id = str(item.get('node_id') or '').strip()
                    if not leaf_id:
                        continue
                    hist_node = climb_to_history_node(leaf_id, args.history_structure_slice, order_index, parent_by_id, meta_by_id)
                    if not hist_node or hist_node not in hist_nodes:
                        continue
                    if pub_date <= history_end:
                        hist_papers_by_node[hist_node].add(pid)
                    for window_name, (start, end) in window_dates.items():
                        if start <= pub_date < end:
                            window_papers_by_node[hist_node][window_name].add(pid)
                            current = leaf_id
                            while current and current != hist_node:
                                if order_index.get(created_slice(meta_by_id[current]), 0) > order_index[args.history_structure_slice]:
                                    future_desc_by_node[hist_node][current] += 1
                                current = parent_by_id.get(current)

        packets = []
        for node_id in sorted(hist_nodes):
            meta = meta_by_id[node_id]
            hist_ids = hist_papers_by_node.get(node_id, set())
            if len(hist_ids) < 12:
                continue
            hist_stats = aggregate_paper_stats(hist_ids, paper_map)
            future_windows = {}
            total_future = 0
            total_future_top = 0
            for window_name, ids in window_papers_by_node.get(node_id, {}).items():
                stats = aggregate_paper_stats(ids, paper_map)
                total_future += stats['paper_count']
                total_future_top += stats['top_conf_count']
                future_windows[window_name] = {
                    **stats,
                    'representative_papers': rank_papers(list(ids), paper_map, limit=4),
                }
            desc_counter = future_desc_by_node.get(node_id, Counter())
            desc_rows = []
            for desc_id, count in desc_counter.most_common(8):
                desc_meta = meta_by_id.get(desc_id) or {}
                desc_rows.append({
                    'node_id': desc_id,
                    'display_name': desc_meta.get('display_name'),
                    'created_time_slice': created_slice(desc_meta),
                    'future_paper_count': count,
                })

            hist_top_share = hist_stats['top_conf_share']
            future_half_stats = future_windows.get('halfyear_2025q4_2026q1') or aggregate_paper_stats(set(), paper_map)
            trend_signal = total_future / max(hist_stats['paper_count'], 1)
            venue_gap_signal = round(future_half_stats.get('top_conf_share', 0.0) - hist_top_share, 4)
            split_pressure = sum(x['future_paper_count'] for x in desc_rows[:5])
            packet = {
                'packet_id': f'{domain}::{node_id.replace('/', '__')}',
                'domain': domain,
                'history_structure_slice': args.history_structure_slice,
                'history_end_date': args.history_end,
                'seed_type': 'node',
                'dimension_id': meta['dimension_id'],
                'node_id': node_id,
                'display_name': meta['display_name'],
                'description': meta.get('description'),
                'level': meta['level'],
                'created_time_slice': created_slice(meta),
                'parent_id': meta.get('parent_id'),
                'historical_stats': hist_stats,
                'historical_representative_papers': rank_papers(list(hist_ids), paper_map, limit=6),
                'future_windows': future_windows,
                'emergent_descendants': desc_rows,
                'trend_signal': round(trend_signal, 4),
                'venue_gap_signal': venue_gap_signal,
                'split_pressure': split_pressure,
                'planning_priority_score': round(trend_signal * 2 + max(0.0, venue_gap_signal) + split_pressure / 25.0, 4),
                'direction_score': round((future_half_stats.get('paper_count', 0) * 0.8) + (future_half_stats.get('top_conf_count', 0) * 2.0), 4),
                'bottleneck_pressure_score': round((split_pressure / 20.0) + (hist_stats['paper_count'] / 80.0) + max(0.0, 0.2 - hist_top_share), 4),
                'lineage': [meta_by_id[n]['display_name'] for n in lineage(node_id, parent_by_id, meta_by_id)],
            }
            packets.append(packet)
            all_packets.append(packet)

        packets.sort(key=lambda x: (-x['planning_priority_score'], -x['direction_score'], x['display_name']))
        selected = packets[: args.top_k_per_domain]
        for pkt in selected:
            all_seeds.append({
                'domain': domain,
                'packet_id': pkt['packet_id'],
                'node_id': pkt['node_id'],
                'display_name': pkt['display_name'],
                'dimension_id': pkt['dimension_id'],
                'historical_paper_count': pkt['historical_stats']['paper_count'],
                'future_half_paper_count': pkt['future_windows'].get('halfyear_2025q4_2026q1', {}).get('paper_count', 0),
                'future_half_top_conf_count': pkt['future_windows'].get('halfyear_2025q4_2026q1', {}).get('top_conf_count', 0),
                'direction_score': pkt['direction_score'],
                'bottleneck_pressure_score': pkt['bottleneck_pressure_score'],
                'planning_priority_score': pkt['planning_priority_score'],
            })

        domain_dir = out_dir / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / 'node_support_packets.json').write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding='utf-8')
        (domain_dir / 'selected_seed_nodes.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding='utf-8')
        print(domain, 'packets', len(packets), 'selected', len(selected))

    (out_dir / 'all_node_support_packets.json').write_text(json.dumps(all_packets, ensure_ascii=False, indent=2), encoding='utf-8')
    (out_dir / 'selected_seed_nodes.json').write_text(json.dumps(all_seeds, ensure_ascii=False, indent=2), encoding='utf-8')
    print('wrote', out_dir)
    print('all_packets', len(all_packets))
    print('all_selected', len(all_seeds))


def lineage(node_id: str, parent_by_id: dict[str, str | None], meta_by_id: dict[str, dict[str, Any]]) -> list[str]:
    chain = []
    current = node_id
    while current:
        chain.append(current)
        current = parent_by_id.get(current)
    chain.reverse()
    return chain


if __name__ == '__main__':
    main()
