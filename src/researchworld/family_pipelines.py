from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from researchworld.benchmark_v2 import (
    DEFAULT_DOMAINS,
    ROOT,
    compact_paper,
    compute_trajectory,
    dump_json,
    dump_jsonl,
    format_descendants,
    future_stats,
    join_display_names,
    load_all_packets,
    load_paper_rows,
    summarize_structure_coverage,
    top_future_work_signals,
    top_limitation_signals,
)
from researchworld.corpus import iter_jsonl


def load_selected_seed_rows(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_structure_rows(domain: str) -> Dict[str, Dict[str, Any]]:
    path = ROOT / "data" / "support_packets" / "paper_structures" / domain / "paper_structures.jsonl"
    if not path.exists():
        return {}
    return {str(row["paper_id"]): row for row in iter_jsonl(path)}


def final_taxonomy_dir(domain: str) -> Path:
    return ROOT / "data" / "domains" / domain / "taxoadapt_quarterly" / "quarterly_v1" / "2026Q1"


def load_taxonomy_meta(domain: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str | None], Dict[str, List[str]], Dict[str, int]]:
    final_dir = final_taxonomy_dir(domain)
    nodes = json.loads((final_dir / "taxonomy_nodes.json").read_text(encoding="utf-8"))
    meta_by_id = {str(row["node_id"]): row for row in nodes}
    parent_by_id = {str(row["node_id"]): row.get("parent_id") for row in nodes}
    children_by_id: Dict[str, List[str]] = defaultdict(list)
    for row in nodes:
        parent_id = row.get("parent_id")
        if parent_id:
            children_by_id[str(parent_id)].append(str(row["node_id"]))
    summary = json.loads((ROOT / "data" / "domains" / domain / "taxoadapt_quarterly" / "quarterly_v1" / "final_summary.json").read_text(encoding="utf-8"))
    slice_order = {str(item["time_slice"]): idx for idx, item in enumerate(summary.get("years") or [])}
    return meta_by_id, parent_by_id, children_by_id, slice_order


def normalize_slice(value: Any) -> str:
    text = str(value or "").strip()
    return text or "ROOT"


def slice_rank(value: Any, slice_order: Dict[str, int]) -> int:
    text = normalize_slice(value)
    if text == "ROOT":
        return -1
    return int(slice_order.get(text, 10**6))


def lineage_ids(node_id: str, parent_by_id: Dict[str, str | None]) -> List[str]:
    chain: List[str] = []
    current = str(node_id or "")
    while current:
        chain.append(current)
        current = str(parent_by_id.get(current) or "")
    chain.reverse()
    return chain


def history_lineage(node_id: str, parent_by_id: Dict[str, str | None], meta_by_id: Dict[str, Dict[str, Any]], slice_order: Dict[str, int], history_slice: str) -> List[Dict[str, Any]]:
    history_rank = slice_rank(history_slice, slice_order)
    rows: List[Dict[str, Any]] = []
    for nid in lineage_ids(node_id, parent_by_id):
        meta = meta_by_id.get(nid) or {}
        created = meta.get("created_time_slice")
        if slice_rank(created, slice_order) <= history_rank:
            rows.append(
                {
                    "node_id": nid,
                    "display_name": meta.get("display_name"),
                    "dimension_id": meta.get("dimension_id"),
                    "level": meta.get("level"),
                    "created_time_slice": created,
                    "description": meta.get("description"),
                }
            )
    return rows


def descendant_chain(seed_node_id: str, future_node_id: str, parent_by_id: Dict[str, str | None], meta_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    future_lineage = lineage_ids(future_node_id, parent_by_id)
    rows: List[Dict[str, Any]] = []
    seen_seed = False
    for nid in future_lineage:
        if nid == seed_node_id:
            seen_seed = True
        if not seen_seed:
            continue
        meta = meta_by_id.get(nid) or {}
        rows.append(
            {
                "node_id": nid,
                "display_name": meta.get("display_name"),
                "dimension_id": meta.get("dimension_id"),
                "level": meta.get("level"),
                "created_time_slice": meta.get("created_time_slice"),
                "description": meta.get("description"),
            }
        )
    return rows


def history_structure_rows(packet: Dict[str, Any], structure_by_paper: Dict[str, Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for paper in packet.get("historical_representative_papers") or []:
        paper_id = str(paper.get("paper_id") or "")
        if paper_id in structure_by_paper:
            rows.append(structure_by_paper[paper_id])
        if len(rows) >= limit:
            break
    return rows


def build_chain_packets(
    *,
    domain: str,
    packets: Iterable[Dict[str, Any]],
    max_descendants_per_seed: int = 2,
) -> List[Dict[str, Any]]:
    meta_by_id, parent_by_id, _children_by_id, slice_order = load_taxonomy_meta(domain)
    rows: List[Dict[str, Any]] = []
    for packet in packets:
        descendants = format_descendants(packet, limit=max_descendants_per_seed)
        if not descendants:
            continue
        history_nodes = history_lineage(
            str(packet["node_id"]),
            parent_by_id,
            meta_by_id,
            slice_order,
            str(packet.get("history_structure_slice") or ""),
        )
        for idx, descendant in enumerate(descendants, start=1):
            future_node_id = str(descendant.get("node_id") or "")
            future_chain = descendant_chain(str(packet["node_id"]), future_node_id, parent_by_id, meta_by_id)
            rows.append(
                {
                    "packet_id": f"chain::{packet['packet_id']}::{idx}",
                    "family": "direction_forecasting",
                    "domain": domain,
                    "seed_packet_id": packet["packet_id"],
                    "seed_node_id": packet["node_id"],
                    "history_end_date": packet.get("history_end_date"),
                    "history_structure_slice": packet.get("history_structure_slice"),
                    "history_chain": history_nodes,
                    "future_terminal": descendant,
                    "future_chain": future_chain,
                    "history_representative_papers": list(packet.get("historical_representative_papers") or [])[:5],
                    "future_q4_representative_papers": list(((packet.get("future_windows") or {}).get("quarterly_2025q4") or {}).get("representative_papers") or [])[:3],
                    "future_q1_representative_papers": list(((packet.get("future_windows") or {}).get("quarterly_2026q1") or {}).get("representative_papers") or [])[:3],
                    "trajectory": compute_trajectory(packet),
                    "future_half_stats": (packet.get("future_windows") or {}).get("halfyear_2025q4_2026q1") or {},
                    "historical_stats": packet.get("historical_stats") or {},
                    "public_metadata": {
                        "topic": packet.get("display_name"),
                        "history_chain_labels": join_display_names(history_nodes, limit=6),
                        "future_terminal_label": descendant.get("display_name"),
                    },
                }
            )
    rows.sort(key=lambda row: row["packet_id"])
    return rows


def build_opportunity_packets(
    *,
    domain: str,
    packets: Iterable[Dict[str, Any]],
    max_history_reading_set: int = 8,
    max_future_reading_set: int = 4,
) -> List[Dict[str, Any]]:
    papers = load_paper_rows(domain)
    structure_by_paper = load_structure_rows(domain)
    rows: List[Dict[str, Any]] = []
    for packet in packets:
        hist_rows = history_structure_rows(packet, structure_by_paper, limit=max_history_reading_set)
        hist_papers = []
        for paper in (packet.get("historical_representative_papers") or [])[:max_history_reading_set]:
            row = papers.get(str(paper.get("paper_id") or ""))
            if row:
                hist_papers.append(compact_paper(row))
            else:
                hist_papers.append(paper)
        future_papers = []
        for key in ("quarterly_2025q4", "quarterly_2026q1"):
            for paper in (((packet.get("future_windows") or {}).get(key) or {}).get("representative_papers") or [])[:2]:
                row = papers.get(str(paper.get("paper_id") or ""))
                future_papers.append(compact_paper(row) if row else paper)
        future_papers = future_papers[:max_future_reading_set]
        top_limits = top_limitation_signals(hist_rows, top_k=6)
        top_future_work = top_future_work_signals(hist_rows, top_k=6)
        rows.append(
            {
                "packet_id": f"opportunity::{packet['packet_id']}",
                "family": "bottleneck_opportunity_discovery",
                "domain": domain,
                "seed_packet_id": packet["packet_id"],
                "seed_node_id": packet["node_id"],
                "history_end_date": packet.get("history_end_date"),
                "history_structure_slice": packet.get("history_structure_slice"),
                "topic": packet.get("display_name"),
                "historical_stats": packet.get("historical_stats") or {},
                "history_reading_set": hist_papers,
                "future_validation_set": future_papers,
                "history_structure_coverage": summarize_structure_coverage(hist_rows),
                "historical_limitation_cluster": top_limits,
                "historical_future_work_cluster": top_future_work,
                "realized_opportunity_directions": format_descendants(packet, limit=5),
                "future_half_stats": (packet.get("future_windows") or {}).get("halfyear_2025q4_2026q1") or {},
                "pageindex_priority": {
                    "history_structure_paper_count": len(hist_rows),
                    "has_limitation_signal": bool(top_limits),
                    "has_future_work_signal": bool(top_future_work),
                },
            }
        )
    rows.sort(key=lambda row: row["packet_id"])
    return rows


def build_planning_packets(
    *,
    domain: str,
    packets: Iterable[Dict[str, Any]],
    max_candidate_directions: int = 5,
) -> List[Dict[str, Any]]:
    structure_by_paper = load_structure_rows(domain)
    rows: List[Dict[str, Any]] = []
    for packet in packets:
        target_stats = future_stats(packet, "half_year")
        hist_rows = history_structure_rows(packet, structure_by_paper, limit=8)
        descendants = format_descendants(packet, limit=max_candidate_directions)
        top_future_work = top_future_work_signals(hist_rows, top_k=max_candidate_directions)
        candidate_direction_records = list(descendants)
        if not candidate_direction_records:
            candidate_direction_records = [
                {
                    "display_name": row.get("direction"),
                    "source": "historical_future_work",
                    "count": row.get("count"),
                    "paper_id": row.get("paper_id"),
                    "title": row.get("title"),
                }
                for row in top_future_work
                if str(row.get("direction") or "").strip()
            ]
        candidate_directions = join_display_names(candidate_direction_records, limit=max_candidate_directions)
        hist = packet.get("historical_stats") or {}
        rows.append(
            {
                "packet_id": f"planning::{packet['packet_id']}",
                "family": "strategic_research_planning",
                "domain": domain,
                "seed_packet_id": packet["packet_id"],
                "seed_node_id": packet["node_id"],
                "history_end_date": packet.get("history_end_date"),
                "history_structure_slice": packet.get("history_structure_slice"),
                "topic": packet.get("display_name"),
                "candidate_directions": candidate_directions,
                "direction_records": candidate_direction_records,
                "future_descendant_records": descendants,
                "historical_stats": hist,
                "target_window_stats": target_stats,
                "venue_gap_signal": packet.get("venue_gap_signal"),
                "planning_priority_score": packet.get("planning_priority_score"),
                "historical_future_work_cluster": top_future_work,
                "history_structure_coverage": summarize_structure_coverage(hist_rows),
                "ranking_axes": {
                    "future_volume": target_stats.get("paper_count"),
                    "future_top_conf_count": target_stats.get("top_conf_count"),
                    "historical_top_conf_share": hist.get("top_conf_share"),
                    "future_top_conf_share": target_stats.get("top_conf_share"),
                    "descendant_count": len(descendants),
                    "history_future_work_signal_count": len(top_future_work),
                },
                "history_representative_papers": list(packet.get("historical_representative_papers") or [])[:5],
            }
        )
    rows.sort(key=lambda row: row["packet_id"])
    return rows


def _load_packets_by_domain(selected_seeds_path: str | Path, all_packets_path: str | Path, domains: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    selected = load_selected_seed_rows(selected_seeds_path)
    packets = load_all_packets(all_packets_path)
    out: Dict[str, List[Dict[str, Any]]] = {domain: [] for domain in domains}
    for seed in selected:
        domain = str(seed.get("domain") or "")
        if domain not in out:
            continue
        packet = packets.get(str(seed["packet_id"]))
        if packet:
            out[domain].append(packet)
    return out


def build_all_family_packets(
    *,
    out_root: str | Path,
    selected_seeds_path: str | Path = ROOT / "data" / "support_packets" / "selected_seed_nodes.json",
    all_packets_path: str | Path = ROOT / "data" / "support_packets" / "all_node_support_packets.json",
    domains: List[str] | None = None,
) -> Dict[str, Any]:
    out_root = Path(out_root)
    domains = domains or list(DEFAULT_DOMAINS)
    packets_by_domain = _load_packets_by_domain(selected_seeds_path, all_packets_path, domains)

    manifest: Dict[str, Any] = {"domains": {}, "families": Counter()}
    family_builders = {
        "chain_packets": build_chain_packets,
        "opportunity_packets": build_opportunity_packets,
        "planning_packets": build_planning_packets,
    }
    for domain in domains:
        domain_packets = packets_by_domain.get(domain, [])
        manifest["domains"][domain] = {"seed_packets": len(domain_packets)}
        for family_name, builder in family_builders.items():
            rows = builder(domain=domain, packets=domain_packets)
            dump_jsonl(out_root / family_name / f"{domain}.jsonl", rows)
            manifest["domains"][domain][family_name] = len(rows)
            manifest["families"][family_name] += len(rows)
    manifest["families"] = dict(manifest["families"])
    dump_json(out_root / "manifest.json", manifest)
    return manifest
