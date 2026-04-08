from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.benchmark_v2 import (
    DEFAULT_DOMAINS,
    WINDOW_TO_LABEL,
    compact_paper,
    compute_trajectory,
    dump_json,
    dump_jsonl,
    format_descendants,
    future_stats,
    join_display_names,
    load_all_packets,
    load_json,
    load_label_rows,
    load_paper_rows,
    load_selected_seed_packets,
    quality_band,
    safe_ratio,
    summarize_structure_coverage,
    top_future_work_signals,
    top_limitation_signals,
)
from researchworld.verbalization import public_descendant_names, public_topic_from_packet, title_case_phrase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build benchmark v2 task candidates from support packets and optional full-text structures.")
    parser.add_argument("--selected-seeds", default=str(ROOT / "data" / "support_packets" / "selected_seed_nodes.json"))
    parser.add_argument("--all-packets", default=str(ROOT / "data" / "support_packets" / "all_node_support_packets.json"))
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--structures-root", default=str(ROOT / "data" / "support_packets" / "paper_structures"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "task_candidates"))
    return parser.parse_args()


def get_structure_rows(structures_root: Path, domain: str) -> Dict[str, Dict[str, Any]]:
    path = structures_root / domain / "paper_structures.jsonl"
    if not path.exists():
        return {}
    rows = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = __import__("json").loads(line)
            rows[str(row["paper_id"])] = row
    return rows


def build_context(packet: Dict[str, Any], papers: Dict[str, Dict[str, Any]], structure_by_paper: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    history_rows = [compact_paper(papers[row["paper_id"]]) for row in (packet.get("historical_representative_papers") or []) if row.get("paper_id") in papers][:5]
    q4_rows = [
        compact_paper(papers[row["paper_id"]])
        for row in (((packet.get("future_windows") or {}).get("quarterly_2025q4") or {}).get("representative_papers") or [])
        if row.get("paper_id") in papers
    ][:4]
    q1_rows = [
        compact_paper(papers[row["paper_id"]])
        for row in (((packet.get("future_windows") or {}).get("quarterly_2026q1") or {}).get("representative_papers") or [])
        if row.get("paper_id") in papers
    ][:4]
    history_structures = [structure_by_paper[row["paper_id"]] for row in history_rows if row.get("paper_id") in structure_by_paper][:4]
    return {
        "history_representative_papers": history_rows,
        "future_q4_representative_papers": q4_rows,
        "future_q1_representative_papers": q1_rows,
        "history_structure_coverage": summarize_structure_coverage(history_structures),
        "top_limitations": top_limitation_signals(history_structures, top_k=4),
        "top_future_work": top_future_work_signals(history_structures, top_k=4),
    }


def direction_candidate(packet: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    traj = compute_trajectory(packet)
    hist = packet.get("historical_stats") or {}
    q4 = (packet.get("future_windows") or {}).get("quarterly_2025q4") or {}
    q1 = (packet.get("future_windows") or {}).get("quarterly_2026q1") or {}
    descendants = format_descendants(packet, limit=5)
    public_topic = public_topic_from_packet(packet)
    public_descendants = public_descendant_names(descendants, limit=4)
    question = (
        f"Based on literature available up to 2025-08-31, forecast how research on {public_topic} in {packet['domain']} "
        f"is likely to evolve over the next six months. State whether the area is likely to accelerate, stay steady, cool down, "
        f"or fragment, and identify the concrete subdirections or evaluation shifts that are most likely to define that change."
    )
    answer = (
        f"The realized trajectory is '{traj['trajectory_label']}'. Over 2025-09-01 to 2026-02-28, the node accumulated "
        f"{int((packet.get('future_windows') or {}).get('halfyear_2025q4_2026q1', {}).get('paper_count') or 0)} future papers "
        f"against {int(hist.get('paper_count') or 0)} historical papers before the cutoff, with top-venue share moving from "
        f"{hist.get('top_conf_share', 0.0):.4f} to {((packet.get('future_windows') or {}).get('halfyear_2025q4_2026q1', {}).get('top_conf_share') or 0.0):.4f}. "
        f"Follow-on work concentrated on {', '.join(public_descendants) or 'no major newly separated subdirection'}. "
        f"Quarterly evidence: Q4={int(q4.get('paper_count') or 0)} papers, Q1={int(q1.get('paper_count') or 0)} papers."
    )
    return {
        "family": "direction_forecasting",
        "subtype": "node_momentum",
        "horizon": "half_year",
        "draft_question": question,
        "draft_reference_answer": answer,
        "ground_truth": {
            "trajectory": traj,
            "historical_stats": hist,
            "future_q4_stats": q4,
            "future_q1_stats": q1,
            "future_half_stats": (packet.get("future_windows") or {}).get("halfyear_2025q4_2026q1") or {},
            "emergent_descendants": descendants,
            "reference_papers": {
                "history": context["history_representative_papers"][:4],
                "future_q4": context["future_q4_representative_papers"][:3],
                "future_q1": context["future_q1_representative_papers"][:3],
            },
        },
        "quality_signals": {
            "heuristic_score": round(
                0.40
                + min(0.25, (packet.get("direction_score") or 0.0) / 80.0)
                + min(0.15, abs(traj["venue_share_delta"]) * 2.5)
                + min(0.20, len(descendants) / 5.0),
                4,
            ),
        },
        "public_metadata": {
            "topic": public_topic,
            "topic_title": title_case_phrase(public_topic),
            "future_themes": public_descendants,
        },
    }


def bottleneck_candidate(packet: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    top_limits = context["top_limitations"]
    future_work = context["top_future_work"]
    descendants = format_descendants(packet, limit=5)
    hist = packet.get("historical_stats") or {}
    half = (packet.get("future_windows") or {}).get("halfyear_2025q4_2026q1") or {}
    public_topic = public_topic_from_packet(packet)
    opportunity = public_descendant_names(descendants, limit=3) or [row.get("direction") for row in future_work[:3] if row.get("direction")]
    question = (
        f"Using literature available before 2025-09-01, identify the most consequential unresolved bottleneck in {public_topic} "
        f"and the associated opportunity that was most likely to open over the next six months. The answer should connect a concrete technical bottleneck "
        f"to a plausible research move rather than giving generic future-work advice."
    )
    answer = (
        f"A strong realized bottleneck-opportunity pair should be anchored in historical limitations such as "
        f"{', '.join(row['name'] for row in top_limits[:3]) or 'insufficiently explicit limitation evidence'}, "
        f"and linked to follow-on work on {', '.join(opportunity[:3]) or 'the main follow-on directions that later appeared'}. "
        f"The area had {int(hist.get('paper_count') or 0)} historical papers and {int(half.get('paper_count') or 0)} papers in the following half year, "
        f"showing that this opportunity was subsequently taken up by the literature."
    )
    coverage = context["history_structure_coverage"]
    heuristic_score = 0.35 + min(0.30, coverage["paper_count"] / 8.0) + min(0.20, len(top_limits) / 5.0) + min(0.15, len(opportunity) / 4.0)
    return {
        "family": "bottleneck_opportunity_discovery",
        "subtype": "bottleneck_linked_opportunity",
        "horizon": "half_year",
        "draft_question": question,
        "draft_reference_answer": answer,
        "ground_truth": {
            "historical_limitation_signals": top_limits,
            "historical_future_work_signals": future_work,
            "future_descendants": descendants,
            "future_half_stats": half,
            "structure_coverage": coverage,
            "reference_papers": {
                "history": context["history_representative_papers"][:4],
                "future_q4": context["future_q4_representative_papers"][:2],
                "future_q1": context["future_q1_representative_papers"][:2],
            },
        },
        "quality_signals": {
            "heuristic_score": round(heuristic_score, 4),
            "has_fulltext_evidence": coverage["paper_count"] >= 2,
        },
        "public_metadata": {
            "topic": public_topic,
            "topic_title": title_case_phrase(public_topic),
            "future_themes": opportunity[:3],
        },
    }


def planning_candidate(packet: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    hist = packet.get("historical_stats") or {}
    q4 = (packet.get("future_windows") or {}).get("quarterly_2025q4") or {}
    q1 = (packet.get("future_windows") or {}).get("quarterly_2026q1") or {}
    descendants = format_descendants(packet, limit=4)
    public_topic = public_topic_from_packet(packet)
    public_descendants = public_descendant_names(descendants, limit=4)
    venue_gap = float(packet.get("venue_gap_signal") or 0.0)
    subtype = "submission_oriented_planning" if (int(q4.get("top_conf_count") or 0) + int(q1.get("top_conf_count") or 0)) >= 2 else "medium_horizon_agenda_synthesis"
    horizon = "quarterly" if subtype == "submission_oriented_planning" else "half_year"
    target_stats = future_stats(packet, horizon)
    question = (
        f"Assume you are preparing a research plan for work on {public_topic} at the 2025-08-31 cutoff. "
        f"What research agenda would have the highest strategic value for the next {'quarter' if horizon == 'quarterly' else 'six months'}, "
        f"given venue selectivity, citation concentration, and emerging subtopics?"
    )
    answer = (
        f"A high-value plan should prioritize directions such as {', '.join(public_descendants[:3]) or 'the strongest emerging follow-on themes'}, "
        f"while accounting for top-venue share moving from {hist.get('top_conf_share', 0.0):.4f} historically to {target_stats.get('top_conf_share', 0.0):.4f} in the target window. "
        f"Realized target-window volume was {int(target_stats.get('paper_count') or 0)} papers and {int(target_stats.get('top_conf_count') or 0)} top-venue papers."
    )
    score = 0.40 + min(0.20, (packet.get("planning_priority_score") or 0.0) / 60.0) + min(0.20, max(0.0, venue_gap) * 3.0) + min(0.20, len(descendants) / 4.0)
    return {
        "family": "strategic_research_planning",
        "subtype": subtype,
        "horizon": horizon,
        "draft_question": question,
        "draft_reference_answer": answer,
        "ground_truth": {
            "target_window_stats": target_stats,
            "venue_gap_signal": venue_gap,
            "planning_priority_score": packet.get("planning_priority_score"),
            "emergent_descendants": descendants,
            "reference_papers": {
                "history": context["history_representative_papers"][:4],
                "future_q4": context["future_q4_representative_papers"][:3],
                "future_q1": context["future_q1_representative_papers"][:3],
            },
        },
        "quality_signals": {
            "heuristic_score": round(score, 4),
        },
        "public_metadata": {
            "topic": public_topic,
            "topic_title": title_case_phrase(public_topic),
            "future_themes": public_descendants[:3],
        },
    }


def enrich_common_fields(base: Dict[str, Any], packet: Dict[str, Any], context: Dict[str, Any], index: int) -> Dict[str, Any]:
    history_stats = packet.get("historical_stats") or {}
    base["task_id"] = (
        f"{base['family']}::{packet['domain']}::{packet['node_id'].replace('/', '__')}::{base['horizon']}::{index}"
    )
    base["benchmark_version"] = "v2_candidate_20260329"
    base["domain"] = packet["domain"]
    base["seed"] = {
        "packet_id": packet["packet_id"],
        "node_id": packet["node_id"],
        "display_name": packet["display_name"],
        "dimension_id": packet["dimension_id"],
        "lineage": packet.get("lineage") or [],
    }
    base["time_context"] = {
        "history_end": packet["history_end_date"],
        "history_structure_slice": packet["history_structure_slice"],
        "future_windows": {
            "quarterly_2025q4": WINDOW_TO_LABEL["quarterly_2025q4"],
            "quarterly_2026q1": WINDOW_TO_LABEL["quarterly_2026q1"],
            "halfyear_2025q4_2026q1": WINDOW_TO_LABEL["halfyear_2025q4_2026q1"],
        },
    }
    base["support_context"] = {
        "node_description": packet.get("description"),
        "historical_stats": history_stats,
        "history_representative_papers": context["history_representative_papers"],
        "future_q4_representative_papers": context["future_q4_representative_papers"],
        "future_q1_representative_papers": context["future_q1_representative_papers"],
        "history_structure_coverage": context["history_structure_coverage"],
        "top_limitations": context["top_limitations"],
        "top_future_work": context["top_future_work"],
    }
    base["evaluation_rubric"] = {
        "core_dimensions": rubric_for_family(base["family"]),
        "judge_mode": "llm_as_judge_with_structured_reference",
        "non_leakage_rule": "The prompt only exposes evidence available before 2025-09-01. Ground truth uses 2025-09-01 to 2026-02-28 outcomes.",
    }
    score = float((base.get("quality_signals") or {}).get("heuristic_score") or 0.0)
    base["quality_signals"]["heuristic_band"] = quality_band(score)
    base["quality_signals"]["history_paper_count"] = int(history_stats.get("paper_count") or 0)
    return base


def rubric_for_family(family: str) -> List[Dict[str, Any]]:
    if family == "direction_forecasting":
        return [
            {"name": "trajectory_call", "weight": 0.35, "description": "Correctly identifies the realized trajectory label and its sign."},
            {"name": "emerging_subdirections", "weight": 0.30, "description": "Names the major descendant or shifted subdirections that actually emerged."},
            {"name": "venue_or_evaluation_shift", "weight": 0.20, "description": "Captures realized venue-share or evaluation emphasis change."},
            {"name": "evidence_linkage", "weight": 0.15, "description": "Grounds the forecast in historical evidence rather than generic statements."},
        ]
    if family == "bottleneck_opportunity_discovery":
        return [
            {"name": "bottleneck_specificity", "weight": 0.30, "description": "Identifies a concrete unresolved bottleneck, not a vague complaint."},
            {"name": "opportunity_linkage", "weight": 0.30, "description": "Connects the bottleneck to a plausible opportunity later realized in the future window."},
            {"name": "historical_evidence_use", "weight": 0.25, "description": "Uses limitations/future-work evidence from historical papers."},
            {"name": "forward_value", "weight": 0.15, "description": "Opportunity is consequential for the domain, not just incremental."},
        ]
    return [
        {"name": "strategic_priority", "weight": 0.30, "description": "Prioritizes directions that later proved consequential."},
        {"name": "venue_awareness", "weight": 0.25, "description": "Reflects where selective venue activity actually concentrated."},
        {"name": "plan_specificity", "weight": 0.25, "description": "Provides concrete agenda elements, milestones, or decision points."},
        {"name": "evidence_alignment", "weight": 0.20, "description": "Aligns with historical signals and realized future evidence."},
    ]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    structures_root = Path(args.structures_root)
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    selected = load_selected_seed_packets(args.selected_seeds)
    packets = load_all_packets(args.all_packets)
    manifest = {"benchmark_version": "v2_candidate_20260329", "domains": {}, "family_counts": defaultdict(int)}

    all_rows: List[Dict[str, Any]] = []
    for domain in domains:
        papers = load_paper_rows(domain)
        _ = load_label_rows(domain)
        structure_by_paper = get_structure_rows(structures_root, domain)
        per_family_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        seed_count = 0
        for seed in selected:
            if seed.get("domain") != domain:
                continue
            packet = packets.get(str(seed["packet_id"]))
            if not packet:
                continue
            seed_count += 1
            context = build_context(packet, papers, structure_by_paper)
            builders = [direction_candidate, bottleneck_candidate, planning_candidate]
            for idx, builder in enumerate(builders, start=1):
                row = builder(packet, context)
                row = enrich_common_fields(row, packet, context, idx)
                per_family_rows[row["family"]].append(row)
                all_rows.append(row)
                manifest["family_counts"][row["family"]] += 1
        for family, rows in per_family_rows.items():
            rows.sort(key=lambda row: (-float(row["quality_signals"]["heuristic_score"]), row["task_id"]))
            dump_jsonl(out_dir / family / f"{domain}.jsonl", rows)
        manifest["domains"][domain] = {
            "seed_count": seed_count,
            "structure_paper_count": len(structure_by_paper),
            "family_counts": {family: len(rows) for family, rows in per_family_rows.items()},
        }
        print(domain, manifest["domains"][domain])
    dump_json(out_dir / "manifest.json", {**manifest, "family_counts": dict(manifest["family_counts"])})
    dump_jsonl(out_dir / "all_candidates.jsonl", all_rows)
    print("total_candidates", len(all_rows))


if __name__ == "__main__":
    main()
