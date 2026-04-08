from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.benchmark_v2 import DEFAULT_DOMAINS, dump_json, dump_jsonl
from researchworld.verbalization import public_descendant_names, public_topic_from_packet, sentence_case_phrase, title_case_phrase


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_direction_candidate(packet: Dict[str, Any]) -> Dict[str, Any]:
    history_chain = packet.get("history_chain") or []
    chain_labels = [sentence_case_phrase(str(row.get("display_name") or "")) for row in history_chain if str(row.get("display_name") or "").strip()]
    future_terminal = packet.get("future_terminal") or {}
    topic = public_topic_from_packet({"display_name": packet.get("public_metadata", {}).get("topic") or packet.get("seed_node_id")})
    future_descendants = list((packet.get("future_chain") or [future_terminal])[1:] or [future_terminal])
    future_themes = [x for x in public_descendant_names(future_descendants, limit=4) if x.lower() != topic.lower()]
    trajectory = packet.get("trajectory") or {}
    question = (
        f"Based on literature available up to {packet.get('history_end_date')}, consider the historical idea chain "
        f"{' -> '.join(chain_labels[-4:]) or topic}. "
        f"What specific next-step direction is most likely to emerge over the next six months, and should the trajectory of this area be characterized as "
        f"accelerating, fragmenting, steady, or cooling?"
    )
    answer = (
        f"The realized next-step direction was {sentence_case_phrase(str(future_terminal.get('display_name') or 'a more specialized subdirection'))}, "
        f"and the trajectory was {trajectory.get('trajectory_label')}. This evolution was supported by "
        f"{int((packet.get('future_half_stats') or {}).get('paper_count') or 0)} future papers."
    )
    return {
        "task_id": packet["packet_id"],
        "benchmark_version": "v3_family_candidate_20260401",
        "family": "direction_forecasting",
        "subtype": "chain_terminal_forecast",
        "domain": packet["domain"],
        "horizon": "half_year",
        "draft_question": question,
        "draft_reference_answer": answer,
        "time_context": {
            "history_end": packet.get("history_end_date"),
            "history_structure_slice": packet.get("history_structure_slice"),
            "future_window": "2025-09-01_to_2026-02-28",
        },
        "seed": {
            "packet_id": packet.get("packet_id"),
            "node_id": packet.get("seed_node_id"),
        },
        "support_context": {
            "history_chain": packet.get("history_chain"),
            "history_representative_papers": packet.get("history_representative_papers"),
            "historical_stats": packet.get("historical_stats"),
        },
        "ground_truth": {
            "future_terminal": packet.get("future_terminal"),
            "future_chain": packet.get("future_chain"),
            "emergent_descendants": future_descendants,
            "trajectory": packet.get("trajectory"),
            "historical_stats": packet.get("historical_stats"),
            "future_half_stats": packet.get("future_half_stats"),
            "reference_papers": {
                "history": packet.get("history_representative_papers"),
                "future_q4": packet.get("future_q4_representative_papers"),
                "future_q1": packet.get("future_q1_representative_papers"),
            },
        },
        "public_metadata": {
            "topic": topic,
            "topic_title": title_case_phrase(topic),
            "future_themes": future_themes,
        },
        "quality_signals": {
            "heuristic_score": round(
                0.45
                + min(0.20, len(history_chain) / 6.0)
                + min(0.20, len(packet.get("future_chain") or []) / 4.0)
                + min(0.15, abs(float(trajectory.get("venue_share_delta") or 0.0)) * 2.0),
                4,
            )
        },
    }


def build_bottleneck_candidate(packet: Dict[str, Any]) -> Dict[str, Any]:
    limits = packet.get("historical_limitation_cluster") or []
    future_dirs = packet.get("realized_opportunity_directions") or []
    topic = public_topic_from_packet({"display_name": packet.get("topic") or packet.get("seed_node_id")})
    future_themes = public_descendant_names(future_dirs, limit=4)
    question = (
        f"Using literature available before {packet.get('history_end_date')}, identify the most consequential unresolved bottleneck in "
        f"{topic} and explain which concrete research opportunity would be most likely to open if that bottleneck were addressed over the next six months. "
        f"Ground the answer in recurring limitations or failure evidence from the historical literature."
    )
    answer = (
        f"A historically grounded bottleneck-opportunity answer should center on limitations such as "
        f"{', '.join(str(x.get('name') or '') for x in limits[:3] if str(x.get('name') or '').strip()) or 'insufficient explicit limitation evidence'}, "
        f"and connect them to later-opened directions such as "
        f"{', '.join(future_themes[:3]) or 'subsequent follow-on directions'}. "
        f"The realized future window contained {int((packet.get('future_half_stats') or {}).get('paper_count') or 0)} papers."
    )
    return {
        "task_id": packet["packet_id"],
        "benchmark_version": "v3_family_candidate_20260401",
        "family": "bottleneck_opportunity_discovery",
        "subtype": "pageindex_grounded_bottleneck",
        "domain": packet["domain"],
        "horizon": "half_year",
        "draft_question": question,
        "draft_reference_answer": answer,
        "time_context": {
            "history_end": packet.get("history_end_date"),
            "history_structure_slice": packet.get("history_structure_slice"),
            "future_window": "2025-09-01_to_2026-02-28",
        },
        "seed": {
            "packet_id": packet.get("packet_id"),
            "node_id": packet.get("seed_node_id"),
        },
        "support_context": {
            "history_reading_set": packet.get("history_reading_set"),
            "future_validation_set": packet.get("future_validation_set"),
            "historical_stats": packet.get("historical_stats"),
            "top_limitations": packet.get("historical_limitation_cluster"),
            "top_future_work": packet.get("historical_future_work_cluster"),
            "history_structure_coverage": packet.get("history_structure_coverage"),
            "pageindex_priority": packet.get("pageindex_priority"),
        },
        "ground_truth": {
            "historical_limitation_cluster": packet.get("historical_limitation_cluster"),
            "historical_limitation_signals": packet.get("historical_limitation_cluster"),
            "historical_future_work_cluster": packet.get("historical_future_work_cluster"),
            "historical_future_work_signals": packet.get("historical_future_work_cluster"),
            "realized_opportunity_directions": packet.get("realized_opportunity_directions"),
            "future_descendants": packet.get("realized_opportunity_directions"),
            "future_half_stats": packet.get("future_half_stats"),
            "reference_papers": {
                "history": packet.get("history_reading_set"),
                "future_q4": packet.get("future_validation_set"),
                "future_q1": [],
            },
        },
        "public_metadata": {
            "topic": topic,
            "topic_title": title_case_phrase(topic),
            "future_themes": future_themes,
        },
        "quality_signals": {
            "heuristic_score": round(
                0.45
                + min(0.20, len(packet.get("historical_limitation_cluster") or []) / 5.0)
                + min(0.20, len(packet.get("history_reading_set") or []) / 8.0)
                + (0.15 if bool((packet.get("pageindex_priority") or {}).get("has_limitation_signal")) else 0.0),
                4,
            )
        },
    }


def build_planning_candidate(packet: Dict[str, Any]) -> Dict[str, Any]:
    topic = public_topic_from_packet({"display_name": packet.get("topic") or packet.get("seed_node_id")})
    candidate_directions = [sentence_case_phrase(x) for x in (packet.get("candidate_directions") or []) if str(x).strip()]
    future_themes = public_descendant_names(packet.get("future_descendant_records") or packet.get("direction_records") or [], limit=4)
    question = (
        f"Assume you are planning research on {topic} at the cutoff {packet.get('history_end_date')}. "
        f"If you could prioritize only a small number of next-step directions for the next six months, which directions should receive priority, "
        f"and what evidence-based rationale supports that ranking?"
    )
    answer = (
        f"A high-value plan should prioritize directions such as "
        f"{', '.join(candidate_directions) or 'the strongest candidate directions'}, "
        f"while accounting for a realized future volume of {int((packet.get('target_window_stats') or {}).get('paper_count') or 0)} papers "
        f"and a realized top-venue share of {float((packet.get('target_window_stats') or {}).get('top_conf_share') or 0.0):.4f}."
    )
    return {
        "task_id": packet["packet_id"],
        "benchmark_version": "v3_family_candidate_20260401",
        "family": "strategic_research_planning",
        "subtype": "agenda_priority_selection",
        "domain": packet["domain"],
        "horizon": "half_year",
        "draft_question": question,
        "draft_reference_answer": answer,
        "time_context": {
            "history_end": packet.get("history_end_date"),
            "history_structure_slice": packet.get("history_structure_slice"),
            "future_window": "2025-09-01_to_2026-02-28",
        },
        "seed": {
            "packet_id": packet.get("packet_id"),
            "node_id": packet.get("seed_node_id"),
        },
        "support_context": {
            "candidate_directions": candidate_directions,
            "history_representative_papers": packet.get("history_representative_papers"),
            "historical_stats": packet.get("historical_stats"),
            "top_future_work": packet.get("historical_future_work_cluster"),
            "history_structure_coverage": packet.get("history_structure_coverage"),
            "ranking_axes": packet.get("ranking_axes"),
        },
        "ground_truth": {
            "candidate_directions": candidate_directions,
            "direction_records": packet.get("direction_records"),
            "emergent_descendants": packet.get("future_descendant_records") or packet.get("direction_records"),
            "target_window_stats": packet.get("target_window_stats"),
            "venue_gap_signal": packet.get("venue_gap_signal"),
            "planning_priority_score": packet.get("planning_priority_score"),
        },
        "public_metadata": {
            "topic": topic,
            "topic_title": title_case_phrase(topic),
            "future_themes": future_themes or candidate_directions[:4],
        },
        "quality_signals": {
            "heuristic_score": round(
                0.45
                + min(0.20, len(candidate_directions) / 5.0)
                + min(0.20, max(0.0, float(packet.get("planning_priority_score") or 0.0)) / 80.0)
                + min(0.15, abs(float(packet.get("venue_gap_signal") or 0.0)) * 2.0),
                4,
            )
        },
    }


BUILDERS = {
    "chain_packets": build_direction_candidate,
    "opportunity_packets": build_bottleneck_candidate,
    "planning_packets": build_planning_candidate,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 family-specific task candidates from family packets.")
    parser.add_argument("--packets-root", default=str(ROOT / "data" / "family_packets" / "v1"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "task_candidates_v3"))
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packets_root = Path(args.packets_root)
    out_dir = Path(args.out_dir)
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    manifest: Dict[str, Any] = {"domains": {}, "family_counts": Counter()}
    all_rows: List[Dict[str, Any]] = []
    for packet_family, builder in BUILDERS.items():
        family_rows: List[Dict[str, Any]] = []
        for domain in domains:
            path = packets_root / packet_family / f"{domain}.jsonl"
            rows = [builder(row) for row in iter_jsonl(path)] if path.exists() else []
            family_rows.extend(rows)
            manifest["domains"].setdefault(domain, {})[packet_family] = len(rows)
        family = family_rows[0]["family"] if family_rows else packet_family
        manifest["family_counts"][family] += len(family_rows)
        dump_jsonl(out_dir / family / "all.jsonl", family_rows)
        all_rows.extend(family_rows)
    manifest["family_counts"] = dict(manifest["family_counts"])
    manifest["task_count"] = len(all_rows)
    dump_json(out_dir / "manifest.json", manifest)
    dump_jsonl(out_dir / "all_candidates.jsonl", all_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
