from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import researchworld.benchmark_v2 as benchmark_v2
import researchworld.family_pipelines as family_pipelines
from researchworld.corpus import iter_jsonl
from researchworld.verbalization import public_descendant_names, public_topic_from_packet, sentence_case_phrase, title_case_phrase


SETTING_ID = "q1_3m"
HISTORY_END = "2025-11-30"
FUTURE_WINDOW = "2025-12-01_to_2026-02-28"
FUTURE_WINDOW_KEY = "quarterly_2026q1"
DEFAULT_DOMAINS = [
    "llm_agent",
    "llm_finetuning_post_training",
    "rag_and_retrieval_structuring",
    "visual_generative_modeling_and_diffusion",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build q1_3m short-horizon candidates across four domains.")
    p.add_argument(
        "--trajectorylab-root",
        default="/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab",
    )
    p.add_argument(
        "--support-root",
        default=str(ROOT / "tmp" / "q1_support_packets_20251130"),
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "tmp" / "q1_short_candidates"),
    )
    p.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def patch_roots(trajectorylab_root: Path) -> None:
    benchmark_v2.ROOT = trajectorylab_root
    family_pipelines.ROOT = trajectorylab_root


def effective_q1_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    q1 = json.loads(json.dumps(((packet.get("future_windows") or {}).get(FUTURE_WINDOW_KEY) or {}), ensure_ascii=False))
    out = json.loads(json.dumps(packet, ensure_ascii=False))
    out["future_windows"] = {
        "quarterly_2026q1": q1,
        "halfyear_2025q4_2026q1": q1,
    }
    out["history_end_date"] = HISTORY_END
    out["setting_id"] = SETTING_ID
    return out


def iter_support_packets(support_root: Path, domains: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    selected = load_json(support_root / "selected_seed_nodes.json")
    packets_list = load_json(support_root / "all_node_support_packets.json")
    packet_by_id = {str(row["packet_id"]): row for row in packets_list}
    out = {domain: [] for domain in domains}
    for seed in selected:
        domain = str(seed.get("domain") or "")
        if domain not in out:
            continue
        pkt = packet_by_id.get(str(seed.get("packet_id") or ""))
        if pkt:
            out[domain].append(effective_q1_packet(pkt))
    return out


def build_q1_direction_candidate(packet: Dict[str, Any]) -> Dict[str, Any]:
    history_chain = packet.get("history_chain") or []
    chain_labels = [sentence_case_phrase(str(row.get("display_name") or "")) for row in history_chain if str(row.get("display_name") or "").strip()]
    future_terminal = packet.get("future_terminal") or {}
    topic = public_topic_from_packet({"display_name": packet.get("public_metadata", {}).get("topic") or packet.get("seed_node_id")})
    future_descendants = list((packet.get("future_chain") or [future_terminal])[1:] or [future_terminal])
    future_themes = [x for x in public_descendant_names(future_descendants, limit=4) if x.lower() != topic.lower()]
    trajectory = packet.get("trajectory") or {}
    future_stats = packet.get("future_half_stats") or {}
    question = (
        f"Based on literature available up to {packet.get('history_end_date')}, consider the historical idea chain "
        f"{' -> '.join(chain_labels[-4:]) or topic}. What specific next-step direction is most likely to emerge over the next three months, "
        f"and should the trajectory of this area be characterized as accelerating, fragmenting, steady, or cooling?"
    )
    answer = (
        f"The realized next-step direction was {sentence_case_phrase(str(future_terminal.get('display_name') or 'a more specialized subdirection'))}, "
        f"and the trajectory was {trajectory.get('trajectory_label')}. This evolution was supported by "
        f"{int(future_stats.get('paper_count') or 0)} papers in the following quarter."
    )
    return {
        "task_id": f"{SETTING_ID}::{packet['packet_id']}",
        "setting_id": SETTING_ID,
        "benchmark_version": "v4_q1_short_candidate_20260409",
        "family": "direction_forecasting",
        "subtype": "q1_terminal_forecast",
        "domain": packet["domain"],
        "horizon": "quarter",
        "draft_question": question,
        "draft_reference_answer": answer,
        "time_context": {
            "history_end": packet.get("history_end_date"),
            "history_structure_slice": packet.get("history_structure_slice"),
            "future_window": FUTURE_WINDOW,
            "setting_id": SETTING_ID,
        },
        "seed": {
            "packet_id": packet.get("packet_id"),
            "node_id": packet.get("seed_node_id"),
            "seed_group_id": f"{packet['domain']}::{packet.get('seed_node_id')}",
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
            "future_half_stats": future_stats,
            "future_quarter_stats": future_stats,
            "reference_papers": {
                "history": packet.get("history_representative_papers"),
                "future_q1": packet.get("future_q1_representative_papers"),
            },
        },
        "public_metadata": {
            "topic": topic,
            "topic_title": title_case_phrase(topic),
            "future_themes": future_themes,
            "setting_id": SETTING_ID,
        },
        "quality_signals": {
            "heuristic_score": round(
                0.42
                + min(0.20, len(history_chain) / 6.0)
                + min(0.20, len(packet.get("future_chain") or []) / 4.0)
                + min(0.18, abs(float(trajectory.get("venue_share_delta") or 0.0)) * 2.0),
                4,
            ),
        },
    }


def build_q1_bottleneck_candidate(packet: Dict[str, Any]) -> Dict[str, Any]:
    limits = packet.get("historical_limitation_cluster") or []
    future_dirs = packet.get("realized_opportunity_directions") or []
    topic = public_topic_from_packet({"display_name": packet.get("topic") or packet.get("seed_node_id")})
    future_themes = public_descendant_names(future_dirs, limit=4)
    future_stats = packet.get("future_half_stats") or {}
    question = (
        f"Using literature available before {packet.get('history_end_date')}, identify the most consequential unresolved bottleneck in "
        f"{topic} and explain which concrete research opportunity would be most likely to open if that bottleneck were addressed over the next three months. "
        f"Ground the answer in recurring limitations or failure evidence from the historical literature."
    )
    answer = (
        f"A historically grounded bottleneck-opportunity answer should center on limitations such as "
        f"{', '.join(str(x.get('name') or '') for x in limits[:3] if str(x.get('name') or '').strip()) or 'insufficient explicit limitation evidence'}, "
        f"and connect them to later-opened directions such as "
        f"{', '.join(future_themes[:3]) or 'subsequent follow-on directions'}. "
        f"The realized quarter contained {int(future_stats.get('paper_count') or 0)} papers."
    )
    return {
        "task_id": f"{SETTING_ID}::{packet['packet_id']}",
        "setting_id": SETTING_ID,
        "benchmark_version": "v4_q1_short_candidate_20260409",
        "family": "bottleneck_opportunity_discovery",
        "subtype": "q1_pageindex_grounded_bottleneck",
        "domain": packet["domain"],
        "horizon": "quarter",
        "draft_question": question,
        "draft_reference_answer": answer,
        "time_context": {
            "history_end": packet.get("history_end_date"),
            "history_structure_slice": packet.get("history_structure_slice"),
            "future_window": FUTURE_WINDOW,
            "setting_id": SETTING_ID,
        },
        "seed": {
            "packet_id": packet.get("packet_id"),
            "node_id": packet.get("seed_node_id"),
            "seed_group_id": f"{packet['domain']}::{packet.get('seed_node_id')}",
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
            "future_half_stats": future_stats,
            "future_quarter_stats": future_stats,
            "reference_papers": {
                "history": packet.get("history_reading_set"),
                "future_q1": packet.get("future_validation_set"),
            },
        },
        "public_metadata": {
            "topic": topic,
            "topic_title": title_case_phrase(topic),
            "future_themes": future_themes,
            "setting_id": SETTING_ID,
        },
        "quality_signals": {
            "heuristic_score": round(
                0.40
                + min(0.20, len(packet.get("historical_limitation_cluster") or []) / 5.0)
                + min(0.20, len(packet.get("history_reading_set") or []) / 8.0)
                + (0.15 if bool((packet.get("pageindex_priority") or {}).get("has_limitation_signal")) else 0.0),
                4,
            ),
        },
    }


def build_q1_planning_candidate(packet: Dict[str, Any]) -> Dict[str, Any]:
    topic = public_topic_from_packet({"display_name": packet.get("topic") or packet.get("seed_node_id")})
    candidate_directions = [sentence_case_phrase(x) for x in (packet.get("candidate_directions") or []) if str(x).strip()]
    future_themes = public_descendant_names(packet.get("future_descendant_records") or packet.get("direction_records") or [], limit=4)
    target_stats = packet.get("target_window_stats") or {}
    question = (
        f"Assume you are planning research on {topic} at the cutoff {packet.get('history_end_date')}. "
        f"If you could prioritize only a small number of next-step directions for the next three months, which directions should receive priority, "
        f"and what evidence-based rationale supports that ranking?"
    )
    answer = (
        f"A high-value near-term plan should prioritize directions such as "
        f"{', '.join(candidate_directions) or 'the strongest candidate directions'}, "
        f"while accounting for a realized quarter with {int(target_stats.get('paper_count') or 0)} papers "
        f"and a realized top-venue share of {float(target_stats.get('top_conf_share') or 0.0):.4f}."
    )
    return {
        "task_id": f"{SETTING_ID}::{packet['packet_id']}",
        "setting_id": SETTING_ID,
        "benchmark_version": "v4_q1_short_candidate_20260409",
        "family": "strategic_research_planning",
        "subtype": "q1_agenda_priority_selection",
        "domain": packet["domain"],
        "horizon": "quarter",
        "draft_question": question,
        "draft_reference_answer": answer,
        "time_context": {
            "history_end": packet.get("history_end_date"),
            "history_structure_slice": packet.get("history_structure_slice"),
            "future_window": FUTURE_WINDOW,
            "setting_id": SETTING_ID,
        },
        "seed": {
            "packet_id": packet.get("packet_id"),
            "node_id": packet.get("seed_node_id"),
            "seed_group_id": f"{packet['domain']}::{packet.get('seed_node_id')}",
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
            "target_window_stats": target_stats,
            "target_quarter_stats": target_stats,
            "venue_gap_signal": packet.get("venue_gap_signal"),
            "planning_priority_score": packet.get("planning_priority_score"),
        },
        "public_metadata": {
            "topic": topic,
            "topic_title": title_case_phrase(topic),
            "future_themes": future_themes or candidate_directions[:4],
            "setting_id": SETTING_ID,
        },
        "quality_signals": {
            "heuristic_score": round(
                0.42
                + min(0.20, len(candidate_directions) / 5.0)
                + min(0.20, max(0.0, float(packet.get("planning_priority_score") or 0.0)) / 80.0)
                + min(0.15, abs(float(packet.get("venue_gap_signal") or 0.0)) * 2.0),
                4,
            ),
        },
    }


def main() -> None:
    args = parse_args()
    trajectorylab_root = Path(args.trajectorylab_root)
    support_root = Path(args.support_root)
    out_dir = Path(args.out_dir)
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]

    patch_roots(trajectorylab_root)
    packets_by_domain = iter_support_packets(support_root, domains)

    all_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {"setting_id": SETTING_ID, "domains": {}, "family_counts": {}}

    for domain in domains:
        packets = packets_by_domain.get(domain, [])
        chain_packets = family_pipelines.build_chain_packets(domain=domain, packets=packets)
        opportunity_packets = family_pipelines.build_opportunity_packets(domain=domain, packets=packets)
        planning_packets = family_pipelines.build_planning_packets(domain=domain, packets=packets)

        direction_rows = [build_q1_direction_candidate(row) for row in chain_packets]
        bottleneck_rows = [build_q1_bottleneck_candidate(row) for row in opportunity_packets]
        planning_rows = [build_q1_planning_candidate(row) for row in planning_packets]

        manifest["domains"][domain] = {
            "selected_seed_packets": len(packets),
            "direction_forecasting": len(direction_rows),
            "bottleneck_opportunity_discovery": len(bottleneck_rows),
            "strategic_research_planning": len(planning_rows),
        }

        dump_jsonl(out_dir / domain / "direction_forecasting.jsonl", direction_rows)
        dump_jsonl(out_dir / domain / "bottleneck_opportunity_discovery.jsonl", bottleneck_rows)
        dump_jsonl(out_dir / domain / "strategic_research_planning.jsonl", planning_rows)
        all_rows.extend(direction_rows)
        all_rows.extend(bottleneck_rows)
        all_rows.extend(planning_rows)

    from collections import Counter
    manifest["family_counts"] = dict(Counter(row["family"] for row in all_rows))
    manifest["candidate_count"] = len(all_rows)
    dump_json(out_dir / "manifest.json", manifest)
    dump_jsonl(out_dir / "all_candidates.jsonl", all_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
