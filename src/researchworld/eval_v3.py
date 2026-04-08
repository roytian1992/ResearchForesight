from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


FAMILY_SCORE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "bottleneck_opportunity_discovery": {
        "fact_precision": 0.40,
        "task_fulfillment": 0.30,
        "insight_judge": 0.30,
    },
    "direction_forecasting": {
        "fact_precision": 0.30,
        "task_fulfillment": 0.35,
        "insight_judge": 0.35,
    },
    "strategic_research_planning": {
        "fact_precision": 0.20,
        "task_fulfillment": 0.20,
        "insight_judge": 0.60,
    },
}


JUDGE_DIMENSIONS: Dict[str, List[Dict[str, Any]]] = {
    "bottleneck_opportunity_discovery": [
        {"name": "historical_grounding", "weight": 0.20},
        {"name": "bottleneck_specificity", "weight": 0.20},
        {"name": "opportunity_linkage", "weight": 0.25},
        {"name": "future_realization_alignment", "weight": 0.20},
        {"name": "insight_value", "weight": 0.15},
    ],
    "direction_forecasting": [
        {"name": "trajectory_call", "weight": 0.20},
        {"name": "history_to_future_reasoning", "weight": 0.20},
        {"name": "future_structure_alignment", "weight": 0.20},
        {"name": "statistical_alignment", "weight": 0.20},
        {"name": "temporal_discipline", "weight": 0.20},
    ],
    "strategic_research_planning": [
        {"name": "evidence_grounded_planning", "weight": 0.20},
        {"name": "priority_selection", "weight": 0.20},
        {"name": "strategic_value", "weight": 0.25},
        {"name": "future_alignment", "weight": 0.20},
        {"name": "temporal_discipline", "weight": 0.15},
    ],
}


def normalize_text(text: Any) -> str:
    value = str(text or "").replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


def slugify(text: Any) -> str:
    value = normalize_text(text).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        item = normalize_text(value)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def aliases_for_label(label: str) -> List[str]:
    label = normalize_text(label)
    if not label:
        return []
    variants = [
        label,
        label.lower(),
        label.replace("-", " "),
        label.replace("_", " "),
        slugify(label).replace("_", " "),
    ]
    return dedupe_keep_order(variants)


def _first_names(rows: Iterable[Dict[str, Any]], key: str, limit: int = 3) -> List[str]:
    out: List[str] = []
    for row in rows:
        value = normalize_text((row or {}).get(key))
        if value:
            out.append(value)
        if len(out) >= limit:
            break
    return dedupe_keep_order(out)


def _reference_titles(rows: Iterable[Dict[str, Any]], limit: int = 3) -> List[str]:
    return _first_names(rows, "title", limit=limit)


def build_slot_targets(hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> Dict[str, Any]:
    family = str(hidden_row.get("family") or "")
    gt = trace_row.get("ground_truth") or {}
    support = trace_row.get("support_context") or {}
    public = hidden_row.get("public_metadata") or trace_row.get("public_metadata") or {}
    history_stats = support.get("historical_stats") or {}
    slots: Dict[str, Any] = {
        "family": family,
        "topic": public.get("topic"),
        "topic_title": public.get("topic_title"),
        "future_themes": public.get("future_themes") or [],
        "history_cutoff": (trace_row.get("time_context") or {}).get("history_end"),
        "historical_stats": {
            "paper_count": history_stats.get("paper_count"),
            "top_conf_share": history_stats.get("top_conf_share"),
            "citation_median": history_stats.get("citation_median"),
        },
    }
    if family == "bottleneck_opportunity_discovery":
        slots.update(
            {
                "core_bottleneck_labels": _first_names(gt.get("historical_limitation_signals") or support.get("top_limitations") or [], "name", limit=3),
                "core_opportunity_labels": dedupe_keep_order(public.get("future_themes") or _first_names(gt.get("future_descendants") or [], "display_name", limit=3)),
                "future_realization_paper_count": (gt.get("future_half_stats") or {}).get("paper_count"),
                "future_realization_examples": _reference_titles(
                    ((gt.get("reference_papers") or {}).get("future_q4") or []) + ((gt.get("reference_papers") or {}).get("future_q1") or []),
                    limit=4,
                ),
            }
        )
    elif family == "direction_forecasting":
        traj = gt.get("trajectory") or {}
        venue_forecast = gt.get("venue_forecast") or {}
        slots.update(
            {
                "trajectory_label": traj.get("trajectory_label"),
                "future_half_stats": {
                    "paper_count": (gt.get("future_half_stats") or {}).get("paper_count"),
                    "top_conf_share": (gt.get("future_half_stats") or {}).get("top_conf_share"),
                },
                "emergent_direction_labels": _first_names(gt.get("emergent_descendants") or [], "display_name", limit=4),
                "trajectory_support": {
                    "future_to_history_ratio": traj.get("future_to_history_ratio"),
                    "venue_share_delta": traj.get("venue_share_delta"),
                    "split_pressure": traj.get("split_pressure"),
                },
            }
        )
        if venue_forecast:
            slots["venue_forecast"] = {
                "likely_bucket": venue_forecast.get("likely_bucket"),
                "likely_venue": venue_forecast.get("likely_venue"),
                "future_top_conf_count": venue_forecast.get("future_top_conf_count"),
                "future_top_conf_share": venue_forecast.get("future_top_conf_share"),
            }
    elif family == "strategic_research_planning":
        slots.update(
            {
                "priority_direction_labels": dedupe_keep_order(public.get("future_themes") or _first_names(gt.get("emergent_descendants") or [], "display_name", limit=3)),
                "target_window_stats": {
                    "paper_count": (gt.get("target_window_stats") or {}).get("paper_count"),
                    "top_conf_count": (gt.get("target_window_stats") or {}).get("top_conf_count"),
                    "top_conf_share": (gt.get("target_window_stats") or {}).get("top_conf_share"),
                },
                "venue_gap_signal": gt.get("venue_gap_signal"),
                "priority_score": gt.get("planning_priority_score"),
            }
        )
        if gt.get("target_venue_bucket"):
            slots["target_venue_bucket"] = gt.get("target_venue_bucket")
            slots["target_venue_name"] = gt.get("target_venue_name")
    return slots


def _make_claim(
    *,
    claim_id: str,
    text: str,
    claim_type: str,
    time_scope: str,
    importance: float,
    canonical_objects: List[str],
    source_pointer: str,
    match_policy: str = "all_of",
    min_match_count: int | None = None,
) -> Dict[str, Any]:
    canonical_objects = dedupe_keep_order(canonical_objects)
    return {
        "claim_id": claim_id,
        "text": normalize_text(text),
        "claim_type": claim_type,
        "time_scope": time_scope,
        "importance": importance,
        "canonical_objects": canonical_objects,
        "aliases": {obj: aliases_for_label(obj) for obj in canonical_objects},
        "source_pointer": source_pointer,
        "match_policy": match_policy,
        "min_match_count": min_match_count,
    }


def build_claim_bank(hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    family = str(hidden_row.get("family") or "")
    gt = trace_row.get("ground_truth") or {}
    support = trace_row.get("support_context") or {}
    public = hidden_row.get("public_metadata") or trace_row.get("public_metadata") or {}
    claims: List[Dict[str, Any]] = []

    if family == "bottleneck_opportunity_discovery":
        bottlenecks = _first_names(gt.get("historical_limitation_signals") or support.get("top_limitations") or [], "name", limit=2)
        opportunities = dedupe_keep_order(public.get("future_themes") or _first_names(gt.get("future_descendants") or [], "display_name", limit=3))
        if bottlenecks:
            claims.append(
                _make_claim(
                    claim_id="hist_bottleneck_core",
                    text=f"Before the cutoff, one central unresolved bottleneck was among: {', '.join(bottlenecks)}.",
                    claim_type="historical_fact",
                    time_scope="history",
                    importance=1.0,
                    canonical_objects=bottlenecks,
                    source_pointer="ground_truth.historical_limitation_signals",
                    match_policy="at_least_one",
                    min_match_count=1,
                )
            )
        if opportunities:
            claims.append(
                _make_claim(
                    claim_id="future_opportunity_core",
                    text=f"In the realized future window, work opened around directions such as {', '.join(opportunities[:3])}.",
                    claim_type="future_realization_fact",
                    time_scope="future",
                    importance=1.0,
                    canonical_objects=opportunities[:3],
                    source_pointer="public_metadata.future_themes",
                    match_policy="at_least_one",
                    min_match_count=1,
                )
            )
        future_half = gt.get("future_half_stats") or {}
        if future_half.get("paper_count") is not None:
            claims.append(
                _make_claim(
                    claim_id="future_half_volume",
                    text=f"The realized six-month future window contained {int(future_half.get('paper_count') or 0)} papers for this topic.",
                    claim_type="statistical_fact",
                    time_scope="future",
                    importance=0.7,
                    canonical_objects=[str(int(future_half.get("paper_count") or 0))],
                    source_pointer="ground_truth.future_half_stats.paper_count",
                )
            )

    elif family == "direction_forecasting":
        traj = gt.get("trajectory") or {}
        if traj.get("trajectory_label"):
            claims.append(
                _make_claim(
                    claim_id="trajectory_label",
                    text=f"The realized trajectory label was {traj.get('trajectory_label')}.",
                    claim_type="taxonomy_fact",
                    time_scope="future",
                    importance=1.0,
                    canonical_objects=[str(traj.get("trajectory_label"))],
                    source_pointer="ground_truth.trajectory.trajectory_label",
                )
            )
        venue_forecast = gt.get("venue_forecast") or {}
        if venue_forecast.get("likely_bucket"):
            claims.append(
                _make_claim(
                    claim_id="likely_venue_bucket",
                    text=f"The most likely top-tier venue bucket for this direction was {venue_forecast.get('likely_bucket')}.",
                    claim_type="venue_fact",
                    time_scope="future",
                    importance=0.9,
                    canonical_objects=[str(venue_forecast.get("likely_bucket"))],
                    source_pointer="ground_truth.venue_forecast.likely_bucket",
                )
            )
        descendants = _first_names(gt.get("emergent_descendants") or [], "display_name", limit=4)
        if descendants:
            claims.append(
                _make_claim(
                    claim_id="emergent_directions",
                    text=f"Realized future evolution included subdirections such as {', '.join(descendants[:4])}.",
                    claim_type="taxonomy_fact",
                    time_scope="future",
                    importance=1.0,
                    canonical_objects=descendants[:4],
                    source_pointer="ground_truth.emergent_descendants",
                    match_policy="at_least_one",
                    min_match_count=1,
                )
            )
        future_half = gt.get("future_half_stats") or {}
        hist = gt.get("historical_stats") or {}
        if future_half.get("paper_count") is not None and hist.get("paper_count") is not None:
            claims.append(
                _make_claim(
                    claim_id="history_future_volume",
                    text=(
                        f"The topic had {int(hist.get('paper_count') or 0)} historical papers before the cutoff "
                        f"and {int(future_half.get('paper_count') or 0)} papers in the realized six-month future window."
                    ),
                    claim_type="statistical_fact",
                    time_scope="cross_temporal",
                    importance=0.8,
                    canonical_objects=[
                        str(int(hist.get("paper_count") or 0)),
                        str(int(future_half.get("paper_count") or 0)),
                    ],
                    source_pointer="ground_truth.historical_stats.paper_count + ground_truth.future_half_stats.paper_count",
                )
            )

    elif family == "strategic_research_planning":
        target = gt.get("target_window_stats") or {}
        directions = dedupe_keep_order(public.get("future_themes") or _first_names(gt.get("emergent_descendants") or [], "display_name", limit=3))
        if directions:
            claims.append(
                _make_claim(
                    claim_id="priority_directions",
                    text=f"High-value planning should center on directions such as {', '.join(directions[:3])}.",
                    claim_type="future_realization_fact",
                    time_scope="future",
                    importance=1.0,
                    canonical_objects=directions[:3],
                    source_pointer="public_metadata.future_themes",
                    match_policy="at_least_one",
                    min_match_count=1,
                )
            )
        if gt.get("target_venue_bucket"):
            claims.append(
                _make_claim(
                    claim_id="target_venue_bucket",
                    text=f"The planning target venue bucket was {gt.get('target_venue_bucket')}.",
                    claim_type="venue_fact",
                    time_scope="future",
                    importance=0.8,
                    canonical_objects=[str(gt.get("target_venue_bucket"))],
                    source_pointer="ground_truth.target_venue_bucket",
                )
            )
        if target.get("paper_count") is not None:
            claims.append(
                _make_claim(
                    claim_id="target_window_volume",
                    text=f"In the realized target window, the topic produced {int(target.get('paper_count') or 0)} papers.",
                    claim_type="statistical_fact",
                    time_scope="future",
                    importance=0.8,
                    canonical_objects=[str(int(target.get("paper_count") or 0))],
                    source_pointer="ground_truth.target_window_stats.paper_count",
                )
            )
        if target.get("top_conf_share") is not None:
            claims.append(
                _make_claim(
                    claim_id="target_window_venue_share",
                    text=f"The realized target-window top-venue share was {float(target.get('top_conf_share') or 0.0):.4f}.",
                    claim_type="venue_fact",
                    time_scope="future",
                    importance=0.7,
                    canonical_objects=[f"{float(target.get('top_conf_share') or 0.0):.4f}"],
                    source_pointer="ground_truth.target_window_stats.top_conf_share",
                )
            )

    return claims


def build_hidden_eval_v3_row(hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> Dict[str, Any]:
    family = str(hidden_row.get("family") or "")
    slot_targets = build_slot_targets(hidden_row, trace_row)
    time_context = trace_row.get("time_context") or {}
    future_label = str(((time_context.get("future_windows") or {}).get("halfyear_2025q4_2026q1")) or "")
    future_start = ""
    future_end = ""
    if "_to_" in future_label:
        future_start, future_end = future_label.split("_to_", 1)
    return {
        "task_id": hidden_row.get("task_id"),
        "internal_task_id": hidden_row.get("internal_task_id"),
        "family": family,
        "domain": hidden_row.get("domain"),
        "title": hidden_row.get("title"),
        "gold_answer": hidden_row.get("gold_answer"),
        "expected_answer_points": hidden_row.get("expected_answer_points") or [],
        "slot_targets": slot_targets,
        "claim_bank": build_claim_bank(hidden_row, trace_row),
        "judge_profile": {
            "mode": "benchmark_aware_structured_judge",
            "dimensions": JUDGE_DIMENSIONS.get(family, []),
            "score_weights": FAMILY_SCORE_WEIGHTS.get(family, {}),
        },
        "temporal_policy": {
            "history_cutoff": time_context.get("history_end"),
            "history_slice": time_context.get("history_structure_slice"),
            "future_start": future_start,
            "future_end": future_end,
            "future_window_label": future_label,
            "history_evidence_only_for_reasoning": True,
            "future_window_hidden_for_fact_verification": True,
        },
        "raw_trace_links": {
            "seed": trace_row.get("seed"),
            "time_context": time_context,
            "support_context": trace_row.get("support_context"),
            "ground_truth": trace_row.get("ground_truth"),
        },
    }
