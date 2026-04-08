from __future__ import annotations

import json
from typing import Any, Dict, List

from researchworld.eval_v3 import aliases_for_label, dedupe_keep_order, normalize_text


def _top_titles(rows: List[Dict[str, Any]], limit: int = 4) -> List[str]:
    out: List[str] = []
    for row in rows[:limit]:
        value = normalize_text((row or {}).get("title") or "")
        if value:
            out.append(value)
    return dedupe_keep_order(out)


def _top_names(rows: List[Dict[str, Any]], key: str, limit: int = 4) -> List[str]:
    out: List[str] = []
    for row in rows[:limit]:
        value = normalize_text((row or {}).get(key) or "")
        if value:
            out.append(value)
    return dedupe_keep_order(out)


def _component(name: str, description: str, values: List[str], *, importance: float = 1.0, notes: str = "") -> Dict[str, Any]:
    canonical = dedupe_keep_order(values)
    return {
        "name": name,
        "description": description,
        "importance": importance,
        "canonical_values": canonical,
        "aliases": {value: aliases_for_label(value) for value in canonical},
        "notes": normalize_text(notes),
    }


def build_component_targets_v3_1(hidden_v3_row: Dict[str, Any], trace_row: Dict[str, Any]) -> Dict[str, Any]:
    family = str(hidden_v3_row.get("family") or "")
    slots = hidden_v3_row.get("slot_targets") or {}
    gt = (trace_row.get("ground_truth") or {})
    support = (trace_row.get("support_context") or {})
    components: List[Dict[str, Any]] = []

    if family == "bottleneck_opportunity_discovery":
        components = [
            _component(
                "bottleneck",
                "The answer should identify a concrete unresolved bottleneck rather than a broad area.",
                list(slots.get("core_bottleneck_labels") or []),
                importance=1.0,
                notes="Prefer technically discriminative bottlenecks grounded in historical limitations or failures.",
            ),
            _component(
                "opportunity",
                "The answer should identify a concrete opportunity that becomes viable if the bottleneck is addressed.",
                list(slots.get("core_opportunity_labels") or []),
                importance=1.0,
                notes="Opportunity should be specific and future-facing, not just a generic adjacent topic.",
            ),
            _component(
                "linkage",
                "The answer should explicitly connect the opportunity to the bottleneck through a mechanism or enabling shift.",
                list(slots.get("core_bottleneck_labels") or []) + list(slots.get("core_opportunity_labels") or []),
                importance=0.9,
                notes="Reward answers that explain why solving the bottleneck opens the stated opportunity.",
            ),
            _component(
                "mechanism_depth",
                "The answer should explain why the bottleneck remains unresolved and why the opportunity is strategically meaningful.",
                _top_names(list(support.get("top_limitations") or []), "name", limit=4),
                importance=0.8,
                notes="Look for causal drivers, trade-offs, evaluation gaps, or empirical failure patterns.",
            ),
        ]
    elif family == "direction_forecasting":
        trajectory = normalize_text(slots.get("trajectory_label") or "")
        components = [
            _component(
                "trajectory",
                "The answer should make one clear trajectory call.",
                [trajectory] if trajectory else [],
                importance=1.0,
                notes="Expected labels include accelerating, steady, fragmenting, or cooling.",
            ),
            _component(
                "next_directions",
                "The answer should name one to three concrete future directions.",
                list(slots.get("emergent_direction_labels") or []),
                importance=1.0,
                notes="Prefer specific future subdirections over broad parent topics.",
            ),
            _component(
                "signals",
                "The answer should identify enabling or friction signals that justify the trajectory.",
                list(slots.get("future_themes") or []) + list(slots.get("emergent_direction_labels") or []),
                importance=0.8,
                notes="Reward momentum, enabling shifts, evaluation pressure, or friction awareness.",
            ),
            _component(
                "calibration",
                "The answer should show uncertainty discipline rather than overclaiming.",
                [trajectory] if trajectory else [],
                importance=0.6,
                notes="Reward answers that distinguish strong signals from speculative ones.",
            ),
        ]
    elif family == "strategic_research_planning":
        gt_records = list(gt.get("direction_records") or [])
        ranked = _top_names(gt_records, "display_name", limit=4) or list(slots.get("priority_direction_labels") or [])
        components = [
            _component(
                "ranked_directions",
                "The answer should provide a prioritized set of research directions rather than an unranked list.",
                ranked,
                importance=1.0,
                notes="Ranking matters; directions should be differentiated rather than repetitive.",
            ),
            _component(
                "why_now",
                "The answer should explain why the selected directions are timely and tractable.",
                list(slots.get("priority_direction_labels") or []),
                importance=0.9,
                notes="Reward answers that tie prioritization to momentum, feasibility, or structural openings.",
            ),
            _component(
                "risk_awareness",
                "The answer should acknowledge crowded areas, uncertainty, or implementation risks.",
                _top_names(gt_records, "display_name", limit=4),
                importance=0.7,
                notes="Reward explicit trade-off and risk language.",
            ),
            _component(
                "venue_strategy",
                "If relevant, the answer should reflect venue-fit or publication-likelihood considerations.",
                [normalize_text(slots.get("target_venue_bucket") or ""), normalize_text(slots.get("target_venue_name") or "")],
                importance=0.5,
                notes="Only score when the answer actually addresses venue-aware planning.",
            ),
        ]

    return {"family": family, "components": components}


def build_future_alignment_targets_v3_1(hidden_v3_row: Dict[str, Any], trace_row: Dict[str, Any]) -> Dict[str, Any]:
    family = str(hidden_v3_row.get("family") or "")
    gt = trace_row.get("ground_truth") or {}
    support = trace_row.get("support_context") or {}
    slots = hidden_v3_row.get("slot_targets") or {}

    units: List[Dict[str, Any]] = []
    future_papers = _top_titles(list(support.get("future_validation_set") or []), limit=6)

    if family == "bottleneck_opportunity_discovery":
        for idx, value in enumerate(list(slots.get("core_opportunity_labels") or [])[:3], start=1):
            units.append(
                {
                    "unit_id": f"opp_{idx}",
                    "unit_type": "opportunity",
                    "text": value,
                    "aliases": aliases_for_label(value),
                    "importance": 1.0 if idx == 1 else 0.8,
                    "future_paper_titles": future_papers[:4],
                }
            )
    elif family == "direction_forecasting":
        venue_forecast = gt.get("venue_forecast") or {}
        for idx, value in enumerate(list(slots.get("emergent_direction_labels") or [])[:4], start=1):
            units.append(
                {
                    "unit_id": f"dir_{idx}",
                    "unit_type": "direction",
                    "text": value,
                    "aliases": aliases_for_label(value),
                    "importance": 1.0 if idx == 1 else 0.85,
                    "future_paper_titles": future_papers[:4],
                    "target_venue_bucket": normalize_text(venue_forecast.get("likely_bucket") or ""),
                }
            )
    elif family == "strategic_research_planning":
        for idx, row in enumerate(list(gt.get("direction_records") or [])[:4], start=1):
            value = normalize_text((row or {}).get("display_name") or "")
            if not value:
                continue
            units.append(
                {
                    "unit_id": f"plan_{idx}",
                    "unit_type": "priority_direction",
                    "text": value,
                    "aliases": aliases_for_label(value),
                    "importance": 1.0 if idx == 1 else 0.85,
                    "future_paper_titles": future_papers[:4],
                    "future_paper_count": (row or {}).get("future_paper_count"),
                    "target_venue_bucket": normalize_text(gt.get("target_venue_bucket") or ""),
                }
            )

    return {
        "enabled": family in {"bottleneck_opportunity_discovery", "direction_forecasting", "strategic_research_planning"},
        "alignment_units": units,
        "future_window_stats": {
            "paper_count": (gt.get("future_half_stats") or gt.get("target_window_stats") or {}).get("paper_count"),
            "top_conf_share": (gt.get("future_half_stats") or gt.get("target_window_stats") or {}).get("top_conf_share"),
        },
        "reference_future_papers": future_papers,
    }


def build_hidden_eval_v3_1_row(hidden_v3_row: Dict[str, Any], trace_row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(hidden_v3_row)
    row["schema_version"] = "v3.1"
    row["component_targets"] = build_component_targets_v3_1(hidden_v3_row, trace_row)
    row["future_alignment_targets"] = build_future_alignment_targets_v3_1(hidden_v3_row, trace_row)
    row["eval_profiles_v3_1"] = {
        "task_fulfillment": {
            "mode": "component_aware_task_fulfillment",
            "core_components": [str(x.get("name") or "") for x in (row.get("component_targets") or {}).get("components", [])],
        },
        "strategic_intelligence": {
            "mode": "anchored_strategic_intelligence_judge",
            "anchors": {
                "0.2": "Generic, weakly structured, or mostly descriptive.",
                "0.4": "Some useful reasoning, but shallow or weakly prioritized.",
                "0.6": "Structured and reasonably insightful, with clear but limited strategic value.",
                "0.8": "Strong multi-hop synthesis, good trade-off reasoning, and clear strategic utility.",
                "1.0": "Senior-researcher quality synthesis with non-obvious, decision-useful insight.",
            },
        },
        "future_alignment": {
            "mode": "future_unit_alignment",
            "unit_count": len((row.get("future_alignment_targets") or {}).get("alignment_units", [])),
        },
    }
    return row
