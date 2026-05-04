from __future__ import annotations

from typing import Any, Dict, List


FAMILY_EVALUATION_RUBRICS: Dict[str, Dict[str, Any]] = {
    "bottleneck_opportunity_discovery": {
        "mode": "manual_single_file_v2",
        "score_scale": [0.0, 1.0],
        "dimensions": [
            {
                "name": "bottleneck_specificity",
                "weight": 0.35,
                "description": "Names one concrete unresolved bottleneck rather than a generic trend.",
            },
            {
                "name": "causal_linkage",
                "weight": 0.25,
                "description": "Explains how solving the bottleneck unlocks the stated opportunity.",
            },
            {
                "name": "historical_grounding",
                "weight": 0.25,
                "description": "Anchors the bottleneck in pre-cutoff limitations, failures, or gaps.",
            },
            {
                "name": "opportunity_quality",
                "weight": 0.15,
                "description": "Proposes a concrete near-term opportunity rather than a broad future trend.",
            },
        ],
    },
    "direction_forecasting": {
        "mode": "manual_single_file_v2",
        "score_scale": [0.0, 1.0],
        "dimensions": [
            {
                "name": "primary_direction",
                "weight": 0.35,
                "description": "Commits to one concrete next-step direction.",
            },
            {
                "name": "trajectory_call",
                "weight": 0.2,
                "description": "Makes one explicit trajectory call and keeps it internally consistent.",
            },
            {
                "name": "signal_chain",
                "weight": 0.3,
                "description": "Uses pre-cutoff technical signals to justify why this direction is next.",
            },
            {
                "name": "calibration",
                "weight": 0.15,
                "description": "Avoids overclaiming and distinguishes dominant signals from secondary ones.",
            },
        ],
    },
    "strategic_research_planning": {
        "mode": "manual_single_file_v2",
        "score_scale": [0.0, 1.0],
        "dimensions": [
            {
                "name": "ranking_quality",
                "weight": 0.3,
                "description": "Provides a complete ordering over the listed candidate directions.",
            },
            {
                "name": "prioritization_rationale",
                "weight": 0.35,
                "description": "Explains why higher-ranked items should move first in the near term.",
            },
            {
                "name": "dependency_awareness",
                "weight": 0.2,
                "description": "Accounts for gating dependencies, tractability, or scope differences.",
            },
            {
                "name": "deprioritization_logic",
                "weight": 0.15,
                "description": "Explains why lower-ranked items trail rather than merely praising all options.",
            },
        ],
    },
    "venue_aware_research_positioning": {
        "mode": "manual_single_file_v2",
        "score_scale": [0.0, 1.0],
        "dimensions": [
            {
                "name": "direction_specificity",
                "weight": 0.3,
                "description": "Names one concrete direction rather than a broad topic area.",
            },
            {
                "name": "technical_fit",
                "weight": 0.25,
                "description": "Explains the technical contribution package behind the forecast.",
            },
            {
                "name": "venue_bucket",
                "weight": 0.15,
                "description": "Names one venue bucket clearly and consistently.",
            },
            {
                "name": "venue_fit_reasoning",
                "weight": 0.3,
                "description": "Justifies venue fit using reviewer expectations, framing style, or evaluation emphasis.",
            },
        ],
    },
}


def default_evaluation_rubric(family: str) -> Dict[str, Any]:
    rubric = FAMILY_EVALUATION_RUBRICS.get(str(family or ""))
    if rubric is None:
        return {"mode": "manual_single_file_v2", "score_scale": [0.0, 1.0], "dimensions": []}
    return {
        "mode": rubric.get("mode"),
        "score_scale": list(rubric.get("score_scale") or [0.0, 1.0]),
        "dimensions": [dict(item) for item in (rubric.get("dimensions") or []) if isinstance(item, dict)],
    }


def rubric_dimension_names(family: str) -> List[str]:
    return [
        str(item.get("name") or "").strip()
        for item in (default_evaluation_rubric(family).get("dimensions") or [])
        if str(item.get("name") or "").strip()
    ]
