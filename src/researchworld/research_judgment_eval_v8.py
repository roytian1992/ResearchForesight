from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

from researchworld.experiment_eval_v3 import infer_domain_id
from researchworld.llm import OpenAICompatChatClient, complete_json_object


RUBRIC_VERSION = "v8_research_judgment_strict_deliberative_decision_20260430"

RESEARCH_JUDGMENT_WEIGHTS: Dict[str, float] = {
    "core_decision_quality": 0.30,
    "mechanistic_reasoning": 0.20,
    "comparative_judgment": 0.25,
    "decision_robustness": 0.25,
}

JUDGE_DIMENSIONS = [
    "core_decision_quality",
    "mechanistic_reasoning",
    "comparative_judgment",
    "decision_robustness",
]

NEIGHBORHOOD_RELATIONS = {"equivalent", "adjacent", "broader_valid", "narrower_valid"}
ADVANCED_NEIGHBORHOOD_LABELS = {"more_forward_than_gold", "stronger_than_gold"}


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _mean(values: Sequence[float]) -> float:
    import statistics

    return round(statistics.mean(values), 4) if values else 0.0


def _compact_task(public_task: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "task_id",
        "family",
        "subtype",
        "domain",
        "time_cutoff",
        "horizon",
        "title",
        "question",
        "answer_contract",
    ]
    return {key: public_task.get(key) for key in keep if key in public_task}


def _compact_reference(hidden_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "gold_answer": hidden_row.get("gold_answer"),
        "expected_answer_points": hidden_row.get("expected_answer_points") or [],
        "component_targets": hidden_row.get("component_targets") or {},
        "future_alignment_targets": hidden_row.get("future_alignment_targets") or {},
        "temporal_policy": hidden_row.get("temporal_policy") or {},
    }


def _family_policy(family: str) -> str:
    if family == "bottleneck_opportunity_discovery":
        return (
            "Reward a concrete bottleneck only when the answer explains why this bottleneck dominates nearby limitations, "
            "what dependency makes it hard, and what failure mode would invalidate the opportunity."
        )
    if family == "direction_forecasting":
        return (
            "Reward a forecast only when the answer distinguishes the selected direction from plausible adjacent futures, "
            "states the signal chain, and gives conditions under which the forecast would fail."
        )
    if family == "strategic_research_planning":
        return (
            "Reward planning only when the answer makes a priority/ranking decision, explains trade-offs and dependencies, "
            "and gives actionable milestones, deferral logic, or kill criteria."
        )
    if family == "venue_aware_research_positioning":
        return (
            "Reward venue positioning only when the answer compares candidate directions or venue buckets, explains reviewer-fit "
            "trade-offs, and states how to position the work within the near-term venue horizon."
        )
    return "Reward explicit research decision-making, mechanism, comparison, and robustness."


def _score_from_judge_obj(obj: Dict[str, Any]) -> Dict[str, float]:
    raw = obj.get("scores") if isinstance(obj.get("scores"), dict) else obj
    dimensions = {name: _clamp01(raw.get(name)) for name in JUDGE_DIMENSIONS}
    relation = str(obj.get("relation_to_gold") or "").strip().lower()
    scope_relation = str(obj.get("scope_relation") or "").strip().lower()
    advancement = str(obj.get("neighborhood_advancement") or "").strip().lower()
    neighborhood_valid = relation in NEIGHBORHOOD_RELATIONS and scope_relation != "off_target"
    more_forward = _bool(obj.get("more_forward_than_gold")) or advancement in ADVANCED_NEIGHBORHOOD_LABELS

    has_explicit_comparison = _bool(obj.get("has_explicit_comparison"))
    has_risk_or_dependency = _bool(obj.get("has_risk_or_dependency"))
    has_actionable_decision_rule = _bool(obj.get("has_actionable_decision_rule"))
    has_future_specificity = _bool(obj.get("has_future_specificity"))

    # Prevent fluent one-path answers from being scored as deliberative.
    if not has_explicit_comparison:
        dimensions["comparative_judgment"] = min(dimensions["comparative_judgment"], 0.55)
    if not has_risk_or_dependency:
        dimensions["decision_robustness"] = min(dimensions["decision_robustness"], 0.55)
    if not has_actionable_decision_rule:
        dimensions["decision_robustness"] = min(dimensions["decision_robustness"], 0.70)
    if not has_future_specificity:
        dimensions["mechanistic_reasoning"] = min(dimensions["mechanistic_reasoning"], 0.75)

    if relation == "equivalent":
        dimensions["core_decision_quality"] = max(dimensions["core_decision_quality"], 0.76)
    elif relation == "narrower_valid":
        dimensions["core_decision_quality"] = max(dimensions["core_decision_quality"], 0.68)
    elif relation == "broader_valid":
        dimensions["core_decision_quality"] = max(dimensions["core_decision_quality"], 0.60)
    elif relation == "adjacent" and scope_relation in {"right_scope", "mixed", "too_broad", "too_narrow"}:
        dimensions["core_decision_quality"] = max(dimensions["core_decision_quality"], 0.56)
        if more_forward and has_explicit_comparison and has_risk_or_dependency and not _bool(obj.get("kitchen_sink_or_noncommittal")):
            dimensions["core_decision_quality"] = max(dimensions["core_decision_quality"], 0.70)
            dimensions["mechanistic_reasoning"] = max(dimensions["mechanistic_reasoning"], 0.64)
            dimensions["comparative_judgment"] = max(dimensions["comparative_judgment"], 0.64)
            dimensions["decision_robustness"] = max(dimensions["decision_robustness"], 0.68)

    overall = sum(RESEARCH_JUDGMENT_WEIGHTS[name] * dimensions[name] for name in RESEARCH_JUDGMENT_WEIGHTS)

    if _bool(obj.get("non_answer_or_refusal")):
        overall = min(overall, 0.25)
    if _bool(obj.get("critical_core_error")) and not neighborhood_valid:
        overall = min(overall, 0.49)
    if _bool(obj.get("wrong_required_label_or_ranking")) and not neighborhood_valid:
        overall = min(overall, 0.59)
    if _bool(obj.get("kitchen_sink_or_noncommittal")):
        overall = min(overall, 0.66)
    if more_forward and neighborhood_valid and has_explicit_comparison and has_risk_or_dependency and not _bool(obj.get("kitchen_sink_or_noncommittal")):
        overall = min(1.0, overall + 0.025)

    dimensions["research_judgment_score"] = _clamp01(overall)
    return {key: round(value, 4) for key, value in dimensions.items()}


def _judge_prompt(public_task: Dict[str, Any], hidden_row: Dict[str, Any], answer: str) -> str:
    family = str(public_task.get("family") or hidden_row.get("family") or "")
    return f"""# Role
You are a strict ResearchForesight deliberative-judgment evaluator.

# Objective
Score whether the candidate answer demonstrates high-quality research decision-making. This metric is separate from factuality, future-alignment similarity, citation support, and literal gold-point coverage.

# Non-Negotiable Rules
- Use only the public task, reference fields, and candidate answer below. Do not use external knowledge or a KB.
- Do not reward fluency alone. A polished single-path explanation is not enough for high deliberative judgment.
- Do not score citation support or factual correctness here.
- The gold answer is a reference anchor. Neighboring answers can receive credit, but only when the answer explicitly justifies why that neighbor is the better research decision.
- Mark more_forward_than_gold=true only if the answer explicitly compares against the reference neighborhood or plausible alternatives and explains why its decision is more actionable, more consequential, or more robust.
- Penalize answers that make a plausible claim without explicit alternatives, dependencies, risks, failure modes, milestones, or decision criteria.

# Family-Specific Policy
{_family_policy(family)}

# Public Task
{json.dumps(_compact_task(public_task), ensure_ascii=False, indent=2)}

# Reference Anchor
{json.dumps(_compact_reference(hidden_row), ensure_ascii=False, indent=2)}

# Candidate Answer
{answer}

# Required Analysis
Infer:
- required_decision: the task-local bottleneck, direction, ranking, venue-positioning, or trajectory decision.
- candidate_decision: the answer's committed decision.
- explicit_comparison: explicit contrast against alternatives, lower-ranked options, nearby venues, adjacent mechanisms, or the reference neighborhood.
- risk_or_dependency: explicit dependency, bottleneck, uncertainty, failure mode, kill criterion, or gating condition.
- actionable_decision_rule: concrete next step, milestone, ranking/deferral rule, venue-positioning rule, or near-term consequence.
- future_specificity: whether the reasoning is tied to the task horizon rather than a timeless generic argument.

# Score Dimensions
Return each score in [0, 1]:
- core_decision_quality: quality of the committed research decision relative to the task and reference anchor. Do not require literal gold-point overlap.
- mechanistic_reasoning: quality of the causal, technical, venue-fit, trajectory, or dependency mechanism.
- comparative_judgment: explicit quality of trade-off reasoning among neighboring options. If the answer does not explicitly compare alternatives, this should be <= 0.55.
- decision_robustness: quality of decision criteria, risks, dependencies, milestones, deferral logic, kill criteria, or conditions under which the decision would fail. If these are absent, this should be <= 0.55.

# Output JSON
{{
  "reference_units": {{
    "required_decision": ["..."],
    "mechanism_units": ["..."],
    "comparison_units": ["..."],
    "robustness_units": ["..."]
  }},
  "candidate_units": {{
    "candidate_decision": ["..."],
    "mechanism_chain": ["..."],
    "explicit_comparison": ["..."],
    "risk_or_dependency": ["..."],
    "actionable_decision_rule": ["..."]
  }},
  "scores": {{
    "core_decision_quality": 0.0,
    "mechanistic_reasoning": 0.0,
    "comparative_judgment": 0.0,
    "decision_robustness": 0.0
  }},
  "relation_to_gold": "equivalent | narrower_valid | broader_valid | adjacent | drift",
  "neighborhood_advancement": "same_level | more_forward_than_gold | less_specific | weaker_than_gold | not_applicable",
  "more_forward_than_gold": false,
  "scope_relation": "right_scope | too_broad | too_narrow | mixed | off_target",
  "critical_core_error": false,
  "wrong_required_label_or_ranking": false,
  "kitchen_sink_or_noncommittal": false,
  "non_answer_or_refusal": false,
  "has_explicit_comparison": false,
  "has_risk_or_dependency": false,
  "has_actionable_decision_rule": false,
  "has_future_specificity": false,
  "decision_strengths": ["..."],
  "decision_weaknesses": ["..."],
  "rationale": "Brief explanation focused on deliberative research judgment."
}}
"""


def evaluate_research_judgment_v8(
    *,
    judge_client: OpenAICompatChatClient,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    result_row: Dict[str, Any],
) -> Dict[str, Any]:
    answer = str(result_row.get("answer") or "").strip()
    if not answer:
        return {
            "research_judgment_score": 0.0,
            "dimension_scores": {name: 0.0 for name in JUDGE_DIMENSIONS},
            "relation_to_gold": "drift",
            "scope_relation": "off_target",
            "critical_core_error": True,
            "wrong_required_label_or_ranking": False,
            "kitchen_sink_or_noncommittal": False,
            "non_answer_or_refusal": True,
            "neighborhood_advancement": "not_applicable",
            "more_forward_than_gold": False,
            "has_explicit_comparison": False,
            "has_risk_or_dependency": False,
            "has_actionable_decision_rule": False,
            "has_future_specificity": False,
            "decision_strengths": [],
            "decision_weaknesses": [],
            "rationale": "Candidate answer is empty.",
            "evaluator_scope": "task_refined_json_only_no_kb",
            "rubric_version": RUBRIC_VERSION,
        }

    obj = complete_json_object(
        judge_client,
        [
            {"role": "system", "content": "You are a strict ResearchForesight deliberative-judgment evaluator. Return JSON only."},
            {"role": "user", "content": _judge_prompt(public_task, hidden_row, answer)},
        ],
        response_format={"type": "json_object"},
        max_tokens=2200,
        timeout=240,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Return exactly one valid JSON object following the requested schema.",
    )
    scores = _score_from_judge_obj(obj)
    return {
        "research_judgment_score": scores["research_judgment_score"],
        "dimension_scores": {name: scores[name] for name in JUDGE_DIMENSIONS},
        "relation_to_gold": str(obj.get("relation_to_gold") or "").strip(),
        "neighborhood_advancement": str(obj.get("neighborhood_advancement") or "").strip(),
        "more_forward_than_gold": _bool(obj.get("more_forward_than_gold")),
        "scope_relation": str(obj.get("scope_relation") or "").strip(),
        "critical_core_error": _bool(obj.get("critical_core_error")),
        "wrong_required_label_or_ranking": _bool(obj.get("wrong_required_label_or_ranking")),
        "kitchen_sink_or_noncommittal": _bool(obj.get("kitchen_sink_or_noncommittal")),
        "non_answer_or_refusal": _bool(obj.get("non_answer_or_refusal")),
        "has_explicit_comparison": _bool(obj.get("has_explicit_comparison")),
        "has_risk_or_dependency": _bool(obj.get("has_risk_or_dependency")),
        "has_actionable_decision_rule": _bool(obj.get("has_actionable_decision_rule")),
        "has_future_specificity": _bool(obj.get("has_future_specificity")),
        "reference_units": obj.get("reference_units") or {},
        "candidate_units": obj.get("candidate_units") or {},
        "decision_strengths": [str(x) for x in (obj.get("decision_strengths") or []) if str(x).strip()],
        "decision_weaknesses": [str(x) for x in (obj.get("decision_weaknesses") or []) if str(x).strip()],
        "rationale": str(obj.get("rationale") or "").strip(),
        "evaluator_scope": "task_refined_json_only_no_kb",
        "rubric_version": RUBRIC_VERSION,
    }


def build_research_judgment_result_row_v8(
    *,
    run_id: str,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
    judgment_eval: Dict[str, Any],
) -> Dict[str, Any]:
    scores = {"research_judgment_score": round(float(judgment_eval.get("research_judgment_score") or 0.0), 4)}
    for name, value in (judgment_eval.get("dimension_scores") or {}).items():
        scores[str(name)] = round(float(value or 0.0), 4)
    return {
        "run_id": run_id,
        "task_id": public_task.get("task_id"),
        "family": public_task.get("family"),
        "domain": infer_domain_id(result_row),
        "method": str(result_row.get("agent") or result_row.get("baseline") or result_row.get("method") or "unknown"),
        "answer": str(result_row.get("answer") or ""),
        "metadata": {
            "task_title": public_task.get("title"),
            "time_cutoff": public_task.get("time_cutoff"),
            "schema_version": RUBRIC_VERSION,
            "evaluator_scope": "task_refined_json_only_no_kb",
        },
        "scores": scores,
        "judgment_eval": judgment_eval,
    }


def summarize_research_judgment_results_v8(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family") or "")].append(row)
        by_domain[str(row.get("domain") or "")].append(row)
        by_method[str(row.get("method") or "")].append(row)

    def group_summary(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        dim_summary = {
            name: _mean([float(((row.get("scores") or {}).get(name) or 0.0)) for row in group])
            for name in JUDGE_DIMENSIONS
        }
        return {
            "count": len(group),
            "mean_research_judgment_score": _mean(
                [float(((row.get("scores") or {}).get("research_judgment_score") or 0.0)) for row in group]
            ),
            "dimension_summary": dim_summary,
            "relation_counts": dict(Counter(str((row.get("judgment_eval") or {}).get("relation_to_gold") or "") for row in group)),
            "scope_relation_counts": dict(Counter(str((row.get("judgment_eval") or {}).get("scope_relation") or "") for row in group)),
            "critical_core_error_count": sum(1 for row in group if (row.get("judgment_eval") or {}).get("critical_core_error")),
            "wrong_required_label_or_ranking_count": sum(1 for row in group if (row.get("judgment_eval") or {}).get("wrong_required_label_or_ranking")),
            "kitchen_sink_or_noncommittal_count": sum(1 for row in group if (row.get("judgment_eval") or {}).get("kitchen_sink_or_noncommittal")),
            "has_explicit_comparison_count": sum(1 for row in group if (row.get("judgment_eval") or {}).get("has_explicit_comparison")),
            "has_risk_or_dependency_count": sum(1 for row in group if (row.get("judgment_eval") or {}).get("has_risk_or_dependency")),
            "has_actionable_decision_rule_count": sum(1 for row in group if (row.get("judgment_eval") or {}).get("has_actionable_decision_rule")),
        }

    return {
        "task_count": len(rows),
        "rubric_version": RUBRIC_VERSION,
        "mean_research_judgment_score": group_summary(rows)["mean_research_judgment_score"] if rows else 0.0,
        "dimension_summary": group_summary(rows)["dimension_summary"] if rows else {},
        "method_summary": {key: group_summary(group) for key, group in sorted(by_method.items())},
        "family_summary": {key: group_summary(group) for key, group in sorted(by_family.items())},
        "domain_summary": {key: group_summary(group) for key, group in sorted(by_domain.items())},
    }


def write_main_table_csv_v8(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method") or "")].append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Method", "ResearchJudgment"])
        writer.writeheader()
        for method, group in sorted(by_method.items()):
            summary = summarize_research_judgment_results_v8(group)
            writer.writerow({"Method": method, "ResearchJudgment": summary["mean_research_judgment_score"]})


def write_dimension_csv_v8(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_research_judgment_results_v8(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope_type", "scope_value", "component", "mean_score", "count"])
        writer.writeheader()

        def emit(scope_type: str, scope_value: str, group_summary: Dict[str, Any]) -> None:
            for component, score in (group_summary.get("dimension_summary") or {}).items():
                writer.writerow(
                    {
                        "scope_type": scope_type,
                        "scope_value": scope_value,
                        "component": component,
                        "mean_score": score,
                        "count": int(group_summary.get("count") or 0),
                    }
                )

        emit("overall", "all", {"count": summary.get("task_count"), "dimension_summary": summary.get("dimension_summary")})
        for family, group in sorted((summary.get("family_summary") or {}).items()):
            emit("family", family, group)
        for domain, group in sorted((summary.get("domain_summary") or {}).items()):
            emit("domain", domain, group)
