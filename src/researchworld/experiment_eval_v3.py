from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import PUBLIC_DOMAIN_TO_ID


TASK_FULFILLMENT_DIMENSIONS: List[Dict[str, Any]] = [
    {"name": "task_alignment", "weight": 0.30},
    {"name": "deliverable_completeness", "weight": 0.30},
    {"name": "conclusion_specificity", "weight": 0.20},
    {"name": "constraint_compliance", "weight": 0.20},
]


def infer_domain_id(row: Dict[str, Any]) -> str:
    domain_id = str(row.get("domain_id") or "").strip()
    if domain_id:
        return domain_id
    domain = str(row.get("domain") or "").strip()
    return PUBLIC_DOMAIN_TO_ID.get(domain, domain)


def _task_family_guidance(family: str) -> List[str]:
    if family == "direction_forecasting":
        return [
            "The answer should make a concrete directional call, not just summarize background.",
            "It should name the predicted direction, shift, or emerging line of work and justify why it is likely.",
        ]
    if family == "bottleneck_opportunity_discovery":
        return [
            "The answer should identify at least one concrete unresolved bottleneck.",
            "It should connect that bottleneck to a specific downstream opportunity that would open up if the bottleneck were addressed.",
        ]
    if family == "strategic_research_planning":
        return [
            "The answer should prioritize a small set of research directions or actions rather than provide an unranked list.",
            "It should justify why the proposed priorities deserve attention under the stated task setting.",
        ]
    return ["Judge whether the answer directly completes the requested task rather than loosely discussing the topic."]


def evaluate_task_fulfillment_judge(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    candidate_answer: str,
) -> Dict[str, Any]:
    family = str(public_task.get("family") or hidden_row.get("family") or "")
    dimensions = TASK_FULFILLMENT_DIMENSIONS
    dim_names = [str(item["name"]) for item in dimensions]
    guidance = _task_family_guidance(family)
    prompt = f"""# Role
You are a Task Fulfillment Auditor for a research benchmark. Your job is to judge whether the candidate answer actually completes the public task.

# Core Principle
- Evaluate task completion quality, not factual correctness.
- You are NOT given any hidden reference answer and must NOT infer one.
- Judge only from the public task definition, its stated deliverables, and the candidate answer itself.

# What to Ignore
- Do NOT score based on agreement with any presumed ground truth.
- Do NOT reward verbosity, polished prose, or generic academic tone by themselves.
- Do NOT check citation accuracy or external factual correctness. That is handled by another module.

# Input Data
- Public Task Definition: {json.dumps(public_task, ensure_ascii=False, indent=2)}
- Task Family: {family}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}
- Family-Specific Fulfillment Guidance: {json.dumps(guidance, ensure_ascii=False)}

# Candidate Answer under Review
{candidate_answer}

# Rubric
1. task_alignment: Does the answer directly address the asked question, rather than drifting into nearby background or generic discussion?
2. deliverable_completeness: Does the answer cover the requested deliverables in the public task and deliverable spec?
3. conclusion_specificity: Does the answer commit to concrete conclusions, priorities, bottlenecks, opportunities, or forecasts, instead of staying vague?
4. constraint_compliance: Does the answer respect explicit constraints in the task, such as time framing, requested output style, or the need to connect multiple required elements?

# Scoring Guidance
- 0.9-1.0: Fully completes the task with clear and concrete deliverables.
- 0.7-0.8: Substantially fulfills the task, with only minor missing pieces.
- 0.4-0.6: Partially fulfills the task, but misses important requested elements or stays too generic.
- 0.1-0.3: Weak fulfillment; only loosely related to the task or missing core deliverables.
- 0.0: Off-task or effectively non-responsive.

# Instructions
- Score each dimension from 0.0 to 1.0.
- Let strengths and weaknesses focus on task completion behavior, not fact checking.
- Penalize answers that sound intelligent but do not actually complete the requested deliverable.
- If the answer is concise but fully responsive, score it well.

# Output (Strict JSON)
{{
  "dimension_scores": {{
    "dimension_name": 0.0
  }},
  "task_fulfillment_score": 0.0,
  "strengths": [
    "Specific examples of strong task completion"
  ],
  "weaknesses": [
    "Specific missing deliverables or constraint violations"
  ]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict research benchmark task-fulfillment judge. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=900,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, task_fulfillment_score, strengths, weaknesses.",
    )
    raw_scores = obj.get("dimension_scores") or {}
    dim_scores = {name: float(raw_scores.get(name) or 0.0) for name in dim_names}
    if dim_scores:
        weighted = 0.0
        total_w = 0.0
        for item in dimensions:
            name = str(item.get("name") or "")
            weight = float(item.get("weight") or 0.0)
            total_w += weight
            weighted += weight * float(dim_scores.get(name) or 0.0)
        overall = weighted / total_w if total_w else sum(dim_scores.values()) / max(1, len(dim_scores))
    else:
        overall = float(obj.get("task_fulfillment_score") or 0.0)
    return {
        "task_fulfillment_score": round(float(obj.get("task_fulfillment_score") or overall), 4),
        "rubric_scores": {key: round(float(value), 4) for key, value in dim_scores.items()},
        "strengths": [str(x) for x in (obj.get("strengths") or []) if str(x).strip()],
        "weaknesses": [str(x) for x in (obj.get("weaknesses") or []) if str(x).strip()],
    }


def evaluate_insight_judge(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    candidate_answer: str,
) -> Dict[str, Any]:
    dimensions = list(((hidden_row.get("judge_profile") or {}).get("dimensions") or []))
    dim_names = [str(x.get("name") or "") for x in dimensions if str(x.get("name") or "").strip()]
    prompt = f"""# Role
You are an Advanced Research Auditor and Strategy Consultant. Your mission is to evaluate the "Intellectual Grade" of a candidate's research output without being biased by pre-defined answer keys.

# Evaluation Philosophy: Independent Quality Assessment
- Objective: Evaluate the latent quality of the reasoning. Does the response exhibit "Senior Researcher" level synthesis, or is it closer to "Junior Assistant" level data-dumping?
- No Ground Truth: You are intentionally not provided with a reference answer. Do NOT infer or imagine one.
- Judge the answer only by its internal logical quality, strategic usefulness, and alignment with the public task.

# Hard Constraints
- DO NOT evaluate factual correctness or data accuracy. Those are handled separately by another module.
- DO NOT reward answers merely for being long, polished, or terminology-heavy.
- DO NOT compare the answer against any hidden reference framing.

# Input Data
- Public Task Definition: {json.dumps(public_task, ensure_ascii=False, indent=2)}
- Task Family: {hidden_row.get('family')}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}

# Candidate Answer under Review
{candidate_answer}

# Evaluation Pillars (Research Signal)
1. Analytical Depth: Does the answer move beyond "what" into "why" and "how"? Does it identify mechanisms, causal drivers, bottlenecks, or trade-offs?
2. Framework Sophistication: Does the answer organize ideas into a meaningful structure (e.g., layered bottlenecks, ranked priorities, strategic decomposition, competing hypotheses), instead of presenting a flat list?
3. Temporal Intelligence: Does the answer cleanly separate historical signals, current status, and future trajectories? Are extrapolations disciplined rather than hand-wavy?
4. Task-Family Fit: Does the answer truly satisfy the demands of a "{hidden_row.get('family')}" task? For example, forecasting should make a concrete trajectory call; planning should prioritize and justify; bottleneck analysis should connect obstacles to actionable opportunities.

# Scoring Guidance
- 0.9-1.0: Publishable research judgment. Highly structured, strategically sharp, non-obvious, and well reasoned.
- 0.7-0.8: Strong senior-level analysis. Solid structure and clear insight, with only minor gaps.
- 0.4-0.6: Competent but limited. Some useful reasoning, but too generic, shallow, or weakly structured.
- 0.1-0.3: Poor analytical quality. Mostly surface-level statements, weak framework, weak task fit.
- 0.0: Off-task, logically incoherent, or almost entirely uninformative.

# Reward
- Nuance
- Identification of trade-offs and research friction
- Synthesis of disparate signals into a coherent strategic view
- Concrete, non-obvious, decision-useful conclusions

# Penalize
- Generic or safe statements
- Flat lists without hierarchy
- Superficial summaries that restate the task
- Temporal confusion
- Strategic vagueness
- Domain drift or failure to match task family

# Instructions
- Evaluate each provided dimension on a scale from 0.0 to 1.0.
- Let "strengths" and "weaknesses" focus on reasoning quality, structure, and strategic usefulness, not verbosity.
- If the answer is articulate but analytically shallow, score it low.
- If the answer is unconventional but logically strong and strategically useful, score it high.

# Output (Strict JSON)
{{
  "dimension_scores": {{
    "dimension_name": 0.0
  }},
  "insight_judge_score": 0.0,
  "strengths": [
    "Specific examples of high-order reasoning, structural excellence, or strategic clarity"
  ],
  "weaknesses": [
    "Specific instances where the analysis is surface-level, logically thin, or structurally disorganized"
  ]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict benchmark-aware insight judge. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, insight_judge_score, strengths, weaknesses.",
    )
    raw_scores = obj.get("dimension_scores") or {}
    dim_scores = {name: float(raw_scores.get(name) or 0.0) for name in dim_names}
    if dim_scores:
        weighted = 0.0
        total_w = 0.0
        for item in dimensions:
            name = str(item.get("name") or "")
            weight = float(item.get("weight") or 0.0)
            total_w += weight
            weighted += weight * float(dim_scores.get(name) or 0.0)
        overall = weighted / total_w if total_w else sum(dim_scores.values()) / max(1, len(dim_scores))
    else:
        overall = float(obj.get("insight_judge_score") or 0.0)
    return {
        "insight_judge_score": round(float(obj.get("insight_judge_score") or overall), 4),
        "rubric_scores": {key: round(float(value), 4) for key, value in dim_scores.items()},
        "strengths": [str(x) for x in (obj.get("strengths") or []) if str(x).strip()],
        "weaknesses": [str(x) for x in (obj.get("weaknesses") or []) if str(x).strip()],
    }


def _scope_precision(fact_eval: Dict[str, Any], scope: str) -> float:
    claims = [row for row in (fact_eval.get("claims") or []) if str(row.get("matched_time_scope") or "") == scope]
    if not claims:
        return 0.0
    total = sum(float(row.get("weight") or 0.0) for row in claims)
    supported = sum(
        float(row.get("weight") or 0.0)
        for row in claims
        if ((row.get("verdict") or {}).get("label") == "supported") and ((row.get("verdict") or {}).get("temporal_consistency") != "inconsistent")
    )
    return round(supported / total, 4) if total else 0.0


def build_experiment_result_row(
    *,
    run_id: str,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    result_row: Dict[str, Any],
    fact_eval: Dict[str, Any],
    task_fulfillment_eval: Dict[str, Any],
    judge_eval: Dict[str, Any],
) -> Dict[str, Any]:
    fact_precision_score = float(fact_eval.get("benchmark_factscore") or 0.0)
    task_fulfillment_score = float(task_fulfillment_eval.get("task_fulfillment_score") or 0.0)
    insight_judge_score = float(judge_eval.get("insight_judge_score") or 0.0)
    unsupported = sum(1 for row in (fact_eval.get("claims") or []) if ((row.get("verdict") or {}).get("label") == "unsupported"))
    claim_count = int(fact_eval.get("claim_count") or 0)
    leakage = any((row.get("verdict") or {}).get("temporal_consistency") == "inconsistent" for row in (fact_eval.get("claims") or []))
    trace = result_row.get("trace") or {}
    diagnostics = trace.get("diagnostics") or {}
    evidence = trace.get("evidence") or {}
    retrieval_diag = {
        "retrieval_mode": trace.get("retrieval_mode") or ((result_row.get("evidence") or {}).get("retrieval_mode") if isinstance(result_row.get("evidence"), dict) else ""),
        "retrieved_doc_count": len((evidence.get("papers") or [])) + len((evidence.get("structures") or [])) + len((evidence.get("pageindex") or [])) + len((evidence.get("fulltext") or [])),
        "unique_paper_count": len({str(x.get("paper_id") or "") for x in (evidence.get("papers") or []) if str(x.get("paper_id") or "").strip()}),
        "pageindex_hit_count": len(evidence.get("pageindex") or []),
        "fulltext_hit_count": len(evidence.get("fulltext") or []),
        "tool_call_count": int(diagnostics.get("tool_calls") or 0),
    }
    agent_diag = {
        "reflection_steps": int(diagnostics.get("reflection_steps") or 0),
        "memory_updates": int(diagnostics.get("memory_updates") or 0),
        "revision_rounds": int(diagnostics.get("revision_rounds") or 0),
        "answer_changed_after_revision": bool(diagnostics.get("answer_changed_after_revision") or False),
    }
    return {
        "run_id": run_id,
        "task_id": public_task.get("task_id"),
        "family": public_task.get("family"),
        "domain": infer_domain_id(result_row),
        "method": str(result_row.get("agent") or result_row.get("baseline") or "unknown"),
        "answer": str(result_row.get("answer") or ""),
        "metadata": {
            "task_title": public_task.get("title"),
            "time_cutoff": public_task.get("time_cutoff"),
        },
        "retrieval_diagnostics": retrieval_diag,
        "agent_diagnostics": agent_diag,
        "scores": {
            "fact_precision_score": round(fact_precision_score, 4),
            "task_fulfillment_score": round(task_fulfillment_score, 4),
            "insight_judge_score": round(insight_judge_score, 4),
        },
        "fact_eval": {
            "weighted_claim_precision": round(float(fact_eval.get("precision_score") or 0.0), 4),
            "weighted_claim_coverage": round(float(fact_eval.get("coverage_score") or 0.0), 4),
            "benchmark_factscore": round(float(fact_eval.get("benchmark_factscore") or 0.0), 4),
            "history_claim_precision": _scope_precision(fact_eval, "history"),
            "future_claim_precision": _scope_precision(fact_eval, "future"),
            "cross_temporal_claim_precision": _scope_precision(fact_eval, "cross_temporal"),
            "unsupported_claim_rate": round(unsupported / claim_count, 4) if claim_count else 0.0,
            "temporal_leakage_violation": leakage,
            "claims": fact_eval.get("claims") or [],
        },
        "task_fulfillment_eval": task_fulfillment_eval,
        "judge_eval": judge_eval,
    }


def summarize_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _mean(values: List[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    fact_scores = [float((row.get("scores") or {}).get("fact_precision_score") or 0.0) for row in rows]
    task_fulfillment_scores = [float((row.get("scores") or {}).get("task_fulfillment_score") or 0.0) for row in rows]
    judge_scores = [float((row.get("scores") or {}).get("insight_judge_score") or 0.0) for row in rows]
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("family") or "")].append(row)
        by_domain[str(row.get("domain") or "")].append(row)
    return {
        "task_count": len(rows),
        "mean_fact_precision_score": _mean(fact_scores),
        "mean_task_fulfillment_score": _mean(task_fulfillment_scores),
        "mean_insight_judge_score": _mean(judge_scores),
        "family_summary": {
            key: {
                "count": len(group),
                "mean_fact_precision_score": _mean([float((row.get("scores") or {}).get("fact_precision_score") or 0.0) for row in group]),
                "mean_task_fulfillment_score": _mean([float((row.get("scores") or {}).get("task_fulfillment_score") or 0.0) for row in group]),
                "mean_insight_judge_score": _mean([float((row.get("scores") or {}).get("insight_judge_score") or 0.0) for row in group]),
            }
            for key, group in sorted(by_family.items())
        },
        "domain_summary": {
            key: {
                "count": len(group),
                "mean_fact_precision_score": _mean([float((row.get("scores") or {}).get("fact_precision_score") or 0.0) for row in group]),
                "mean_task_fulfillment_score": _mean([float((row.get("scores") or {}).get("task_fulfillment_score") or 0.0) for row in group]),
                "mean_insight_judge_score": _mean([float((row.get("scores") or {}).get("insight_judge_score") or 0.0) for row in group]),
            }
            for key, group in sorted(by_domain.items())
        },
    }


def write_main_table_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method") or "")].append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Method", "FactScore", "TaskFulfillment", "InsightJudge"])
        writer.writeheader()
        for method, group in sorted(by_method.items()):
            summary = summarize_results(group)
            writer.writerow(
                {
                    "Method": method,
                    "FactScore": summary["mean_fact_precision_score"],
                    "TaskFulfillment": summary["mean_task_fulfillment_score"],
                    "InsightJudge": summary["mean_insight_judge_score"],
                }
            )
