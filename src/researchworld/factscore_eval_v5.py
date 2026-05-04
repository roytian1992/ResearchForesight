from __future__ import annotations

from typing import Any, Dict, List, Optional

from researchworld.factscore_eval_v3 import (
    FactScoreV3Config,
    _clamp01,
    _embedded_trace_evidence,
    extract_atomic_claims,
    match_answer_claim_to_gt,
    normalize_ws,
    verify_claim_v3,
)
from researchworld.llm import OpenAICompatChatClient


def retrieve_claim_evidence_v5(
    *,
    answer_claim: str,
    matched_gt_claim: Optional[Dict[str, Any]],
    temporal_policy: Dict[str, Any],
    gt_row: Dict[str, Any],
    cfg: FactScoreV3Config,
) -> List[Dict[str, Any]]:
    """Retrieve only evidence embedded in task_refined.jsonl.

    v5 intentionally does not open the offline KB. The evaluator can inspect the
    task JSON evaluation packet and the submitted answer, but it must not run its
    own retrieval over the benchmark corpus.
    """

    del answer_claim
    if not matched_gt_claim:
        return []
    evidence_ids = [
        str(eid).strip()
        for eid in ((matched_gt_claim or {}).get("reference_evidence_ids") or [])
        if str(eid).strip()
    ]
    if evidence_ids:
        direct_rows = _embedded_gold_set_evidence(gt_row, evidence_ids=evidence_ids, max_rows=cfg.max_evidence_rows)
        if direct_rows:
            return direct_rows
    scope = str((matched_gt_claim or {}).get("time_scope") or "cross_temporal")
    evidence_rows: List[Dict[str, Any]] = []
    if scope in {"history", "cross_temporal"}:
        evidence_rows.extend(
            _embedded_trace_evidence(
                gt_row,
                matched_gt_claim=matched_gt_claim,
                temporal_policy=temporal_policy,
                source_name="history",
                max_rows=cfg.max_evidence_rows,
            )
        )
    if scope in {"future", "cross_temporal"}:
        evidence_rows.extend(
            _embedded_trace_evidence(
                gt_row,
                matched_gt_claim=matched_gt_claim,
                temporal_policy=temporal_policy,
                source_name="future",
                max_rows=cfg.max_evidence_rows,
            )
        )
    return evidence_rows[: cfg.max_evidence_rows]


def _embedded_gold_set_evidence(
    gt_row: Dict[str, Any],
    *,
    evidence_ids: List[str],
    max_rows: int,
) -> List[Dict[str, Any]]:
    wanted = {str(eid).strip() for eid in evidence_ids if str(eid).strip()}
    if not wanted:
        return []
    trace = gt_row.get("trace") or {}
    rows: List[Dict[str, Any]] = []
    for prefix, source_name, trace_key in (
        ("HE", "history", "history_evidence"),
        ("FE", "future", "future_evidence"),
    ):
        for idx, item in enumerate(trace.get(trace_key) or [], start=1):
            evidence_id = f"{prefix}{idx}"
            if evidence_id not in wanted or not isinstance(item, dict):
                continue
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "evidence_source": f"gold_set_{source_name}",
                    "paper_id": item.get("paper_id"),
                    "paper_title": item.get("title"),
                    "published_date": item.get("published_date"),
                    "snippet": normalize_ws(item.get("why_it_matters") or item.get("title") or ""),
                    "scores": {"combined_score": 1.0},
                }
            )
    return rows[:max_rows]


def _expanded_claim_bank(gt_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Materialize an evidence-bound claim bank from gold_sets.

    The original claim bank remains the strict core. Variant claims add
    benchmark-valid paraphrases or nearby evidence-grounded facts, with evidence
    tied back to trace rows embedded in task_refined.jsonl.
    """

    claims: List[Dict[str, Any]] = [dict(claim) for claim in (gt_row.get("claim_bank") or []) if isinstance(claim, dict)]
    gold_sets = gt_row.get("gold_sets") or {}
    grounded = gold_sets.get("evidence_grounded_claims") or {}
    trace_rows = {
        row["evidence_id"]: row
        for row in _embedded_gold_set_evidence(
            gt_row,
            evidence_ids=[
                f"{prefix}{idx}"
                for prefix, trace_key in (("HE", "history_evidence"), ("FE", "future_evidence"))
                for idx, _ in enumerate((gt_row.get("trace") or {}).get(trace_key) or [], start=1)
            ],
            max_rows=10_000,
        )
    }
    seen_ids = {str(claim.get("claim_id") or "") for claim in claims}
    for idx, item in enumerate(grounded.get("variant_claims") or [], start=1):
        if not isinstance(item, dict) or not normalize_ws(item.get("text")):
            continue
        raw_id = str(item.get("claim_id") or f"variant_{idx}").strip()
        claim_id = f"goldset_{raw_id}"
        if claim_id in seen_ids:
            continue
        evidence_ids = [
            str(eid).strip()
            for eid in (item.get("evidence_ids") or [])
            if str(eid).strip() in trace_rows
        ]
        reference_paper_ids = [
            str((trace_rows.get(eid) or {}).get("paper_id") or "").strip()
            for eid in evidence_ids
            if str((trace_rows.get(eid) or {}).get("paper_id") or "").strip()
        ]
        paraphrases = [normalize_ws(x) for x in (item.get("acceptable_paraphrases") or []) if normalize_ws(x)]
        claims.append(
            {
                "claim_id": claim_id,
                "text": normalize_ws(item.get("text")),
                "claim_type": item.get("claim_type") or "expanded_gold_variant",
                "time_scope": item.get("time_scope") or "cross_temporal",
                "importance": float(item.get("importance") or 0.7),
                "reference_paper_ids": reference_paper_ids,
                "reference_evidence_ids": evidence_ids,
                "aliases": {"acceptable_paraphrases": paraphrases},
                "gold_set_source": "evidence_grounded_claims.variant_claims",
            }
        )
        seen_ids.add(claim_id)
    return claims


def evaluate_answer_factscore_v5(
    *,
    judge_client: OpenAICompatChatClient,
    result_row: Dict[str, Any],
    gt_row: Dict[str, Any],
    cfg: Optional[FactScoreV3Config] = None,
    claim_bank_override: Optional[List[Dict[str, Any]]] = None,
    evaluator_scope: str = "task_json_only",
) -> Dict[str, Any]:
    cfg = cfg or FactScoreV3Config()
    answer = normalize_ws(result_row.get("answer"))
    if not answer:
        return {
            "claim_count": 0,
            "supported_count": 0,
            "precision_score": 0.0,
            "coverage_score": 0.0,
            "benchmark_factscore": 0.0,
            "claims": [],
            "evaluator_scope": evaluator_scope,
        }

    family = str(gt_row.get("family") or result_row.get("family") or "")
    answer_claims = extract_atomic_claims(
        judge_client,
        answer=answer,
        max_claims=cfg.max_claims,
        family=family,
    )
    gt_claim_bank = list(claim_bank_override) if claim_bank_override is not None else list(gt_row.get("claim_bank") or [])
    temporal_policy = gt_row.get("temporal_policy") or {}

    claim_rows = []
    total_weight = 0.0
    supported_weight = 0.0
    covered_gt_claims = set()
    gt_total_weight = sum(float(claim.get("importance") or 0.0) for claim in gt_claim_bank)

    for claim in answer_claims:
        matched_gt_claim, match_score = match_answer_claim_to_gt(claim, gt_claim_bank, cfg.gt_match_threshold)
        weight = float((matched_gt_claim or {}).get("importance") or cfg.unmatched_claim_weight)
        total_weight += weight
        evidence_rows = retrieve_claim_evidence_v5(
            answer_claim=claim,
            matched_gt_claim=matched_gt_claim,
            temporal_policy=temporal_policy,
            gt_row=gt_row,
            cfg=cfg,
        )
        verdict = verify_claim_v3(
            judge_client,
            claim=claim,
            matched_gt_claim=matched_gt_claim,
            evidence_rows=evidence_rows,
        )
        if verdict["label"] == "supported" and verdict["temporal_consistency"] != "inconsistent":
            supported_weight += weight
            if matched_gt_claim:
                covered_gt_claims.add(str(matched_gt_claim.get("claim_id")))
        claim_rows.append(
            {
                "claim": claim,
                "matched_gt_claim_id": (matched_gt_claim or {}).get("claim_id"),
                "matched_gt_claim_type": (matched_gt_claim or {}).get("claim_type"),
                "matched_time_scope": (matched_gt_claim or {}).get("time_scope"),
                "match_score": round(match_score, 4),
                "weight": weight,
                "verdict": verdict,
                "evidence": evidence_rows,
            }
        )

    covered_weight = sum(
        float(claim.get("importance") or 0.0)
        for claim in gt_claim_bank
        if str(claim.get("claim_id")) in covered_gt_claims
    )
    precision_score = round(_clamp01(supported_weight / total_weight), 4) if total_weight else 0.0
    coverage_score = round(_clamp01(covered_weight / gt_total_weight), 4) if gt_total_weight else 0.0
    benchmark_factscore = round(
        _clamp01(cfg.precision_weight * precision_score + cfg.coverage_weight * coverage_score),
        4,
    )
    return {
        "claim_count": len(claim_rows),
        "supported_count": sum(
            1
            for row in claim_rows
            if row["verdict"]["label"] == "supported"
            and row["verdict"]["temporal_consistency"] != "inconsistent"
        ),
        "precision_score": precision_score,
        "coverage_score": coverage_score,
        "benchmark_factscore": benchmark_factscore,
        "weighted_supported": round(supported_weight, 4),
        "weighted_total": round(total_weight, 4),
        "weighted_gt_covered": round(covered_weight, 4),
        "weighted_gt_total": round(gt_total_weight, 4),
        "claims": claim_rows,
        "claim_bank_count": len(gt_claim_bank),
        "expanded_claim_bank_count": max(0, len(gt_claim_bank) - len(gt_row.get("claim_bank") or [])),
        "evaluator_scope": evaluator_scope,
    }


def evaluate_answer_expanded_factscore_v5(
    *,
    judge_client: OpenAICompatChatClient,
    result_row: Dict[str, Any],
    gt_row: Dict[str, Any],
    cfg: Optional[FactScoreV3Config] = None,
) -> Dict[str, Any]:
    claim_bank = _expanded_claim_bank(gt_row)
    detail = evaluate_answer_factscore_v5(
        judge_client=judge_client,
        result_row=result_row,
        gt_row=gt_row,
        cfg=cfg,
        claim_bank_override=claim_bank,
        evaluator_scope="task_json_gold_sets_expanded_claim_bank",
    )
    detail["claim_bank_mode"] = "strict_core_plus_gold_sets_variant_claims"
    return detail
