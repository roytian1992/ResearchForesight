from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from researchworld.future_alignment_eval_v3_1 import (
    FutureAlignmentV3_1Config,
    _apply_public_leakage_guard,
    _clamp01,
    _embedded_future_evidence,
    _float_or_default,
    _future_alignment_json,
    _future_unit_public_leakage,
    _label_score,
)
from researchworld.llm import OpenAICompatChatClient, OpenAICompatEmbeddingClient, complete_json_object
from researchworld.offline_kb import normalize_ws


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _rescale_cosine(score: float) -> float:
    # Embedding cosine can be slightly negative. FAS is reported on [0, 1].
    return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))


def _future_signal_text(unit: Dict[str, Any], evidence_rows: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    if normalize_ws(unit.get("text")):
        parts.append(f"Future unit: {normalize_ws(unit.get('text'))}")
    aliases = [normalize_ws(x) for x in (unit.get("aliases") or []) if normalize_ws(x)]
    if aliases:
        parts.append("Aliases: " + "; ".join(aliases[:6]))
    titles = [normalize_ws(x) for x in (unit.get("future_paper_titles") or []) if normalize_ws(x)]
    if titles:
        parts.append("Reference future papers: " + "; ".join(titles[:6]))
    snippets = []
    for row in evidence_rows[:4]:
        title = normalize_ws(row.get("paper_title"))
        snippet = normalize_ws(row.get("snippet"))
        if title or snippet:
            snippets.append(f"{title}: {snippet}".strip(": "))
    if snippets:
        parts.append("Future evidence summaries: " + " ".join(snippets))
    return normalize_ws(" ".join(parts))


def _reference_answer_text(hidden_row: Dict[str, Any]) -> str:
    parts: List[str] = []
    gold = normalize_ws(hidden_row.get("gold_answer"))
    if gold:
        parts.append(f"Reference answer: {gold}")
    expected = [normalize_ws(x) for x in (hidden_row.get("expected_answer_points") or []) if normalize_ws(x)]
    if expected:
        parts.append("Expected answer points: " + "; ".join(expected[:6]))
    return normalize_ws(" ".join(parts))


def _embedding_label(score: float) -> str:
    if score >= 0.78:
        return "aligned"
    if score >= 0.62:
        return "partial"
    return "not_aligned"


def _embedding_specificity(score: float) -> float:
    if score >= 0.78:
        return max(0.72, min(1.0, score))
    if score >= 0.62:
        return max(0.42, min(0.78, score))
    return max(0.0, min(0.45, score))


_SCOPE_CAPS = {
    "equivalent": 1.0,
    "narrower_valid": 0.82,
    "broader_valid": 0.68,
    "adjacent": 0.45,
    "drift": 0.12,
}


def _scope_label(score: float) -> str:
    if score >= 0.78:
        return "aligned"
    if score >= 0.42:
        return "partial"
    return "not_aligned"


def _scope_relation_json(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    family: str,
    candidate_answer: str,
    unit: Dict[str, Any],
    future_signal_text: str,
    embedding_fas: float,
) -> Dict[str, Any]:
    prompt = f"""# Role
You are a scope-calibration judge for a time-sliced Future Alignment Score.

# Objective
Classify whether the candidate answer's main future-facing point matches the hidden future unit at the right granularity.

# Important
This is NOT a factuality audit and you must not use external knowledge. Use only the public task, candidate answer, hidden future unit, and stored future signal below.

# Public Task
{public_task}

# Task Family
{family}

# Candidate Answer
{candidate_answer}

# Hidden Future Unit
{unit}

# Stored Future Signal
{future_signal_text}

# Embedding FAS
{embedding_fas:.4f}

# Scope Relations
- equivalent: same technical direction/mechanism/priority as the hidden future unit, allowing wording differences.
- narrower_valid: answer names a more specific subcase, implementation, benchmark, or mechanism that is a plausible child of the hidden future unit.
- broader_valid: answer names a parent umbrella area that contains the hidden future unit but misses the discriminative mechanism, constraint, or priority.
- adjacent: answer is in the same topic neighborhood but focuses on a different mechanism, priority, or venue rationale.
- drift: answer is mostly unrelated or selects a conflicting direction.

# Operation Fit
Also judge whether the answer performs the required operation for this family:
- bottleneck: names a bottleneck/opportunity pair relevant to the future unit.
- forecasting: forecasts/selects the future direction and trajectory.
- strategic planning: assesses/prioritizes the relevant direction rather than merely mentioning it.
- venue positioning: ranks/selects the relevant direction with the requested venue framing.

# Output JSON
{{
  "scope_relation": "equivalent | narrower_valid | broader_valid | adjacent | drift",
  "operation_fit": true,
  "rationale": "Briefly explain the granularity relation and whether this should receive full or partial FAS credit."
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict scope-calibration judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=500,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Return exactly one valid JSON object with keys scope_relation, operation_fit, rationale.",
    )
    relation = str(obj.get("scope_relation") or "adjacent").strip().lower()
    if relation not in _SCOPE_CAPS:
        relation = "adjacent"
    return {
        "scope_relation": relation,
        "operation_fit": bool(obj.get("operation_fit")),
        "rationale": str(obj.get("rationale") or "").strip(),
    }


def _apply_scope_calibration(
    *,
    embedding_fas: float,
    scope_obj: Dict[str, Any],
) -> tuple[str, float, float, Dict[str, Any]]:
    relation = str(scope_obj.get("scope_relation") or "adjacent")
    cap = float(_SCOPE_CAPS.get(relation, 0.45))
    if not bool(scope_obj.get("operation_fit")):
        cap = min(cap, 0.50)
    calibrated = round(max(0.0, min(float(embedding_fas), cap)), 4)
    label = _scope_label(calibrated)
    specificity = calibrated
    details = {
        "embedding_fas": round(float(embedding_fas), 4),
        "scope_relation": relation,
        "operation_fit": bool(scope_obj.get("operation_fit")),
        "scope_cap": round(cap, 4),
        "calibrated_fas": calibrated,
        "rationale": str(scope_obj.get("rationale") or "").strip(),
    }
    return label, specificity, calibrated, details


def _fas_label(score: float) -> str:
    if score >= 0.78:
        return "aligned"
    if score >= 0.62:
        return "partial"
    return "not_aligned"


def evaluate_future_alignment_v5(
    *,
    judge_client: OpenAICompatChatClient,
    embedding_client: Optional[OpenAICompatEmbeddingClient] = None,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
    hidden_row: Dict[str, Any],
    cfg: Optional[FutureAlignmentV3_1Config] = None,
) -> Dict[str, Any]:
    """Evaluate future alignment using only task_refined.jsonl evidence.

    v5 does not accept a future KB. Future signals are constructed from
    hidden_row["future_alignment_targets"] plus hidden_row["trace"]["future_evidence"].
    The primary score follows the paper's benchmark adaptation of FAS: a
    unit-level semantic audit against hidden future units, aggregated as
    0.7 * WUA + 0.2 * coverage + 0.1 * specificity. Embedding similarity to
    gold_answer/reference_answer is retained only as a diagnostic.
    """

    cfg = cfg or FutureAlignmentV3_1Config()
    targets = hidden_row.get("future_alignment_targets") or {}
    units = list((targets.get("alignment_units") or []))[: cfg.max_units]
    if not units:
        return {
            "future_alignment_score": 0.0,
            "weighted_unit_alignment": 0.0,
            "alignment_coverage": 0.0,
            "mean_specificity": 0.0,
            "units": [],
            "evaluator_scope": "task_json_only",
        }

    family = str(hidden_row.get("family") or public_task.get("family") or "")
    answer = normalize_ws(result_row.get("answer"))
    temporal_policy = hidden_row.get("temporal_policy") or {}
    future_start = str(temporal_policy.get("future_start") or "")
    cutoff = str(temporal_policy.get("future_end") or "")
    unit_rows: List[Dict[str, Any]] = []
    total_importance = 0.0
    weighted_alignment = 0.0
    weighted_raw_embedding_fas = 0.0
    specificity_values: List[float] = []
    covered = 0
    used_embedding_diagnostics = embedding_client is not None
    reference_text = _reference_answer_text(hidden_row)
    reference_similarity: Optional[float] = None
    reference_cosine: Optional[float] = None
    if embedding_client is not None and answer and reference_text:
        try:
            answer_vec, reference_vec = embedding_client.embed([answer, reference_text], transport_retries=2)
            reference_cosine = _cosine_similarity(answer_vec, reference_vec)
            reference_similarity = round(_rescale_cosine(reference_cosine), 4)
        except Exception:
            reference_similarity = None
            reference_cosine = None

    for unit in units:
        evidence_rows = _embedded_future_evidence(
            hidden_row,
            unit,
            max_rows=cfg.max_evidence_rows,
            future_start=future_start,
            future_end=cutoff,
        )[: cfg.max_evidence_rows]
        public_leakage = _future_unit_public_leakage(public_task, unit)
        signal_text = _future_signal_text(unit, evidence_rows)
        label = "not_aligned"
        specificity = 0.0
        raw_fas_for_unit = 0.0
        leakage_guard = {"applied": False, "reason": ""}
        obj: Dict[str, Any] = {}
        scope_calibration: Dict[str, Any] = {
            "embedding_fas": None,
            "scope_relation": "not_run",
            "operation_fit": None,
            "scope_cap": None,
            "calibrated_fas": None,
            "rationale": "",
        }
        evaluator_mode = "llm_unit_audit_task_json_only"
        if embedding_client is not None and answer and signal_text:
            try:
                answer_vec, signal_vec = embedding_client.embed([answer, signal_text], transport_retries=2)
                raw_cosine = _cosine_similarity(answer_vec, signal_vec)
                raw_fas_for_unit = round(_rescale_cosine(raw_cosine), 4)
                scope_calibration = {
                    "embedding_fas": raw_fas_for_unit,
                    "scope_relation": "diagnostic_only",
                    "operation_fit": None,
                    "scope_cap": None,
                    "calibrated_fas": None,
                    "rationale": "Embedding FAS to the stored future signal is recorded as a diagnostic only.",
                }
            except Exception as exc:
                embedding_client = None
                used_embedding_diagnostics = False
                raw_fas_for_unit = 0.0
                scope_calibration = {
                    "embedding_fas": 0.0,
                    "scope_relation": "diagnostic_unavailable",
                    "operation_fit": None,
                    "scope_cap": 0.0,
                    "calibrated_fas": None,
                    "rationale": str(exc),
                }

        if not evidence_rows:
            label = "not_aligned"
            specificity = 0.0
            leakage_guard = {"applied": False, "reason": "no_future_evidence"}
            obj = {
                "rationale": "No future evidence rows were available in task_refined.jsonl for this alignment unit.",
                "cited_evidence_ids": [],
            }
            evaluator_mode = "llm_unit_audit_task_json_only"
        else:
            obj = _future_alignment_json(
                judge_client,
                public_task=public_task,
                family=family,
                candidate_answer=answer,
                unit=unit,
                evidence_rows=evidence_rows,
                public_leakage=public_leakage,
            )
            label = str(obj.get("label") or "not_aligned").strip().lower()
            if label not in {"aligned", "partial", "not_aligned"}:
                label = "not_aligned"
            specificity = _clamp01(obj.get("specificity"))
            label, specificity, guarded_rationale, leakage_guard = _apply_public_leakage_guard(
                label=label,
                specificity=specificity,
                rationale=str(obj.get("rationale") or "").strip(),
                public_leakage=public_leakage,
                candidate_answer=answer,
                family=family,
                unit=unit,
            )
            obj["rationale"] = guarded_rationale
            evaluator_mode = "llm_unit_audit_task_json_only"
            if not scope_calibration.get("rationale"):
                scope_calibration = {
                    "embedding_fas": None,
                    "scope_relation": "diagnostic_not_run",
                    "operation_fit": None,
                    "scope_cap": None,
                    "calibrated_fas": None,
                    "rationale": "Embedding diagnostic unavailable; used JSON-only LLM semantic judge.",
                }

        valid_evidence_ids = {str(row.get("evidence_id") or "") for row in evidence_rows}
        importance = max(0.0, _float_or_default(unit.get("importance"), 1.0))
        base = _label_score(label)
        unit_score = round(base * (0.5 + 0.5 * specificity), 4)
        total_importance += importance
        weighted_alignment += importance * unit_score
        if raw_fas_for_unit:
            weighted_raw_embedding_fas += importance * float(raw_fas_for_unit)
        specificity_values.append(specificity)
        if label in {"aligned", "partial"}:
            covered += 1
        unit_rows.append(
            {
                "unit_id": unit.get("unit_id"),
                "unit_type": unit.get("unit_type"),
                "text": unit.get("text"),
                "importance": importance,
                "label": label,
                "specificity": round(specificity, 4),
                "unit_score": unit_score,
                "embedding_fas": round(float(raw_fas_for_unit), 4) if raw_fas_for_unit else None,
                "rationale": str(obj.get("rationale") or "").strip(),
                "cited_evidence_ids": [
                    str(x)
                    for x in (obj.get("cited_evidence_ids") or [])
                    if str(x).strip() and str(x).strip() in valid_evidence_ids
                ],
                "public_leakage": public_leakage,
                "leakage_guard": leakage_guard,
                "future_signal_text": signal_text,
                "evaluator_mode": evaluator_mode,
                "scope_calibration": scope_calibration,
                "evidence": evidence_rows,
            }
        )

    weighted_unit_alignment = round(weighted_alignment / total_importance, 4) if total_importance else 0.0
    raw_weighted_embedding_fas = round(weighted_raw_embedding_fas / total_importance, 4) if total_importance and used_embedding_diagnostics else None
    alignment_coverage = round(covered / len(units), 4) if units else 0.0
    mean_specificity = round(sum(specificity_values) / len(specificity_values), 4) if specificity_values else 0.0
    future_alignment_score = round(
        0.70 * weighted_unit_alignment + 0.20 * alignment_coverage + 0.10 * mean_specificity,
        4,
    )
    return {
        "future_alignment_score": future_alignment_score,
        "canonical_fas_score": future_alignment_score,
        "reference_answer_similarity": reference_similarity,
        "reference_answer_raw_cosine": round(reference_cosine, 4) if reference_cosine is not None else None,
        "reference_answer_text": reference_text,
        "raw_weighted_embedding_fas": raw_weighted_embedding_fas,
        "scope_calibrated_future_signal_fas": None,
        "weighted_unit_alignment": weighted_unit_alignment,
        "alignment_coverage": alignment_coverage,
        "mean_specificity": mean_specificity,
        "unit_count": len(units),
        "units": unit_rows,
        "evaluator_scope": "task_json_only",
        "fas_mode": "official_inspired_unit_semantic_audit_with_reference_diagnostics",
    }


def _gold_set_evidence_rows(
    hidden_row: Dict[str, Any],
    *,
    evidence_ids: List[str],
    max_rows: int,
) -> List[Dict[str, Any]]:
    wanted = {str(eid).strip() for eid in evidence_ids if str(eid).strip()}
    if not wanted:
        return []
    trace = hidden_row.get("trace") or {}
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


def _score_gold_set_future_item(
    *,
    judge_client: OpenAICompatChatClient,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    answer: str,
    item: Dict[str, Any],
    item_kind: str,
    cfg: FutureAlignmentV3_1Config,
) -> Dict[str, Any]:
    evidence_rows = _gold_set_evidence_rows(
        hidden_row,
        evidence_ids=[str(x) for x in (item.get("evidence_ids") or [])],
        max_rows=cfg.max_evidence_rows,
    )
    unit = {
        "unit_id": item.get("neighbor_id") or item.get("confusion_id") or item.get("unit_id") or item_kind,
        "unit_type": item_kind,
        "text": item.get("text"),
        "importance": float(item.get("importance") or 1.0),
    }
    if not evidence_rows:
        return {
            "item_id": unit["unit_id"],
            "text": unit["text"],
            "label": "not_aligned",
            "specificity": 0.0,
            "credit_cap": _clamp01(item.get("credit_cap") or 0.0),
            "item_score": 0.0,
            "rationale": "No usable gold_sets evidence rows were available for this item.",
            "evidence": [],
            "item_kind": item_kind,
        }
    public_leakage = _future_unit_public_leakage(public_task, unit)
    obj = _future_alignment_json(
        judge_client,
        public_task=public_task,
        family=str(hidden_row.get("family") or public_task.get("family") or ""),
        candidate_answer=answer,
        unit=unit,
        evidence_rows=evidence_rows,
        public_leakage=public_leakage,
    )
    label = str(obj.get("label") or "not_aligned").strip().lower()
    if label not in {"aligned", "partial", "not_aligned"}:
        label = "not_aligned"
    specificity = _clamp01(obj.get("specificity"))
    label, specificity, guarded_rationale, leakage_guard = _apply_public_leakage_guard(
        label=label,
        specificity=specificity,
        rationale=str(obj.get("rationale") or "").strip(),
        public_leakage=public_leakage,
        candidate_answer=answer,
        family=str(hidden_row.get("family") or public_task.get("family") or ""),
        unit=unit,
    )
    cap = _clamp01(item.get("credit_cap") if item.get("credit_cap") is not None else 1.0)
    item_score = round(_label_score(label) * (0.5 + 0.5 * specificity) * cap, 4)
    valid_evidence_ids = {str(row.get("evidence_id") or "") for row in evidence_rows}
    return {
        "item_id": unit["unit_id"],
        "text": unit["text"],
        "relation": item.get("relation"),
        "label": label,
        "specificity": round(specificity, 4),
        "credit_cap": cap,
        "item_score": item_score,
        "rationale": guarded_rationale,
        "cited_evidence_ids": [
            str(x)
            for x in (obj.get("cited_evidence_ids") or [])
            if str(x).strip() and str(x).strip() in valid_evidence_ids
        ],
        "public_leakage": public_leakage,
        "leakage_guard": leakage_guard,
        "evidence": evidence_rows,
        "item_kind": item_kind,
    }


def _select_gold_set_items_for_llm(
    *,
    embedding_client: Optional[OpenAICompatEmbeddingClient],
    answer: str,
    items: List[Dict[str, Any]],
    limit: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if limit <= 0 or not items:
        return [], []
    if embedding_client is None or not answer:
        selected = [dict(item, embedding_fas=None) for item in items[:limit]]
        return selected, selected
    scored: List[Dict[str, Any]] = []
    try:
        texts = [answer] + [normalize_ws(item.get("text")) for item in items]
        vectors = embedding_client.embed(texts, transport_retries=2)
        answer_vec = vectors[0]
        for item, vec in zip(items, vectors[1:]):
            row = dict(item)
            row["embedding_fas"] = round(_rescale_cosine(_cosine_similarity(answer_vec, vec)), 4)
            scored.append(row)
    except Exception:
        scored = [dict(item, embedding_fas=None) for item in items]
    selected = sorted(scored, key=lambda item: float(item.get("embedding_fas") or 0.0), reverse=True)[:limit]
    return selected, scored


def evaluate_expanded_future_alignment_v5(
    *,
    judge_client: OpenAICompatChatClient,
    embedding_client: Optional[OpenAICompatEmbeddingClient] = None,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
    hidden_row: Dict[str, Any],
    cfg: Optional[FutureAlignmentV3_1Config] = None,
) -> Dict[str, Any]:
    """Future alignment that credits evidence-backed acceptable neighbors.

    The strict unit audit remains intact. gold_sets acceptable neighbors expand
    what counts as a valid future-facing answer, while negative confusions cap
    answers that merely drift into tempting but unsupported directions.
    """

    cfg = cfg or FutureAlignmentV3_1Config()
    strict = evaluate_future_alignment_v5(
        judge_client=judge_client,
        embedding_client=embedding_client,
        public_task=public_task,
        result_row=result_row,
        hidden_row=hidden_row,
        cfg=cfg,
    )
    gold_sets = hidden_row.get("gold_sets") or {}
    expanded = gold_sets.get("expanded_future") or {}
    answer = normalize_ws(result_row.get("answer"))
    neighbors = [
        item
        for item in (expanded.get("acceptable_neighbors") or [])
        if isinstance(item, dict) and normalize_ws(item.get("text"))
    ]
    confusions = [
        item
        for item in (expanded.get("negative_confusions") or [])
        if isinstance(item, dict) and normalize_ws(item.get("text"))
    ]
    selected_neighbors, all_neighbor_candidates = _select_gold_set_items_for_llm(
        embedding_client=embedding_client,
        answer=answer,
        items=neighbors,
        limit=3,
    )
    selected_confusions, all_confusion_candidates = _select_gold_set_items_for_llm(
        embedding_client=embedding_client,
        answer=answer,
        items=confusions,
        limit=2,
    )
    neighbor_rows = [
        _score_gold_set_future_item(
            judge_client=judge_client,
            public_task=public_task,
            hidden_row=hidden_row,
            answer=answer,
            item=item,
            item_kind="acceptable_neighbor",
            cfg=cfg,
        )
        for item in selected_neighbors
    ]
    confusion_rows = [
        _score_gold_set_future_item(
            judge_client=judge_client,
            public_task=public_task,
            hidden_row=hidden_row,
            answer=answer,
            item=item,
            item_kind="negative_confusion",
            cfg=cfg,
        )
        for item in selected_confusions
    ]
    strict_score = _clamp01(strict.get("future_alignment_score"))
    best_neighbor_score = max([0.0] + [float(row.get("item_score") or 0.0) for row in neighbor_rows])
    negative_hits = [
        row
        for row in confusion_rows
        if row.get("label") == "aligned" or float(row.get("item_score") or 0.0) >= 0.45
    ]
    negative_cap = 1.0
    if negative_hits:
        negative_cap = 0.55 if any(row.get("label") == "aligned" for row in negative_hits) else 0.70
    expanded_score = round(min(max(strict_score, best_neighbor_score), negative_cap), 4)
    return {
        "future_alignment_score": expanded_score,
        "canonical_fas_score": expanded_score,
        "strict_future_alignment_score": strict_score,
        "strict_detail": strict,
        "best_acceptable_neighbor_score": round(best_neighbor_score, 4),
        "acceptable_neighbors": neighbor_rows,
        "negative_confusions": confusion_rows,
        "candidate_selection": {
            "mode": "embedding_topk_then_llm_audit" if embedding_client is not None else "first_k_then_llm_audit",
            "acceptable_neighbor_count": len(neighbors),
            "selected_acceptable_neighbor_count": len(selected_neighbors),
            "negative_confusion_count": len(confusions),
            "selected_negative_confusion_count": len(selected_confusions),
            "all_acceptable_neighbor_candidates": [
                {
                    "neighbor_id": item.get("neighbor_id"),
                    "text": item.get("text"),
                    "relation": item.get("relation"),
                    "credit_cap": item.get("credit_cap"),
                    "embedding_fas": item.get("embedding_fas"),
                }
                for item in all_neighbor_candidates
            ],
            "all_negative_confusion_candidates": [
                {
                    "confusion_id": item.get("confusion_id"),
                    "text": item.get("text"),
                    "embedding_fas": item.get("embedding_fas"),
                }
                for item in all_confusion_candidates
            ],
        },
        "negative_confusion_cap": negative_cap,
        "negative_confusion_hits": negative_hits,
        "reference_answer_similarity": strict.get("reference_answer_similarity"),
        "reference_answer_raw_cosine": strict.get("reference_answer_raw_cosine"),
        "raw_weighted_embedding_fas": strict.get("raw_weighted_embedding_fas"),
        "weighted_unit_alignment": strict.get("weighted_unit_alignment"),
        "alignment_coverage": strict.get("alignment_coverage"),
        "mean_specificity": strict.get("mean_specificity"),
        "units": strict.get("units") or [],
        "evaluator_scope": "task_json_gold_sets_expanded_future",
        "fas_mode": "strict_unit_fas_plus_evidence_bound_acceptable_neighbors",
    }
