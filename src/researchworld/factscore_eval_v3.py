from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import OfflineKnowledgeBase, clip_text, merge_multi_query_results, normalize_ws


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\./+]{0,63}")


@dataclass
class FactScoreV3Config:
    max_claims: int = 8
    evidence_per_view: int = 3
    max_evidence_rows: int = 8
    gt_match_threshold: float = 0.24
    unmatched_claim_weight: float = 0.4
    precision_weight: float = 0.6
    coverage_weight: float = 0.4


def _complete_json(client: OpenAICompatChatClient, *, system: str, prompt: str, max_tokens: int = 900) -> Dict[str, Any]:
    return complete_json_object(
        client,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=max_tokens,
        timeout=90,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Your previous response was malformed JSON. Return exactly one valid JSON object matching the requested schema. No markdown, no extra text.",
    )


def tokenize(text: Any) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or "").replace("_", " "))]


def extract_atomic_claims(client: OpenAICompatChatClient, *, answer: str, max_claims: int) -> List[str]:
    prompt = f"""Decompose the answer into a small set of atomic benchmark-relevant factual claims.

Rules:
- Keep only claims that could be grounded in a research-paper benchmark.
- Prefer claims about bottlenecks, opportunities, trajectories, emergent directions, venues, metrics, counts, or representative papers.
- Ignore purely stylistic statements and generic advice.
- Ignore duplicates.
- At most {max_claims} claims.

Answer:
{answer}

Return JSON with key "claims" containing a list of strings.
"""
    obj = _complete_json(
        client,
        system="You extract benchmark-relevant atomic factual claims. Return only JSON.",
        prompt=prompt,
        max_tokens=700,
    )
    claims = [normalize_ws(x) for x in (obj.get("claims") or []) if normalize_ws(x)]
    deduped: List[str] = []
    seen = set()
    for claim in claims:
        key = claim.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped[:max_claims]


def render_evidence(rows: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        head = [f"[{row['evidence_id']}] source={row.get('evidence_source')} {row.get('paper_title') or ''}"]
        if row.get("section_title"):
            head.append(f"section={row['section_title']}")
        if row.get("kind"):
            head.append(f"kind={row['kind']}")
        parts.append(" | ".join(head))
        parts.append(str(row.get("snippet") or ""))
        parts.append("")
    return "\n".join(parts).strip()


def _claim_candidates(gt_claim: Dict[str, Any]) -> List[str]:
    values: List[str] = [str(gt_claim.get("text") or "")]
    values.extend(str(x) for x in (gt_claim.get("canonical_objects") or []))
    aliases = gt_claim.get("aliases") or {}
    for rows in aliases.values():
        values.extend(str(x) for x in (rows or []))
    out: List[str] = []
    seen = set()
    for value in values:
        norm = normalize_ws(value)
        key = norm.lower()
        if not norm or key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def _text_match_score(a: str, b: str) -> float:
    a_norm = normalize_ws(a).lower().replace("_", " ")
    b_norm = normalize_ws(b).lower().replace("_", " ")
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    if a_norm in b_norm or b_norm in a_norm:
        return 0.82
    a_tokens = set(tokenize(a_norm))
    b_tokens = set(tokenize(b_norm))
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    jaccard = inter / union
    recall = inter / len(b_tokens)
    precision = inter / len(a_tokens)
    return max(jaccard, 0.65 * recall + 0.35 * precision)


def match_answer_claim_to_gt(answer_claim: str, claim_bank: List[Dict[str, Any]], threshold: float) -> Tuple[Optional[Dict[str, Any]], float]:
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for gt_claim in claim_bank:
        local_best = 0.0
        for candidate in _claim_candidates(gt_claim):
            local_best = max(local_best, _text_match_score(answer_claim, candidate))
        if local_best > best_score:
            best_score = local_best
            best = gt_claim
    if best is None or best_score < threshold:
        return None, best_score
    return best, best_score


def _collect_evidence_from_domain(domain_kb, queries: List[str], *, cutoff_date: Optional[str], cfg: FactScoreV3Config, source_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    paper_hits = merge_multi_query_results(domain_kb.paper_retriever(cutoff_date=cutoff_date), queries, top_k_per_query=cfg.evidence_per_view, limit=cfg.evidence_per_view)
    for idx, (doc, scores) in enumerate(paper_hits, start=1):
        rows.append(
            {
                "evidence_id": f"{source_name[0].upper()}P{idx}",
                "evidence_source": source_name,
                "paper_id": doc.paper_id,
                "paper_title": doc.title,
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    structure_hits = merge_multi_query_results(domain_kb.structure_retriever(cutoff_date=cutoff_date), queries, top_k_per_query=cfg.evidence_per_view, limit=cfg.evidence_per_view)
    for idx, (doc, scores) in enumerate(structure_hits, start=1):
        rows.append(
            {
                "evidence_id": f"{source_name[0].upper()}T{idx}",
                "evidence_source": source_name,
                "paper_id": doc.paper_id,
                "paper_title": doc.title,
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    section_hits = merge_multi_query_results(domain_kb.section_retriever(cutoff_date=cutoff_date), queries, top_k_per_query=cfg.evidence_per_view, limit=cfg.evidence_per_view)
    for idx, (doc, scores) in enumerate(section_hits, start=1):
        rows.append(
            {
                "evidence_id": f"{source_name[0].upper()}S{idx}",
                "evidence_source": source_name,
                "paper_id": doc.paper_id,
                "paper_title": doc.meta.get("paper_title") or doc.title,
                "section_title": doc.meta.get("section_title"),
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    page_hits = merge_multi_query_results(domain_kb.pageindex_retriever(cutoff_date=cutoff_date), queries, top_k_per_query=cfg.evidence_per_view, limit=cfg.evidence_per_view)
    for idx, (doc, scores) in enumerate(page_hits, start=1):
        rows.append(
            {
                "evidence_id": f"{source_name[0].upper()}G{idx}",
                "evidence_source": source_name,
                "paper_id": doc.paper_id,
                "paper_title": doc.meta.get("paper_title") or doc.title,
                "section_title": doc.meta.get("section_title"),
                "kind": doc.meta.get("kind"),
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    return rows


def retrieve_claim_evidence_v3(
    history_kb: OfflineKnowledgeBase,
    future_kb: OfflineKnowledgeBase,
    *,
    domain_id: str,
    answer_claim: str,
    matched_gt_claim: Optional[Dict[str, Any]],
    temporal_policy: Dict[str, Any],
    cfg: FactScoreV3Config,
) -> List[Dict[str, Any]]:
    queries = [answer_claim]
    if matched_gt_claim:
        queries.extend(str(x) for x in (matched_gt_claim.get("canonical_objects") or []))
        queries.append(str(matched_gt_claim.get("text") or ""))
    deduped_queries: List[str] = []
    seen = set()
    for query in queries:
        norm = normalize_ws(query)
        key = norm.lower()
        if not norm or key in seen:
            continue
        seen.add(key)
        deduped_queries.append(norm)

    evidence_rows: List[Dict[str, Any]] = []
    scope = str((matched_gt_claim or {}).get("time_scope") or "cross_temporal")
    if scope in {"history", "cross_temporal"}:
        evidence_rows.extend(
            _collect_evidence_from_domain(
                history_kb.domain(domain_id),
                deduped_queries,
                cutoff_date=str(temporal_policy.get("history_cutoff") or ""),
                cfg=cfg,
                source_name="history",
            )
        )
    if scope in {"future", "cross_temporal"}:
        evidence_rows.extend(
            _collect_evidence_from_domain(
                future_kb.domain(domain_id),
                deduped_queries,
                cutoff_date=str(temporal_policy.get("future_end") or ""),
                cfg=cfg,
                source_name="future",
            )
        )
    if not matched_gt_claim:
        evidence_rows.extend(
            _collect_evidence_from_domain(
                history_kb.domain(domain_id),
                deduped_queries,
                cutoff_date=str(temporal_policy.get("history_cutoff") or ""),
                cfg=cfg,
                source_name="history",
            )
        )
        evidence_rows.extend(
            _collect_evidence_from_domain(
                future_kb.domain(domain_id),
                deduped_queries,
                cutoff_date=str(temporal_policy.get("future_end") or ""),
                cfg=cfg,
                source_name="future",
            )
        )
    evidence_rows.sort(key=lambda x: -float((x.get("scores") or {}).get("combined_score") or 0.0))

    deduped_rows: List[Dict[str, Any]] = []
    seen_keys = set()
    for row in evidence_rows:
        key = (row.get("evidence_source"), row.get("paper_id"), row.get("section_title"), row.get("kind"))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped_rows.append(row)
    return deduped_rows[: cfg.max_evidence_rows]


def verify_claim_v3(
    client: OpenAICompatChatClient,
    *,
    claim: str,
    matched_gt_claim: Optional[Dict[str, Any]],
    evidence_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    gt_note = ""
    if matched_gt_claim:
        gt_note = json.dumps(
            {
                "claim_type": matched_gt_claim.get("claim_type"),
                "time_scope": matched_gt_claim.get("time_scope"),
                "canonical_objects": matched_gt_claim.get("canonical_objects"),
                "match_policy": matched_gt_claim.get("match_policy"),
            },
            ensure_ascii=False,
        )
    prompt = f"""# Role
You are a High-Precision Research Fact-Verifier. Your task is to audit whether an atomic claim extracted from a research answer is justified by the provided evidence.

# Strategic Intent
A claim may be a "Generalization" or a "Synthesized Point" that covers multiple ground-truth aspects. You should evaluate it with "rigorous flexibility":
1. If the claim is a high-level summary, it is supported as long as the evidence confirms its constituent parts or the overarching logic.
2. Avoid being pedantic about exact wording; focus on whether the research substance is corroborated.

# Input Context
- Answer Claim: {claim}
- Benchmark Context: {gt_note}
- Evidence: {render_evidence(evidence_rows)}

# Judgment Labels
- supported: The evidence directly confirms the claim OR provides enough granular data to justify this summary/generalization.
- unsupported: The evidence explicitly contradicts the claim or supports a fundamentally different conclusion.
- insufficient: The evidence is too peripheral, vague, or lacks the necessary depth to confirm even a generalized version of the claim.

# Temporal & Causal Logic
Evaluate the Temporal Consistency:
- consistent: The claim's timing (e.g., "current bottleneck" vs "future direction") matches the evidence.
- inconsistent: The claim describes as "future" what evidence shows is "historical," or vice-versa.
- unclear: The evidence or claim lacks specific temporal markers.

# Rules of Engagement
- Do NOT use external knowledge; rely solely on the provided Evidence.
- In the rationale, explain how the specific evidence IDs roll up to support the claim, especially if the claim is a generalization.

# Output (Strict JSON)
{{
  "label": "supported | unsupported | insufficient",
  "rationale": "Explain the logical mapping between the claim and the evidence. If the claim is a summary, clarify how the specific evidence points support the broader statement.",
  "cited_evidence_ids": ["id1", "id2"],
  "temporal_consistency": "consistent | inconsistent | unclear"
}}
"""
    obj = _complete_json(
        client,
        system="You are a strict benchmark-aware fact verification judge. Use only the provided evidence. Return only JSON.",
        prompt=prompt,
        max_tokens=550,
    )
    label = str(obj.get("label") or "insufficient").strip().lower()
    if label not in {"supported", "unsupported", "insufficient"}:
        label = "insufficient"
    temporal_consistency = str(obj.get("temporal_consistency") or "clear").strip().lower()
    if temporal_consistency not in {"consistent", "inconsistent", "unclear"}:
        temporal_consistency = "unclear"
    return {
        "label": label,
        "rationale": str(obj.get("rationale") or "").strip(),
        "cited_evidence_ids": [str(x) for x in (obj.get("cited_evidence_ids") or []) if str(x).strip()],
        "temporal_consistency": temporal_consistency,
    }


def evaluate_answer_factscore_v3(
    *,
    history_kb: OfflineKnowledgeBase,
    future_kb: OfflineKnowledgeBase,
    judge_client: OpenAICompatChatClient,
    result_row: Dict[str, Any],
    gt_row: Dict[str, Any],
    cfg: Optional[FactScoreV3Config] = None,
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
        }

    answer_claims = extract_atomic_claims(judge_client, answer=answer, max_claims=cfg.max_claims)
    gt_claim_bank = list(gt_row.get("claim_bank") or [])
    temporal_policy = gt_row.get("temporal_policy") or {}
    domain_id = str(result_row.get("domain_id") or "")

    claim_rows = []
    total_weight = 0.0
    supported_weight = 0.0
    covered_gt_claims = set()
    gt_total_weight = sum(float(claim.get("importance") or 0.0) for claim in gt_claim_bank)

    for claim in answer_claims:
        matched_gt_claim, match_score = match_answer_claim_to_gt(claim, gt_claim_bank, cfg.gt_match_threshold)
        weight = float((matched_gt_claim or {}).get("importance") or cfg.unmatched_claim_weight)
        total_weight += weight
        evidence_rows = retrieve_claim_evidence_v3(
            history_kb,
            future_kb,
            domain_id=domain_id,
            answer_claim=claim,
            matched_gt_claim=matched_gt_claim,
            temporal_policy=temporal_policy,
            cfg=cfg,
        )
        verdict = verify_claim_v3(judge_client, claim=claim, matched_gt_claim=matched_gt_claim, evidence_rows=evidence_rows)
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
    precision_score = round(supported_weight / total_weight, 4) if total_weight else 0.0
    coverage_score = round(covered_weight / gt_total_weight, 4) if gt_total_weight else 0.0
    benchmark_factscore = round(
        cfg.precision_weight * precision_score + cfg.coverage_weight * coverage_score,
        4,
    )
    return {
        "claim_count": len(claim_rows),
        "supported_count": sum(1 for row in claim_rows if row["verdict"]["label"] == "supported"),
        "precision_score": precision_score,
        "coverage_score": coverage_score,
        "benchmark_factscore": benchmark_factscore,
        "weighted_supported": round(supported_weight, 4),
        "weighted_total": round(total_weight, 4),
        "weighted_gt_covered": round(covered_weight, 4),
        "weighted_gt_total": round(gt_total_weight, 4),
        "claims": claim_rows,
    }
