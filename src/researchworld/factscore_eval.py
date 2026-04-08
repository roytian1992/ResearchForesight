from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import OfflineKnowledgeBase, clip_text, merge_multi_query_results


@dataclass
class FactScoreConfig:
    max_claims: int = 8
    evidence_per_view: int = 3
    max_evidence_rows: int = 8


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


def extract_atomic_claims(client: OpenAICompatChatClient, *, answer: str, max_claims: int) -> List[str]:
    prompt = f"""Decompose the answer into a small set of atomic factual claims that are intended to be grounded in historical evidence.

Rules:
- Keep only factual claims that could in principle be checked against a paper database.
- Ignore stylistic text, hedging, and purely normative statements.
- Ignore duplicate claims.
- At most {max_claims} claims.

Answer:
{answer}

Return JSON with key "claims" containing a list of strings.
"""
    obj = _complete_json(
        client,
        system="You extract atomic factual claims for evidence-grounding evaluation. Return only JSON.",
        prompt=prompt,
        max_tokens=700,
    )
    claims = [str(x).strip() for x in (obj.get("claims") or []) if str(x).strip()]
    return claims[:max_claims]


def render_evidence(rows: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        head = [f"[{row['evidence_id']}] {row.get('paper_title') or ''}"]
        if row.get("section_title"):
            head.append(f"section={row['section_title']}")
        if row.get("kind"):
            head.append(f"kind={row['kind']}")
        parts.append(" | ".join(head))
        parts.append(str(row.get("snippet") or ""))
        parts.append("")
    return "\n".join(parts).strip()


def retrieve_claim_evidence(
    kb: OfflineKnowledgeBase,
    *,
    domain_id: str,
    cutoff_date: str,
    claim: str,
    cfg: FactScoreConfig,
) -> List[Dict[str, Any]]:
    domain = kb.domain(domain_id)
    rows: List[Dict[str, Any]] = []
    paper_hits = merge_multi_query_results(domain.paper_retriever(cutoff_date=cutoff_date), [claim], top_k_per_query=cfg.evidence_per_view, limit=cfg.evidence_per_view)
    for idx, (doc, scores) in enumerate(paper_hits, start=1):
        rows.append(
            {
                "evidence_id": f"P{idx}",
                "paper_id": doc.paper_id,
                "paper_title": doc.title,
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    structure_hits = merge_multi_query_results(domain.structure_retriever(cutoff_date=cutoff_date), [claim], top_k_per_query=cfg.evidence_per_view, limit=cfg.evidence_per_view)
    for idx, (doc, scores) in enumerate(structure_hits, start=1):
        rows.append(
            {
                "evidence_id": f"T{idx}",
                "paper_id": doc.paper_id,
                "paper_title": doc.title,
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    section_hits = merge_multi_query_results(domain.section_retriever(cutoff_date=cutoff_date), [claim], top_k_per_query=cfg.evidence_per_view, limit=cfg.evidence_per_view)
    for idx, (doc, scores) in enumerate(section_hits, start=1):
        rows.append(
            {
                "evidence_id": f"S{idx}",
                "paper_id": doc.paper_id,
                "paper_title": doc.meta.get("paper_title") or doc.title,
                "section_title": doc.meta.get("section_title"),
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    page_hits = merge_multi_query_results(domain.pageindex_retriever(cutoff_date=cutoff_date), [claim], top_k_per_query=cfg.evidence_per_view, limit=cfg.evidence_per_view)
    for idx, (doc, scores) in enumerate(page_hits, start=1):
        rows.append(
            {
                "evidence_id": f"G{idx}",
                "paper_id": doc.paper_id,
                "paper_title": doc.meta.get("paper_title") or doc.title,
                "section_title": doc.meta.get("section_title"),
                "kind": doc.meta.get("kind"),
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    rows.sort(key=lambda x: -float((x.get("scores") or {}).get("combined_score") or 0.0))
    return rows[: cfg.max_evidence_rows]


def verify_claim(
    client: OpenAICompatChatClient,
    *,
    claim: str,
    evidence_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = f"""Decide whether the claim is supported by the provided evidence.

Claim:
{claim}

Evidence:
{render_evidence(evidence_rows)}

Return JSON with keys:
- label: supported | unsupported | insufficient
- rationale
- cited_evidence_ids: list of evidence ids

Rules:
- supported: the claim is directly backed by the evidence.
- unsupported: the evidence contradicts the claim or clearly points away from it.
- insufficient: the evidence is too weak or only partially related.
"""
    obj = _complete_json(
        client,
        system="You are a strict fact-verification judge. Use only the provided evidence. Return only JSON.",
        prompt=prompt,
        max_tokens=500,
    )
    label = str(obj.get("label") or "insufficient").strip().lower()
    if label not in {"supported", "unsupported", "insufficient"}:
        label = "insufficient"
    return {
        "label": label,
        "rationale": str(obj.get("rationale") or "").strip(),
        "cited_evidence_ids": [str(x) for x in (obj.get("cited_evidence_ids") or []) if str(x).strip()],
    }


def evaluate_answer_factscore(
    *,
    kb: OfflineKnowledgeBase,
    judge_client: OpenAICompatChatClient,
    row: Dict[str, Any],
    cfg: Optional[FactScoreConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or FactScoreConfig()
    answer = str(row.get("answer") or "").strip()
    if not answer:
        return {"claim_count": 0, "supported_count": 0, "factscore": 0.0, "claims": []}
    claims = extract_atomic_claims(judge_client, answer=answer, max_claims=cfg.max_claims)
    claim_rows = []
    for claim in claims:
        evidence_rows = retrieve_claim_evidence(
            kb,
            domain_id=str(row.get("domain_id") or ""),
            cutoff_date=str(row.get("time_cutoff") or ""),
            claim=claim,
            cfg=cfg,
        )
        verdict = verify_claim(judge_client, claim=claim, evidence_rows=evidence_rows)
        claim_rows.append(
            {
                "claim": claim,
                "verdict": verdict,
                "evidence": evidence_rows,
            }
        )
    supported = sum(1 for row in claim_rows if row["verdict"]["label"] == "supported")
    total = len(claim_rows)
    return {
        "claim_count": total,
        "supported_count": supported,
        "factscore": round(supported / total, 4) if total else 0.0,
        "claims": claim_rows,
    }
