from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import OfflineKnowledgeBase, clip_text, merge_multi_query_results, normalize_ws


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\./+]{0,63}")
NUMERIC_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

GENERIC_FACT_TOKENS = {
    "a",
    "an",
    "and",
    "approach",
    "approaches",
    "challenge",
    "challenges",
    "direction",
    "directions",
    "for",
    "framework",
    "frameworks",
    "in",
    "is",
    "method",
    "methods",
    "model",
    "models",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "research",
    "study",
    "studies",
    "system",
    "systems",
    "task",
    "tasks",
    "the",
    "to",
    "toward",
    "towards",
    "using",
    "with",
}

SOFT_GENERIC_FACT_TOKENS = {
    "agent",
    "agents",
    "augmented",
    "benchmark",
    "benchmarks",
    "control",
    "evaluation",
    "fine",
    "finegrained",
    "generation",
    "graph",
    "llm",
    "memory",
    "multi",
    "rag",
    "retrieval",
    "semantic",
    "training",
    "video",
}

BOTTLENECK_CUES = [
    "bottleneck",
    "barrier",
    "challenge",
    "fails",
    "failure",
    "friction",
    "hinders",
    "insufficient",
    "lack of",
    "limitation",
    "limited",
    "underperform",
    "unresolved",
]

OPPORTUNITY_CUES = [
    "becomes viable",
    "downstream opportunity",
    "enable",
    "enables",
    "opportunity",
    "opens room",
    "opens the door",
    "unlock",
    "unlocks",
]

DIRECTION_CUES = [
    "emerging as",
    "future work",
    "immediate next",
    "most likely next",
    "next direction",
    "next focus",
    "next step",
    "primary focus",
    "research is shifting",
    "shifting toward",
]

TRAJECTORY_CUES = [
    "accelerating",
    "consolidating",
    "fragmenting",
    "plateauing",
    "slowing",
    "trajectory",
]

VENUE_CUES = [
    "aaai",
    "acl",
    "conference",
    "emnlp",
    "iclr",
    "icml",
    "ijcai",
    "kdd",
    "naacl",
    "neurips",
    "sigir",
    "venue",
]

STATISTICAL_CUES = [
    "citation",
    "count",
    "paper count",
    "papers",
    "percent",
    "share",
]


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


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def tokenize(text: Any) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or "").replace("_", " "))]


def _normalize_text(text: Any) -> str:
    return normalize_ws(text).lower().replace("_", " ")


def _has_numeric_fact(text: Any) -> bool:
    return NUMERIC_TOKEN_RE.search(str(text or "")) is not None


def _target_anchor_tokens(text: Any) -> List[str]:
    tokens: List[str] = []
    for tok in tokenize(text):
        if tok in GENERIC_FACT_TOKENS:
            continue
        if len(tok) <= 2:
            continue
        tokens.append(tok)
    out: List[str] = []
    seen = set()
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def _distinctive_tokens(text: Any) -> List[str]:
    return [tok for tok in _target_anchor_tokens(text) if tok not in SOFT_GENERIC_FACT_TOKENS]


def _phrase_hits(text: Any, phrases: List[str]) -> List[str]:
    norm = _normalize_text(text)
    hits: List[str] = []
    for phrase in phrases:
        if phrase in norm:
            hits.append(phrase)
    return hits


def _answer_claim_profile(answer_claim: str) -> Dict[str, bool]:
    return {
        "bottleneck": bool(_phrase_hits(answer_claim, BOTTLENECK_CUES)),
        "opportunity": bool(_phrase_hits(answer_claim, OPPORTUNITY_CUES)),
        "direction": bool(_phrase_hits(answer_claim, DIRECTION_CUES)),
        "trajectory": bool(_phrase_hits(answer_claim, TRAJECTORY_CUES)),
        "venue": bool(_phrase_hits(answer_claim, VENUE_CUES)),
        "statistical": bool(_phrase_hits(answer_claim, STATISTICAL_CUES)) or _has_numeric_fact(answer_claim),
    }


def _gt_claim_family(gt_claim: Dict[str, Any]) -> str:
    claim_type = str(gt_claim.get("claim_type") or "").lower()
    if "bottleneck" in claim_type:
        return "bottleneck"
    if "opportunity" in claim_type:
        return "opportunity"
    if "direction" in claim_type or "ranked" in claim_type:
        return "direction"
    if "trajectory" in claim_type:
        return "trajectory"
    if "venue" in claim_type:
        return "venue"
    if "statistical" in claim_type or "volume" in claim_type or "share" in claim_type:
        return "statistical"
    return "other"


def _claim_family_compatibility(answer_claim: str, gt_claim: Dict[str, Any]) -> float:
    profile = _answer_claim_profile(answer_claim)
    family = _gt_claim_family(gt_claim)
    if family == "bottleneck":
        if profile["trajectory"] or profile["venue"]:
            return 0.72
        if profile["opportunity"] and not profile["bottleneck"]:
            return 0.78
        if profile["bottleneck"]:
            return 1.06
    elif family == "opportunity":
        if profile["trajectory"] or profile["venue"]:
            return 0.74
        if profile["bottleneck"] and not profile["opportunity"] and not profile["direction"]:
            return 0.78
        if profile["opportunity"] or profile["direction"]:
            return 1.05
    elif family == "direction":
        if profile["trajectory"]:
            return 0.82
        if profile["bottleneck"] and not profile["direction"] and not profile["opportunity"]:
            return 0.76
        if profile["direction"] or profile["opportunity"]:
            return 1.06
    elif family == "trajectory":
        if profile["trajectory"]:
            return 1.08
        if profile["direction"]:
            return 0.86
        return 0.72
    elif family == "venue":
        return 1.08 if profile["venue"] else 0.74
    elif family == "statistical":
        return 1.08 if profile["statistical"] else 0.72
    return 1.0


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
    current = max(jaccard, 0.65 * recall + 0.35 * precision)
    if _has_numeric_fact(a) or _has_numeric_fact(b):
        return current

    a_distinctive = _distinctive_tokens(a)
    b_distinctive = _distinctive_tokens(b)
    if not a_distinctive or not b_distinctive:
        return current
    overlap = [tok for tok in a_distinctive if tok in set(b_distinctive)]
    if not overlap:
        return current

    distinct_recall = len(overlap) / max(1, len(b_distinctive))
    target_bigrams = {
        f"{b_distinctive[idx]} {b_distinctive[idx + 1]}"
        for idx in range(len(b_distinctive) - 1)
        if b_distinctive[idx] and b_distinctive[idx + 1]
    }
    bigram_hits = [bg for bg in target_bigrams if bg in a_norm]
    if bigram_hits:
        current = max(current, min(0.86, 0.64 + 0.08 * len(bigram_hits) + 0.12 * distinct_recall))
    elif len(overlap) >= 2:
        if distinct_recall >= 0.5:
            current = max(current, min(0.8, 0.58 + 0.22 * distinct_recall))
        else:
            current = max(current, 0.54)
    return current


def match_answer_claim_to_gt(answer_claim: str, claim_bank: List[Dict[str, Any]], threshold: float) -> Tuple[Optional[Dict[str, Any]], float]:
    best: Optional[Dict[str, Any]] = None
    best_score = 0.0
    for gt_claim in claim_bank:
        local_best = 0.0
        for candidate in _claim_candidates(gt_claim):
            score = _text_match_score(answer_claim, candidate)
            score = min(1.0, score * _claim_family_compatibility(answer_claim, gt_claim))
            local_best = max(local_best, score)
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
                "published_date": doc.meta.get("published_date"),
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
                "published_date": (domain_kb.get_paper(doc.paper_id) or {}).get("published_date"),
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
                "published_date": doc.meta.get("published_date") or (domain_kb.get_paper(doc.paper_id) or {}).get("published_date"),
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
                "published_date": (domain_kb.get_paper(doc.paper_id) or {}).get("published_date"),
                "snippet": clip_text(doc.text, 1000),
                "scores": scores,
            }
        )
    return rows


def _date_in_window(value: Any, *, start: str = "", end: str = "") -> bool:
    day = normalize_ws(value)[:10]
    if not day:
        return True
    if start and day < start:
        return False
    if end and day > end:
        return False
    return True


def _filter_evidence_window(rows: List[Dict[str, Any]], *, start: str = "", end: str = "") -> List[Dict[str, Any]]:
    return [row for row in rows if _date_in_window(row.get("published_date"), start=start, end=end)]


def _embedded_trace_evidence(
    gt_row: Optional[Dict[str, Any]],
    *,
    matched_gt_claim: Optional[Dict[str, Any]],
    temporal_policy: Dict[str, Any],
    source_name: str,
    max_rows: int,
) -> List[Dict[str, Any]]:
    if not gt_row:
        return []
    trace = gt_row.get("trace") or {}
    trace_key = "future_evidence" if source_name == "future" else "history_evidence"
    items = [item for item in (trace.get(trace_key) or []) if isinstance(item, dict)]
    if not items:
        return []
    ref_ids = {
        str(x).strip()
        for x in ((matched_gt_claim or {}).get("reference_paper_ids") or [])
        if str(x).strip()
    }
    if ref_ids:
        matched = [item for item in items if str(item.get("paper_id") or "").strip() in ref_ids]
        if matched:
            items = matched
    if source_name == "history":
        items = [
            item
            for item in items
            if _date_in_window(item.get("published_date"), end=str(temporal_policy.get("history_cutoff") or ""))
        ]
    else:
        items = [
            item
            for item in items
            if _date_in_window(
                item.get("published_date"),
                start=str(temporal_policy.get("future_start") or ""),
                end=str(temporal_policy.get("future_end") or ""),
            )
        ]
    rows: List[Dict[str, Any]] = []
    prefix = "FE" if source_name == "future" else "HE"
    for idx, item in enumerate(items[:max_rows], start=1):
        rows.append(
            {
                "evidence_id": f"{prefix}{idx}",
                "evidence_source": f"embedded_{source_name}",
                "paper_id": item.get("paper_id"),
                "paper_title": item.get("title"),
                "published_date": item.get("published_date"),
                "snippet": clip_text(item.get("why_it_matters") or item.get("title") or "", 1000),
                "scores": {"combined_score": 1.0},
            }
        )
    return rows


def retrieve_claim_evidence_v3(
    history_kb: OfflineKnowledgeBase,
    future_kb: Optional[OfflineKnowledgeBase],
    *,
    domain_id: str,
    answer_claim: str,
    matched_gt_claim: Optional[Dict[str, Any]],
    temporal_policy: Dict[str, Any],
    gt_row: Optional[Dict[str, Any]],
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
        history_rows = _collect_evidence_from_domain(
            history_kb.domain(domain_id),
            deduped_queries,
            cutoff_date=str(temporal_policy.get("history_cutoff") or ""),
            cfg=cfg,
            source_name="history",
        )
        evidence_rows.extend(
            _filter_evidence_window(history_rows, end=str(temporal_policy.get("history_cutoff") or ""))
        )
        evidence_rows.extend(
            _embedded_trace_evidence(
                gt_row,
                matched_gt_claim=matched_gt_claim,
                temporal_policy=temporal_policy,
                source_name="history",
                max_rows=cfg.max_evidence_rows,
            )
        )
    if scope in {"future", "cross_temporal"} and future_kb is not None:
        future_rows = _collect_evidence_from_domain(
            future_kb.domain(domain_id),
            deduped_queries,
            cutoff_date=str(temporal_policy.get("future_end") or ""),
            cfg=cfg,
            source_name="future",
        )
        evidence_rows.extend(
            _filter_evidence_window(
                future_rows,
                start=str(temporal_policy.get("future_start") or ""),
                end=str(temporal_policy.get("future_end") or ""),
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
    if not matched_gt_claim:
        history_rows = _collect_evidence_from_domain(
            history_kb.domain(domain_id),
            deduped_queries,
            cutoff_date=str(temporal_policy.get("history_cutoff") or ""),
            cfg=cfg,
            source_name="history",
        )
        evidence_rows.extend(
            _filter_evidence_window(history_rows, end=str(temporal_policy.get("history_cutoff") or ""))
        )
        if future_kb is not None:
            future_rows = _collect_evidence_from_domain(
                future_kb.domain(domain_id),
                deduped_queries,
                cutoff_date=str(temporal_policy.get("future_end") or ""),
                cfg=cfg,
                source_name="future",
            )
            evidence_rows.extend(
                _filter_evidence_window(
                    future_rows,
                    start=str(temporal_policy.get("future_start") or ""),
                    end=str(temporal_policy.get("future_end") or ""),
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
    if not evidence_rows:
        return {
            "label": "insufficient",
            "rationale": "No evidence rows were available for this claim.",
            "cited_evidence_ids": [],
            "temporal_consistency": "unclear",
        }
    gt_note = ""
    if matched_gt_claim:
        gt_note = json.dumps(
            {
                "claim_text": matched_gt_claim.get("text"),
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
3. If the claim is only slightly broader or narrower than the matched benchmark target, but stays inside the same immediate technical cluster or mechanism family, treat that as acceptable alignment rather than as a mismatch.
4. Penalize only when the claim drifts to a materially different mechanism family, future direction family, venue family, or trajectory label.

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
- For benchmark taxonomy-like claims about bottlenecks, opportunities, directions, trajectories, or venues, exact canonical phrase equality is NOT required.
- If the evidence supports the same technical mechanism but the claim uses a nearby parent/child abstraction, mark it as supported rather than insufficient.

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
    valid_evidence_ids = {str(row.get("evidence_id") or "") for row in evidence_rows}
    return {
        "label": label,
        "rationale": str(obj.get("rationale") or "").strip(),
        "cited_evidence_ids": [
            str(x)
            for x in (obj.get("cited_evidence_ids") or [])
            if str(x).strip() and str(x).strip() in valid_evidence_ids
        ],
        "temporal_consistency": temporal_consistency,
    }


def evaluate_answer_factscore_v3(
    *,
    history_kb: OfflineKnowledgeBase,
    future_kb: Optional[OfflineKnowledgeBase],
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
            gt_row=gt_row,
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
    }
