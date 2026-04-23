from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import clip_text, normalize_ws


TRAJECTORY_LABELS = {"accelerating", "steady", "cooling", "fragmenting"}
_LIST_ITEM_RE = re.compile(r"\((\d+)\)\s*")
_WS_RE = re.compile(r"\s+")


def _clean_text(text: Any) -> str:
    return normalize_ws(str(text or ""))


def _clean_sentence(text: Any) -> str:
    return _clean_text(text).strip(" .;")


def _norm(text: Any) -> str:
    text = _clean_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return _WS_RE.sub(" ", text).strip()


def _clip_json(obj: Any, *, limit: int = 6000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    return text if len(text) <= limit else text[:limit] + "\n...<truncated>"


def _family_guidance(family: str) -> List[str]:
    if family == "bottleneck_opportunity_discovery":
        return [
            "Choose one mechanism-level bottleneck rather than a broad area label.",
            "Choose one immediate downstream opportunity that becomes more viable if that bottleneck is resolved.",
            "Make the causal linkage explicit and evidence-grounded.",
        ]
    if family == "direction_forecasting":
        return [
            "Choose exactly one next-step direction and one trajectory label.",
            "Keep the direction concrete and technically visible in the evidence.",
            "Explain why the direction follows now rather than giving a generic trend summary.",
        ]
    if family == "strategic_research_planning":
        return [
            "Rank all listed candidate directions exactly once.",
            "Do not introduce any new direction or umbrella substitute.",
            "Give why-now logic plus one dependency, defer reason, or trade-off for each item.",
        ]
    if family == "venue_aware_research_positioning":
        return [
            "Rank all listed candidate directions exactly once when the task is comparative.",
            "Separate technical rationale from venue-fit rationale.",
            "Make the target venue's methodological preference explicit.",
        ]
    return ["Directly fulfill the task and keep the answer concrete."]


def _normalize_evidence_item(row: Dict[str, Any], evidence_id: str) -> Dict[str, Any]:
    title = (
        row.get("paper_title")
        or row.get("title")
        or row.get("display_name")
        or row.get("section_title")
        or row.get("node_id")
        or ""
    )
    snippet = (
        row.get("snippet")
        or row.get("text")
        or row.get("abstract")
        or row.get("problem_statement")
        or row.get("description")
        or ""
    )
    if not snippet:
        pieces = []
        for key, label in (
            ("limitations", "Limitations"),
            ("future_work", "Future work"),
            ("core_ideas", "Core ideas"),
        ):
            values = row.get(key) or []
            if values:
                pieces.append(f"{label}: " + "; ".join(str(x) for x in values[:3] if str(x).strip()))
        snippet = " ".join(pieces)
    return {
        "evidence_id": _clean_text(row.get("evidence_id") or evidence_id),
        "title": _clean_text(title),
        "snippet": clip_text(snippet, 900),
        "paper_id": row.get("paper_id"),
        "venue": row.get("venue"),
        "published_date": row.get("published_date"),
    }


def extract_adapter_evidence(result_row: Dict[str, Any], *, limit: int = 12) -> List[Dict[str, Any]]:
    evidence_rows: List[Dict[str, Any]] = []
    top_evidence = result_row.get("evidence")
    if isinstance(top_evidence, dict):
        for key in ("retrieved", "papers", "fulltext", "structures", "pageindex"):
            for item in (top_evidence.get(key) or []):
                if isinstance(item, dict):
                    evidence_rows.append(item)
    trace = result_row.get("trace") or {}
    trace_evidence = trace.get("evidence") or {}
    if isinstance(trace_evidence, dict):
        for key in (
            "retrieved",
            "papers",
            "fulltext",
            "structures",
            "pageindex",
            "paper_evidence",
            "section_evidence",
            "structure_evidence",
            "candidate_node_evidence",
        ):
            for item in (trace_evidence.get(key) or []):
                if isinstance(item, dict):
                    evidence_rows.append(item)
    formatted = []
    seen = set()
    for idx, row in enumerate(evidence_rows, start=1):
        norm = _normalize_evidence_item(row, str(row.get("evidence_id") or f"P{idx}"))
        key = (str(norm.get("paper_id") or ""), norm["evidence_id"], norm["title"], norm["snippet"])
        if key in seen:
            continue
        if not norm["title"] and not norm["snippet"]:
            continue
        seen.add(key)
        formatted.append(norm)
        if len(formatted) >= limit:
            break
    return formatted


def _extract_listed_candidates(text: str) -> List[str]:
    matches = list(_LIST_ITEM_RE.finditer(text or ""))
    if len(matches) < 2:
        return []
    out: List[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end]
        chunk = re.split(
            r"(?i)\b(?:provide|rank|justify|consider only|do not introduce|do not add|limit your ranking|your response)\b",
            chunk,
            maxsplit=1,
        )[0]
        chunk = chunk.strip(" \t\r\n:;,.")
        chunk = normalize_ws(chunk)
        if chunk:
            out.append(chunk)
    deduped = []
    seen = set()
    for item in out:
        key = _norm(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    collapsed: List[str] = []
    for item in deduped:
        item_norm = _norm(item)
        replaced = False
        for idx, existing in enumerate(collapsed):
            existing_norm = _norm(existing)
            if item_norm == existing_norm or item_norm in existing_norm or existing_norm in item_norm:
                if len(item_norm) > len(existing_norm):
                    collapsed[idx] = item
                replaced = True
                break
        if not replaced:
            collapsed.append(item)
    return collapsed


def extract_task_candidates(public_task: Dict[str, Any]) -> List[str]:
    candidates = _extract_listed_candidates(str(public_task.get("question") or ""))
    if len(candidates) >= 2:
        return candidates
    for req in (public_task.get("deliverable_spec") or {}).get("requirements") or []:
        candidates = _extract_listed_candidates(str(req or ""))
        if len(candidates) >= 2:
            return candidates
    return []


def _align_to_candidates(text: str, candidates: Sequence[str]) -> str:
    if not candidates:
        return _clean_text(text)
    norm_text = _norm(text)
    if not norm_text:
        return ""
    norm_map = {_norm(candidate): candidate for candidate in candidates}
    if norm_text in norm_map:
        return norm_map[norm_text]
    for candidate in candidates:
        cand_norm = _norm(candidate)
        if norm_text in cand_norm or cand_norm in norm_text:
            return candidate
    best: Optional[Tuple[float, str]] = None
    for candidate in candidates:
        score = difflib.SequenceMatcher(None, norm_text, _norm(candidate)).ratio()
        if best is None or score > best[0]:
            best = (score, candidate)
    if best and best[0] >= 0.62:
        return best[1]
    return _clean_text(text)


def _sanitize_evidence_ids(raw_ids: Sequence[Any], evidence_map: Dict[str, Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in raw_ids or []:
        if raw is None:
            continue
        text = str(raw)
        for token in re.findall(r"(?:P|F|T|S|E)\d+", text):
            if token in evidence_map and token not in seen:
                seen.add(token)
                out.append(token)
    if out:
        return out[:4]
    return list(evidence_map.keys())[:3]


def _trace_bundle(result_row: Dict[str, Any]) -> Dict[str, Any]:
    trace = result_row.get("trace") or {}
    if not isinstance(trace, dict):
        return {}
    bundle: Dict[str, Any] = {}
    for key in ("task_frame", "decision_packet", "render", "family_head", "family_packet", "review", "survey", "ideation", "trend", "future", "draft_answer", "critique"):
        value = trace.get(key)
        if value is not None:
            if key == "survey" and isinstance(value, dict):
                bundle[key] = {
                    "paper_table": (value.get("paper_table") or [])[:8],
                    "themes": (value.get("themes") or [])[:6],
                    "gaps": (value.get("gaps") or [])[:5],
                    "momentum_signals": (value.get("momentum_signals") or [])[:5],
                    "signal_map": value.get("signal_map") or {},
                }
            elif key == "ideation" and isinstance(value, dict):
                idea = dict(value)
                if isinstance(idea.get("candidates"), list):
                    idea["candidates"] = idea["candidates"][:6]
                if isinstance(idea.get("agenda"), list):
                    idea["agenda"] = idea["agenda"][:8]
                bundle[key] = idea
            else:
                bundle[key] = value
    if trace.get("workflow") is not None:
        bundle["workflow"] = trace.get("workflow")
    if trace.get("retrieval_mode") is not None:
        bundle["retrieval_mode"] = trace.get("retrieval_mode")
    return bundle


def build_adapter_bundle(public_task: Dict[str, Any], result_row: Dict[str, Any]) -> Dict[str, Any]:
    method = str(result_row.get("agent") or result_row.get("baseline") or result_row.get("method_key") or "unknown")
    evidence = extract_adapter_evidence(result_row)
    raw_answer = str(result_row.get("raw_answer") or result_row.get("answer") or "").strip()
    return {
        "method": method,
        "public_task": {
            "task_id": public_task.get("task_id"),
            "family": public_task.get("family"),
            "domain": public_task.get("domain"),
            "horizon": public_task.get("horizon"),
            "time_cutoff": public_task.get("time_cutoff"),
            "title": public_task.get("title"),
            "question": public_task.get("question"),
            "deliverable_spec": public_task.get("deliverable_spec") or {},
            "answer_contract": public_task.get("answer_contract") or {},
            "listed_candidates": extract_task_candidates(public_task),
        },
        "raw_answer": raw_answer,
        "trace_bundle": _trace_bundle(result_row),
        "evidence": evidence,
    }


def _fallback_rewrite_prompt(bundle: Dict[str, Any], guidance: Sequence[str]) -> str:
    return f"""# Role
You are a benchmark answer adapter. Improve task fulfillment and output alignment.

# Inputs
- Method: {bundle['method']}
- Public task: {json.dumps(bundle['public_task'], ensure_ascii=False, indent=2)}
- Family-specific guidance: {json.dumps(list(guidance), ensure_ascii=False)}
- Raw answer: {bundle['raw_answer']}
- Evidence snippets: {json.dumps(bundle['evidence'], ensure_ascii=False, indent=2)}

# Rules
- Use only the raw answer and evidence.
- Do not add unsupported claims.
- Make the answer benchmark-facing, concrete, and concise.
- If the task is a ranking task, keep the candidate labels faithful to the task wording.

# Output JSON
{{
  "adapted_answer": "...",
  "preserved_core_points": ["..."],
  "omitted_or_softened_points": ["..."]
}}
"""


def _strong_renderer_prompt(bundle: Dict[str, Any], guidance: Sequence[str], family: str) -> str:
    listed_candidates = bundle["public_task"].get("listed_candidates") or []
    return f"""# Role
You are the final benchmark-facing renderer for an offline research benchmark.

# Mission
Produce the strongest final answer you can from the task, structured intermediate reasoning, and retrieved evidence.

# Authority
- You may revise, replace, or sharpen the raw answer if a better-grounded conclusion is supported by the structured packet and evidence.
- Do NOT preserve a weak draft merely because it appeared in the raw answer.

# Hard Rules
- Use only the provided evidence and structured packet.
- Do not introduce new papers, dates, claims, or directions that are absent from the provided materials.
- Prefer structured intermediate reasoning over the raw answer when they conflict.
- Use evidence ids exactly as provided.
- Keep the answer benchmark-facing rather than conversational.

# Family Requirements
{json.dumps(list(guidance), ensure_ascii=False, indent=2)}

# Public Task
{json.dumps(bundle['public_task'], ensure_ascii=False, indent=2)}

# Raw Answer
{bundle['raw_answer']}

# Structured Packet
{_clip_json(bundle['trace_bundle'])}

# Evidence
{_clip_json(bundle['evidence'])}

# Ranking Contract
- If `listed_candidates` is non-empty, output a complete ranking over those exact labels only.
- Every listed candidate must appear exactly once.
- Do not collapse, paraphrase away, or replace listed candidates with umbrella labels.

# Output Schema
- For `bottleneck_opportunity_discovery`:
{{
  "bottleneck": "...",
  "opportunity": "...",
  "historical_basis": "...",
  "linkage": "...",
  "why_now": "...",
  "evidence_ids": ["P1", "P2"]
}}
- For `direction_forecasting`:
{{
  "trajectory_label": "accelerating|steady|cooling|fragmenting",
  "primary_direction": "...",
  "supporting_directions": ["..."],
  "historical_basis": "...",
  "why_now": "...",
  "evidence_ids": ["P1", "P2"]
}}
- For `strategic_research_planning`:
{{
  "ranked_items": [
    {{
      "rank": 1,
      "direction": "...",
      "why_now": "...",
      "dependency_or_tradeoff": "...",
      "evidence_ids": ["P1", "P2"]
    }}
  ],
  "overall_rationale": "...",
  "first_milestone": "...",
  "defer_rationale": "...",
  "risk_or_kill_criterion": "..."
}}
- For `venue_aware_research_positioning`:
{{
  "ranked_items": [
    {{
      "rank": 1,
      "direction": "...",
      "technical_rationale": "...",
      "venue_fit_rationale": "...",
      "evidence_ids": ["P1", "P2"]
    }}
  ],
  "contribution_package": "...",
  "evaluation_signature": "...",
  "contrast_or_avoid": "..."
}}

Return exactly one JSON object using the schema for family `{family}`.
"""


def _title_map(evidence: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in evidence:
        evidence_id = _clean_text(item.get("evidence_id"))
        if evidence_id:
            out[evidence_id] = item
    return out


def _format_citations(evidence_ids: Sequence[str]) -> str:
    ids = [str(x).strip() for x in evidence_ids if str(x).strip()]
    return f" [{', '.join(ids)}]" if ids else ""


def _format_refs(evidence_ids: Sequence[str], evidence_map: Dict[str, Dict[str, Any]]) -> str:
    refs = []
    for evidence_id in evidence_ids[:4]:
        item = evidence_map.get(evidence_id) or {}
        title = _clean_text(item.get("title"))
        if title:
            refs.append(f"[{evidence_id}] {title}")
        else:
            refs.append(f"[{evidence_id}]")
    return "; ".join(refs)


def _ranked_items_from_obj(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for raw in obj.get("ranked_items") or []:
        if isinstance(raw, dict):
            items.append(dict(raw))
    return items


def _fallback_ranked_items(candidates: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for idx, candidate in enumerate(candidates, start=1):
        out.append(
            {
                "rank": idx,
                "direction": candidate,
                "why_now": "Lower priority in the provided reasoning packet than the higher-ranked directions.",
                "dependency_or_tradeoff": "Weaker direct support in the provided evidence than the higher-ranked directions.",
                "technical_rationale": "Lower priority in the provided reasoning packet than the higher-ranked directions.",
                "venue_fit_rationale": "Weaker direct venue-fit support in the provided evidence than the higher-ranked directions.",
                "evidence_ids": [],
            }
        )
    return out


def _normalize_ranked_items(
    obj: Dict[str, Any],
    *,
    candidates: Sequence[str],
    evidence_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    raw_items = _ranked_items_from_obj(obj)
    seen = set()
    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        direction = _align_to_candidates(str(item.get("direction") or ""), candidates)
        if not direction:
            continue
        key = _norm(direction)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "rank": len(normalized) + 1,
                "direction": direction,
                "why_now": _clean_sentence(item.get("why_now")),
                "dependency_or_tradeoff": _clean_sentence(item.get("dependency_or_tradeoff")),
                "technical_rationale": _clean_sentence(item.get("technical_rationale")),
                "venue_fit_rationale": _clean_sentence(item.get("venue_fit_rationale")),
                "evidence_ids": _sanitize_evidence_ids(item.get("evidence_ids") or [], evidence_map),
            }
        )
    if candidates:
        if not normalized:
            normalized = _fallback_ranked_items(candidates)
        missing = [candidate for candidate in candidates if _norm(candidate) not in {_norm(item["direction"]) for item in normalized}]
        for candidate in missing:
            normalized.append(
                {
                    "rank": len(normalized) + 1,
                    "direction": candidate,
                    "why_now": "Lower priority in the provided reasoning packet than the higher-ranked directions.",
                    "dependency_or_tradeoff": "Weaker direct support in the provided evidence than the higher-ranked directions.",
                    "technical_rationale": "Lower priority in the provided reasoning packet than the higher-ranked directions.",
                    "venue_fit_rationale": "Weaker direct venue-fit support in the provided evidence than the higher-ranked directions.",
                    "evidence_ids": list(evidence_map.keys())[:2],
                }
            )
        normalized = sorted(normalized, key=lambda row: int(row.get("rank") or 9999))[: len(candidates)]
        for idx, item in enumerate(normalized, start=1):
            item["rank"] = idx
    return normalized


def _render_bottleneck(obj: Dict[str, Any], evidence_map: Dict[str, Dict[str, Any]]) -> str:
    evidence_ids = _sanitize_evidence_ids(obj.get("evidence_ids") or [], evidence_map)
    citations = _format_citations(evidence_ids)
    refs = _format_refs(evidence_ids, evidence_map)
    parts = [
        f"The key bottleneck is {_clean_sentence(obj.get('bottleneck'))}.",
        f"If that bottleneck is resolved, the clearest immediate opportunity is {_clean_sentence(obj.get('opportunity'))}.",
        f"Historical basis: {_clean_sentence(obj.get('historical_basis'))}{citations}.",
        f"Linkage: {_clean_sentence(obj.get('linkage'))}.",
    ]
    why_now = _clean_sentence(obj.get("why_now"))
    if why_now:
        parts.append(f"Why now: {why_now}{citations}.")
    if refs:
        parts.append(f"Evidence: {refs}.")
    return " ".join(part for part in parts if part and part.strip())


def _render_direction(obj: Dict[str, Any], evidence_map: Dict[str, Dict[str, Any]]) -> str:
    evidence_ids = _sanitize_evidence_ids(obj.get("evidence_ids") or [], evidence_map)
    citations = _format_citations(evidence_ids)
    refs = _format_refs(evidence_ids, evidence_map)
    trajectory = _clean_text(obj.get("trajectory_label")).lower()
    if trajectory not in TRAJECTORY_LABELS:
        trajectory = "steady"
    supporting = [_clean_text(x) for x in (obj.get("supporting_directions") or []) if _clean_text(x)]
    parts = [
        f"The trajectory is {trajectory}, and the most likely next direction is {_clean_sentence(obj.get('primary_direction'))}.",
        f"Historical basis: {_clean_sentence(obj.get('historical_basis'))}{citations}.",
        f"Why now: {_clean_sentence(obj.get('why_now'))}{citations}.",
    ]
    if supporting:
        parts.append(f"Supporting signals: {'; '.join(supporting[:2])}.")
    if refs:
        parts.append(f"Evidence: {refs}.")
    return " ".join(part for part in parts if part and part.strip())


def _render_planning(obj: Dict[str, Any], evidence_map: Dict[str, Dict[str, Any]], candidates: Sequence[str]) -> str:
    items = _normalize_ranked_items(obj, candidates=candidates, evidence_map=evidence_map)
    lines = []
    for item in items:
        citations = _format_citations(item.get("evidence_ids") or [])
        lines.append(
            f"{item['rank']}. {item['direction']} - Why now: {item.get('why_now') or 'Most directly supported by the provided evidence.'} "
            f"Dependency/trade-off: {item.get('dependency_or_tradeoff') or 'Depends on the higher-ranked prerequisites and evidence-supported bottlenecks.'}{citations}."
        )
    overall = _clean_sentence(obj.get("overall_rationale"))
    first_milestone = _clean_sentence(obj.get("first_milestone"))
    defer_rationale = _clean_sentence(obj.get("defer_rationale"))
    risk = _clean_sentence(obj.get("risk_or_kill_criterion"))
    ref_ids: List[str] = []
    for item in items:
        for evidence_id in item.get("evidence_ids") or []:
            if evidence_id not in ref_ids:
                ref_ids.append(evidence_id)
    refs = _format_refs(ref_ids, evidence_map)
    trailer = []
    if overall:
        trailer.append(f"Overall rationale: {overall}.")
    if first_milestone:
        trailer.append(f"First milestone: {first_milestone}.")
    if defer_rationale:
        trailer.append(f"Defer rationale: {defer_rationale}.")
    if risk:
        trailer.append(f"Risk/kill criterion: {risk}.")
    if refs:
        trailer.append(f"Evidence: {refs}.")
    return "\n".join(lines + trailer)


def _render_venue(obj: Dict[str, Any], evidence_map: Dict[str, Dict[str, Any]], candidates: Sequence[str]) -> str:
    items = _normalize_ranked_items(obj, candidates=candidates, evidence_map=evidence_map)
    lines = []
    for item in items:
        citations = _format_citations(item.get("evidence_ids") or [])
        lines.append(
            f"{item['rank']}. {item['direction']} - Technical rationale: {item.get('technical_rationale') or 'Best aligned with the evidence-supported technical momentum.'} "
            f"Venue fit: {item.get('venue_fit_rationale') or 'Most aligned with the target venue preferences visible in the provided evidence.'}{citations}."
        )
    package = _clean_sentence(obj.get("contribution_package"))
    evaluation = _clean_sentence(obj.get("evaluation_signature"))
    contrast = _clean_sentence(obj.get("contrast_or_avoid"))
    ref_ids: List[str] = []
    for item in items:
        for evidence_id in item.get("evidence_ids") or []:
            if evidence_id not in ref_ids:
                ref_ids.append(evidence_id)
    refs = _format_refs(ref_ids, evidence_map)
    trailer = []
    if package:
        trailer.append(f"Contribution package: {package}.")
    if evaluation:
        trailer.append(f"Evaluation signature: {evaluation}.")
    if contrast:
        trailer.append(f"Avoid: {contrast}.")
    if refs:
        trailer.append(f"Evidence: {refs}.")
    return "\n".join(lines + trailer)


def _render_answer_from_payload(
    *,
    family: str,
    obj: Dict[str, Any],
    evidence: Sequence[Dict[str, Any]],
    candidates: Sequence[str],
) -> str:
    evidence_map = _title_map(evidence)
    if family == "bottleneck_opportunity_discovery":
        return _render_bottleneck(obj, evidence_map)
    if family == "direction_forecasting":
        return _render_direction(obj, evidence_map)
    if family == "strategic_research_planning":
        return _render_planning(obj, evidence_map, candidates)
    if family == "venue_aware_research_positioning":
        return _render_venue(obj, evidence_map, candidates)
    return _clean_text(obj.get("adapted_answer"))


def _adapt_with_strong_renderer(
    client: OpenAICompatChatClient,
    *,
    bundle: Dict[str, Any],
    family: str,
) -> Dict[str, Any]:
    prompt = _strong_renderer_prompt(bundle, _family_guidance(family), family)
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict benchmark answer renderer. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=2200,
        timeout=180,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Return exactly one valid JSON object following the family-specific schema.",
    )
    adapted_answer = _render_answer_from_payload(
        family=family,
        obj=obj,
        evidence=bundle["evidence"],
        candidates=bundle["public_task"].get("listed_candidates") or [],
    )
    return {
        "adapted_answer": adapted_answer,
        "structured_payload": obj,
        "renderer_mode": "strong_family_renderer",
    }


def _adapt_with_fallback_rewrite(
    client: OpenAICompatChatClient,
    *,
    bundle: Dict[str, Any],
    family: str,
) -> Dict[str, Any]:
    prompt = _fallback_rewrite_prompt(bundle, _family_guidance(family))
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict answer adapter for a research benchmark. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1200,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction="Return exactly one valid JSON object with keys adapted_answer, preserved_core_points, omitted_or_softened_points.",
    )
    return {
        "adapted_answer": normalize_ws(obj.get("adapted_answer") or bundle["raw_answer"]),
        "structured_payload": obj,
        "renderer_mode": "fallback_rewrite",
    }


def adapt_answer(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
) -> Dict[str, Any]:
    bundle = build_adapter_bundle(public_task, result_row)
    family = str(public_task.get("family") or result_row.get("family") or "")
    if bundle["trace_bundle"]:
        adapted = _adapt_with_strong_renderer(client, bundle=bundle, family=family)
    else:
        adapted = _adapt_with_fallback_rewrite(client, bundle=bundle, family=family)
    return {
        "adapted_answer": adapted["adapted_answer"],
        "structured_payload": adapted.get("structured_payload") or {},
        "renderer_mode": adapted.get("renderer_mode") or "unknown",
        "input_bundle": bundle,
    }


def shared_adapter_name() -> str:
    return "shared_family_final_renderer_v2"


def has_shared_final_adapter(result_row: Dict[str, Any]) -> bool:
    adapter = result_row.get("adapter") or {}
    return str(adapter.get("name") or "").strip() == shared_adapter_name()


def apply_shared_final_adapter(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
) -> Dict[str, Any]:
    if has_shared_final_adapter(result_row):
        return dict(result_row)
    adapted = adapt_answer(client, public_task=public_task, result_row=result_row)
    out_row = dict(result_row)
    out_row["raw_answer"] = str(result_row.get("raw_answer") or result_row.get("answer") or "")
    out_row["answer"] = adapted["adapted_answer"]
    out_row["adapter"] = {
        "name": shared_adapter_name(),
        "renderer_mode": adapted.get("renderer_mode"),
        "structured_payload": adapted.get("structured_payload") or {},
        "input_bundle": adapted.get("input_bundle") or {},
    }
    return out_row


def apply_shared_final_adapter_to_trace_result(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    trace_result: Dict[str, Any],
) -> Dict[str, Any]:
    if has_shared_final_adapter(trace_result):
        return dict(trace_result)
    seed_row = {
        "task_id": trace_result.get("task_id") or public_task.get("task_id"),
        "family": trace_result.get("family") or public_task.get("family"),
        "domain": trace_result.get("domain") or public_task.get("domain"),
        "title": trace_result.get("title") or public_task.get("title"),
        "question": trace_result.get("question") or public_task.get("question"),
        "time_cutoff": trace_result.get("time_cutoff") or public_task.get("time_cutoff"),
        "answer": trace_result.get("answer") or "",
        "evidence": trace_result.get("evidence"),
        "trace": trace_result,
    }
    adapted_row = apply_shared_final_adapter(client, public_task=public_task, result_row=seed_row)
    out = dict(trace_result)
    out["raw_answer"] = str(adapted_row.get("raw_answer") or seed_row["answer"] or "")
    out["answer"] = str(adapted_row.get("answer") or "")
    out["adapter"] = adapted_row.get("adapter") or {}
    return out
