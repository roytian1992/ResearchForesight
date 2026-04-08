from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import clip_text, normalize_ws


def _family_guidance(family: str) -> List[str]:
    if family == "bottleneck_opportunity_discovery":
        return [
            "State one concrete unresolved bottleneck.",
            "State one concrete downstream opportunity enabled by addressing that bottleneck.",
            "Explicitly explain the linkage between the bottleneck and the opportunity.",
        ]
    if family == "direction_forecasting":
        return [
            "Make a concrete trajectory call rather than only summarizing background.",
            "Name one to three concrete emerging directions or shifts.",
            "Explain why the historical evidence supports that trajectory call.",
        ]
    if family == "strategic_research_planning":
        return [
            "Return a prioritized plan, not an unranked brainstorm.",
            "Each priority should be concrete and justified.",
            "Include why-now logic and executable emphasis when supported by the input.",
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
        or row.get("problem_statement")
        or row.get("description")
        or ""
    )
    if not snippet:
        pieces = []
        limitations = row.get("limitations") or []
        future_work = row.get("future_work") or []
        core_ideas = row.get("core_ideas") or []
        if limitations:
            pieces.append("Limitations: " + "; ".join(str(x) for x in limitations[:3] if str(x).strip()))
        if future_work:
            pieces.append("Future work: " + "; ".join(str(x) for x in future_work[:3] if str(x).strip()))
        if core_ideas:
            pieces.append("Core ideas: " + "; ".join(str(x) for x in core_ideas[:3] if str(x).strip()))
        snippet = " ".join(pieces)
    return {
        "evidence_id": evidence_id,
        "title": normalize_ws(title),
        "snippet": clip_text(snippet, 500),
        "paper_id": row.get("paper_id"),
    }


def extract_adapter_evidence(result_row: Dict[str, Any], *, limit: int = 10) -> List[Dict[str, Any]]:
    evidence_rows: List[Dict[str, Any]] = []
    top_evidence = result_row.get("evidence")
    if isinstance(top_evidence, dict):
        retrieved = top_evidence.get("retrieved") or []
        for item in retrieved:
            if isinstance(item, dict):
                evidence_rows.append(item)
    trace = result_row.get("trace") or {}
    trace_evidence = trace.get("evidence") or {}
    if isinstance(trace_evidence, dict):
        for key in ["papers", "fulltext", "structures", "pageindex", "paper_evidence", "section_evidence", "structure_evidence", "candidate_node_evidence"]:
            for item in (trace_evidence.get(key) or []):
                if isinstance(item, dict):
                    evidence_rows.append(item)
    formatted = []
    seen = set()
    for idx, row in enumerate(evidence_rows, start=1):
        norm = _normalize_evidence_item(row, str(row.get("evidence_id") or f"E{idx}"))
        key = (str(norm.get("paper_id") or ""), norm["title"], norm["snippet"])
        if key in seen:
            continue
        if not norm["title"] and not norm["snippet"]:
            continue
        seen.add(key)
        formatted.append(norm)
        if len(formatted) >= limit:
            break
    return formatted


def build_adapter_bundle(public_task: Dict[str, Any], result_row: Dict[str, Any]) -> Dict[str, Any]:
    method = str(result_row.get("agent") or result_row.get("baseline") or "unknown")
    evidence = extract_adapter_evidence(result_row)
    raw_answer = str(result_row.get("raw_answer") or result_row.get("answer") or "").strip()
    bundle = {
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
        },
        "raw_answer": raw_answer,
        "evidence": evidence,
    }
    return bundle


def adapt_answer(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    result_row: Dict[str, Any],
) -> Dict[str, Any]:
    bundle = build_adapter_bundle(public_task, result_row)
    family = str(public_task.get("family") or "")
    guidance = _family_guidance(family)
    prompt = f"""# Role
You are a benchmark answer adapter. Your job is to improve task fulfillment and output alignment without changing the underlying method's substantive position.

# Mission
Rewrite the candidate answer into a cleaner benchmark-facing final answer that better matches the public task.

# Hard Constraints
- Do NOT introduce new papers, dates, metrics, claims, or conclusions that are absent from the raw answer and the provided evidence snippets.
- Do NOT use external knowledge.
- Do NOT strengthen weak claims beyond what the raw answer supports.
- If the raw answer is missing a required element, only infer it when strongly implied by the raw answer or evidence; otherwise state uncertainty or incompleteness explicitly.
- Preserve the method's core conclusion whenever possible.

# Inputs
- Method: {bundle['method']}
- Public task: {json.dumps(bundle['public_task'], ensure_ascii=False, indent=2)}
- Family-specific guidance: {json.dumps(guidance, ensure_ascii=False)}
- Raw answer: {bundle['raw_answer']}
- Evidence snippets: {json.dumps(bundle['evidence'], ensure_ascii=False, indent=2)}

# Adaptation goals
1. Improve task-family fit.
2. Make the answer more concrete and better structured.
3. Remove generic filler and unsupported overreach.
4. Preserve uncertainty when evidence is weak.

# Additional style requirements
- Output natural benchmark answer prose, not JSON-like fragments.
- Be concise but substantive.
- Prefer one compact paragraph for bottleneck/direction tasks.
- Prefer a short ranked list for planning tasks if ranking is requested or strongly implied.

# Output (Strict JSON)
{{
  "adapted_answer": "...",
  "preserved_core_points": ["..."],
  "omitted_or_softened_points": ["..."]
}}
"""
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
    adapted_answer = normalize_ws(obj.get("adapted_answer") or bundle["raw_answer"])
    return {
        "adapted_answer": adapted_answer,
        "preserved_core_points": [str(x) for x in (obj.get("preserved_core_points") or []) if str(x).strip()],
        "omitted_or_softened_points": [str(x) for x in (obj.get("omitted_or_softened_points") or []) if str(x).strip()],
        "input_bundle": bundle,
    }


def shared_adapter_name() -> str:
    return "shared_final_adapter_v1"


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
        "preserved_core_points": adapted.get("preserved_core_points") or [],
        "omitted_or_softened_points": adapted.get("omitted_or_softened_points") or [],
        "input_bundle": adapted.get("input_bundle") or {},
    }
    return out_row
