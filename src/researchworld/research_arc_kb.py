from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import (
    OfflineKnowledgeBase,
    clip_text,
    dedupe,
    merge_multi_query_results,
)


STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "over", "under", "based", "using", "available", "before",
    "after", "through", "within", "large", "language", "model", "models", "systems", "system", "research",
    "trajectory", "forecasting", "forecast", "strategic", "planning", "agenda", "discovery", "opportunity",
    "bottleneck", "evaluation", "frameworks", "framework", "methods", "method", "tasks", "task", "domain",
    "domains", "specific", "period", "literature", "identify", "concrete", "change", "direction", "directions",
    "next", "quarter", "post", "training", "agentic", "agents", "retrieval", "augmented", "generation", "rag",
    "llm", "llms", "knowledge", "available", "future", "likely", "would",
}


TITLE_PATTERNS = [
    r"^Bottleneck and Opportunity Discovery in\s+",
    r"^Bottleneck and Opportunity Discovery for\s+",
    r"^Forecasting the Trajectory of\s+",
    r"^Forecasting Research Trajectory in\s+",
    r"^Forecasting Research Trajectory for\s+",
    r"^Forecasting Trajectory and Subdirections in\s+",
    r"^Forecast for\s+",
    r"^Strategic Research Planning for\s+",
    r"^Strategic Research Agenda for\s+",
]


def norm_tokens(text: Any) -> List[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", str(text or "").lower())
    return [t for t in raw if t not in STOPWORDS]


def extract_focus_text(task: Dict[str, Any]) -> str:
    title = str(task.get("title") or "").strip()
    text = title
    for pattern in TITLE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    if text and text != title:
        return text
    question = str(task.get("question") or "").strip()
    m = re.search(
        r"(?:within|for|on)\s+(.+?)(?:\s+over the next|\s+for the next|\s+considering|\.|$)",
        question,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else (title or question)


def render_rows(rows: List[Dict[str, Any]], *, limit: int = 8) -> str:
    parts = []
    for row in rows[:limit]:
        head = [f"[{row.get('evidence_id')}] {row.get('paper_title') or row.get('title') or ''}"]
        if row.get("section_title"):
            head.append(f"section={row['section_title']}")
        if row.get("venue"):
            head.append(f"venue={row['venue']}")
        if row.get("citations") is not None:
            head.append(f"citations={row['citations']}")
        parts.append(" | ".join(head))
        if row.get("problem_statement"):
            parts.append(f"Problem: {row['problem_statement']}")
        if row.get("limitations"):
            parts.append(f"Limitations: {'; '.join(row['limitations'])}")
        if row.get("future_work"):
            parts.append(f"Future work: {'; '.join(row['future_work'])}")
        if row.get("core_ideas"):
            parts.append(f"Core ideas: {'; '.join(row['core_ideas'])}")
        if row.get("snippet"):
            parts.append(row["snippet"])
        parts.append("")
    return "\n".join(parts).strip()


class ResearchArcKB:
    def __init__(
        self,
        *,
        kb: OfflineKnowledgeBase,
        answer_client: OpenAICompatChatClient,
        critic_client: Optional[OpenAICompatChatClient] = None,
    ):
        self.kb = kb
        self.answer_client = answer_client
        self.critic_client = critic_client or answer_client

    def build_queries(self, task: Dict[str, Any]) -> List[str]:
        focus = extract_focus_text(task)
        family = str(task.get("family") or "")
        queries = [task.get("question") or "", task.get("title") or "", focus]
        focus_terms = norm_tokens(focus)[:8]
        if focus_terms:
            queries.append(" ".join(focus_terms))
        if family == "bottleneck_opportunity_discovery":
            queries += [
                f"{focus} limitation bottleneck challenge",
                f"{focus} future work opportunity",
                f"{focus} evaluation limitation long-term challenge",
            ]
        elif family == "direction_forecasting":
            queries += [
                f"{focus} benchmark evaluation trend",
                f"{focus} emerging direction",
                f"{focus} survey recent methods",
            ]
        elif family == "strategic_research_planning":
            queries += [
                f"{focus} open problem method evaluation",
                f"{focus} limitation future work benchmark",
                f"{focus} top venue citations benchmark",
            ]
        return dedupe([str(x) for x in queries if str(x or "").strip()])[:8]

    def gather_evidence(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        domain = self.kb.domain(domain_id)
        queries = self.build_queries(task)

        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        paper_hits = merge_multi_query_results(domain.paper_retriever(cutoff_date=cutoff_date), queries, top_k_per_query=8, limit=12)
        paper_rows = []
        paper_ids = []
        for idx, (doc, scores) in enumerate(paper_hits, start=1):
            paper = domain.get_paper(doc.paper_id) or {}
            pub = paper.get("publication") or {}
            paper_ids.append(doc.paper_id)
            paper_rows.append(
                {
                    "evidence_id": f"P{idx}",
                    "paper_id": doc.paper_id,
                    "paper_title": doc.title,
                    "published_date": paper.get("published_date"),
                    "venue": pub.get("venue_name"),
                    "citations": pub.get("citation_count"),
                    "is_top_ai_venue": pub.get("is_top_ai_venue"),
                    "snippet": clip_text(doc.text, 1400),
                    "scores": scores,
                }
            )

        structure_hits = merge_multi_query_results(
            domain.structure_retriever(cutoff_date=cutoff_date),
            queries,
            top_k_per_query=6,
            limit=8,
        )
        structure_rows = []
        for idx, (doc, scores) in enumerate(structure_hits, start=1):
            if doc.paper_id not in paper_ids:
                paper_ids.append(doc.paper_id)
            structure_rows.append(
                {
                    "evidence_id": f"T{idx}",
                    "paper_id": doc.paper_id,
                    "paper_title": doc.title,
                    "problem_statement": clip_text(doc.meta.get("problem_statement"), 280),
                    "limitations": list(doc.meta.get("limitations") or [])[:4],
                    "future_work": list(doc.meta.get("future_work") or [])[:4],
                    "core_ideas": list(doc.meta.get("core_ideas") or [])[:4],
                    "snippet": clip_text(doc.text, 1000),
                    "scores": scores,
                }
            )

        section_hits = merge_multi_query_results(
            domain.section_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids[:10]),
            queries,
            top_k_per_query=8,
            limit=10,
        )
        section_rows = []
        for idx, (doc, scores) in enumerate(section_hits, start=1):
            section_rows.append(
                {
                    "evidence_id": f"S{idx}",
                    "paper_id": doc.paper_id,
                    "paper_title": doc.meta.get("paper_title") or doc.title,
                    "section_title": doc.meta.get("section_title"),
                    "snippet": clip_text(doc.text, 1000),
                    "scores": scores,
                }
            )

        summary = self._summarize_retrieved_papers(paper_rows, structure_rows)
        return {
            "queries": queries,
            "paper_evidence": paper_rows,
            "structure_evidence": structure_rows,
            "section_evidence": section_rows,
            "summary": summary,
        }

    def _summarize_retrieved_papers(
        self,
        paper_rows: List[Dict[str, Any]],
        structure_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        venue_counts: Dict[str, int] = {}
        top_venue_count = 0
        citations: List[float] = []
        for row in paper_rows:
            venue = str(row.get("venue") or "").strip()
            if venue:
                venue_counts[venue] = venue_counts.get(venue, 0) + 1
            if row.get("is_top_ai_venue"):
                top_venue_count += 1
            if row.get("citations") is not None:
                try:
                    citations.append(float(row.get("citations")))
                except Exception:
                    pass
        top_limitations = []
        top_future_work = []
        for row in structure_rows:
            top_limitations.extend(row.get("limitations") or [])
            top_future_work.extend(row.get("future_work") or [])
        return {
            "retrieved_paper_count": len(paper_rows),
            "retrieved_structure_count": len(structure_rows),
            "top_venue_share_in_retrieved": round(top_venue_count / len(paper_rows), 4) if paper_rows else 0.0,
            "top_venues": sorted(venue_counts.items(), key=lambda x: (-x[1], x[0]))[:5],
            "citation_max": max(citations) if citations else 0.0,
            "citation_median_proxy": sorted(citations)[len(citations) // 2] if citations else 0.0,
            "frequent_limitations": dedupe(top_limitations)[:8],
            "frequent_future_work": dedupe(top_future_work)[:8],
        }

    def run_task(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        evidence = self.gather_evidence(task=task, domain_id=domain_id)
        family = str(task.get("family") or "")
        if family == "direction_forecasting":
            head = self._run_direction(task=task, evidence=evidence)
        elif family == "strategic_research_planning":
            head = self._run_planning(task=task, evidence=evidence)
        else:
            head = self._run_bottleneck(task=task, evidence=evidence)
        return {
            "queries": evidence["queries"],
            "summary": evidence["summary"],
            "evidence": evidence,
            "head_result": head,
            "answer": head.get("answer") or "",
        }

    def _run_direction(self, *, task: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are answering an offline research benchmark with a frozen historical KB.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Retrieved summary:
{json.dumps(evidence['summary'], ensure_ascii=False, indent=2)}

Paper evidence:
{render_rows(evidence['paper_evidence'], limit=8)}

Structure evidence:
{render_rows(evidence['structure_evidence'], limit=6)}

Section evidence:
{render_rows(evidence['section_evidence'], limit=6)}

Return JSON:
{{
  "trajectory_label": "accelerating|steady|cooling|fragmenting",
  "subdirections": ["...", "..."],
  "venue_or_evaluation_shift": "...",
  "evidence_linkage": "...",
  "final_answer": "..."
}}

Rules:
- Use only the retrieved historical evidence.
- Prefer fragmenting only when the retrieved evidence points to multiple concrete emerging variants.
- Mention at most two subdirections.
- Keep the answer compact, technical, and ex ante.
"""
        draft = self._complete_json(self.answer_client, prompt, system="You are a precise research forecasting model. Output JSON only.")
        repair = f"""Critique and repair the forecast.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Retrieved summary:
{json.dumps(evidence['summary'], ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Evidence:
{render_rows(evidence['paper_evidence'], limit=6)}

Return JSON with the same schema. Fix vague subdirections and unsupported claims."""
        final = self._complete_json(self.critic_client, repair, system="You are a strict benchmark evaluator. Output JSON only.")
        return {"draft": draft, "final": final, "answer": str(final.get("final_answer") or draft.get("final_answer") or "").strip()}

    def _run_bottleneck(self, *, task: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are answering an offline research benchmark with a frozen historical KB.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Retrieved summary:
{json.dumps(evidence['summary'], ensure_ascii=False, indent=2)}

Structure evidence:
{render_rows(evidence['structure_evidence'], limit=8)}

Section evidence:
{render_rows(evidence['section_evidence'], limit=6)}

Paper evidence:
{render_rows(evidence['paper_evidence'], limit=6)}

Return JSON:
{{
  "bottleneck": "...",
  "opportunity": "...",
  "evidence_titles": ["...", "..."],
  "linkage": "...",
  "final_answer": "..."
}}

Rules:
- The bottleneck must be a specific unresolved technical constraint.
- The opportunity must directly follow from that bottleneck.
- Reuse natural language from the evidence when possible.
- Keep the final answer compact, technical, and ex ante.
"""
        draft = self._complete_json(self.answer_client, prompt, system="You are a precise research synthesis model. Output JSON only.")
        repair = f"""Critique and repair the bottleneck-opportunity answer.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Evidence:
{render_rows(evidence['structure_evidence'], limit=8)}

Return JSON with the same schema. Reject vague bottlenecks and generic opportunities."""
        final = self._complete_json(self.critic_client, repair, system="You are a strict benchmark evaluator. Output JSON only.")
        return {"draft": draft, "final": final, "answer": str(final.get("final_answer") or draft.get("final_answer") or "").strip()}

    def _run_planning(self, *, task: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are answering an offline research benchmark with a frozen historical KB.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Retrieved summary:
{json.dumps(evidence['summary'], ensure_ascii=False, indent=2)}

Paper evidence:
{render_rows(evidence['paper_evidence'], limit=8)}

Structure evidence:
{render_rows(evidence['structure_evidence'], limit=8)}

Section evidence:
{render_rows(evidence['section_evidence'], limit=4)}

Return JSON:
{{
  "target_problem": "...",
  "method_angle": "...",
  "evaluation_plan": "...",
  "why_now": "...",
  "milestones": ["...", "..."],
  "final_answer": "..."
}}

Rules:
- Propose one focused agenda, not a list of unrelated ideas.
- Ground target problem and method angle in the retrieved evidence.
- Make the evaluation plan concrete.
- Keep the final answer compact and strategic.
"""
        draft = self._complete_json(self.answer_client, prompt, system="You are a precise research planning model. Output JSON only.")
        repair = f"""Critique and repair the planning answer.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Evidence:
{render_rows(evidence['structure_evidence'], limit=8)}

Return JSON with the same schema. Reject diffuse agendas and weak evaluation plans."""
        final = self._complete_json(self.critic_client, repair, system="You are a strict benchmark evaluator. Output JSON only.")
        return {"draft": draft, "final": final, "answer": str(final.get("final_answer") or draft.get("final_answer") or "").strip()}

    @staticmethod
    def _complete_json(client: OpenAICompatChatClient, prompt: str, *, system: str) -> Dict[str, Any]:
        return complete_json_object(
            client,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            transport_retries=2,
            max_parse_attempts=3,
        )
