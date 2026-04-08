from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import HybridRetriever, OfflineKnowledgeBase, RetrievalDoc, clip_text, dedupe, merge_multi_query_results, normalize_ws
from researchworld.research_arc_kb import extract_focus_text


ROOT = Path(__file__).resolve().parents[2]


def _humanize_label(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = raw.split("/")[-1]
    raw = raw.replace("__", " / ")
    raw = raw.replace("_", " ")
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _packet_text(packet: Dict[str, Any]) -> str:
    parts: List[str] = [
        f"Display name: {_humanize_label(packet.get('display_name') or '')}",
        f"Description: {packet.get('description') or ''}",
        f"Dimension: {_humanize_label(packet.get('dimension_id') or '')}",
        f"Node path: {_humanize_label(packet.get('node_id') or '')}",
    ]
    rep_titles = [str(x.get("title") or "") for x in (packet.get("historical_representative_papers") or [])[:8]]
    if rep_titles:
        parts.append("Representative papers: " + "; ".join(rep_titles))
    descendants = [str((x or {}).get("display_name") or "") for x in (packet.get("emergent_descendants") or [])[:8] if str((x or {}).get("display_name") or "").strip()]
    if descendants:
        parts.append("Emergent descendants: " + "; ".join(_humanize_label(x) for x in descendants))
    if packet.get("split_pressure") is not None:
        parts.append(f"Split pressure: {packet.get('split_pressure')}")
    top_limits = [str((x or {}).get("name") or "") for x in (packet.get("top_limitations") or [])[:6] if str((x or {}).get("name") or "").strip()]
    if top_limits:
        parts.append("Historical bottlenecks: " + "; ".join(top_limits))
    top_future = [str((x or {}).get("direction") or (x or {}).get("name") or "") for x in (packet.get("top_future_work") or [])[:6]]
    top_future = [x for x in top_future if x.strip()]
    if top_future:
        parts.append("Historical future work: " + "; ".join(top_future))
    return "\n".join(part for part in parts if normalize_ws(part))


def _paper_text(paper: Dict[str, Any], structure: Optional[Dict[str, Any]] = None) -> str:
    pub = paper.get("publication") or {}
    limitations = [
        str(x.get("name") or "")
        for x in ((structure or {}).get("explicit_limitations") or [])
        if str(x.get("name") or "").strip()
    ][:4]
    future = [
        str(x.get("direction") or "")
        for x in ((structure or {}).get("future_work") or [])
        if str(x.get("direction") or "").strip()
    ][:4]
    return "\n".join(
        part
        for part in [
            f"Title: {paper.get('title') or ''}",
            f"Published: {paper.get('published_date') or paper.get('published') or ''}",
            f"Venue: {pub.get('venue_name') or ''}",
            f"Abstract: {paper.get('abstract') or ''}",
            f"Problem statement: {(structure or {}).get('problem_statement') or ''}",
            f"Limitations: {'; '.join(limitations)}",
            f"Future work: {'; '.join(future)}",
        ]
        if normalize_ws(part)
    )


def _focus_terms(task: Dict[str, Any], packets: Iterable[Dict[str, Any]]) -> List[str]:
    text_parts = [str(task.get("title") or ""), str(task.get("question") or "")]
    for packet in packets:
        text_parts.extend([_humanize_label(packet.get("display_name") or ""), str(packet.get("description") or "")])
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", " ".join(text_parts).lower())
    stop = {
        "based", "identify", "identifying", "specific", "concrete", "technical", "unresolved", "research",
        "literature", "published", "before", "after", "using", "development", "agents", "agent", "large",
        "language", "model", "models", "framework", "frameworks", "evaluation", "evaluating", "systems",
        "subsequent", "period", "response", "historical", "future", "task", "next", "step", "priority",
        "prioritized", "planning", "pre", "cutoff", "available", "following", "month", "months",
    }
    out: List[str] = []
    seen = set()
    for tok in tokens:
        if tok in stop or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out[:24]


def _focus_score(text: str, focus_terms: Iterable[str]) -> int:
    lower = str(text or "").lower()
    return sum(1 for term in focus_terms if term and term in lower)


def _family_keywords(family: str) -> Dict[str, List[str]]:
    if family == "bottleneck_opportunity_discovery":
        return {
            "query": [
                "limitation",
                "failure mode",
                "error analysis",
                "benchmark",
                "evaluation",
                "challenge",
                "bottleneck",
                "open problem",
            ],
            "signal": [
                "limitation",
                "limitations",
                "failure",
                "fails",
                "error analysis",
                "benchmark",
                "evaluation",
                "challenge",
                "open problem",
                "future work",
            ],
        }
    if family == "direction_forecasting":
        return {
            "query": [
                "trend",
                "trajectory",
                "emerging direction",
                "scaling",
                "benchmark",
                "evaluation",
                "survey",
                "next step",
            ],
            "signal": [
                "trend",
                "trajectory",
                "emerging",
                "survey",
                "benchmark",
                "evaluation",
                "future work",
                "scaling",
                "toward",
                "towards",
            ],
        }
    if family == "strategic_research_planning":
        return {
            "query": [
                "open problem",
                "future work",
                "trade-off",
                "evaluation",
                "benchmark",
                "optimization",
                "scaling",
                "priority",
            ],
            "signal": [
                "open problem",
                "future work",
                "trade-off",
                "tradeoff",
                "evaluation",
                "benchmark",
                "scaling",
                "efficiency",
                "robustness",
                "generalization",
            ],
        }
    return {"query": [], "signal": []}


def _family_signal_score(*, family: str, text: str) -> int:
    lower = str(text or "").lower()
    return sum(1 for kw in _family_keywords(family).get("signal", []) if kw in lower)


@dataclass
class PacketSelection:
    packets: List[Dict[str, Any]]
    retrieval_rows: List[Tuple[RetrievalDoc, Dict[str, Any]]]
    llm_selected_packet_ids: List[str]


@dataclass
class TaskCandidatePool:
    packets: List[Dict[str, Any]]
    packet_retrieval_rows: List[Tuple[RetrievalDoc, Dict[str, Any]]]
    papers: List[Dict[str, Any]]
    paper_scores: Dict[str, Dict[str, Any]]
    packet_ids: List[str]


class CoIOfflineRetrievalAdaptor:
    def __init__(
        self,
        *,
        kb: OfflineKnowledgeBase,
        cheap_client: OpenAICompatChatClient,
        main_client: OpenAICompatChatClient,
        support_root: Path = ROOT / "data" / "support_packets",
    ):
        self.kb = kb
        self.cheap_client = cheap_client
        self.main_client = main_client
        self.support_root = support_root
        self._packet_rows_by_domain: Dict[str, List[Dict[str, Any]]] = {}
        self._packet_retriever_by_domain: Dict[str, HybridRetriever] = {}

    def _load_domain_packets(self, domain_id: str) -> List[Dict[str, Any]]:
        if domain_id not in self._packet_rows_by_domain:
            path = self.support_root / domain_id / "selected_seed_nodes.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            self._packet_rows_by_domain[domain_id] = rows
        return self._packet_rows_by_domain[domain_id]

    def _packet_retriever(self, domain_id: str) -> HybridRetriever:
        if domain_id not in self._packet_retriever_by_domain:
            docs: List[RetrievalDoc] = []
            for row in self._load_domain_packets(domain_id):
                docs.append(
                    RetrievalDoc(
                        doc_id=str(row["packet_id"]),
                        paper_id="",
                        title=_humanize_label(row.get("display_name") or row.get("node_id") or ""),
                        text=_packet_text(row),
                        meta={
                            "packet_id": row.get("packet_id"),
                            "node_id": row.get("node_id"),
                            "dimension_id": row.get("dimension_id"),
                            "description": row.get("description"),
                        },
                    )
                )
            self._packet_retriever_by_domain[domain_id] = HybridRetriever(docs)
        return self._packet_retriever_by_domain[domain_id]

    def task_queries(self, task: Dict[str, Any]) -> List[str]:
        family = str(task.get("family") or "")
        topic = extract_focus_text(task)
        queries = [
            str(task.get("title") or ""),
            str(task.get("question") or ""),
            topic,
        ]
        if family == "bottleneck_opportunity_discovery":
            queries.extend(
                [
                    f"{topic} limitation evaluation bottleneck",
                    f"{topic} benchmark failure mode",
                    f"{topic} error analysis challenge",
                    f"{topic} open problem future work",
                ]
            )
        elif family == "direction_forecasting":
            queries.extend(
                [
                    f"{topic} next step direction",
                    f"{topic} technical trajectory",
                    f"{topic} emerging direction evaluation",
                    f"{topic} benchmark trend survey",
                ]
            )
        elif family == "strategic_research_planning":
            queries.extend(
                [
                    f"{topic} priority research directions",
                    f"{topic} roadmap agenda",
                    f"{topic} open problems trade-offs",
                    f"{topic} future work evaluation bottleneck",
                ]
            )
        queries.extend(f"{topic} {kw}" for kw in _family_keywords(family).get("query", [])[:4])
        return dedupe(queries)

    def _llm_select_packets(self, *, task: Dict[str, Any], packet_rows: List[Dict[str, Any]]) -> List[str]:
        if not packet_rows:
            return []
        family = str(task.get("family") or "")
        max_packets = 3 if family == "strategic_research_planning" else 2
        prompt = {
            "bottleneck_opportunity_discovery": "Pick packets whose core topic is exactly the technical bottleneck/opportunity target, not just broad adjacent agent themes.",
            "direction_forecasting": "Pick packets that best match the exact technical direction being forecast, not broad neighboring topics.",
            "strategic_research_planning": "Pick packets that best represent the exact topic for which the ranked agenda should be planned. Include closely related sibling or descendant packets only when they help support a ranked list of multiple directions.",
        }.get(family, "Pick the exact core topic packets.")
        packet_block = []
        for idx, row in enumerate(packet_rows, start=1):
            packet_block.append(
                f"[{idx}] packet_id={row.get('packet_id')}\n"
                f"display_name={_humanize_label(row.get('display_name'))}\n"
                f"description={row.get('description') or ''}\n"
                f"dimension={_humanize_label(row.get('dimension_id'))}\n"
                f"representative_papers={'; '.join(str(x.get('title') or '') for x in (row.get('historical_representative_papers') or [])[:5])}"
            )
        user_prompt = f"""Task title: {task.get('title')}
Task question: {task.get('question')}
Task family: {family}

Candidate taxonomy packets:
{chr(10).join(packet_block)}

Instruction:
- {prompt}
- Prefer exact subfield matches over broad parent topics.
- Return at most {max_packets} packet ids.
- If one packet is clearly best, return only one.

Return JSON only:
{{
  "selected_packet_ids": ["packet_id_1", "packet_id_2"],
  "reason": "short reason"
}}
"""
        try:
            obj = complete_json_object(
                self.cheap_client,
                [
                    {"role": "system", "content": "You are a precise benchmark retrieval router. Output JSON only."},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
                temperature=0.0,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            selected = [str(x).strip() for x in (obj.get("selected_packet_ids") or []) if str(x).strip()]
            valid = {str(row.get("packet_id")) for row in packet_rows}
            selected = [x for x in selected if x in valid]
            return selected[:max_packets]
        except Exception:
            return []

    def select_packets(self, *, task: Dict[str, Any], domain_id: str) -> PacketSelection:
        family = str(task.get("family") or "")
        queries = self.task_queries(task)
        retrieval_rows = merge_multi_query_results(self._packet_retriever(domain_id), queries, top_k_per_query=8, limit=8)
        packet_map = {str(row.get("packet_id")): row for row in self._load_domain_packets(domain_id)}
        candidate_packets = [packet_map.get(str(doc.doc_id)) for doc, _ in retrieval_rows]
        candidate_packets = [row for row in candidate_packets if row]
        selected_ids = self._llm_select_packets(task=task, packet_rows=candidate_packets[:6])
        if not selected_ids and candidate_packets:
            selected_ids = [str(candidate_packets[0]["packet_id"])]
        if family == "strategic_research_planning":
            for row in candidate_packets:
                packet_id = str(row.get("packet_id") or "")
                if not packet_id or packet_id in selected_ids:
                    continue
                selected_ids.append(packet_id)
                if len(selected_ids) >= 2:
                    break
        selected = [packet_map[x] for x in selected_ids if x in packet_map]
        return PacketSelection(packets=selected, retrieval_rows=retrieval_rows, llm_selected_packet_ids=selected_ids)

    def _expand_queries_from_packets(self, packets: Iterable[Dict[str, Any]], base_queries: Iterable[str]) -> List[str]:
        queries = list(base_queries)
        for packet in packets:
            queries.append(_humanize_label(packet.get("display_name") or ""))
            queries.append(str(packet.get("description") or ""))
            for item in (packet.get("top_limitations") or [])[:4]:
                queries.append(str((item or {}).get("name") or ""))
            for item in (packet.get("top_future_work") or [])[:4]:
                queries.append(str((item or {}).get("direction") or (item or {}).get("name") or ""))
            for item in (packet.get("emergent_descendants") or [])[:4]:
                queries.append(_humanize_label((item or {}).get("display_name") or (item or {}).get("node_id") or ""))
            for paper in (packet.get("historical_representative_papers") or [])[:5]:
                queries.append(str(paper.get("title") or ""))
        return dedupe(queries)

    def _llm_select_core_papers(
        self,
        *,
        task: Dict[str, Any],
        packets: List[Dict[str, Any]],
        candidate_rows: List[Dict[str, Any]],
    ) -> List[str]:
        if not candidate_rows:
            return []
        packet_desc = "\n".join(
            f"- {_humanize_label(row.get('display_name'))}: {row.get('description') or ''}"
            for row in packets[:2]
        )
        blocks = []
        for idx, row in enumerate(candidate_rows, start=1):
            blocks.append(
                f"[{idx}] paper_id={row.get('paper_id')}\n"
                f"title={row.get('title')}\n"
                f"published={row.get('published_date')}\n"
                f"packet_match={row.get('packet_match')}\n"
                f"abstract={clip_text(row.get('abstract') or '', 700)}\n"
                f"problem={clip_text(row.get('problem_statement') or '', 240)}\n"
                f"limitations={'; '.join(row.get('limitations') or [])}"
            )
        prompt = f"""Task title: {task.get('title')}
Task question: {task.get('question')}
Task family: {task.get('family')}

Selected taxonomy packets:
{packet_desc}

Candidate papers:
{chr(10).join(blocks)}

Choose up to 14 papers that are core to this task's exact technical topic.
Rules:
- Prefer papers squarely inside the selected packets.
- Reject broad adjacent papers that would cause topic drift.
- For bottleneck tasks, prefer evaluation, limitation, failure mode, benchmark, or memory-specific papers when relevant.
- For planning tasks, prefer papers that define clear technical trajectories or optimization tensions.

Return JSON only:
{{
  "core_paper_ids": ["id1", "id2"],
  "reason": "short reason"
}}
"""
        try:
            obj = complete_json_object(
                self.cheap_client,
                [
                    {"role": "system", "content": "You are a precise benchmark retrieval filter. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=900,
                temperature=0.0,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            selected = [str(x).strip() for x in (obj.get("core_paper_ids") or []) if str(x).strip()]
            valid = {str(row.get("paper_id")) for row in candidate_rows}
            return [x for x in selected if x in valid][:14]
        except Exception:
            return []

    def build_candidate_pool(self, *, task: Dict[str, Any], domain_id: str) -> TaskCandidatePool:
        domain = self.kb.domain(domain_id)
        packet_selection = self.select_packets(task=task, domain_id=domain_id)
        packets = packet_selection.packets
        queries = self._expand_queries_from_packets(packets, self.task_queries(task))
        focus_terms = _focus_terms(task, packets)

        seeded_ids = []
        packet_match_by_paper: Dict[str, List[str]] = {}
        for packet in packets:
            packet_name = _humanize_label(packet.get("display_name") or packet.get("node_id") or "")
            for paper in (packet.get("historical_representative_papers") or [])[:10]:
                paper_id = str(paper.get("paper_id") or "").strip()
                if paper_id:
                    seeded_ids.append(paper_id)
                    packet_match_by_paper.setdefault(paper_id, []).append(packet_name)

        paper_rows = merge_multi_query_results(domain.paper_retriever(cutoff_date=str(task.get("time_cutoff") or "")), queries, top_k_per_query=12, limit=40)
        candidate_map: Dict[str, Dict[str, Any]] = {}
        for paper_id in seeded_ids:
            paper = domain.get_paper(paper_id) or {}
            if paper:
                candidate_map[paper_id] = {
                    "paper_id": paper_id,
                    "title": str(paper.get("title") or ""),
                    "abstract": str(paper.get("abstract") or ""),
                    "published_date": str(paper.get("published_date") or ""),
                    "packet_match": packet_match_by_paper.get(paper_id) or [],
                    "focus_score": 100,
                    "scores": {"seed_bonus": 1.0},
                }
        for doc, scores in paper_rows:
            paper = domain.get_paper(doc.paper_id) or {}
            structure = domain.get_structure(doc.paper_id) or {}
            family = str(task.get("family") or "")
            family_signal_text = "\n".join(
                [
                    str(doc.title or ""),
                    str(paper.get("abstract") or ""),
                    str(structure.get("problem_statement") or ""),
                    " ".join(
                        str(x.get("name") or "")
                        for x in (structure.get("explicit_limitations") or [])
                        if str(x.get("name") or "").strip()
                    ),
                    " ".join(
                        str(x.get("direction") or "")
                        for x in (structure.get("future_work") or [])
                        if str(x.get("direction") or "").strip()
                    ),
                    " ".join(
                        str(x.get("name") or "")
                        for x in (structure.get("core_ideas") or [])
                        if str(x.get("name") or "").strip()
                    ),
                ]
            )
            existing = candidate_map.get(doc.paper_id)
            focus_score = _focus_score(
                family_signal_text,
                focus_terms,
            )
            family_score = _family_signal_score(family=family, text=family_signal_text)
            row = {
                "paper_id": doc.paper_id,
                "title": doc.title,
                "abstract": str(paper.get("abstract") or ""),
                "published_date": str(paper.get("published_date") or ""),
                "problem_statement": str(structure.get("problem_statement") or ""),
                "limitations": [
                    str(x.get("name") or "")
                    for x in (structure.get("explicit_limitations") or [])
                    if str(x.get("name") or "").strip()
                ][:4],
                "packet_match": packet_match_by_paper.get(doc.paper_id) or [],
                "focus_score": focus_score,
                "family_score": family_score,
                "scores": scores,
            }
            if existing is None:
                candidate_map[doc.paper_id] = row
            else:
                existing["scores"].update(scores)
                existing["problem_statement"] = row.get("problem_statement")
                existing["limitations"] = row.get("limitations")
                existing["focus_score"] = max(int(existing.get("focus_score") or 0), focus_score)
                existing["family_score"] = max(int(existing.get("family_score") or 0), family_score)

        ranked_candidates = sorted(
            candidate_map.values(),
            key=lambda row: (
                len(row.get("packet_match") or []),
                int(row.get("family_score") or 0),
                int(row.get("focus_score") or 0),
                float((row.get("scores") or {}).get("combined_score") or (row.get("scores") or {}).get("seed_bonus") or 0.0),
            ),
            reverse=True,
        )
        strict_candidates = [
            row for row in ranked_candidates
            if (row.get("packet_match") or []) or int(row.get("focus_score") or 0) >= 2 or int(row.get("family_score") or 0) >= 2
        ]
        if len(strict_candidates) < 6:
            strict_candidates = [
                row for row in ranked_candidates
                if (row.get("packet_match") or []) or int(row.get("focus_score") or 0) >= 1 or int(row.get("family_score") or 0) >= 1
            ]
        candidate_rows_for_llm = strict_candidates[:18] if strict_candidates else ranked_candidates[:18]
        selected_core_ids = self._llm_select_core_papers(task=task, packets=packets, candidate_rows=candidate_rows_for_llm)
        valid_selected_ids = {str(row.get("paper_id") or "") for row in candidate_rows_for_llm}
        selected_core_ids = [x for x in selected_core_ids if x in valid_selected_ids]
        if not selected_core_ids:
            backfill_rows = strict_candidates if strict_candidates else ranked_candidates
            selected_core_ids = [str(row.get("paper_id")) for row in backfill_rows[:12]]
        final_papers = []
        paper_scores: Dict[str, Dict[str, Any]] = {}
        for paper_id in selected_core_ids:
            paper = domain.get_paper(paper_id) or {}
            if not paper:
                continue
            final_papers.append(paper)
            paper_scores[paper_id] = candidate_map.get(paper_id, {}).get("scores") or {}
        if len(final_papers) < 8:
            backfill_rows = strict_candidates if strict_candidates else ranked_candidates
            for row in backfill_rows:
                paper_id = str(row.get("paper_id") or "")
                if paper_id in paper_scores:
                    continue
                paper = domain.get_paper(paper_id) or {}
                if not paper:
                    continue
                final_papers.append(paper)
                paper_scores[paper_id] = row.get("scores") or {}
                if len(final_papers) >= 16:
                    break
        return TaskCandidatePool(
            packets=packets,
            packet_retrieval_rows=packet_selection.retrieval_rows,
            papers=final_papers,
            paper_scores=paper_scores,
            packet_ids=[str(row.get("packet_id")) for row in packets],
        )
