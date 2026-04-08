from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.fulltext_cache import LocalFulltextCache
from researchworld.coi_offline_retrieval import CoIOfflineRetrievalAdaptor, TaskCandidatePool
from researchworld.offline_kb import OfflineKnowledgeBase, clip_text, dedupe, merge_multi_query_results, normalize_ws
from researchworld.research_arc_kb import extract_focus_text
from researchworld.research_arc_v2 import extract_task_contract


ROOT = Path(__file__).resolve().parents[2]
COI_PATH = ROOT / "external" / "CoI-Agent"
if COI_PATH.exists() and str(COI_PATH) not in sys.path:
    sys.path.insert(0, str(COI_PATH))

from prompts.deep_research_agent_prompts import (  # type: ignore
    get_deep_reference_prompt,
    get_deep_generate_future_direciton_prompt,
    get_deep_judge_relevant_prompt,
    get_deep_search_query_prompt,
    get_deep_trend_idea_chains_prompt,
)


TAG_RE_TEMPLATE = r"<{tag}>(.*?)</{tag}>"
GENERIC_TAG_RE = re.compile(r"</?[A-Za-z_][A-Za-z0-9_:-]*>")


def extract_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    pattern = re.compile(TAG_RE_TEMPLATE.format(tag=re.escape(tag)), re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return " ".join(str(x).strip() for x in matches if str(x).strip()).strip()
    return ""


def parse_query_list(text: str) -> List[str]:
    raw = extract_tag(text, "queries") or text
    raw = raw.strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return dedupe(str(x) for x in value if str(x).strip())
    except Exception:
        pass
    quoted = re.findall(r'"([^"]+)"', raw)
    if quoted:
        return dedupe(quoted)
    return dedupe(re.split(r"[\n;,]+", raw))


def strip_xmlish_tags(text: str) -> str:
    return normalize_ws(GENERIC_TAG_RE.sub(" ", str(text or "")))


def _join_lines(parts: Iterable[str]) -> str:
    return "\n".join(str(x).strip() for x in parts if str(x).strip())


def _normalize_label(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ")).strip().lower()


def _humanize_label(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = raw.split("/")[-1].replace("__", " / ").replace("_", " ")
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def short_focus_terms(text: str, *, limit: int = 10) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", str(text or "").lower())
    stop = {
        "based", "before", "after", "using", "identify", "concrete", "research", "literature", "published",
        "long", "large", "model", "models", "agent", "agents", "framework", "frameworks", "technical",
        "unresolved", "subsequent", "period", "response", "grounded", "exclusively", "evidence", "signals",
    }
    picked: List[str] = []
    for token in tokens:
        if token in stop:
            continue
        picked.append(token)
        if len(picked) >= limit:
            break
    return " ".join(picked).strip()


class CoIAgentOffline:
    def __init__(
        self,
        *,
        kb: OfflineKnowledgeBase,
        main_client: OpenAICompatChatClient,
        cheap_client: Optional[OpenAICompatChatClient] = None,
        fulltext_cache: Optional[LocalFulltextCache] = None,
        allow_fulltext_fetch: bool = False,
    ):
        self.kb = kb
        self.main_client = main_client
        self.cheap_client = cheap_client or main_client
        self.fulltext_cache = fulltext_cache
        self.allow_fulltext_fetch = allow_fulltext_fetch
        self.max_chain_length = 5
        self.min_chain_length = 3
        self.max_anchor_papers = 4
        self.verbose = os.environ.get("RTL_VERBOSE_COI", "").strip().lower() in {"1", "true", "yes", "y"}
        self.retrieval_adaptor = CoIOfflineRetrievalAdaptor(kb=kb, cheap_client=self.cheap_client, main_client=self.main_client)

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[CoI-Agent-Offline][debug] {message}", flush=True)

    def _complete(self, client: OpenAICompatChatClient, prompt: str, *, max_tokens: int = 1400, temperature: float = 0.2) -> str:
        return client.complete_text(
            [
                {"role": "system", "content": "You are a precise research agent. Follow the requested format exactly."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=180,
            transport_retries=2,
        )

    def _contract(self, task: Dict[str, Any]) -> Dict[str, Any]:
        contract = extract_task_contract(task)
        contract["topic_text"] = str(contract.get("topic_text") or extract_focus_text(task) or "").strip()
        return contract

    def _chain_candidate_labels(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        packet_labels: List[str] = []
        descendant_labels: List[str] = []
        if candidate_pool is not None:
            for packet in candidate_pool.packets:
                packet_labels.append(_humanize_label(packet.get("display_name") or packet.get("node_id") or ""))
                for row in (packet.get("emergent_descendants") or [])[:10]:
                    descendant_labels.append(_humanize_label(row.get("display_name") or row.get("node_id") or ""))
        profile_titles = [_humanize_label((row or {}).get("title") or "") for row in (chain.get("profiles") or [])]
        profile_ideas = [normalize_ws((row or {}).get("idea") or "") for row in (chain.get("profiles") or [])]
        family = str(task.get("family") or "")
        planning_candidates = dedupe(
            [
                *descendant_labels,
                *packet_labels,
                *profile_titles,
                *profile_ideas,
            ]
        )[:18]
        return {
            "packet_labels": [x for x in dedupe(packet_labels) if x][:8],
            "descendant_labels": [x for x in dedupe(descendant_labels) if x][:12],
            "planning_candidates": [x for x in planning_candidates if x][:18],
            "family": [family],
        }

    def _canonicalize_text(self, text: str, pool: Iterable[str]) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        norm = _normalize_label(value)
        best = value
        for candidate in pool:
            cand = str(candidate or "").strip()
            if not cand:
                continue
            cand_norm = _normalize_label(cand)
            if norm == cand_norm or norm in cand_norm or cand_norm in norm:
                best = cand
                break
        return best

    def _project_structured_answer(
        self,
        *,
        task: Dict[str, Any],
        chain_text: str,
        trend: str,
        future: str,
        human: str,
        evidence: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Dict[str, Any],
    ) -> Dict[str, Any]:
        contract = self._contract(task)
        family = str(task.get("family") or "")
        labels = self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain)
        evidence_refs = []
        for group in ["papers", "fulltext"]:
            for row in (evidence.get(group) or [])[:5]:
                evidence_refs.append(
                    {
                        "evidence_id": row.get("evidence_id"),
                        "paper_id": row.get("paper_id"),
                        "title": row.get("title"),
                    }
                )
        if family == "strategic_research_planning":
            schema_hint = {
                "ranked_directions": [
                    {"rank": 1, "direction": "...", "rationale": "...", "evidence_ids": ["P1", "F2"]}
                ]
            }
            family_rule = f"Return up to {int(contract.get('max_items') or 3)} ranked directions. Keep direction wording close to candidate labels when possible."
            candidate_block = labels["planning_candidates"]
        elif family == "direction_forecasting":
            schema_hint = {
                "trajectory_label": "accelerating",
                "next_directions": ["..."],
                "rationale": "...",
                "evidence_ids": ["P1"],
            }
            family_rule = "Return exactly one trajectory_label from accelerating, fragmenting, steady, cooling, and 1-3 next_directions."
            candidate_block = labels["descendant_labels"] or labels["planning_candidates"]
        else:
            schema_hint = {
                "bottleneck": "...",
                "opportunity": "...",
                "linkage": "...",
                "evidence_ids": ["P1", "F2"],
            }
            family_rule = "Return exactly one bottleneck and one opportunity. The opportunity must directly answer the bottleneck."
            candidate_block = labels["descendant_labels"] or labels["planning_candidates"]
        prompt = f"""Adapt the Chain-of-Ideas reasoning to the benchmark schema.

Task:
{task.get('question')}

Task contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Chain of ideas:
{chain_text}

Trend:
{trend}

Future direction:
{future}

Human reasoning:
{human}

Candidate labels:
{json.dumps(candidate_block[:16], ensure_ascii=False, indent=2)}

Evidence refs:
{json.dumps(evidence_refs, ensure_ascii=False, indent=2)}

Rules:
- Preserve the CoI reasoning; do not invent a different line of argument.
- {family_rule}
- Prefer terminology visible in candidate labels or evidence titles.
- Cite evidence ids when possible.
- Output JSON only.

Schema:
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}
"""
        draft = complete_json_object(
            self.main_client,
            [
                {"role": "system", "content": "You are a precise benchmark schema adapter. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.1,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
        critique_prompt = f"""Repair the structured answer so that it strictly matches the task family and benchmark contract.

Task:
{task.get('question')}

Contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Candidate labels:
{json.dumps(candidate_block[:16], ensure_ascii=False, indent=2)}

Draft JSON:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Rules:
- Keep the output faithful to the Chain-of-Ideas reasoning.
- Reject vague generic directions when more specific candidate labels are available.
- Ensure the schema is complete and concise.
- Output JSON only.
"""
        final = complete_json_object(
            self.main_client,
            [
                {"role": "system", "content": "You are a strict benchmark schema critic. Output JSON only."},
                {"role": "user", "content": critique_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.0,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
        return {"draft": draft, "final": final, "candidate_labels": candidate_block}

    def _verbalize_structured_answer(
        self,
        *,
        task: Dict[str, Any],
        projected: Dict[str, Any],
    ) -> str:
        family = str(task.get("family") or "")
        payload = projected.get("final") or projected.get("draft") or {}
        if isinstance(payload, dict) and isinstance(payload.get("items"), list) and payload.get("items"):
            first = payload["items"][0]
            if isinstance(first, dict):
                normalized = dict(first)
                for key in ("trajectory_label", "next_directions", "rationale", "ranked_directions", "bottleneck", "opportunity", "linkage", "evidence_ids"):
                    if key not in normalized and key in payload:
                        normalized[key] = payload.get(key)
                payload = normalized
        pool = projected.get("candidate_labels") or []
        if family == "strategic_research_planning":
            rows = payload.get("ranked_directions") or []
            lines: List[str] = []
            for idx, row in enumerate(rows, start=1):
                direction = self._canonicalize_text(str((row or {}).get("direction") or ""), pool)
                rationale = normalize_ws((row or {}).get("rationale") or "")
                evidence_ids = [str(x).strip() for x in ((row or {}).get("evidence_ids") or []) if str(x).strip()]
                if not direction:
                    continue
                suffix = f" Evidence anchors: {', '.join(evidence_ids[:3])}." if evidence_ids else ""
                lines.append(f"{idx}. {direction} — {rationale}.{suffix}".replace("..", "."))
            return "\n".join(lines).strip()
        if family == "direction_forecasting":
            label = str(payload.get("trajectory_label") or "").strip().lower()
            directions = [self._canonicalize_text(str(x or ""), pool) for x in (payload.get("next_directions") or [])]
            directions = [x for x in directions if x]
            rationale = normalize_ws(payload.get("rationale") or "")
            if directions:
                return f"The area is most likely {label}. The next technical directions are {', '.join(directions[:3])}. {rationale}".strip()
            return f"The area is most likely {label}. {rationale}".strip()
        bottleneck = self._canonicalize_text(str(payload.get("bottleneck") or ""), pool)
        opportunity = self._canonicalize_text(str(payload.get("opportunity") or ""), pool)
        linkage = normalize_ws(payload.get("linkage") or "")
        evidence_ids = [str(x).strip() for x in (payload.get("evidence_ids") or []) if str(x).strip()]
        suffix = f" Evidence anchors: {', '.join(evidence_ids[:3])}." if evidence_ids else ""
        return f"A concrete unresolved bottleneck is {bottleneck}. The associated opportunity is {opportunity}. {linkage}.{suffix}".replace("..", ".").strip()

    def _score_candidate_run(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Dict[str, Any],
        projected: Dict[str, Any],
        trend: str,
        future: str,
    ) -> float:
        family = str(task.get("family") or "")
        contract = self._contract(task)
        payload = projected.get("final") or projected.get("draft") or {}
        chain_profiles = list(chain.get("profiles") or [])
        score = 0.0
        score += min(2.0, 0.5 * len(chain_profiles))
        score += min(0.8, len(normalize_ws(trend)) / 500.0)
        score += min(0.8, len(normalize_ws(future)) / 500.0)
        pool_labels = self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain)["planning_candidates"]
        if family == "strategic_research_planning":
            rows = payload.get("ranked_directions") or []
            score += 1.2 if len(rows) >= min(2, int(contract.get("max_items") or 3)) else 0.0
            score += min(1.0, len(rows) / max(1, int(contract.get("max_items") or 3)))
            hit = 0
            for row in rows:
                direction = str((row or {}).get("direction") or "")
                if self._canonicalize_text(direction, pool_labels):
                    hit += 1
            score += 0.4 * hit
        elif family == "direction_forecasting":
            label = str(payload.get("trajectory_label") or "").strip().lower()
            score += 1.2 if label in {"accelerating", "fragmenting", "steady", "cooling"} else 0.0
            score += 0.4 * min(3, len(payload.get("next_directions") or []))
        else:
            score += 1.0 if normalize_ws(payload.get("bottleneck") or "") else 0.0
            score += 1.0 if normalize_ws(payload.get("opportunity") or "") else 0.0
            score += 0.6 if normalize_ws(payload.get("linkage") or "") else 0.0
        return round(score, 4)

    def _candidate_anchor_count(self, task: Dict[str, Any]) -> int:
        family = str(task.get("family") or "")
        if family == "direction_forecasting":
            return 3
        if family == "strategic_research_planning":
            return 4
        return 3

    def _family_anchor_score(
        self,
        *,
        task: Dict[str, Any],
        anchor: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
    ) -> float:
        family = str(task.get("family") or "")
        text = "\n".join([str(anchor.get("title") or ""), str(anchor.get("abstract") or "")]).lower()
        score = 0.0
        score += 1.8 * float(anchor.get("pool_bonus") or 0.0)
        score += 1.2 * len(anchor.get("packet_match") or [])
        score += 0.5 * int(anchor.get("query_hit_count") or 0)
        score += 0.25 * max(0, 20 - int(anchor.get("best_rank") or 20))
        score += 0.4 * float((anchor.get("scores") or {}).get("combined_score") or 0.0)
        if family == "bottleneck_opportunity_discovery":
            for kw in ["limitation", "benchmark", "evaluation", "failure", "error", "challenge", "memory", "alignment"]:
                if kw in text:
                    score += 0.45
        elif family == "direction_forecasting":
            for kw in ["trend", "emerging", "survey", "benchmark", "evaluation", "scaling", "trajectory"]:
                if kw in text:
                    score += 0.35
            cutoff_dt = self._parse_date(str(task.get("time_cutoff") or ""))
            published_dt = self._parse_date(str(anchor.get("published_date") or ""))
            if cutoff_dt is not None and published_dt is not None:
                score += max(0.0, 1.5 - abs((cutoff_dt - published_dt).days) / 365.0)
        elif family == "strategic_research_planning":
            for kw in ["future work", "trade-off", "tradeoff", "efficiency", "scaling", "robust", "generalization", "benchmark"]:
                if kw in text:
                    score += 0.35
            if candidate_pool is not None:
                score += 0.15 * min(4, len(candidate_pool.packet_ids))
        return round(score, 4)

    def _support_queries_from_runs(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> List[str]:
        family = str(task.get("family") or "")
        focus = extract_focus_text(task)
        queries = [str(task.get("question") or ""), str(task.get("title") or ""), focus]
        for run in candidate_runs[:4]:
            for profile in (run.get("chain", {}) or {}).get("profiles", [])[:5]:
                queries.extend(
                    [
                        str(profile.get("title") or ""),
                        str(profile.get("idea") or ""),
                        str(profile.get("entities") or ""),
                    ]
                )
        if candidate_pool is not None:
            for packet in candidate_pool.packets[:3]:
                queries.extend(
                    [
                        _humanize_label(packet.get("display_name") or packet.get("node_id") or ""),
                        str(packet.get("description") or ""),
                    ]
                )
                for row in (packet.get("emergent_descendants") or [])[:5]:
                    queries.append(_humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or ""))
                for row in (packet.get("top_limitations") or [])[:4]:
                    queries.append(str((row or {}).get("name") or ""))
                for row in (packet.get("top_future_work") or [])[:4]:
                    queries.append(str((row or {}).get("direction") or (row or {}).get("name") or ""))
        if family == "bottleneck_opportunity_discovery":
            queries.extend(
                [
                    f"{focus} limitation failure mode benchmark",
                    f"{focus} error analysis challenge",
                    f"{focus} evaluation gap future work",
                ]
            )
        elif family == "direction_forecasting":
            queries.extend(
                [
                    f"{focus} technical trajectory",
                    f"{focus} emerging direction evaluation",
                    f"{focus} benchmark trend scaling",
                ]
            )
        elif family == "strategic_research_planning":
            queries.extend(
                [
                    f"{focus} future work priority",
                    f"{focus} open problems trade-offs",
                    f"{focus} benchmark bottleneck opportunity",
                ]
            )
        return dedupe([q for q in queries if normalize_ws(q)])[:16]

    def _collect_family_evidence(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        family = str(task.get("family") or "")
        paper_ids: List[str] = []
        for run in candidate_runs[: self._candidate_anchor_count(task)]:
            for profile in (run.get("chain", {}) or {}).get("profiles", [])[:5]:
                pid = str(profile.get("paper_id") or "").strip()
                if pid:
                    paper_ids.append(pid)
        if candidate_pool is not None and family in {"direction_forecasting", "strategic_research_planning"}:
            for paper in candidate_pool.papers[:6]:
                pid = str(paper.get("paper_id") or "").strip()
                if pid:
                    paper_ids.append(pid)
        paper_ids = dedupe(paper_ids)[:14]
        paper_rows: List[Dict[str, Any]] = []
        for idx, paper_id in enumerate(paper_ids, start=1):
            paper = domain.get_paper(paper_id) or {}
            pub = paper.get("publication") or {}
            paper_rows.append(
                {
                    "evidence_id": f"P{idx}",
                    "paper_id": paper_id,
                    "title": str(paper.get("title") or ""),
                    "published_date": paper.get("published_date"),
                    "venue": pub.get("venue_name"),
                    "citations": pub.get("citation_count"),
                    "abstract_snippet": clip_text(paper.get("abstract") or "", 700),
                }
            )

        queries = self._support_queries_from_runs(task=task, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        structure_rows: List[Dict[str, Any]] = []
        if paper_ids:
            structure_hits = merge_multi_query_results(
                domain.structure_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                queries,
                top_k_per_query=8,
                limit=10,
            )
            for idx, (doc, scores) in enumerate(structure_hits, start=1):
                structure_rows.append(
                    {
                        "evidence_id": f"T{idx}",
                        "paper_id": doc.paper_id,
                        "title": doc.title,
                        "problem_statement": clip_text(doc.meta.get("problem_statement"), 260),
                        "limitations": list(doc.meta.get("limitations") or [])[:4],
                        "future_work": list(doc.meta.get("future_work") or [])[:4],
                        "core_ideas": list(doc.meta.get("core_ideas") or [])[:4],
                        "snippet": clip_text(doc.text, 900),
                        "scores": scores,
                    }
                )
        page_rows: List[Dict[str, Any]] = []
        if paper_ids and family == "bottleneck_opportunity_discovery":
            page_hits = merge_multi_query_results(
                domain.pageindex_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                queries,
                top_k_per_query=8,
                limit=10,
            )
            for idx, (doc, scores) in enumerate(page_hits, start=1):
                page_rows.append(
                    {
                        "evidence_id": f"G{idx}",
                        "paper_id": doc.paper_id,
                        "title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "kind": doc.meta.get("kind"),
                        "snippet": clip_text(doc.text, 1000),
                        "scores": scores,
                    }
                )
        fulltext_rows: List[Dict[str, Any]] = []
        if paper_ids:
            section_hits = merge_multi_query_results(
                domain.section_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                queries,
                top_k_per_query=8,
                limit=10,
            )
            for idx, (doc, scores) in enumerate(section_hits, start=1):
                fulltext_rows.append(
                    {
                        "evidence_id": f"F{idx}",
                        "paper_id": doc.paper_id,
                        "title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "level": doc.meta.get("level"),
                        "source_type": "kb_section",
                        "snippet": clip_text(doc.text, 1200),
                        "scores": scores,
                    }
                )
        packet_rows: List[Dict[str, Any]] = []
        if candidate_pool is not None:
            for idx, packet in enumerate(candidate_pool.packets[:4], start=1):
                packet_rows.append(
                    {
                        "evidence_id": f"N{idx}",
                        "packet_id": packet.get("packet_id"),
                        "display_name": _humanize_label(packet.get("display_name") or packet.get("node_id") or ""),
                        "description": packet.get("description"),
                        "snippet": _join_lines(
                            [
                                f"Display name: {_humanize_label(packet.get('display_name') or packet.get('node_id') or '')}",
                                f"Description: {packet.get('description') or ''}",
                                "Emergent descendants: "
                                + "; ".join(
                                    _humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or "")
                                    for row in (packet.get("emergent_descendants") or [])[:6]
                                    if _humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or "")
                                ),
                            ]
                        ),
                    }
                )
        return {
            "queries": queries,
            "papers": paper_rows,
            "structures": structure_rows,
            "pageindex": page_rows,
            "fulltext": fulltext_rows,
            "candidate_node_evidence": packet_rows,
        }

    def _aggregate_candidate_labels(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> List[str]:
        labels: List[str] = []
        if candidate_pool is not None:
            for packet in candidate_pool.packets[:4]:
                labels.append(_humanize_label(packet.get("display_name") or packet.get("node_id") or ""))
                for row in (packet.get("emergent_descendants") or [])[:8]:
                    labels.append(_humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or ""))
                for row in (packet.get("top_limitations") or [])[:6]:
                    labels.append(str((row or {}).get("name") or ""))
                for row in (packet.get("top_future_work") or [])[:6]:
                    labels.append(str((row or {}).get("direction") or (row or {}).get("name") or ""))
        for run in candidate_runs[:4]:
            labels.extend(self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=run.get("chain") or {}).get("planning_candidates") or [])
        return [x for x in dedupe(labels) if x][:24]

    def _render_candidate_runs(self, candidate_runs: List[Dict[str, Any]], *, limit: int = 3) -> str:
        blocks: List[str] = []
        for idx, run in enumerate(candidate_runs[:limit], start=1):
            chain = run.get("chain") or {}
            profiles = list(chain.get("profiles") or [])
            paper_line = " -> ".join(
                f"{str(p.get('published_date') or '')[:10]} {str(p.get('title') or '')}".strip()
                for p in profiles[:5]
                if str(p.get("title") or "").strip()
            )
            idea_line = " | ".join(
                clip_text(str(p.get("idea") or ""), 180)
                for p in profiles[:4]
                if normalize_ws(p.get("idea") or "")
            )
            blocks.append(
                _join_lines(
                    [
                        f"Branch {idx}",
                        f"Anchor: {run.get('anchor', {}).get('title') or ''}",
                        f"Paper chain: {paper_line}",
                        f"Key ideas: {idea_line}",
                        f"Trend summary: {run.get('trend') or ''}",
                        f"Future summary: {run.get('future') or ''}",
                        f"Provisional answer: {run.get('answer') or ''}",
                    ]
                )
            )
        return "\n\n".join(blocks).strip()

    def _coi_family_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            obj = complete_json_object(
                self.main_client,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=1400,
                temperature=0.15,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            if isinstance(obj, dict) and obj:
                return obj
        except Exception:
            pass
        return fallback

    def _render_packet_context(self, candidate_pool: Optional[TaskCandidatePool], *, limit: int = 4) -> str:
        if candidate_pool is None:
            return ""
        blocks: List[str] = []
        for idx, packet in enumerate(candidate_pool.packets[:limit], start=1):
            descendants = [
                _humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or "")
                for row in (packet.get("emergent_descendants") or [])[:6]
            ]
            descendants = [x for x in descendants if x]
            limitations = [
                normalize_ws((row or {}).get("name") or "")
                for row in (packet.get("top_limitations") or [])[:5]
                if normalize_ws((row or {}).get("name") or "")
            ]
            future_work = [
                normalize_ws((row or {}).get("direction") or (row or {}).get("name") or "")
                for row in (packet.get("top_future_work") or [])[:5]
                if normalize_ws((row or {}).get("direction") or (row or {}).get("name") or "")
            ]
            blocks.append(
                _join_lines(
                    [
                        f"Packet {idx}: {_humanize_label(packet.get('display_name') or packet.get('node_id') or '')}",
                        f"Description: {packet.get('description') or ''}",
                        f"Historical bottlenecks: {'; '.join(limitations)}",
                        f"Emergent descendants: {'; '.join(descendants)}",
                        f"Historical future work: {'; '.join(future_work)}",
                    ]
                )
            )
        return "\n\n".join(blocks).strip()

    def _render_coi_backbone_state(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
        evidence: Dict[str, Any],
    ) -> str:
        run_blocks: List[str] = []
        for idx, run in enumerate(candidate_runs[:4], start=1):
            chain = run.get("chain") or {}
            profiles = list(chain.get("profiles") or [])
            transition_line = " -> ".join(
                f"{str(row.get('published_date') or '')[:10]} {str(row.get('title') or '')}".strip()
                for row in profiles[:5]
                if normalize_ws(row.get("title") or "")
            )
            idea_line = " | ".join(
                clip_text(str(row.get("idea") or ""), 160)
                for row in profiles[:4]
                if normalize_ws(row.get("idea") or "")
            )
            run_blocks.append(
                _join_lines(
                    [
                        f"Branch {idx}",
                        f"Anchor: {(run.get('anchor') or {}).get('title') or ''}",
                        f"Paper transitions: {transition_line}",
                        f"Key ideas: {idea_line}",
                        f"Trend summary: {run.get('trend') or ''}",
                        f"Forward-looking hypothesis: {run.get('future') or ''}",
                        f"Provisional answer: {run.get('answer') or ''}",
                    ]
                )
            )
        counts = {
            "candidate_runs": len(candidate_runs),
            "packet_count": len((candidate_pool.packets if candidate_pool is not None else []) or []),
            "paper_evidence": len(evidence.get("papers") or []),
            "structure_evidence": len(evidence.get("structures") or []),
            "pageindex_evidence": len(evidence.get("pageindex") or []),
            "fulltext_evidence": len(evidence.get("fulltext") or []),
        }
        return _join_lines(
            [
                f"Task family: {task.get('family')}",
                f"Question: {task.get('question')}",
                f"Backbone coverage: {json.dumps(counts, ensure_ascii=False)}",
                "",
                "Packet context:",
                self._render_packet_context(candidate_pool),
                "",
                "CoI candidate branches:",
                "\n\n".join(run_blocks).strip(),
                "",
                "Evidence block:",
                self._render_evidence_block(evidence),
            ]
        ).strip()

    def _adapt_head_output(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
        evidence: Dict[str, Any],
        head_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task.get("family") or "")
        labels = self._aggregate_candidate_labels(task=task, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        evidence_refs = []
        for group in ["papers", "structures", "pageindex", "fulltext"]:
            for row in (evidence.get(group) or [])[:6]:
                evidence_refs.append(
                    {
                        "evidence_id": row.get("evidence_id"),
                        "paper_id": row.get("paper_id"),
                        "title": row.get("title"),
                    }
                )
        if family == "strategic_research_planning":
            schema_hint = {
                "ranked_directions": [
                    {
                        "rank": 1,
                        "direction": "...",
                        "rationale": "...",
                        "evidence_ids": ["P1", "T1"],
                    }
                ]
            }
            family_rule = "Return up to 3 ranked directions. Each direction should be specific, non-redundant, and phrased as a real research agenda item."
        elif family == "direction_forecasting":
            schema_hint = {
                "trajectory_label": "accelerating",
                "next_directions": ["..."],
                "rationale": "...",
                "evidence_ids": ["P1", "T1"],
            }
            family_rule = "Return exactly one trajectory_label from accelerating, fragmenting, steady, cooling and 1-3 next_directions."
        else:
            schema_hint = {
                "bottleneck": "...",
                "opportunity": "...",
                "linkage": "...",
                "evidence_ids": ["T1", "F1"],
            }
            family_rule = "Return exactly one bottleneck and one opportunity. The opportunity must directly respond to the bottleneck."
        prompt = f"""You are the task-specific answer adapter that sits on top of a Chain-of-Ideas evidence backbone.

Task:
{task.get('question')}

Task family:
{family}

Family-specific analysis output:
{json.dumps(head_analysis, ensure_ascii=False, indent=2)}

Candidate labels:
{json.dumps(labels[:24], ensure_ascii=False, indent=2)}

Evidence refs:
{json.dumps(evidence_refs[:18], ensure_ascii=False, indent=2)}

Rules:
- Preserve the backbone reasoning and the family-specific analysis; do not invent a new line of argument.
- Keep wording technically specific and close to supported labels when possible.
- Avoid generic management language.
- {family_rule}
- Output JSON only.

Schema:
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}
"""
        draft = complete_json_object(
            self.main_client,
            [
                {"role": "system", "content": "You are a precise benchmark answer adapter. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.1,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
        critique_prompt = f"""Review and repair the draft benchmark JSON so that it is specific, evidence-grounded, and family-aligned.

Task:
{task.get('question')}

Task family:
{family}

Family-specific analysis:
{json.dumps(head_analysis, ensure_ascii=False, indent=2)}

Draft JSON:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Candidate labels:
{json.dumps(labels[:24], ensure_ascii=False, indent=2)}

Rules:
- Keep the draft faithful to the backbone analysis.
- Remove vague, duplicate, or overly broad items.
- Prefer technically discriminative labels over broad parent topics.
- Output repaired JSON only.
"""
        final = complete_json_object(
            self.main_client,
            [
                {"role": "system", "content": "You are a strict benchmark answer critic. Output JSON only."},
                {"role": "user", "content": critique_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.0,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
        return {"draft": draft, "final": final, "candidate_labels": labels}

    def _finalize_head_answer(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
        evidence: Dict[str, Any],
        head_analysis: Dict[str, Any],
        fallback_answer: str,
    ) -> Tuple[str, Dict[str, Any]]:
        adapted = self._adapt_head_output(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=head_analysis,
        )
        answer = self._verbalize_structured_answer(task=task, projected=adapted).strip()
        if not answer:
            answer = normalize_ws(fallback_answer)
        return answer, adapted

    def _run_bottleneck_head(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        evidence = self._collect_family_evidence(task=task, domain_id=domain_id, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        fallback_answer = candidate_runs[0].get("answer") if candidate_runs else ""
        backbone_state = self._render_coi_backbone_state(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
        )
        prompt = f"""You are the bottleneck-opportunity analysis head that sits on top of a Chain-of-Ideas evidence backbone.

Backbone state:
{backbone_state}

Instructions:
- Preserve the Chain-of-Ideas reasoning and compare multiple branches instead of reading one branch in isolation.
- Extract recurring technical frictions, evaluation gaps, and unresolved trade-offs.
- Then decide which bottleneck is the most central unresolved constraint.
- Finally identify which opportunity is genuinely opened if that bottleneck is addressed.
- Distinguish the bottleneck itself from its downstream opportunity.

Return JSON only:
{{
  "recurring_bottlenecks": [
    {{"name": "...", "mechanism": "...", "evidence_ids": ["T1", "F1"]}}
  ],
  "opportunity_candidates": [
    {{"name": "...", "why_now": "...", "evidence_ids": ["P1", "T2"]}}
  ],
  "selected_bottleneck": "...",
  "selected_opportunity": "...",
  "linkage": "...",
  "strategic_implication": "..."
}}
"""
        fallback = {
            "recurring_bottlenecks": [],
            "opportunity_candidates": [],
            "selected_bottleneck": "",
            "selected_opportunity": "",
            "linkage": "",
            "strategic_implication": "",
        }
        analysis = self._coi_family_json(
            system_prompt="You are a precise Chain-of-Ideas benchmark synthesizer for bottleneck discovery. Output JSON only.",
            user_prompt=prompt,
            fallback=fallback,
        )
        answer, adapted = self._finalize_head_answer(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=analysis,
            fallback_answer=fallback_answer,
        )
        return answer, {"analysis": analysis, "adapted": adapted}, evidence

    def _run_direction_head(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        evidence = self._collect_family_evidence(task=task, domain_id=domain_id, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        fallback_answer = candidate_runs[0].get("answer") if candidate_runs else ""
        backbone_state = self._render_coi_backbone_state(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
        )
        prompt = f"""You are the direction-forecasting head that sits on top of a Chain-of-Ideas evidence backbone.

Backbone state:
{backbone_state}

Instructions:
- Preserve the backbone reasoning and compare multiple branches for convergence, divergence, momentum, and recombination.
- Separate momentum signals from enabling shifts and unresolved frictions.
- Make one trajectory call and explain why.
- Forecast one to three next directions that naturally extend the strongest historical signals.
- Avoid generic trend language; focus on technically discriminative directions.

Return JSON only:
{{
  "momentum_signals": [
    {{"signal": "...", "why_it_matters": "...", "evidence_ids": ["P1", "T1"]}}
  ],
  "enabling_shifts": [
    {{"shift": "...", "evidence_ids": ["P2"]}}
  ],
  "friction_points": [
    {{"friction": "...", "evidence_ids": ["T2", "F1"]}}
  ],
  "trajectory_label": "accelerating",
  "next_directions": ["...", "..."],
  "rationale": "...",
  "uncertainty_notes": "..."
}}
"""
        fallback = {
            "momentum_signals": [],
            "enabling_shifts": [],
            "friction_points": [],
            "trajectory_label": "",
            "next_directions": [],
            "rationale": "",
            "uncertainty_notes": "",
        }
        analysis = self._coi_family_json(
            system_prompt="You are a precise Chain-of-Ideas benchmark synthesizer for trajectory forecasting. Output JSON only.",
            user_prompt=prompt,
            fallback=fallback,
        )
        answer, adapted = self._finalize_head_answer(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=analysis,
            fallback_answer=fallback_answer,
        )
        return answer, {"analysis": analysis, "adapted": adapted}, evidence

    def _run_planning_head(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        evidence = self._collect_family_evidence(task=task, domain_id=domain_id, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        fallback_answer = candidate_runs[0].get("answer") if candidate_runs else ""
        contract = self._contract(task)
        max_items = int(contract.get("max_items") or 3)
        backbone_state = self._render_coi_backbone_state(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
        )
        prompt = f"""You are the strategic-planning head that sits on top of a Chain-of-Ideas evidence backbone.

Backbone state:
{backbone_state}

Instructions:
- Preserve the backbone reasoning and translate it into a ranked research plan.
- Distinguish promising directions, crowded directions, and risky bets.
- Rank only directions that are technically actionable in the next six months.
- Use why-now value, tractability, and non-redundancy to determine rank.
- Keep the final ranked list to at most {max_items} items.

Return JSON only:
{{
  "promising_directions": [
    {{"direction": "...", "why_now": "...", "evidence_ids": ["P1", "T1"]}}
  ],
  "crowded_or_saturated_areas": [
    {{"direction": "...", "reason": "..."}}
  ],
  "risky_bets": [
    {{"direction": "...", "risk": "..."}}
  ],
  "ranked_directions": [
    {{"rank": 1, "direction": "...", "rationale": "...", "evidence_ids": ["T1", "P2"]}}
  ]
}}
"""
        fallback = {
            "promising_directions": [],
            "crowded_or_saturated_areas": [],
            "risky_bets": [],
            "ranked_directions": [],
        }
        analysis = self._coi_family_json(
            system_prompt="You are a precise Chain-of-Ideas benchmark synthesizer for strategic research planning. Output JSON only.",
            user_prompt=prompt,
            fallback=fallback,
        )
        answer, adapted = self._finalize_head_answer(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=analysis,
            fallback_answer=fallback_answer,
        )
        return answer, {"analysis": analysis, "adapted": adapted}, evidence

    def build_queries(self, task: Dict[str, Any]) -> List[str]:
        topic = extract_focus_text(task)
        short_topic = short_focus_terms(topic) or topic
        title_focus = short_focus_terms(str(task.get("title") or "")) or str(task.get("title") or "")
        prompt = get_deep_search_query_prompt(topic=topic)
        try:
            response = self._complete(self.main_client, prompt, max_tokens=500, temperature=0.2)
            queries = parse_query_list(response)
        except Exception:
            queries = []
        fallbacks = [
            task.get("question") or "",
            task.get("title") or "",
            topic,
            short_topic,
            title_focus,
        ]
        family = str(task.get("family") or "")
        if family == "bottleneck_opportunity_discovery":
            fallbacks += [f"{short_topic} limitation bottleneck", f"{short_topic} future work"]
        elif family == "direction_forecasting":
            fallbacks += [f"{short_topic} emerging direction", f"{short_topic} trend evaluation"]
        elif family == "strategic_research_planning":
            fallbacks += [f"{short_topic} open problems", f"{short_topic} future direction priority"]
        return dedupe([*queries, *fallbacks])[:8]

    def render_paper_content(self, *, domain_id: str, paper_id: str) -> str:
        domain = self.kb.domain(domain_id)
        paper = domain.get_paper(paper_id) or {}
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        sections: List[Dict[str, Any]] = []
        if self.fulltext_cache is not None:
            cached = self.fulltext_cache.get_content(paper_id) or self.fulltext_cache.ensure_content(
                paper_id,
                allow_fetch=self.allow_fulltext_fetch,
            )
            if cached:
                sections = list(cached.get("sections") or [])
        if not sections:
            sections = list(domain.sections_by_paper.get(str(paper_id), []))
        parts = [f"Title: {title}", f"Abstract: {abstract}"]
        for idx, section in enumerate(sections[:12], start=1):
            section_title = str(section.get("title") or section.get("section_title") or f"Section {idx}")
            section_text = clip_text(section.get("text") or "", 2200)
            if not normalize_ws(section_text):
                continue
            parts.append(f"Section {idx}: {section_title}\n{section_text}")
        return "\n\n".join(part for part in parts if normalize_ws(part))

    def extract_paper_profile(self, *, task: Dict[str, Any], domain_id: str, paper_id: str) -> Dict[str, Any]:
        domain = self.kb.domain(domain_id)
        paper = domain.get_paper(paper_id) or {}
        self._log(f"extract_profile paper_id={paper_id} title={str(paper.get('title') or '')[:120]}")
        content = self.render_paper_content(domain_id=domain_id, paper_id=paper_id)
        prompt = get_deep_reference_prompt(content, extract_focus_text(task))
        response = self._complete(self.cheap_client, prompt, max_tokens=2200, temperature=0.1)
        return {
            "paper_id": paper_id,
            "title": str(paper.get("title") or ""),
            "abstract": str(paper.get("abstract") or ""),
            "published_date": str(paper.get("published_date") or paper.get("published") or ""),
            "idea": extract_tag(response, "idea").strip(),
            "experiment": extract_tag(response, "experiment").strip(),
            "entities": extract_tag(response, "entities").strip(),
            "references": parse_query_list(extract_tag(response, "references").strip()),
            "raw": response,
        }

    def judge_relevant(
        self,
        *,
        candidate_title: str,
        candidate_abstract: str,
        topic: str,
        current_title: str = "",
        current_idea: str = "",
        direction: str = "",
    ) -> bool:
        if current_title or current_idea or direction:
            prompt = f"""You are validating one step in a Chain-of-Ideas research trajectory.

Topic:
{topic}

Direction:
{direction or "same-line continuation"}

Current paper:
Title: {current_title}
Idea: {current_idea}

Candidate paper:
Title: {candidate_title}
Abstract: {candidate_abstract}

Return <relevant>1</relevant> only if the candidate is a plausible adjacent step in the same technical line of work.
Reject broad neighboring papers that share the topic but do not continue the same line.
Output only the XML tag.
"""
        else:
            prompt = get_deep_judge_relevant_prompt(candidate_title, candidate_abstract, topic)
        response = self._complete(self.main_client, prompt, max_tokens=300, temperature=0.0)
        return extract_tag(response, "relevant").strip() != "0"

    def _parse_date(self, text: str) -> Optional[datetime]:
        value = str(text or "").strip()
        if not value:
            return None
        candidates = [value[:10], value[:7], value[:4], value]
        for sample, fmt in (
            (candidates[0], "%Y-%m-%d"),
            (candidates[1], "%Y-%m"),
            (candidates[2], "%Y"),
        ):
            try:
                return datetime.strptime(sample, fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(value[:10])
        except Exception:
            return None

    def _paper_pool_features(
        self,
        *,
        paper_id: str,
        candidate_pool: Optional[TaskCandidatePool] = None,
    ) -> Dict[str, Any]:
        packet_match: List[str] = []
        pool_score = {}
        in_pool = False
        if candidate_pool is not None:
            in_pool = paper_id in {str(row.get("paper_id") or "") for row in candidate_pool.papers}
            pool_score = candidate_pool.paper_scores.get(paper_id) or {}
            for packet in candidate_pool.packets:
                packet_name = str(packet.get("display_name") or packet.get("node_id") or "")
                rep_ids = {str(x.get("paper_id") or "") for x in (packet.get("historical_representative_papers") or [])}
                if paper_id in rep_ids:
                    packet_match.append(packet_name)
        return {
            "packet_match": packet_match,
            "pool_bonus": 1.0 if in_pool else 0.0,
            "pool_score": pool_score,
        }

    def _transition_queries(self, *, task: Dict[str, Any], current: Dict[str, Any], direction: str) -> List[str]:
        topic = extract_focus_text(task)
        title = normalize_ws(current.get("title") or "")
        idea = normalize_ws(current.get("idea") or "")
        entities = normalize_ws(current.get("entities") or "")
        refs = [normalize_ws(x) for x in (current.get("references") or []) if normalize_ws(x)]
        queries = [
            f"{topic} {title}",
            f"{topic} {idea}",
            f"{title} {idea}",
            f"{topic} {entities}",
        ]
        if direction == "backward":
            queries = [*refs[:3], *queries]
        else:
            queries = [*queries, *refs[:2]]
        return dedupe([q for q in queries if normalize_ws(q)])[:6]

    def _select_transition_candidate(
        self,
        *,
        task: Dict[str, Any],
        current: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        direction: str,
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        topic = extract_focus_text(task)
        heuristic_fallback = None
        for cand in candidates:
            self._log(f"judge_{direction} candidate={cand.get('paper_id')} title={str(cand.get('title') or '')[:120]}")
            try:
                if self.judge_relevant(
                    candidate_title=str(cand.get("title") or ""),
                    candidate_abstract=str(cand.get("abstract") or ""),
                    topic=topic,
                    current_title=str(current.get("title") or ""),
                    current_idea=str(current.get("idea") or ""),
                    direction=direction,
                ):
                    return cand
            except Exception:
                pass
            if heuristic_fallback is None:
                if (
                    float(cand.get("pool_bonus") or 0.0) > 0.0
                    or len(cand.get("packet_match") or []) > 0
                    or int(cand.get("query_hit_count") or 0) >= 2
                ):
                    heuristic_fallback = cand
        return heuristic_fallback or candidates[0]

    def search_anchor_papers(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        queries: List[str],
        candidate_pool: Optional[TaskCandidatePool] = None,
    ) -> List[Dict[str, Any]]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        anchor_map: Dict[str, Dict[str, Any]] = {}
        for query in queries[:6]:
            self._log(f"anchor_query={query[:160]}")
            rows = domain.paper_retriever(cutoff_date=cutoff_date).retrieve(query, top_k=20)
            for rank, (doc, scores) in enumerate(rows, start=1):
                paper = domain.get_paper(doc.paper_id) or {}
                features = self._paper_pool_features(paper_id=doc.paper_id, candidate_pool=candidate_pool)
                existing = anchor_map.get(doc.paper_id)
                record = {
                    "paper_id": doc.paper_id,
                    "title": doc.title,
                    "abstract": str(paper.get("abstract") or ""),
                    "published_date": str(paper.get("published_date") or ""),
                    "scores": dict(scores),
                    "packet_match": features["packet_match"],
                    "pool_bonus": features["pool_bonus"],
                    "query_hit_count": 1,
                    "best_rank": rank,
                }
                if existing is None:
                    anchor_map[doc.paper_id] = record
                else:
                    existing["query_hit_count"] = int(existing.get("query_hit_count") or 0) + 1
                    existing["best_rank"] = min(int(existing.get("best_rank") or rank), rank)
                    existing["scores"].update(scores)
                    existing["packet_match"] = dedupe([*(existing.get("packet_match") or []), *features["packet_match"]])
                    existing["pool_bonus"] = max(float(existing.get("pool_bonus") or 0.0), float(features["pool_bonus"] or 0.0))
        anchors = sorted(
            anchor_map.values(),
            key=lambda row: (
                len(row.get("packet_match") or []),
                int(row.get("query_hit_count") or 0),
                float(row.get("pool_bonus") or 0.0),
                float((row.get("scores") or {}).get("combined_score") or 0.0),
                str(row.get("published_date") or ""),
            ),
            reverse=True,
        )[: self.max_anchor_papers]
        if not anchors and candidate_pool is not None:
            for paper in candidate_pool.papers[: self.max_anchor_papers]:
                paper_id = str(paper.get("paper_id") or "")
                if not paper_id:
                    continue
                anchors.append(
                    {
                        "paper_id": paper_id,
                        "title": str(paper.get("title") or ""),
                        "abstract": str(paper.get("abstract") or ""),
                        "published_date": str(paper.get("published_date") or ""),
                        "scores": candidate_pool.paper_scores.get(paper_id) or {},
                        "packet_match": [],
                        "pool_bonus": 1.0,
                        "query_hit_count": 0,
                        "best_rank": 999,
                    }
                )
        return anchors

    def _search_candidate_papers(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        query: Any,
        seen_paper_ids: Iterable[str],
        current_published_date: str,
        direction: str,
        candidate_pool: Optional[TaskCandidatePool] = None,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        seen = {str(x) for x in seen_paper_ids}
        queries = [str(query)] if isinstance(query, str) else [str(x) for x in (query or []) if str(x).strip()]
        if not queries:
            return []
        rows = merge_multi_query_results(
            domain.paper_retriever(cutoff_date=cutoff_date),
            queries,
            top_k_per_query=max(top_k, 10),
            limit=max(24, top_k * 3),
        )
        out: List[Dict[str, Any]] = []
        current_dt = self._parse_date(current_published_date)
        for doc, scores in rows:
            if doc.paper_id in seen:
                continue
            paper = domain.get_paper(doc.paper_id) or {}
            published_date = str(paper.get("published_date") or "")
            if direction == "forward" and current_published_date and published_date and published_date < current_published_date:
                continue
            if direction == "backward" and current_published_date and published_date and published_date > current_published_date:
                continue
            features = self._paper_pool_features(paper_id=doc.paper_id, candidate_pool=candidate_pool)
            published_dt = self._parse_date(published_date)
            temporal_priority = -10**9
            if current_dt is not None and published_dt is not None:
                delta = (published_dt - current_dt).days
                if direction == "forward" and delta >= 0:
                    temporal_priority = -abs(delta)
                elif direction == "backward" and delta <= 0:
                    temporal_priority = -abs(delta)
            out.append(
                {
                    "paper_id": doc.paper_id,
                    "title": doc.title,
                    "abstract": str(paper.get("abstract") or ""),
                    "published_date": published_date,
                    "scores": scores,
                    "packet_match": features["packet_match"],
                    "pool_bonus": features["pool_bonus"],
                    "query_hit_count": len(scores.get("matched_queries") or []),
                    "temporal_priority": temporal_priority,
                }
            )
        out.sort(
            key=lambda row: (
                float(row.get("pool_bonus") or 0.0),
                len(row.get("packet_match") or []),
                int(row.get("query_hit_count") or 0),
                float(row.get("temporal_priority") or -10**9),
                float((row.get("scores") or {}).get("combined_score") or 0.0),
            ),
            reverse=True,
        )
        return out

    def build_faithful_chain(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        anchor: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool] = None,
    ) -> Dict[str, Any]:
        topic = extract_focus_text(task)
        self._log(f"build_chain anchor={anchor.get('paper_id')} title={str(anchor.get('title') or '')[:120]}")
        seen_paper_ids: List[str] = []
        profiles: List[Dict[str, Any]] = []
        chain_logs: List[Dict[str, Any]] = []

        def append_profile(profile: Dict[str, Any], position: str) -> None:
            seen_paper_ids.append(str(profile.get("paper_id") or ""))
            profiles.append(profile)
            chain_logs.append(
                {
                    "position": position,
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "published_date": profile.get("published_date"),
                }
            )

        anchor_profile = self.extract_paper_profile(task=task, domain_id=domain_id, paper_id=str(anchor.get("paper_id") or ""))
        append_profile(anchor_profile, "anchor")

        current = anchor_profile
        while len(profiles) < self.max_chain_length:
            query = self._transition_queries(task=task, current=current, direction="forward")
            candidates = self._search_candidate_papers(
                task=task,
                domain_id=domain_id,
                query=query,
                seen_paper_ids=seen_paper_ids,
                current_published_date=str(current.get("published_date") or ""),
                direction="forward",
                candidate_pool=candidate_pool,
            )
            picked = self._select_transition_candidate(task=task, current=current, candidates=candidates, direction="forward")
            if picked is None:
                break
            profile = self.extract_paper_profile(task=task, domain_id=domain_id, paper_id=str(picked.get("paper_id") or ""))
            append_profile(profile, "forward")
            current = profile

        current = anchor_profile
        pending_refs = list(current.get("references") or [])
        while len(profiles) < self.max_chain_length:
            query = [*pending_refs[:3], *self._transition_queries(task=task, current=current, direction="backward")]
            candidates = self._search_candidate_papers(
                task=task,
                domain_id=domain_id,
                query=query,
                seen_paper_ids=seen_paper_ids,
                current_published_date=str(current.get("published_date") or ""),
                direction="backward",
                candidate_pool=candidate_pool,
                top_k=6,
            )
            picked = self._select_transition_candidate(task=task, current=current, candidates=candidates, direction="backward")
            if picked is None:
                break
            profile = self.extract_paper_profile(task=task, domain_id=domain_id, paper_id=str(picked.get("paper_id") or ""))
            profiles = [profile] + profiles
            seen_paper_ids = [str(profile.get("paper_id") or "")] + seen_paper_ids
            chain_logs = [
                {
                    "position": "backward",
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "published_date": profile.get("published_date"),
                }
            ] + chain_logs
            current = profile
            pending_refs = list(profile.get("references") or [])

        profiles = sorted(profiles, key=lambda row: str(row.get("published_date") or ""))
        idea_chain_text = ""
        experiments: List[str] = []
        entities: List[str] = []
        years: List[str] = []
        evidence = {"papers": [], "fulltext": []}
        for idx, profile in enumerate(profiles):
            idea_chain_text += f"{idx}.Paper:{profile.get('title')} idea:{profile.get('idea')}\n \n"
            experiments.append(str(profile.get("experiment") or ""))
            entities.append(str(profile.get("entities") or ""))
            years.append(str(profile.get("published_date") or ""))
            evidence["papers"].append(
                {
                    "evidence_id": f"P{idx+1}",
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "published_date": profile.get("published_date"),
                    "abstract_snippet": clip_text(profile.get("abstract") or "", 700),
                }
            )
            content_text = self.render_paper_content(domain_id=domain_id, paper_id=str(profile.get("paper_id") or ""))
            evidence["fulltext"].append(
                {
                    "evidence_id": f"F{idx+1}",
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "section_title": "paper_content",
                    "source_type": "coi_fulltext",
                    "snippet": clip_text(content_text, 1400),
                }
            )
        return {
            "profiles": profiles,
            "idea_chain_text": idea_chain_text.strip(),
            "experiments": experiments,
            "entities": entities,
            "years": years,
            "chain_logs": chain_logs,
            "evidence": evidence,
        }

    def summarize_entities(self, *, task: Dict[str, Any], entities: List[str]) -> str:
        prompt = f"""The current research topic is: {extract_focus_text(task)}. Please help me summarize and refine the following entities by merging, simplifying, or deleting them : {entities}
Please output strictly in the following format:
<entities>{{cleaned entities}}</entities>
"""
        response = self._complete(self.main_client, prompt, max_tokens=600, temperature=0.1)
        return extract_tag(response, "entities").strip() or " ".join(entities[:4])

    def project_benchmark_answer(
        self,
        *,
        task: Dict[str, Any],
        chain_text: str,
        trend: str,
        future: str,
        human: str,
        evidence: Dict[str, Any],
    ) -> str:
        family = str(task.get("family") or "")
        family_instruction = {
            "direction_forecasting": "State one concrete next-step direction and one trajectory label (accelerating, fragmenting, steady, or cooling).",
            "bottleneck_opportunity_discovery": "State one concrete historical bottleneck and one concrete realized or plausible opportunity opened by addressing it.",
            "strategic_research_planning": "Provide a prioritized 2-4 item research plan for the next six months, with explicit ranking and rationale.",
        }.get(family, "Answer the task concretely.")
        prompt = f"""You are adapting a Chain-of-Ideas research result to an offline benchmark answer.

Task:
{task.get('question')}

CoI chain:
{chain_text}

Trend:
{trend}

Human reasoning:
{human}

Future direction:
{future}

Requirements:
- Use the Chain-of-Ideas reasoning above as the primary basis.
- Do not mention having access to post-cutoff papers.
- Keep the answer concise but technically specific.
- {family_instruction}
- When useful, cite chain evidence inline with ids like [P1] or [F2].

Output only the final answer.
"""
        return self._complete(self.main_client, prompt, max_tokens=1200, temperature=0.15).strip()

    def gather_evidence(self, *, task: Dict[str, Any], domain_id: str, queries: List[str]) -> Dict[str, Any]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None

        paper_hits = merge_multi_query_results(domain.paper_retriever(cutoff_date=cutoff_date), queries, top_k_per_query=8, limit=10)
        paper_rows: List[Dict[str, Any]] = []
        paper_ids: List[str] = []
        for idx, (doc, scores) in enumerate(paper_hits, start=1):
            paper = domain.get_paper(doc.paper_id) or {}
            pub = paper.get("publication") or {}
            paper_ids.append(doc.paper_id)
            paper_rows.append(
                {
                    "evidence_id": f"P{idx}",
                    "paper_id": doc.paper_id,
                    "title": doc.title,
                    "published_date": paper.get("published_date"),
                    "venue": pub.get("venue_name"),
                    "citations": pub.get("citation_count"),
                    "abstract_snippet": clip_text((paper.get("abstract") or doc.text), 700),
                    "scores": scores,
                }
            )

        structure_hits = merge_multi_query_results(domain.structure_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids[:10]), queries, top_k_per_query=6, limit=8)
        structures: Dict[str, Dict[str, Any]] = {}
        structure_rows: List[Dict[str, Any]] = []
        for idx, (doc, scores) in enumerate(structure_hits, start=1):
            row = {
                "evidence_id": f"T{idx}",
                "paper_id": doc.paper_id,
                "title": doc.title,
                "problem_statement": clip_text(doc.meta.get("problem_statement"), 260),
                "limitations": list(doc.meta.get("limitations") or [])[:4],
                "future_work": list(doc.meta.get("future_work") or [])[:4],
                "core_ideas": list(doc.meta.get("core_ideas") or [])[:4],
                "snippet": clip_text(doc.text, 900),
                "scores": scores,
            }
            structures[doc.paper_id] = row
            structure_rows.append(row)

        page_hits = merge_multi_query_results(domain.pageindex_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids[:10]), queries, top_k_per_query=6, limit=8)
        page_rows: List[Dict[str, Any]] = []
        for idx, (doc, scores) in enumerate(page_hits, start=1):
            page_rows.append(
                {
                    "evidence_id": f"G{idx}",
                    "paper_id": doc.paper_id,
                    "title": doc.meta.get("paper_title") or doc.title,
                    "section_title": doc.meta.get("section_title"),
                    "kind": doc.meta.get("kind"),
                    "snippet": clip_text(doc.text, 900),
                    "scores": scores,
                }
            )

        fulltext_rows: List[Dict[str, Any]] = []
        if paper_ids:
            builtin_hits = merge_multi_query_results(
                domain.section_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids[:10]),
                queries,
                top_k_per_query=6,
                limit=8,
            )
            for doc, scores in builtin_hits:
                fulltext_rows.append(
                    {
                        "paper_id": doc.paper_id,
                        "title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "level": doc.meta.get("level"),
                        "source_type": "kb_section",
                        "snippet": clip_text(doc.text, 1200),
                        "scores": scores,
                    }
                )

        if self.fulltext_cache is not None and paper_ids:
            fulltext_queries = list(queries)
            family = str(task.get("family") or "")
            if family == "bottleneck_opportunity_discovery":
                fulltext_queries.extend(
                    [
                        f"{extract_focus_text(task)} limitation failure challenge",
                        f"{extract_focus_text(task)} future work open problem metric benchmark",
                    ]
                )
            elif family == "direction_forecasting":
                fulltext_queries.extend(
                    [
                        f"{extract_focus_text(task)} future work next step direction",
                        f"{extract_focus_text(task)} benchmark evaluation limitation",
                    ]
                )
            section_retriever = self.fulltext_cache.build_section_retriever(
                paper_ids=paper_ids[:4],
                cutoff_date=cutoff_date,
                allow_fetch=self.allow_fulltext_fetch,
            )
            fulltext_hits = merge_multi_query_results(section_retriever, fulltext_queries, top_k_per_query=6, limit=10)
            for doc, scores in fulltext_hits:
                fulltext_rows.append(
                    {
                        "paper_id": doc.paper_id,
                        "title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "level": doc.meta.get("level"),
                        "source_type": doc.meta.get("source_type"),
                        "snippet": clip_text(doc.text, 1200),
                        "scores": scores,
                    }
                )
        deduped_fulltext: List[Dict[str, Any]] = []
        seen_fulltext = set()
        for row in fulltext_rows:
            key = (
                str(row.get("paper_id") or ""),
                str(row.get("section_title") or ""),
                str(row.get("source_type") or ""),
            )
            if key in seen_fulltext:
                continue
            seen_fulltext.add(key)
            deduped_fulltext.append(row)
        fulltext_rows = []
        for idx, row in enumerate(
            sorted(
                deduped_fulltext,
                key=lambda item: float((item.get("scores") or {}).get("combined_score") or 0.0),
                reverse=True,
            )[:10],
            start=1,
        ):
            row = dict(row)
            row["evidence_id"] = f"F{idx}"
            fulltext_rows.append(row)

        return {
            "queries": queries,
            "papers": paper_rows,
            "structures": structure_rows,
            "pageindex": page_rows,
            "fulltext": fulltext_rows,
            "structures_by_paper": structures,
        }

    def build_chain_cards(self, evidence: Dict[str, Any], *, max_papers: int = 4) -> Dict[str, Any]:
        papers = list(evidence.get("papers") or [])
        structures_by_paper = evidence.get("structures_by_paper") or {}
        fulltext_rows = list(evidence.get("fulltext") or [])
        selected = papers[:max_papers]
        selected.sort(key=lambda row: (str(row.get("published_date") or ""), -float((row.get("scores") or {}).get("combined_score") or 0.0)))

        cards: List[Dict[str, Any]] = []
        entity_pool: List[str] = []
        for idx, paper in enumerate(selected):
            structure = structures_by_paper.get(str(paper.get("paper_id") or "")) or {}
            limitations = list(structure.get("limitations") or [])[:3]
            future_work = list(structure.get("future_work") or [])[:3]
            core_ideas = list(structure.get("core_ideas") or [])[:3]
            fulltext_snippets = [
                clip_text(row.get("snippet") or "", 350)
                for row in fulltext_rows
                if str(row.get("paper_id") or "") == str(paper.get("paper_id") or "")
            ][:2]
            entity_pool.extend(limitations)
            entity_pool.extend(future_work)
            entity_pool.extend(core_ideas)
            cards.append(
                {
                    "order": idx,
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "published_date": paper.get("published_date"),
                    "venue": paper.get("venue"),
                    "citations": paper.get("citations"),
                    "abstract_snippet": paper.get("abstract_snippet"),
                    "problem_statement": structure.get("problem_statement"),
                    "limitations": limitations,
                    "future_work": future_work,
                    "core_ideas": core_ideas,
                    "fulltext_snippets": fulltext_snippets,
                }
            )
        card_text = []
        for idx, card in enumerate(cards):
            card_text.append(
                _join_lines(
                    [
                        f"Paper {idx}",
                        f"Title: {card.get('title')}",
                        f"Published: {card.get('published_date')}",
                        f"Venue: {card.get('venue')}",
                        f"Citations: {card.get('citations')}",
                        f"Abstract: {card.get('abstract_snippet')}",
                        f"Problem: {card.get('problem_statement')}",
                        f"Core ideas: {'; '.join(card.get('core_ideas') or [])}",
                        f"Limitations: {'; '.join(card.get('limitations') or [])}",
                        f"Future work: {'; '.join(card.get('future_work') or [])}",
                        f"Fulltext highlights: {' || '.join(card.get('fulltext_snippets') or [])}",
                    ]
                )
            )
        return {
            "cards": cards,
            "card_text": "\n\n".join(card_text).strip(),
            "entities": dedupe(entity_pool)[:16],
        }

    def generate_trend(self, *, task: Dict[str, Any], chain: Dict[str, Any]) -> str:
        topic = extract_focus_text(task)
        prompt = get_deep_trend_idea_chains_prompt(chain.get("card_text") or "", chain.get("entities") or [], topic)
        try:
            response = self._complete(self.main_client, prompt, max_tokens=1200, temperature=0.2)
            trend = extract_tag(response, "trend").strip()
            if trend:
                return trend
        except Exception:
            response = ""
        fallback_prompt = f"""Summarize the historical progression of this topic using the paper chain below.

Topic: {topic}

Paper chain:
{chain.get('card_text')}

Write 3-5 concise sentences describing how the research moved from earlier work to later work. Focus on technical progression, not generic history.
"""
        try:
            return self._complete(self.main_client, fallback_prompt, max_tokens=700, temperature=0.2).strip()
        except Exception:
            return ""

    def generate_future(self, *, task: Dict[str, Any], chain: Dict[str, Any], trend: str) -> Dict[str, str]:
        topic = extract_focus_text(task)
        base_prompt = get_deep_generate_future_direciton_prompt(chain.get("card_text") or "", trend, topic, chain.get("entities") or [])
        prompt = f"""{base_prompt}

Return JSON only:
{{
  "human_reasoning": "2-4 concise sentences explaining why the direction follows from the historical chain",
  "future_direction": "one concrete next-step direction phrase plus one concise explanatory sentence"
}}
"""
        try:
            obj = complete_json_object(
                self.main_client,
                [
                    {"role": "system", "content": "You are a precise research agent. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=900,
                temperature=0.2,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            result = {
                "human_reasoning": str(obj.get("human_reasoning") or "").strip(),
                "future_direction": str(obj.get("future_direction") or "").strip(),
                "raw": json.dumps(obj, ensure_ascii=False),
            }
            if result["human_reasoning"] or result["future_direction"]:
                return result
        except Exception:
            response = ""
        fallback_prompt = f"""Infer one plausible next-step future research direction from the historical chain below.

Topic: {topic}

Historical chain:
{chain.get('card_text')}

Trend summary:
{trend}

Return JSON only:
{{
  "human_reasoning": "2-4 concise sentences about how the future direction follows from the prior chain",
  "future_direction": "one concrete direction phrase and a short explanation"
}}
"""
        try:
            obj = complete_json_object(
                self.main_client,
                [
                    {"role": "system", "content": "You are a precise research agent. Output JSON only."},
                    {"role": "user", "content": fallback_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=700,
                temperature=0.2,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            return {
                "human_reasoning": str(obj.get("human_reasoning") or "").strip(),
                "future_direction": str(obj.get("future_direction") or "").strip(),
                "raw": json.dumps(obj, ensure_ascii=False),
            }
        except Exception:
            return {"human_reasoning": "", "future_direction": "", "raw": response}

    def _render_evidence_block(self, evidence: Dict[str, Any]) -> str:
        parts: List[str] = []
        for row in (evidence.get("papers") or [])[:6]:
            parts.append(_join_lines([f"[{row['evidence_id']}] {row.get('title')}", f"Published: {row.get('published_date')}", f"Venue: {row.get('venue')}", f"Citations: {row.get('citations')}", row.get("abstract_snippet")]))
            parts.append("")
        for row in (evidence.get("structures") or [])[:5]:
            parts.append(_join_lines([f"[{row['evidence_id']}] {row.get('title')}", f"Problem: {row.get('problem_statement')}", f"Limitations: {'; '.join(row.get('limitations') or [])}", f"Future work: {'; '.join(row.get('future_work') or [])}", row.get("snippet")]))
            parts.append("")
        for row in (evidence.get("pageindex") or [])[:4]:
            parts.append(_join_lines([f"[{row['evidence_id']}] {row.get('title')} / {row.get('section_title')}", f"Kind: {row.get('kind')}", row.get("snippet")]))
            parts.append("")
        for row in (evidence.get("fulltext") or [])[:6]:
            parts.append(
                _join_lines(
                    [
                        f"[{row['evidence_id']}] {row.get('title')} / {row.get('section_title')}",
                        f"Source: {row.get('source_type')}",
                        row.get("snippet"),
                    ]
                )
            )
            parts.append("")
        return "\n".join(parts).strip()

    def draft_answer(self, *, task: Dict[str, Any], chain: Dict[str, Any], trend: str, future: Dict[str, str], evidence: Dict[str, Any]) -> str:
        family = str(task.get("family") or "")
        evidence_block = self._render_evidence_block(evidence)
        family_instruction = {
            "direction_forecasting": "State one concrete next-step direction and one trajectory label (accelerating, fragmenting, steady, or cooling).",
            "bottleneck_opportunity_discovery": "State one historically grounded bottleneck and one concrete opportunity that would open if it were addressed.",
            "strategic_research_planning": "Prioritize 2-4 directions for the next six months and justify the ranking.",
        }.get(family, "Answer the task concretely.")
        prompt = f"""You are CoI-Agent-Offline, an offline adaptation of Chain-of-Ideas Agent for benchmark answering.

Task metadata:
- Task ID: {task.get('task_id')}
- Family: {family}
- Domain: {task.get('domain')}
- Time cutoff: {task.get('time_cutoff')}
- Title: {task.get('title')}

Question:
{task.get('question')}

CoI-style research chain:
{chain.get('card_text')}

Inferred research trend:
{trend}

Inferred future direction:
{future.get('future_direction')}

Human-style reasoning trace:
{future.get('human_reasoning')}

Historical evidence (cutoff-safe only):
{evidence_block}

Requirements:
- Use only the historical evidence above.
- Do not claim direct access to post-cutoff papers.
- {family_instruction}
- When the evidence contains explicit limitation, problem, or future-work phrases, preserve that wording where possible instead of paraphrasing it away.
- Prefer concrete technical language over generic trend language.
- Cite evidence inline with ids like [P1], [T2], [G1] when useful.
- Keep the answer concise but substantive.
"""
        return self._complete(self.main_client, prompt, max_tokens=1200, temperature=0.2).strip()

    def critique_answer(self, *, task: Dict[str, Any], answer: str, trend: str, future: Dict[str, str], evidence: Dict[str, Any]) -> str:
        evidence_block = self._render_evidence_block(evidence)
        prompt = f"""You are the review agent in a Chain-of-Ideas style benchmark system.

Question:
{task.get('question')}

Current draft answer:
{answer}

Trend summary:
{trend}

Future direction summary:
{future.get('future_direction')}

Evidence:
{evidence_block}

Give a short critique focused on:
1. missing evidence grounding
2. vague or unsupported claims
3. task-family mismatch
4. weak prioritization / weak bottleneck-opportunity linkage / weak trajectory call

Output 3-5 concise bullet points only.
"""
        return self._complete(self.cheap_client, prompt, max_tokens=600, temperature=0.1).strip()

    def revise_answer(self, *, task: Dict[str, Any], draft: str, critique: str, evidence: Dict[str, Any]) -> str:
        evidence_block = self._render_evidence_block(evidence)
        prompt = f"""Revise the benchmark answer using the critique.

Question:
{task.get('question')}

Draft answer:
{draft}

Critique:
{critique}

Evidence:
{evidence_block}

Constraints:
- Keep only claims supported by historical evidence.
- Make the answer more specific and task-aligned.
- Keep inline evidence ids when useful.
- Output only the revised answer.
"""
        return self._complete(self.main_client, prompt, max_tokens=1200, temperature=0.15).strip()

    def run_task(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        return self.run_task_faithful(task=task, domain_id=domain_id)

    def run_task_faithful(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        self._log(f"run_task task_id={task.get('task_id')} family={task.get('family')} domain_id={domain_id}")
        family = str(task.get("family") or "")
        queries = self.build_queries(task)
        self._log(f"queries={queries}")
        candidate_pool = self.retrieval_adaptor.build_candidate_pool(task=task, domain_id=domain_id)
        self._log(
            f"candidate_pool packets={candidate_pool.packet_ids} paper_count={len(candidate_pool.papers)} "
            f"paper_ids={[str(row.get('paper_id')) for row in candidate_pool.papers[:8]]}"
        )
        anchors = self.search_anchor_papers(task=task, domain_id=domain_id, queries=queries, candidate_pool=candidate_pool)
        anchors = sorted(
            anchors,
            key=lambda row: self._family_anchor_score(task=task, anchor=row, candidate_pool=candidate_pool),
            reverse=True,
        )
        anchors = anchors[: max(1, self._candidate_anchor_count(task))]
        self._log(f"anchor_count={len(anchors)}")
        if not anchors:
            return {
                "answer": "",
                "queries": queries,
                "anchors": [],
                "chain": {},
                "trend": "",
                "future": {"future_direction": "", "human_reasoning": "", "raw": ""},
                "draft_answer": "",
                "critique": "",
                "retrieval_mode": "coi_offline_faithful",
                "evidence": {"papers": [], "fulltext": [], "structures": [], "pageindex": []},
                "diagnostics": {"retrieved_papers": 0, "retrieved_fulltext_sections": 0, "reflection_steps": 0, "revision_rounds": 0, "answer_changed_after_revision": False},
            }

        candidate_runs = []
        for anchor in anchors:
            chain = self.build_faithful_chain(task=task, domain_id=domain_id, anchor=anchor, candidate_pool=candidate_pool)
            self._log(f"chain_built anchor={anchor.get('paper_id')} profile_count={len(chain.get('profiles') or [])}")
            entities = self.summarize_entities(task=task, entities=chain.get("entities") or [])
            trend_prompt = get_deep_trend_idea_chains_prompt(chain.get("idea_chain_text") or "", entities, extract_focus_text(task))
            trend_raw = self._complete(self.main_client, trend_prompt, max_tokens=1600, temperature=0.2)
            trend = strip_xmlish_tags(extract_tag(trend_raw, "trend").strip() or trend_raw.strip())
            future_prompt = get_deep_generate_future_direciton_prompt(chain.get("idea_chain_text") or "", trend, extract_focus_text(task), entities)
            future_raw = self._complete(self.main_client, future_prompt, max_tokens=1400, temperature=0.2)
            future = strip_xmlish_tags(extract_tag(future_raw, "future").strip() or future_raw.strip())
            human = strip_xmlish_tags(extract_tag(future_raw, "human").strip())
            projected = self._project_structured_answer(
                task=task,
                chain_text=chain.get("idea_chain_text") or "",
                trend=trend,
                future=future,
                human=human,
                evidence=chain.get("evidence") or {},
                candidate_pool=candidate_pool,
                chain=chain,
            )
            answer = self._verbalize_structured_answer(task=task, projected=projected)
            run_score = self._score_candidate_run(
                task=task,
                candidate_pool=candidate_pool,
                chain=chain,
                projected=projected,
                trend=trend,
                future=future,
            )
            self._log(
                f"candidate_done anchor={anchor.get('paper_id')} profile_count={len(chain.get('profiles') or [])} "
                f"trend_chars={len(trend)} future_chars={len(future)} answer_chars={len(answer)} run_score={run_score}"
            )
            candidate_runs.append(
                {
                    "anchor": anchor,
                    "chain": chain,
                    "entities": entities,
                    "trend": trend,
                    "future": future,
                    "human": human,
                    "projected": projected,
                    "run_score": run_score,
                    "answer": answer,
                }
            )
        candidate_runs.sort(
            key=lambda item: (
                float(item.get("run_score") or 0.0),
                len(item["chain"].get("profiles") or []),
                len(normalize_ws(item.get("trend") or "")),
            ),
            reverse=True,
        )
        if family == "direction_forecasting":
            final_answer, family_head, evidence = self._run_direction_head(
                task=task,
                domain_id=domain_id,
                candidate_pool=candidate_pool,
                candidate_runs=candidate_runs,
            )
        elif family == "strategic_research_planning":
            final_answer, family_head, evidence = self._run_planning_head(
                task=task,
                domain_id=domain_id,
                candidate_pool=candidate_pool,
                candidate_runs=candidate_runs,
            )
        else:
            final_answer, family_head, evidence = self._run_bottleneck_head(
                task=task,
                domain_id=domain_id,
                candidate_pool=candidate_pool,
                candidate_runs=candidate_runs,
            )
        best = candidate_runs[0]
        evidence = dict(evidence or {})
        evidence.setdefault("structures", [])
        evidence.setdefault("pageindex", [])
        evidence.setdefault("candidate_node_evidence", [])
        return {
            "answer": final_answer or best["answer"],
            "queries": queries,
            "anchors": anchors,
            "chain": {
                "idea_chain_text": best["chain"].get("idea_chain_text"),
                "profiles": best["chain"].get("profiles"),
                "chain_logs": best["chain"].get("chain_logs"),
                "entities": best["entities"],
                "experiments": best["chain"].get("experiments"),
                "years": best["chain"].get("years"),
            },
            "candidate_chains": [
                {
                    "anchor": run.get("anchor"),
                    "run_score": run.get("run_score"),
                    "trend": run.get("trend"),
                    "future": run.get("future"),
                    "answer": run.get("answer"),
                    "chain_logs": (run.get("chain") or {}).get("chain_logs"),
                    "profiles": (run.get("chain") or {}).get("profiles"),
                }
                for run in candidate_runs[: self._candidate_anchor_count(task)]
            ],
            "trend": best["trend"],
            "future": {"future_direction": best["future"], "human_reasoning": best["human"], "raw": ""},
            "draft_answer": self._verbalize_structured_answer(task=task, projected={"final": best.get("projected", {}).get("draft"), "candidate_labels": best.get("projected", {}).get("candidate_labels") or []}),
            "critique": json.dumps(family_head or best.get("projected", {}).get("final") or {}, ensure_ascii=False),
            "family_head": family_head,
            "retrieval_mode": "coi_offline_faithful_v3: query -> family_packet_pool -> family_anchor_selection -> multi_chain_coi_backbone -> family_analysis_head -> unified_answer_adapter",
            "evidence": evidence,
            "diagnostics": {
                "selected_packet_ids": candidate_pool.packet_ids,
                "candidate_pool_papers": [str(row.get("paper_id") or "") for row in candidate_pool.papers[:12]],
                "retrieved_papers": len(evidence.get("papers") or []),
                "retrieved_structures": len(evidence.get("structures") or []),
                "retrieved_pageindex": len(evidence.get("pageindex") or []),
                "retrieved_fulltext_sections": len(evidence.get("fulltext") or []),
                "reflection_steps": 2,
                "revision_rounds": 1,
                "answer_changed_after_revision": normalize_ws(final_answer or best["answer"]) != normalize_ws(self._verbalize_structured_answer(task=task, projected={"final": best.get("projected", {}).get("draft"), "candidate_labels": best.get("projected", {}).get("candidate_labels") or []})),
                "candidate_run_score": float(best.get("run_score") or 0.0),
                "candidate_run_count": len(candidate_runs),
            },
        }

    def run_task_legacy(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        queries = self.build_queries(task)
        evidence = self.gather_evidence(task=task, domain_id=domain_id, queries=queries)
        chain = self.build_chain_cards(evidence)
        trend = self.generate_trend(task=task, chain=chain)
        future = self.generate_future(task=task, chain=chain, trend=trend)
        draft = self.draft_answer(task=task, chain=chain, trend=trend, future=future, evidence=evidence)
        critique = self.critique_answer(task=task, answer=draft, trend=trend, future=future, evidence=evidence)
        final_answer = self.revise_answer(task=task, draft=draft, critique=critique, evidence=evidence)
        return {
            "answer": final_answer,
            "queries": queries,
            "evidence": evidence,
            "chain": chain,
            "trend": trend,
            "future": future,
            "draft_answer": draft,
            "critique": critique,
            "retrieval_mode": "coi_query_generation -> offline_hybrid_retrieval -> chain_cards -> trend_future_reasoning -> review_revision",
            "diagnostics": {
                "retrieved_papers": len(evidence.get("papers") or []),
                "retrieved_structures": len(evidence.get("structures") or []),
                "retrieved_pageindex": len(evidence.get("pageindex") or []),
                "retrieved_fulltext_sections": len(evidence.get("fulltext") or []),
                "reflection_steps": 1,
                "revision_rounds": 1,
                "answer_changed_after_revision": final_answer.strip() != draft.strip(),
            },
        }
