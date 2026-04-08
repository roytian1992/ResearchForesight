from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from researchworld.baseline_runner import (
    DomainCorpus,
    HybridRetriever,
    RetrievalDoc,
    clip_text,
    parse_iso_date,
    tokenize,
)
from researchworld.benchmark_v2 import summarize_structure_coverage, top_future_work_signals, top_limitation_signals
from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.verbalization import normalize_display_name, public_topic_from_packet


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class SupportPacket:
    packet_id: str
    domain: str
    node_id: str
    display_name: str
    dimension_id: str
    text: str
    payload: Dict[str, Any]


class SupportPacketRetriever:
    def __init__(self, domain_id: str):
        rows = load_json(ROOT / "data" / "support_packets" / domain_id / "node_support_packets.json")
        self.packets: List[SupportPacket] = []
        tokenized: List[List[str]] = []
        for row in rows:
            topic = public_topic_from_packet(row)
            hist = row.get("historical_stats") or {}
            historical_papers = row.get("historical_representative_papers") or []
            hist_titles = "; ".join(str(p.get("title") or "") for p in historical_papers[:6])
            lineage = " > ".join(normalize_display_name(x) for x in (row.get("lineage") or []) if x)
            text = "\n".join(
                [
                    f"Display name: {normalize_display_name(row.get('display_name') or '')}",
                    f"Public topic: {topic}",
                    f"Description: {row.get('description')}",
                    f"Dimension: {row.get('dimension_id')}",
                    f"Lineage: {lineage}",
                    f"Historical paper count: {hist.get('paper_count', 0)}",
                    f"Historical top venue share: {hist.get('top_conf_share', 0.0)}",
                    f"Historical citation median: {hist.get('citation_median', 0.0)}",
                    f"Historical papers: {hist_titles}",
                ]
            )
            packet = SupportPacket(
                packet_id=str(row["packet_id"]),
                domain=str(row["domain"]),
                node_id=str(row["node_id"]),
                display_name=str(row.get("display_name") or ""),
                dimension_id=str(row.get("dimension_id") or ""),
                text=text,
                payload=row,
            )
            self.packets.append(packet)
            tokenized.append(tokenize(text))
        self.bm25 = BM25Okapi(tokenized)

    def retrieve_scored(self, query: str, *, top_k: int = 3) -> List[Tuple[SupportPacket, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.packets[idx], float(score)) for idx, score in ranked]

    def retrieve(self, query: str, *, top_k: int = 3) -> List[SupportPacket]:
        return [packet for packet, _ in self.retrieve_scored(query, top_k=top_k)]


class PaperStructureStore:
    def __init__(self, domain_id: str):
        path = ROOT / "data" / "support_packets" / "paper_structures" / domain_id / "paper_structures.jsonl"
        self.rows_by_paper: Dict[str, Dict[str, Any]] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.rows_by_paper[str(row.get("paper_id") or "")] = row

    def get(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.rows_by_paper.get(str(paper_id or ""))

    def summarize(self, paper_ids: Iterable[str]) -> Dict[str, Any]:
        rows = [self.rows_by_paper[pid] for pid in paper_ids if pid in self.rows_by_paper]
        return {
            "structure_coverage": summarize_structure_coverage(rows),
            "top_limitations": top_limitation_signals(rows, top_k=5),
            "top_future_work": top_future_work_signals(rows, top_k=5),
            "top_core_ideas": self._top_core_ideas(rows, top_k=5),
            "paper_cards": [self._paper_card(row) for row in rows[:4]],
            "evidence_refs": self._collect_refs(rows, limit=12),
        }

    def _top_core_ideas(self, rows: List[Dict[str, Any]], *, top_k: int) -> List[Dict[str, Any]]:
        counter: Counter[str] = Counter()
        examples: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            for item in row.get("core_ideas") or []:
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                counter[name] += 1
                if name not in examples:
                    examples[name] = {
                        "name": name,
                        "mechanism": str(item.get("mechanism") or ""),
                        "paper_id": row.get("paper_id"),
                        "title": row.get("title"),
                    }
        out = []
        for name, count in counter.most_common(top_k):
            item = dict(examples.get(name) or {})
            item["count"] = count
            out.append(item)
        return out

    def _paper_card(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "paper_id": row.get("paper_id"),
            "title": row.get("title"),
            "published": row.get("published"),
            "problem_statement": str(row.get("problem_statement") or ""),
            "top_limitations": [item.get("name") for item in (row.get("explicit_limitations") or [])[:3] if item.get("name")],
            "top_future_work": [item.get("direction") for item in (row.get("future_work") or [])[:3] if item.get("direction")],
            "top_core_ideas": [item.get("name") for item in (row.get("core_ideas") or [])[:3] if item.get("name")],
        }

    def _collect_refs(self, rows: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
        refs: List[Dict[str, Any]] = []
        seen = set()
        for row in rows:
            for field in ("explicit_limitations", "future_work", "core_ideas"):
                for item in row.get(field) or []:
                    for ref in item.get("evidence_refs") or []:
                        key = (row.get("paper_id"), ref.get("node_id"), ref.get("section_path"))
                        if key in seen:
                            continue
                        seen.add(key)
                        refs.append(
                            {
                                "paper_id": row.get("paper_id"),
                                "title": row.get("title"),
                                "node_id": ref.get("node_id"),
                                "section_path": ref.get("section_path"),
                                "kind": ref.get("kind"),
                                "quote": ref.get("quote"),
                            }
                        )
                        if len(refs) >= limit:
                            return refs
        return refs


class QuarterlyNodeHistoryStore:
    def __init__(self, domain_id: str):
        path = ROOT / "data" / "aggregates" / "quarterly_node_growth.json"
        rows = load_json(path) if path.exists() else []
        self.rows_by_node: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            if str(row.get("domain") or "") != domain_id:
                continue
            node_id = str(row.get("node_id") or "")
            self.rows_by_node.setdefault(node_id, []).append(row)
        for node_rows in self.rows_by_node.values():
            node_rows.sort(key=lambda row: self._slice_key(str(row.get("time_slice") or "")))

    def summarize(self, node_id: str, *, up_to_slice: str, limit: int = 6) -> List[Dict[str, Any]]:
        target_key = self._slice_key(up_to_slice)
        rows = [
            {
                "time_slice": row.get("time_slice"),
                "paper_count": row.get("paper_count"),
                "paper_growth": row.get("paper_growth"),
                "top_conf_share": row.get("top_conf_share"),
                "citation_median": row.get("citation_median"),
            }
            for row in (self.rows_by_node.get(node_id) or [])
            if self._slice_key(str(row.get("time_slice") or "")) <= target_key
        ]
        return rows[-limit:]

    @staticmethod
    def _slice_key(value: str) -> Tuple[int, int]:
        raw = str(value or "").strip()
        if len(raw) >= 6 and raw[4] == "Q":
            try:
                return int(raw[:4]), int(raw[5:])
            except Exception:
                return (0, 0)
        return (0, 0)


class ResearchArc:
    def __init__(
        self,
        *,
        answer_client: OpenAICompatChatClient,
        critic_client: Optional[OpenAICompatChatClient] = None,
    ):
        self.answer_client = answer_client
        self.critic_client = critic_client or answer_client
        self.domain_corpora: Dict[str, DomainCorpus] = {}
        self.support_retrievers: Dict[str, SupportPacketRetriever] = {}
        self.structure_stores: Dict[str, PaperStructureStore] = {}
        self.node_history_stores: Dict[str, QuarterlyNodeHistoryStore] = {}

    def _domain_corpus(self, domain_id: str) -> DomainCorpus:
        if domain_id not in self.domain_corpora:
            self.domain_corpora[domain_id] = DomainCorpus(domain_id)
        return self.domain_corpora[domain_id]

    def _support_retriever(self, domain_id: str) -> SupportPacketRetriever:
        if domain_id not in self.support_retrievers:
            self.support_retrievers[domain_id] = SupportPacketRetriever(domain_id)
        return self.support_retrievers[domain_id]

    def _structure_store(self, domain_id: str) -> PaperStructureStore:
        if domain_id not in self.structure_stores:
            self.structure_stores[domain_id] = PaperStructureStore(domain_id)
        return self.structure_stores[domain_id]

    def _node_history_store(self, domain_id: str) -> QuarterlyNodeHistoryStore:
        if domain_id not in self.node_history_stores:
            self.node_history_stores[domain_id] = QuarterlyNodeHistoryStore(domain_id)
        return self.node_history_stores[domain_id]

    def run_task(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        task_parse = self.parse_task(task)
        retrieval_plan = self.plan_retrieval(task, task_parse)
        corpus = self._domain_corpus(domain_id)
        support_rows = self._support_retriever(domain_id).retrieve(task["question"], top_k=6)
        benchmark_context = self.build_benchmark_context(task=task, domain_id=domain_id, support_rows=support_rows)
        evidence = self.retrieve_evidence(
            task=task,
            domain_id=domain_id,
            corpus=corpus,
            support_rows=support_rows,
            benchmark_context=benchmark_context,
        )
        coi = self.build_coi(
            task=task,
            task_parse=task_parse,
            domain_id=domain_id,
            support_rows=support_rows,
            benchmark_context=benchmark_context,
            evidence=evidence,
        )
        evidence = self.refine_evidence_by_chains(
            task=task,
            task_parse=task_parse,
            corpus=corpus,
            benchmark_context=benchmark_context,
            seed_coi=coi,
            evidence=evidence,
        )
        coi = self.build_coi(
            task=task,
            task_parse=task_parse,
            domain_id=domain_id,
            support_rows=support_rows,
            benchmark_context=benchmark_context,
            evidence=evidence,
        )
        focused_context = self.build_focused_context(
            task_parse=task_parse,
            support_rows=support_rows,
            benchmark_context=benchmark_context,
            coi=coi,
            evidence=evidence,
        )
        hypotheses = self.generate_hypotheses(
            task=task,
            task_parse=task_parse,
            coi=coi,
            focused_context=focused_context,
            evidence=evidence,
        )
        decision = self.select_hypothesis(
            task=task,
            task_parse=task_parse,
            hypotheses=hypotheses,
            coi=coi,
            focused_context=focused_context,
        )
        return {
            "task_parse": task_parse,
            "retrieval_plan": retrieval_plan,
            "support_packets": [self._sanitize_packet(x.payload) for x in support_rows],
            "benchmark_context": benchmark_context,
            "evidence": evidence,
            "coi": coi,
            "focused_context": focused_context,
            "candidate_hypotheses": hypotheses,
            "decision": decision,
            "answer": decision.get("final_answer") or "",
        }

    def parse_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        family = str(task.get("family") or "")
        subtype = "trend_and_outlook"
        if family == "bottleneck_opportunity_discovery":
            subtype = "bottleneck_opportunity_linkage"
        elif family == "direction_forecasting":
            subtype = "trajectory_forecast"
        elif family == "strategic_research_planning":
            subtype = "research_agenda"
        return {
            "family": family,
            "subtype": subtype,
            "domain": task.get("domain"),
            "horizon": task.get("horizon"),
            "time_cutoff": task.get("time_cutoff"),
        }

    def plan_retrieval(self, task: Dict[str, Any], task_parse: Dict[str, Any]) -> List[Dict[str, Any]]:
        family = task_parse["family"]
        common = [
            {"step": "retrieve_candidate_nodes", "goal": "match the question to benchmark-style support packets"},
            {"step": "global_hybrid_recall", "goal": "collect high-recall historical papers before cutoff"},
            {"step": "collect_node_history", "goal": "recover pre-cutoff quarterly node trajectory used during benchmark construction"},
            {"step": "collect_representative_papers", "goal": "gather the node's historical representative papers before cutoff"},
            {"step": "collect_structure_signals", "goal": "extract historical limitations, future-work, and idea signals from case-paper structures"},
            {"step": "chain_conditioned_rerank", "goal": "use top idea chains to rerank papers and sections under path constraints"},
            {"step": "small_to_big_expansion", "goal": "expand from top papers to their most relevant sections and supporting structures"},
        ]
        if family == "bottleneck_opportunity_discovery":
            common.append({"step": "build_limitation_chain", "goal": "organize recurring bottlenecks and plausible opportunity openings"})
        elif family == "direction_forecasting":
            common.append({"step": "build_transition_chain", "goal": "organize historical transitions and pre-cutoff momentum signals"})
        else:
            common.append({"step": "build_research_agenda_chain", "goal": "organize mature lines, gaps, and combinational opportunities"})
        return common

    def build_benchmark_context(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        support_rows: List[SupportPacket],
    ) -> Dict[str, Any]:
        structure_store = self._structure_store(domain_id)
        history_store = self._node_history_store(domain_id)
        node_cards = []
        all_rep_paper_ids: List[str] = []
        query_terms: List[str] = [task["question"]]
        for packet in support_rows:
            payload = packet.payload
            rep_papers = list(payload.get("historical_representative_papers") or [])[:6]
            rep_paper_ids = [str(row.get("paper_id") or "") for row in rep_papers if row.get("paper_id")]
            all_rep_paper_ids.extend(rep_paper_ids)
            struct = structure_store.summarize(rep_paper_ids[:4])
            topic = public_topic_from_packet(payload)
            node_card = {
                "packet_id": payload.get("packet_id"),
                "node_id": payload.get("node_id"),
                "display_name": normalize_display_name(payload.get("display_name") or ""),
                "public_topic": topic,
                "dimension_id": payload.get("dimension_id"),
                "description": payload.get("description"),
                "historical_stats": payload.get("historical_stats"),
                "split_pressure": payload.get("split_pressure"),
                "direction_score": payload.get("direction_score"),
                "planning_priority_score": payload.get("planning_priority_score"),
                "emergent_descendants": [
                    {
                        "display_name": normalize_display_name(x.get("display_name") or ""),
                        "future_paper_count": x.get("future_paper_count"),
                        "created_time_slice": x.get("created_time_slice"),
                    }
                    for x in (payload.get("emergent_descendants") or [])[:6]
                ],
                "quarterly_history": history_store.summarize(
                    str(payload.get("node_id") or ""),
                    up_to_slice=str(payload.get("history_structure_slice") or ""),
                    limit=6,
                ),
                "historical_representative_papers": rep_papers[:5],
                "top_limitations": struct["top_limitations"][:4],
                "top_future_work": struct["top_future_work"][:4],
                "top_core_ideas": struct["top_core_ideas"][:4],
                "structure_coverage": struct["structure_coverage"],
                "paper_cards": struct["paper_cards"][:3],
            }
            node_cards.append(node_card)
            query_terms.extend([topic, normalize_display_name(payload.get("display_name") or ""), str(payload.get("description") or "")])
            query_terms.extend(item.get("name") or "" for item in struct["top_limitations"][:3])
            query_terms.extend(item.get("direction") or "" for item in struct["top_future_work"][:3])
            query_terms.extend(item.get("name") or "" for item in struct["top_core_ideas"][:2])

        aggregate_structure = structure_store.summarize(self._dedupe_preserve_order(all_rep_paper_ids))
        planning_signals = self._build_planning_signals(node_cards)
        return {
            "retrieval_style": "benchmark_construction_aligned",
            "candidate_nodes": node_cards[:5],
            "aggregate_structure_signals": {
                "structure_coverage": aggregate_structure["structure_coverage"],
                "top_limitations": aggregate_structure["top_limitations"][:6],
                "top_future_work": aggregate_structure["top_future_work"][:6],
                "top_core_ideas": aggregate_structure["top_core_ideas"][:6],
            },
            "planning_signals": planning_signals,
            "structure_evidence_refs": aggregate_structure["evidence_refs"][:12],
            "candidate_paper_ids": self._dedupe_preserve_order(all_rep_paper_ids)[:24],
            "target_queries": [q for q in self._dedupe_preserve_order(query_terms) if q][:10],
        }

    def retrieve_evidence(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        corpus: DomainCorpus,
        support_rows: List[SupportPacket],
        benchmark_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        paper_retriever = corpus.paper_retriever(task["time_cutoff"])
        structure_store = self._structure_store(domain_id)
        query_bundle = list(benchmark_context.get("target_queries") or [task["question"]])
        hybrid_rows = self._multi_query_retrieve_docs(paper_retriever, query_bundle, top_k_per_query=8, limit=14)
        paper_ids = [doc.paper_id for doc, _ in hybrid_rows]
        paper_ids.extend(str(pid) for pid in (benchmark_context.get("candidate_paper_ids") or []))
        paper_ids.extend(str(ref.get("paper_id") or "") for ref in (benchmark_context.get("structure_evidence_refs") or []))
        paper_ids = [pid for pid in self._dedupe_preserve_order(paper_ids) if pid]

        paper_docs: List[Tuple[RetrievalDoc, Dict[str, float]]] = []
        by_id = {doc.paper_id: (doc, scores) for doc, scores in hybrid_rows}
        paper_doc_lookup = {doc.paper_id: doc for doc in corpus.paper_docs(task["time_cutoff"])}
        for paper_id in paper_ids:
            if paper_id in by_id:
                paper_docs.append(by_id[paper_id])
            elif paper_id in paper_doc_lookup:
                paper_docs.append((paper_doc_lookup[paper_id], {"hybrid_score": 0.0, "bm25_score": 0.0, "tfidf_score": 0.0}))

        structure_rows = []
        for paper_id in paper_ids[:16]:
            row = structure_store.get(paper_id)
            if not row:
                continue
            structure_rows.append(
                {
                    "paper_id": paper_id,
                    "paper_title": row.get("title"),
                    "problem_statement": clip_text(str(row.get("problem_statement") or ""), 280),
                    "limitations": [item.get("name") for item in (row.get("explicit_limitations") or [])[:3] if item.get("name")],
                    "future_work": [item.get("direction") for item in (row.get("future_work") or [])[:3] if item.get("direction")],
                    "core_ideas": [item.get("name") for item in (row.get("core_ideas") or [])[:3] if item.get("name")],
                }
            )

        node_docs = corpus.node_docs(task["time_cutoff"], paper_ids={doc.paper_id for doc, _ in paper_docs})
        section_rows: List[Tuple[RetrievalDoc, Dict[str, float]]] = []
        if node_docs:
            node_retriever = HybridRetriever(node_docs)
            section_queries = list(query_bundle)
            for row in structure_rows[:4]:
                section_queries.extend(row["limitations"][:2])
                section_queries.extend(row["future_work"][:2])
            section_rows = self._multi_query_retrieve_docs(node_retriever, section_queries, top_k_per_query=6, limit=16)

        return {
            "retrieval_mode": "candidate_nodes -> representative_papers -> structure_signals -> targeted_expansion",
            "candidate_node_evidence": benchmark_context.get("candidate_nodes") or [],
            "paper_evidence": [
                {
                    "paper_id": doc.paper_id,
                    "paper_title": doc.title,
                    "snippet": clip_text(doc.text, 1200),
                    "scores": scores,
                }
                for doc, scores in paper_docs[:16]
            ],
            "structure_evidence": structure_rows[:10],
            "section_evidence": [
                {
                    "paper_id": doc.paper_id,
                    "paper_title": doc.meta.get("paper_title") or doc.title,
                    "section_path": doc.meta.get("section_path"),
                    "section_kind": doc.meta.get("kind"),
                    "snippet": clip_text(doc.text, 1000),
                    "scores": scores,
                }
                for doc, scores in section_rows[:16]
            ],
        }

    def refine_evidence_by_chains(
        self,
        *,
        task: Dict[str, Any],
        task_parse: Dict[str, Any],
        corpus: DomainCorpus,
        benchmark_context: Dict[str, Any],
        seed_coi: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        top_chains = list(seed_coi.get("idea_chains") or [])[:4]
        if not top_chains:
            return evidence
        chain_terms = self._collect_chain_terms(top_chains)
        path_terms = self._collect_path_terms(benchmark_context=benchmark_context, top_chains=top_chains)
        query_bundle = self._dedupe_preserve_order([task["question"], *chain_terms, *path_terms])[:12]

        paper_retriever = corpus.paper_retriever(task["time_cutoff"])
        extra_papers = self._multi_query_retrieve_docs(paper_retriever, query_bundle, top_k_per_query=6, limit=16)
        merged_papers = self._merge_existing_and_new_paper_evidence(
            existing=evidence.get("paper_evidence") or [],
            extra=extra_papers,
            chain_terms=chain_terms,
            path_terms=path_terms,
        )

        top_paper_ids = [row["paper_id"] for row in merged_papers[:8] if row.get("paper_id")]
        structure_store = self._structure_store(corpus.domain_id)
        merged_structures = self._rerank_structure_evidence(
            existing=evidence.get("structure_evidence") or [],
            structure_store=structure_store,
            paper_ids=top_paper_ids,
            chain_terms=chain_terms,
            path_terms=path_terms,
        )

        node_docs = corpus.node_docs(task["time_cutoff"], paper_ids=set(top_paper_ids))
        merged_sections = self._rerank_section_evidence(
            existing=evidence.get("section_evidence") or [],
            node_docs=node_docs,
            query_bundle=query_bundle,
            chain_terms=chain_terms,
            path_terms=path_terms,
        )
        return {
            **evidence,
            "retrieval_mode": "global_hybrid_recall -> chain_conditioned_rerank -> small_to_big_expansion",
            "paper_evidence": merged_papers[:16],
            "structure_evidence": merged_structures[:10],
            "section_evidence": merged_sections[:16],
            "top_chain_terms": chain_terms[:12],
            "path_terms": path_terms[:12],
        }

    def build_coi(
        self,
        *,
        task: Dict[str, Any],
        task_parse: Dict[str, Any],
        domain_id: str,
        support_rows: List[SupportPacket],
        benchmark_context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        foundation_chain = []
        transition_chain = []
        unresolved_chain = []
        for node in (benchmark_context.get("candidate_nodes") or [])[:4]:
            reps = list(node.get("historical_representative_papers") or [])
            reps_sorted = sorted(reps, key=lambda x: parse_iso_date(x.get("published")) or parse_iso_date("1900-01-01"))
            foundation_chain.append(
                {
                    "node_id": node.get("node_id"),
                    "display_name": node.get("display_name"),
                    "public_topic": node.get("public_topic"),
                    "historical_stats": node.get("historical_stats"),
                    "quarterly_history": (node.get("quarterly_history") or [])[:3],
                    "papers": reps_sorted[:2],
                }
            )
            transition_chain.append(
                {
                    "node_id": node.get("node_id"),
                    "display_name": node.get("display_name"),
                    "quarterly_history_tail": (node.get("quarterly_history") or [])[-3:],
                    "papers": reps_sorted[-2:],
                    "top_core_ideas": (node.get("top_core_ideas") or [])[:3],
                }
            )
            if node.get("top_limitations") or node.get("top_future_work"):
                unresolved_chain.append(
                    {
                        "node_id": node.get("node_id"),
                        "display_name": node.get("display_name"),
                        "limitations": (node.get("top_limitations") or [])[:4],
                        "historical_future_work_signals": (node.get("top_future_work") or [])[:4],
                    }
                )

        recent_rows = []
        for row in evidence.get("paper_evidence") or []:
            recent_rows.append(
                {
                    "paper_id": row["paper_id"],
                    "paper_title": row["paper_title"],
                    "snippet": clip_text(row["snippet"], 280),
                }
            )
        return {
            "foundation_chain": foundation_chain[:3],
            "transition_chain": transition_chain[:3],
            "unresolved_chain": unresolved_chain[:3],
            "recent_evidence_chain": recent_rows[:6],
            "aggregate_structure_signals": benchmark_context.get("aggregate_structure_signals") or {},
            "idea_units": self._extract_idea_units(task_parse=task_parse, benchmark_context=benchmark_context, evidence=evidence),
            "idea_chains": self._build_ranked_idea_chains(task=task, task_parse=task_parse, benchmark_context=benchmark_context, evidence=evidence),
        }

    def generate_hypotheses(
        self,
        *,
        task: Dict[str, Any],
        task_parse: Dict[str, Any],
        coi: Dict[str, Any],
        focused_context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        family = str(task_parse.get("family") or "")
        if family == "direction_forecasting":
            return self._generate_direction_hypotheses(task=task, task_parse=task_parse, coi=coi, focused_context=focused_context, evidence=evidence)
        if family == "strategic_research_planning":
            return self._generate_planning_hypotheses(task=task, task_parse=task_parse, coi=coi, focused_context=focused_context, evidence=evidence)
        family_instructions = self._family_hypothesis_instructions(family)
        prompt = f"""Generate 3 candidate answers for the task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Parsed task:
{json.dumps(task_parse, ensure_ascii=False, indent=2)}

CoI-style trajectory state:
{json.dumps(coi, ensure_ascii=False, indent=2)}

Focused task context:
{json.dumps(focused_context, ensure_ascii=False, indent=2)}

Top Chain-of-Ideas candidates:
{json.dumps((coi.get('idea_chains') or [])[:4], ensure_ascii=False, indent=2)}

Benchmark-aligned retrieval bundle:
{json.dumps((evidence.get('candidate_node_evidence') or [])[:5], ensure_ascii=False, indent=2)}

Paper evidence:
{json.dumps((evidence.get('paper_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Structure evidence:
{json.dumps((evidence.get('structure_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps((evidence.get('section_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Requirements:
- Work only with pre-cutoff evidence.
- Treat the task as a forward-looking inference from history; do not claim knowledge of post-cutoff outcomes.
- Follow the benchmark-construction style of reasoning: identify likely node(s), inspect their historical quarterly trajectory, read representative papers, then synthesize from limitations/future-work and section evidence.
- Use Chain-of-Ideas reasoning explicitly: choose one or two top chains, preserve their causal order, and avoid mixing unrelated fragments from many weak chains.
- Produce concrete, benchmark-style answers.
- Avoid generic trend language.
- {family_instructions}

Return JSON:
{{
  "hypotheses": [
    {{
      "label": "H1",
      "claim": "...",
      "supporting_points": ["...", "..."],
      "draft_answer": "..."
    }}
  ]
}}"""
        obj = complete_json_object(
            self.answer_client,
            [
                {"role": "system", "content": "You are a precise research synthesis model. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            transport_retries=2,
            max_parse_attempts=3,
        )
        return list(obj.get("hypotheses") or [])

    def _generate_direction_hypotheses(
        self,
        *,
        task: Dict[str, Any],
        task_parse: Dict[str, Any],
        coi: Dict[str, Any],
        focused_context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        prompt = f"""Generate 3 candidate answers for a direction forecasting task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Top Chain-of-Ideas candidates:
{json.dumps((coi.get('idea_chains') or [])[:4], ensure_ascii=False, indent=2)}

Focused task context:
{json.dumps(focused_context, ensure_ascii=False, indent=2)}

Paper evidence:
{json.dumps((evidence.get('paper_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps((evidence.get('section_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Requirements:
- Work only with pre-cutoff evidence.
- First choose one trajectory label from: accelerating, steady, cooling, fragmenting.
- Prefer fragmenting when split pressure is high or multiple descendant-like subdirections are visible.
- Each candidate must contain: one trajectory label, two concrete subdirections, one venue/evaluation shift, and one main supporting chain.
- Use one main chain only; do not blend unrelated themes.

Return JSON:
{{
  "hypotheses": [
    {{
      "label": "H1",
      "trajectory_label": "fragmenting",
      "main_chain": ["...", "...", "..."],
      "subdirections": ["...", "..."],
      "venue_or_evaluation_shift": "...",
      "claim": "...",
      "supporting_points": ["...", "..."],
      "draft_answer": "..."
    }}
  ]
}}"""
        obj = complete_json_object(
            self.answer_client,
            [
                {"role": "system", "content": "You are a precise research forecasting model. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            transport_retries=2,
            max_parse_attempts=3,
        )
        return list(obj.get("hypotheses") or [])

    def _generate_planning_hypotheses(
        self,
        *,
        task: Dict[str, Any],
        task_parse: Dict[str, Any],
        coi: Dict[str, Any],
        focused_context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        prompt = f"""Generate 3 candidate answers for a strategic research planning task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Top Chain-of-Ideas candidates:
{json.dumps((coi.get('idea_chains') or [])[:4], ensure_ascii=False, indent=2)}

Focused task context:
{json.dumps(focused_context, ensure_ascii=False, indent=2)}

Paper evidence:
{json.dumps((evidence.get('paper_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps((evidence.get('section_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Requirements:
- Work only with pre-cutoff evidence.
- Each candidate must be one focused agenda, not a list.
- Include: target problem, method angle, evaluation plan, why-now justification, and one milestone sequence.
- Use one main chain only: mature line -> gap -> executable agenda.
- Mention venue/citation/momentum signals when they materially support prioritization.

Return JSON:
{{
  "hypotheses": [
    {{
      "label": "H1",
      "main_chain": ["...", "...", "..."],
      "target_problem": "...",
      "method_angle": "...",
      "evaluation_plan": "...",
      "why_now": "...",
      "milestones": ["...", "..."],
      "claim": "...",
      "supporting_points": ["...", "..."],
      "draft_answer": "..."
    }}
  ]
}}"""
        obj = complete_json_object(
            self.answer_client,
            [
                {"role": "system", "content": "You are a precise research planning model. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            transport_retries=2,
            max_parse_attempts=3,
        )
        return list(obj.get("hypotheses") or [])

    def select_hypothesis(
        self,
        *,
        task: Dict[str, Any],
        task_parse: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
        coi: Dict[str, Any],
        focused_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task_parse.get("family") or "")
        family_criteria = self._family_selection_criteria(family)
        prompt = f"""Select the best hypothesis for the task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

CoI state:
{json.dumps(coi, ensure_ascii=False, indent=2)}

Focused task context:
{json.dumps(focused_context, ensure_ascii=False, indent=2)}

Hypotheses:
{json.dumps(hypotheses, ensure_ascii=False, indent=2)}

Top Chain-of-Ideas candidates:
{json.dumps((coi.get('idea_chains') or [])[:4], ensure_ascii=False, indent=2)}

Evaluation criteria:
- historical evidence faithfulness
- specificity
- non-generic insight
- family fit
- no future leakage
- chain coherence
- {family_criteria}

Return JSON:
{{
  "selected_label": "H1",
  "reasoning": "...",
  "final_answer": "..."
}}"""
        return complete_json_object(
            self.critic_client,
            [
                {"role": "system", "content": "You are a critical research evaluator. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            transport_retries=2,
            max_parse_attempts=3,
        )

    def _sanitize_packet(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "packet_id": payload.get("packet_id"),
            "domain": payload.get("domain"),
            "node_id": payload.get("node_id"),
            "display_name": normalize_display_name(payload.get("display_name") or ""),
            "dimension_id": payload.get("dimension_id"),
            "description": payload.get("description"),
            "historical_stats": payload.get("historical_stats"),
            "historical_representative_papers": (payload.get("historical_representative_papers") or [])[:6],
        }

    def build_focused_context(
        self,
        *,
        task_parse: Dict[str, Any],
        support_rows: List[SupportPacket],
        benchmark_context: Dict[str, Any],
        coi: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task_parse.get("family") or "")
        if family == "bottleneck_opportunity_discovery":
            recurring_limitations = []
            opportunity_candidates = []
            for node in (benchmark_context.get("candidate_nodes") or [])[:5]:
                for item in (node.get("top_limitations") or [])[:4]:
                    recurring_limitations.append(
                        {
                            "node": node.get("display_name"),
                            "name": item.get("name"),
                            "description": item.get("description"),
                            "paper_id": item.get("paper_id"),
                            "title": item.get("title"),
                        }
                    )
                for item in (node.get("top_future_work") or [])[:4]:
                    opportunity_candidates.append(
                        {
                            "node": node.get("display_name"),
                            "direction": item.get("direction"),
                            "paper_id": item.get("paper_id"),
                            "title": item.get("title"),
                        }
                    )
            return {
                "family_focus": "pair a historically grounded bottleneck with a concrete opportunity implied by pre-cutoff future-work signals",
                "candidate_nodes": [node.get("display_name") for node in (benchmark_context.get("candidate_nodes") or [])[:4]],
                "recurring_limitations": recurring_limitations[:8],
                "opportunity_candidates": opportunity_candidates[:8],
                "aggregate_structure_signals": benchmark_context.get("aggregate_structure_signals"),
                "recent_paper_evidence": (evidence.get("paper_evidence") or [])[:6],
            }
        if family == "direction_forecasting":
            trajectory_nodes = []
            for node in (benchmark_context.get("candidate_nodes") or [])[:5]:
                reps = node.get("historical_representative_papers") or []
                trajectory_nodes.append(
                    {
                        "node": node.get("display_name"),
                        "description": node.get("description"),
                        "historical_stats": node.get("historical_stats"),
                        "quarterly_history": node.get("quarterly_history"),
                        "recent_representative_titles": [row.get("title") for row in reps[-3:]],
                    }
                )
            return {
                "family_focus": "infer which direction is likely to accelerate, cool, or branch based on historical evolution only",
                "trajectory_nodes": trajectory_nodes[:5],
                "aggregate_structure_signals": benchmark_context.get("aggregate_structure_signals"),
                "recent_section_evidence": (evidence.get("section_evidence") or [])[:8],
            }
        combined_gaps = []
        for node in (benchmark_context.get("candidate_nodes") or [])[:5]:
            combined_gaps.append(
                {
                    "node": node.get("display_name"),
                    "description": node.get("description"),
                    "quarterly_history": node.get("quarterly_history"),
                    "limitations": [x.get("name") for x in (node.get("top_limitations") or [])[:4]],
                    "future_work": [x.get("direction") for x in (node.get("top_future_work") or [])[:4]],
                    "core_ideas": [x.get("name") for x in (node.get("top_core_ideas") or [])[:4]],
                }
            )
        return {
            "family_focus": "propose a concrete research agenda grounded in current gaps and adjacent historical trajectories",
            "priority_nodes": ((benchmark_context.get("planning_signals") or {}).get("priority_nodes") or [])[:5],
            "cross_node_convergences": ((benchmark_context.get("planning_signals") or {}).get("cross_node_convergences") or {}) ,
            "venue_and_citation_signals": ((benchmark_context.get("planning_signals") or {}).get("venue_and_citation_signals") or [])[:5],
            "agenda_nodes": combined_gaps[:5],
            "aggregate_structure_signals": benchmark_context.get("aggregate_structure_signals"),
            "recent_section_evidence": (evidence.get("section_evidence") or [])[:8],
        }

    def _build_planning_signals(self, node_cards: List[Dict[str, Any]]) -> Dict[str, Any]:
        priority_nodes = []
        limitation_lists: List[List[str]] = []
        future_lists: List[List[str]] = []
        idea_lists: List[List[str]] = []
        venue_and_citation_signals = []
        for node in node_cards[:5]:
            hist = node.get("historical_stats") or {}
            momentum = self._recent_momentum(node.get("quarterly_history") or [])
            priority_score = (
                0.35 * float(hist.get("paper_count") or 0)
                + 18.0 * float(hist.get("top_conf_share") or 0.0)
                + 0.25 * float(hist.get("citation_median") or 0.0)
                + 0.75 * float(momentum.get("recent_growth_sum") or 0.0)
            )
            priority_nodes.append(
                {
                    "node": node.get("display_name"),
                    "public_topic": node.get("public_topic"),
                    "priority_score": round(priority_score, 4),
                    "historical_paper_count": int(hist.get("paper_count") or 0),
                    "historical_top_conf_share": float(hist.get("top_conf_share") or 0.0),
                    "historical_citation_median": float(hist.get("citation_median") or 0.0),
                    "recent_growth_sum": momentum.get("recent_growth_sum"),
                    "recent_growth_mean": momentum.get("recent_growth_mean"),
                    "recent_top_conf_share_mean": momentum.get("recent_top_conf_share_mean"),
                    "top_core_ideas": [x.get("name") for x in (node.get("top_core_ideas") or [])[:3] if x.get("name")],
                    "top_future_work": [x.get("direction") for x in (node.get("top_future_work") or [])[:3] if x.get("direction")],
                }
            )
            venue_and_citation_signals.append(
                {
                    "node": node.get("display_name"),
                    "top_venues": hist.get("top_venues") or {},
                    "top_venue_buckets": hist.get("top_venue_buckets") or {},
                    "historical_citation_median": float(hist.get("citation_median") or 0.0),
                    "representative_papers": [
                        {
                            "title": row.get("title"),
                            "venue": row.get("venue"),
                            "citation": row.get("citation"),
                        }
                        for row in sorted(
                            list(node.get("historical_representative_papers") or []),
                            key=lambda x: int(x.get("citation") or 0),
                            reverse=True,
                        )[:3]
                    ],
                }
            )
            limitation_lists.append([x.get("name") for x in (node.get("top_limitations") or []) if x.get("name")])
            future_lists.append([x.get("direction") for x in (node.get("top_future_work") or []) if x.get("direction")])
            idea_lists.append([x.get("name") for x in (node.get("top_core_ideas") or []) if x.get("name")])
        priority_nodes.sort(key=lambda row: float(row.get("priority_score") or 0.0), reverse=True)
        return {
            "priority_nodes": priority_nodes[:5],
            "cross_node_convergences": {
                "repeated_limitations": self._top_repeated_phrases(limitation_lists, top_k=6),
                "repeated_future_work": self._top_repeated_phrases(future_lists, top_k=6),
                "repeated_core_ideas": self._top_repeated_phrases(idea_lists, top_k=6),
            },
            "venue_and_citation_signals": venue_and_citation_signals[:5],
        }

    @staticmethod
    def _recent_momentum(quarterly_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        tail = list(quarterly_history or [])[-3:]
        growths = [float(row.get("paper_growth") or 0.0) for row in tail]
        shares = [float(row.get("top_conf_share") or 0.0) for row in tail]
        return {
            "recent_growth_sum": round(sum(growths), 4),
            "recent_growth_mean": round(sum(growths) / max(1, len(growths)), 4),
            "recent_top_conf_share_mean": round(sum(shares) / max(1, len(shares)), 4),
        }

    @staticmethod
    def _top_repeated_phrases(value_lists: List[List[str]], *, top_k: int) -> List[Dict[str, Any]]:
        counter: Counter[str] = Counter()
        for values in value_lists:
            seen_local = set()
            for value in values:
                item = str(value or "").strip()
                if not item or item in seen_local:
                    continue
                seen_local.add(item)
                counter[item] += 1
        return [{"name": name, "count": count} for name, count in counter.most_common(top_k)]

    def _family_hypothesis_instructions(self, family: str) -> str:
        if family == "bottleneck_opportunity_discovery":
            return (
                "For bottleneck tasks: identify one precise unresolved bottleneck from the historical literature, "
                "then infer one concrete opportunity that historical future-work signals suggest would open next. "
                "Do not claim the opportunity was already realized before the cutoff. "
                "Answer around one main chain in the form bottleneck -> failed or partial remedy -> concrete opening."
            )
        if family == "direction_forecasting":
            return (
                "For forecasting tasks: state one direction that is most likely to heat up or cool down after the cutoff, "
                "and justify it with historical transition evidence rather than generic popularity claims. "
                "The answer should explicitly commit to one trajectory label and support it with one main chain in the form history -> inflection -> next subdirection."
            )
        return (
            "For planning tasks: propose one agenda with a target problem, a methodological angle, an evaluation plan, "
            "and a why-now justification grounded in historical maturity, recent momentum, venue concentration, citation density, "
            "and cross-node convergence. Avoid generic laundry lists. Use one main chain in the form mature line -> gap -> executable agenda."
        )

    def _family_selection_criteria(self, family: str) -> str:
        if family == "bottleneck_opportunity_discovery":
            return "prefer hypotheses that follow one clear bottleneck-to-opportunity chain with explicit causal linkage"
        if family == "direction_forecasting":
            return "prefer hypotheses that make one clear directional prediction from one coherent chain rather than a bag of hints"
        return "prefer hypotheses that form one focused agenda from one coherent mature-line-to-gap-to-plan chain with explicit why-now justification"

    def _extract_idea_units(
        self,
        *,
        task_parse: Dict[str, Any],
        benchmark_context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        family = str(task_parse.get("family") or "")
        units: List[Dict[str, Any]] = []
        for node in (benchmark_context.get("candidate_nodes") or [])[:5]:
            hist = node.get("historical_stats") or {}
            quarterly = list(node.get("quarterly_history") or [])
            units.append(
                {
                    "kind": "topic",
                    "node": node.get("display_name"),
                    "text": node.get("public_topic") or node.get("display_name"),
                    "score": round(0.2 + 0.01 * float(hist.get("paper_count") or 0), 4),
                }
            )
            if quarterly:
                growth = sum(float(row.get("paper_growth") or 0.0) for row in quarterly[-3:])
                units.append(
                    {
                        "kind": "momentum",
                        "node": node.get("display_name"),
                        "text": f"recent quarterly growth {round(growth, 2)} with top venue share around {round(sum(float(row.get('top_conf_share') or 0.0) for row in quarterly[-3:]) / max(1, len(quarterly[-3:])), 4)}",
                        "score": round(abs(growth) * 0.05 + 0.2, 4),
                    }
                )
            for item in (node.get("top_limitations") or [])[:3]:
                units.append(
                    {
                        "kind": "bottleneck",
                        "node": node.get("display_name"),
                        "text": item.get("name"),
                        "detail": item.get("description"),
                        "paper_id": item.get("paper_id"),
                        "score": round(0.6 + 0.05 * float(item.get("count") or 0), 4),
                    }
                )
            for item in (node.get("top_future_work") or [])[:3]:
                units.append(
                    {
                        "kind": "future_work",
                        "node": node.get("display_name"),
                        "text": item.get("direction"),
                        "paper_id": item.get("paper_id"),
                        "score": round(0.55 + 0.05 * float(item.get("count") or 0), 4),
                    }
                )
            for item in (node.get("top_core_ideas") or [])[:3]:
                units.append(
                    {
                        "kind": "method",
                        "node": node.get("display_name"),
                        "text": item.get("name"),
                        "detail": item.get("mechanism"),
                        "paper_id": item.get("paper_id"),
                        "score": round(0.5 + 0.05 * float(item.get("count") or 0), 4),
                    }
                )
        for row in (evidence.get("structure_evidence") or [])[:8]:
            for value in (row.get("limitations") or [])[:2]:
                units.append({"kind": "bottleneck", "node": row.get("paper_title"), "text": value, "score": 0.45})
            for value in (row.get("future_work") or [])[:2]:
                units.append({"kind": "future_work", "node": row.get("paper_title"), "text": value, "score": 0.4})
            for value in (row.get("core_ideas") or [])[:2]:
                units.append({"kind": "method", "node": row.get("paper_title"), "text": value, "score": 0.35})
        if family == "strategic_research_planning":
            for row in ((benchmark_context.get("planning_signals") or {}).get("priority_nodes") or [])[:5]:
                units.append(
                    {
                        "kind": "priority",
                        "node": row.get("node"),
                        "text": f"{row.get('public_topic') or row.get('node')} with priority score {row.get('priority_score')}",
                        "score": float(row.get("priority_score") or 0.0) / 50.0,
                    }
                )
        units.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        return units[:40]

    def _build_ranked_idea_chains(
        self,
        *,
        task: Dict[str, Any],
        task_parse: Dict[str, Any],
        benchmark_context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        family = str(task_parse.get("family") or "")
        chains: List[Dict[str, Any]] = []
        for node in (benchmark_context.get("candidate_nodes") or [])[:5]:
            if family == "bottleneck_opportunity_discovery":
                chains.extend(self._build_bottleneck_chains_for_node(node=node, evidence=evidence))
            elif family == "direction_forecasting":
                chains.extend(self._build_direction_chains_for_node(node=node, evidence=evidence))
            else:
                chains.extend(
                    self._build_planning_chains_for_node(
                        node=node,
                        planning_signals=benchmark_context.get("planning_signals") or {},
                        evidence=evidence,
                    )
                )
        for chain in chains:
            support = self._score_chain_support(chain=chain, evidence=evidence)
            chain["support_score"] = support["support_score"]
            chain["matched_evidence"] = support["matched_evidence"]
            chain["chain_score"] = round(float(chain.get("base_score") or 0.0) + float(chain["support_score"]), 4)
        chains.sort(key=lambda x: float(x.get("chain_score") or 0.0), reverse=True)
        return chains[:8]

    def _build_bottleneck_chains_for_node(self, *, node: Dict[str, Any], evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        bottlenecks = [x.get("name") for x in (node.get("top_limitations") or [])[:3] if x.get("name")]
        opportunities = [x.get("direction") for x in (node.get("top_future_work") or [])[:3] if x.get("direction")]
        methods = [x.get("name") for x in (node.get("top_core_ideas") or [])[:2] if x.get("name")]
        chains = []
        for bottleneck in bottlenecks or [node.get("public_topic")]:
            for opportunity in opportunities[:2] or methods[:1]:
                steps = [node.get("public_topic"), bottleneck]
                if methods:
                    steps.append(methods[0])
                steps.append(opportunity)
                chains.append(
                    {
                        "family": "bottleneck_opportunity_discovery",
                        "node": node.get("display_name"),
                        "chain_type": "bottleneck_to_opening",
                        "steps": [x for x in steps if x],
                        "base_score": round(0.8 + 0.02 * len(bottlenecks) + 0.02 * len(opportunities), 4),
                    }
                )
        return chains[:4]

    def _build_direction_chains_for_node(self, *, node: Dict[str, Any], evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
        quarterly = list(node.get("quarterly_history") or [])
        hist = node.get("historical_stats") or {}
        growth = sum(float(row.get("paper_growth") or 0.0) for row in quarterly[-3:])
        share = sum(float(row.get("top_conf_share") or 0.0) for row in quarterly[-3:]) / max(1, len(quarterly[-3:])) if quarterly else float(hist.get("top_conf_share") or 0.0)
        descendants = [x.get("display_name") for x in (node.get("emergent_descendants") or [])[:4] if x.get("display_name")]
        split_pressure = float(node.get("split_pressure") or 0.0)
        if split_pressure >= 14 or len(descendants) >= 3:
            trajectory = "fragmenting"
        elif growth >= 20 or share >= float(hist.get("top_conf_share") or 0.0) + 0.03:
            trajectory = "accelerating"
        elif growth <= 2 and float(hist.get("paper_count") or 0) >= 20:
            trajectory = "cooling_or_plateauing"
        else:
            trajectory = "branching_or_steady"
        future_signals = [x.get("direction") for x in (node.get("top_future_work") or [])[:3] if x.get("direction")]
        method_signals = [x.get("name") for x in (node.get("top_core_ideas") or [])[:3] if x.get("name")]
        chains = []
        for next_signal in future_signals[:2] or method_signals[:2]:
            chains.append(
                {
                    "family": "direction_forecasting",
                    "node": node.get("display_name"),
                    "chain_type": "history_inflection_next_direction",
                    "trajectory_label": trajectory,
                    "steps": [
                        node.get("public_topic"),
                        f"historical momentum: growth={round(growth, 2)}, top_venue_share={round(share, 4)}, split_pressure={round(split_pressure, 2)}",
                        next_signal,
                    ],
                    "emergent_descendants": descendants[:4],
                    "base_score": round(
                        0.75
                        + 0.01 * abs(growth)
                        + 0.4 * share
                        + 0.03 * len(future_signals)
                        + 0.03 * len(descendants)
                        + min(0.6, split_pressure / 25.0),
                        4,
                    ),
                }
            )
        if descendants:
            chains.append(
                {
                    "family": "direction_forecasting",
                    "node": node.get("display_name"),
                    "chain_type": "history_split_descendants",
                    "trajectory_label": "fragmenting" if split_pressure >= 8 or len(descendants) >= 2 else trajectory,
                    "steps": [
                        node.get("public_topic"),
                        f"historical momentum: growth={round(growth, 2)}, top_venue_share={round(share, 4)}, split_pressure={round(split_pressure, 2)}",
                        *descendants[:3],
                    ],
                    "emergent_descendants": descendants[:4],
                    "base_score": round(0.95 + 0.02 * len(descendants) + min(0.8, split_pressure / 18.0), 4),
                }
            )
        return chains[:4]

    def _build_planning_chains_for_node(
        self,
        *,
        node: Dict[str, Any],
        planning_signals: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        gaps = [x.get("name") for x in (node.get("top_limitations") or [])[:2] if x.get("name")]
        methods = [x.get("name") for x in (node.get("top_core_ideas") or [])[:2] if x.get("name")]
        future_work = [x.get("direction") for x in (node.get("top_future_work") or [])[:2] if x.get("direction")]
        repeated_future = [x.get("name") for x in ((planning_signals.get("cross_node_convergences") or {}).get("repeated_future_work") or [])[:2] if x.get("name")]
        repeated_ideas = [x.get("name") for x in ((planning_signals.get("cross_node_convergences") or {}).get("repeated_core_ideas") or [])[:2] if x.get("name")]
        hist = node.get("historical_stats") or {}
        why_now = f"historical maturity={int(hist.get('paper_count') or 0)}, top venue share={float(hist.get('top_conf_share') or 0.0):.4f}, citation median={float(hist.get('citation_median') or 0.0):.1f}"
        chains = []
        agenda_candidates = future_work[:2] or repeated_future[:2] or methods[:2]
        method_candidates = methods[:2] or repeated_ideas[:2]
        for agenda in agenda_candidates:
            steps = [node.get("public_topic")]
            if gaps:
                steps.append(gaps[0])
            if method_candidates:
                steps.append(method_candidates[0])
            steps.extend([agenda, why_now])
            chains.append(
                {
                    "family": "strategic_research_planning",
                    "node": node.get("display_name"),
                    "chain_type": "mature_line_gap_agenda",
                    "steps": [x for x in steps if x],
                    "base_score": round(0.8 + 0.015 * float(hist.get("paper_count") or 0) + 0.3 * float(hist.get("top_conf_share") or 0.0), 4),
                }
            )
        return chains[:4]

    def _collect_chain_terms(self, top_chains: List[Dict[str, Any]]) -> List[str]:
        values: List[str] = []
        for chain in top_chains:
            values.extend(str(x) for x in (chain.get("steps") or []) if str(x).strip())
            values.extend(str(x) for x in (chain.get("subdirections") or []) if str(x).strip())
            values.extend(str(x) for x in (chain.get("emergent_descendants") or []) if str(x).strip())
            if chain.get("trajectory_label"):
                values.append(str(chain.get("trajectory_label")))
        return self._dedupe_preserve_order(values)

    def _collect_path_terms(self, *, benchmark_context: Dict[str, Any], top_chains: List[Dict[str, Any]]) -> List[str]:
        values: List[str] = []
        for node in (benchmark_context.get("candidate_nodes") or [])[:5]:
            values.extend(
                [
                    str(node.get("display_name") or ""),
                    str(node.get("public_topic") or ""),
                    str(node.get("description") or ""),
                ]
            )
            values.extend(str(x.get("display_name") or "") for x in (node.get("emergent_descendants") or [])[:4])
        for chain in top_chains:
            if chain.get("node"):
                values.append(str(chain.get("node")))
        return self._dedupe_preserve_order([x for x in values if x])

    def _merge_existing_and_new_paper_evidence(
        self,
        *,
        existing: List[Dict[str, Any]],
        extra: List[Tuple[RetrievalDoc, Dict[str, float]]],
        chain_terms: List[str],
        path_terms: List[str],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for row in existing:
            paper_id = str(row.get("paper_id") or "")
            if not paper_id:
                continue
            item = dict(row)
            item["rerank_score"] = self._text_match_bonus(json.dumps(item, ensure_ascii=False), chain_terms, path_terms)
            merged[paper_id] = item
        for doc, scores in extra:
            item = {
                "paper_id": doc.paper_id,
                "paper_title": doc.title,
                "snippet": clip_text(doc.text, 1200),
                "scores": scores,
                "rerank_score": self._text_match_bonus(doc.text, chain_terms, path_terms),
            }
            prev = merged.get(doc.paper_id)
            if prev is None or float(item["rerank_score"]) + float(scores.get("combined_score") or scores.get("hybrid_score") or 0.0) > float(prev.get("rerank_score") or 0.0) + float((prev.get("scores") or {}).get("combined_score") or (prev.get("scores") or {}).get("hybrid_score") or 0.0):
                merged[doc.paper_id] = item
        rows = list(merged.values())
        rows.sort(
            key=lambda row: (
                -(
                    float((row.get("scores") or {}).get("combined_score") or (row.get("scores") or {}).get("hybrid_score") or 0.0)
                    + float(row.get("rerank_score") or 0.0)
                ),
                -len(str(row.get("snippet") or "")),
            )
        )
        return rows

    def _rerank_structure_evidence(
        self,
        *,
        existing: List[Dict[str, Any]],
        structure_store: PaperStructureStore,
        paper_ids: List[str],
        chain_terms: List[str],
        path_terms: List[str],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for row in existing:
            key = str(row.get("paper_id") or "")
            if not key:
                continue
            item = dict(row)
            item["rerank_score"] = self._text_match_bonus(json.dumps(item, ensure_ascii=False), chain_terms, path_terms)
            merged[key] = item
        for paper_id in paper_ids:
            row = structure_store.get(paper_id)
            if not row:
                continue
            item = {
                "paper_id": paper_id,
                "paper_title": row.get("title"),
                "problem_statement": clip_text(str(row.get("problem_statement") or ""), 280),
                "limitations": [item.get("name") for item in (row.get("explicit_limitations") or [])[:3] if item.get("name")],
                "future_work": [item.get("direction") for item in (row.get("future_work") or [])[:3] if item.get("direction")],
                "core_ideas": [item.get("name") for item in (row.get("core_ideas") or [])[:3] if item.get("name")],
            }
            item["rerank_score"] = self._text_match_bonus(json.dumps(item, ensure_ascii=False), chain_terms, path_terms)
            prev = merged.get(paper_id)
            if prev is None or float(item["rerank_score"]) > float(prev.get("rerank_score") or 0.0):
                merged[paper_id] = item
        rows = list(merged.values())
        rows.sort(key=lambda row: float(row.get("rerank_score") or 0.0), reverse=True)
        return rows

    def _rerank_section_evidence(
        self,
        *,
        existing: List[Dict[str, Any]],
        node_docs: List[RetrievalDoc],
        query_bundle: List[str],
        chain_terms: List[str],
        path_terms: List[str],
    ) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in existing:
            key = (str(row.get("paper_id") or ""), str(row.get("section_path") or ""))
            item = dict(row)
            item["rerank_score"] = self._text_match_bonus(json.dumps(item, ensure_ascii=False), chain_terms, path_terms)
            merged[key] = item
        if node_docs:
            node_retriever = HybridRetriever(node_docs)
            extra = self._multi_query_retrieve_docs(node_retriever, query_bundle, top_k_per_query=5, limit=24)
            for doc, scores in extra:
                key = (str(doc.paper_id or ""), str(doc.meta.get("section_path") or ""))
                item = {
                    "paper_id": doc.paper_id,
                    "paper_title": doc.meta.get("paper_title") or doc.title,
                    "section_path": doc.meta.get("section_path"),
                    "section_kind": doc.meta.get("kind"),
                    "snippet": clip_text(doc.text, 1000),
                    "scores": scores,
                    "rerank_score": self._text_match_bonus(doc.text, chain_terms, path_terms),
                }
                prev = merged.get(key)
                if prev is None or float(item["rerank_score"]) + float(scores.get("combined_score") or scores.get("hybrid_score") or 0.0) > float(prev.get("rerank_score") or 0.0) + float((prev.get("scores") or {}).get("combined_score") or (prev.get("scores") or {}).get("hybrid_score") or 0.0):
                    merged[key] = item
        rows = list(merged.values())
        rows.sort(
            key=lambda row: (
                -(
                    float(row.get("rerank_score") or 0.0)
                    + float((row.get("scores") or {}).get("combined_score") or (row.get("scores") or {}).get("hybrid_score") or 0.0)
                ),
                -len(str(row.get("snippet") or "")),
            )
        )
        return rows

    def _text_match_bonus(self, text: str, chain_terms: List[str], path_terms: List[str]) -> float:
        hay = str(text or "").lower()
        score = 0.0
        for term in chain_terms[:12]:
            t = str(term or "").strip().lower()
            if len(t) >= 4 and t in hay:
                score += 0.18
        for term in path_terms[:12]:
            t = str(term or "").strip().lower()
            if len(t) >= 4 and t in hay:
                score += 0.08
        return round(score, 4)

    def _score_chain_support(self, *, chain: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        terms = [str(x).lower() for x in (chain.get("steps") or []) if str(x).strip()]
        matches = []
        score = 0.0
        for family_name, rows, key in [
            ("paper", evidence.get("paper_evidence") or [], "snippet"),
            ("structure", evidence.get("structure_evidence") or [], "problem_statement"),
            ("section", evidence.get("section_evidence") or [], "snippet"),
        ]:
            for idx, row in enumerate(rows[:10], start=1):
                haystack = json.dumps(row, ensure_ascii=False).lower()
                hit_terms = [term for term in terms if term and term in haystack]
                if not hit_terms:
                    continue
                matches.append(
                    {
                        "source": family_name,
                        "rank": idx,
                        "paper_id": row.get("paper_id"),
                        "paper_title": row.get("paper_title"),
                        "matched_terms": hit_terms[:4],
                    }
                )
                score += 0.12 + 0.04 * min(3, len(hit_terms))
                if len(matches) >= 8:
                    return {"support_score": round(score, 4), "matched_evidence": matches}
        return {"support_score": round(score, 4), "matched_evidence": matches}

    @staticmethod
    def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            item = str(value or "").strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _multi_query_retrieve_docs(
        self,
        retriever: HybridRetriever,
        queries: Iterable[str],
        *,
        top_k_per_query: int,
        limit: int,
    ) -> List[Tuple[RetrievalDoc, Dict[str, float]]]:
        merged: Dict[str, Tuple[RetrievalDoc, Dict[str, float]]] = {}
        for q in self._dedupe_preserve_order(queries):
            rows = retriever.retrieve(q, top_k=top_k_per_query)
            for rank, (doc, scores) in enumerate(rows, start=1):
                bonus = 1.0 / (rank + 1)
                existing = merged.get(doc.doc_id)
                if existing is None:
                    merged[doc.doc_id] = (
                        doc,
                        {
                            **scores,
                            "combined_score": float(scores.get("hybrid_score") or 0.0) + bonus,
                            "matched_queries": [q],
                        },
                    )
                else:
                    _, prev = existing
                    prev["combined_score"] = float(prev.get("combined_score") or 0.0) + float(scores.get("hybrid_score") or 0.0) + bonus
                    prev["bm25_score"] = max(float(prev.get("bm25_score") or 0.0), float(scores.get("bm25_score") or 0.0))
                    prev["tfidf_score"] = max(float(prev.get("tfidf_score") or 0.0), float(scores.get("tfidf_score") or 0.0))
                    prev["hybrid_score"] = max(float(prev.get("hybrid_score") or 0.0), float(scores.get("hybrid_score") or 0.0))
                    prev_queries = list(prev.get("matched_queries") or [])
                    if q not in prev_queries:
                        prev_queries.append(q)
                    prev["matched_queries"] = prev_queries[:5]
        ranked = sorted(merged.values(), key=lambda item: float(item[1].get("combined_score") or 0.0), reverse=True)
        return ranked[:limit]
