from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from researchworld.baseline_runner import (
    DomainCorpus,
    HybridRetriever,
    RetrievalDoc,
    clip_text,
    parse_iso_date,
    tokenize,
)
from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.research_arc import (
    PaperStructureStore,
    QuarterlyNodeHistoryStore,
    SupportPacket,
    SupportPacketRetriever,
)
from researchworld.verbalization import normalize_display_name, public_topic_from_packet


ROOT = Path(__file__).resolve().parents[2]

NUMBER_WORDS = {
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
}

QUESTION_TOPIC_PATTERNS = [
    r'state of research (?:in|on)\s+(.+?)(?:\.\s|,?\s+based|,?\s+over|,?\s+from|,?\s+for the|,?\s+which|,?\s+what|$)',
    r'domain of\s+(.+?)(?:\.\s|,?\s+based|,?\s+over|,?\s+from|,?\s+for the|,?\s+which|,?\s+what|$)',
    r'work on\s+(.+?)(?:\.\s|,?\s+within|,?\s+over|,?\s+from|,?\s+for the|,?\s+which|,?\s+what|$)',
    r'evolution of work on\s+(.+?)(?:\.\s|,?\s+within|,?\s+over|,?\s+from|,?\s+what|$)',
    r'research (?:in|on)\s+(.+?)(?:\.\s|,?\s+based|,?\s+over|,?\s+from|,?\s+for the|,?\s+which|,?\s+what|$)',
]

TITLE_PATTERNS = [
    r'^Identifying a Key Bottleneck in\s+',
    r'^Identifying a Key Bottleneck for\s+',
    r'^Identifying Unresolved Bottlenecks in\s+',
    r'^Bottleneck and Opportunity Discovery in\s+',
    r'^Bottleneck and Opportunity Discovery for\s+',
    r'^Ex Ante Forecast for\s+',
    r'^Ex Ante Forecast on\s+',
    r'^Forecasting the Trajectory of\s+',
    r'^Forecasting Research Trajectory in\s+',
    r'^Forecasting Research Trajectory for\s+',
    r'^Prioritization of Research Directions in\s+',
    r'^Prioritization of Research Directions for\s+',
    r'^Prioritizing Research Directions in\s+',
    r'^Prioritizing Research Directions for\s+',
    r'^Strategic Research Planning for\s+',
    r'^Strategic Research Agenda for\s+',
]


def _clean_topic_text(text: str) -> str:
    value = re.sub(r'\s+', ' ', str(text or '')).strip(' .,:;')
    value = re.sub(r'\s+from pre(?:-| )cutoff.*$', '', value, flags=re.I)
    value = re.sub(r'\s+research trajectory$', '', value, flags=re.I)
    value = re.sub(
        r',?\s+(?:what|which)\s+(?:specific\s+)?(?:technical\s+)?(?:direction|directions|research directions?|candidate directions?|subtopics?).*$',
        '',
        value,
        flags=re.I,
    )
    value = re.sub(r',?\s+(?:is|are)\s+most likely to emerge.*$', '', value, flags=re.I)
    value = re.sub(r'\b(?:published|available|scholarly|literature|cutoff|date|solely|explicitly|historical|existing|corpus)\b', '', value, flags=re.I)
    value = re.sub(r'\s+', ' ', value).strip(' .,:;')
    return value


def extract_task_contract(task: Dict[str, Any]) -> Dict[str, Any]:
    title = str(task.get('title') or '').strip()
    title_topic = title
    for pattern in TITLE_PATTERNS:
        title_topic = re.sub(pattern, '', title_topic, flags=re.I).strip()
    question = str(task.get('question') or '').strip()
    lower_q = question.lower()
    topic = ''
    for pattern in QUESTION_TOPIC_PATTERNS:
        m = re.search(pattern, question, flags=re.I)
        if m:
            topic = m.group(1).strip()
            break
    topic = _clean_topic_text(topic or title_topic or question)
    ranking_required = any(
        token in lower_q
        for token in [
            'ranked list',
            'rank these',
            'how should they be ranked',
            'in what order',
            'relative ranking',
            'rank a small number',
            'rank a small set',
            'rank these directions',
        ]
    )
    max_items = None
    m = re.search(r'no more than\s+(\d+|one|two|three|four|five|six)', lower_q)
    if m:
        raw = m.group(1)
        max_items = int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)
    if max_items is None and ranking_required:
        m = re.search(r'which\s+(\d+|one|two|three|four|five|six)\s+research directions', lower_q)
        if m:
            raw = m.group(1)
            max_items = int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw)
    if max_items is None:
        max_items = 3 if ranking_required else 1
    return {
        'topic_text': topic,
        'ranking_required': ranking_required,
        'max_items': max_items,
    }


def expand_topic_queries(topic_text: str) -> List[str]:
    terms = [t for t in tokenize(str(topic_text or '').lower()) if len(t) > 2]
    out: List[str] = []
    if topic_text:
        out.append(str(topic_text))
    for size in (3, 2):
        for i in range(0, max(0, len(terms) - size + 1)):
            phrase = ' '.join(terms[i:i + size]).strip()
            if phrase:
                out.append(phrase)
            if len(out) >= 6:
                break
        if len(out) >= 6:
            break
    if len(terms) >= 2:
        out.append(f"{terms[0]} {terms[-1]}")
    deduped: List[str] = []
    seen = set()
    for item in out:
        key = item.lower()
        if key in seen or not item.strip():
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:6]


class EvidenceBackbone:
    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        self.corpus = DomainCorpus(domain_id)
        self.support_retriever = SupportPacketRetriever(domain_id)
        self.structure_store = PaperStructureStore(domain_id)
        self.node_history_store = QuarterlyNodeHistoryStore(domain_id)

    def build(self, *, task: Dict[str, Any], top_k_nodes: int = 10) -> Dict[str, Any]:
        task_contract = extract_task_contract(task)
        support_rows = self._retrieve_support_packets(task=task, task_contract=task_contract, top_k_nodes=top_k_nodes)
        candidate_nodes = self._build_candidate_nodes(support_rows)
        query_bundle = self._build_query_bundle(task=task, task_contract=task_contract, candidate_nodes=candidate_nodes)
        paper_rows = self._retrieve_papers(task=task, query_bundle=query_bundle, candidate_nodes=candidate_nodes)
        structure_rows = self._retrieve_structures(paper_rows)
        section_rows = self._retrieve_sections(task=task, query_bundle=query_bundle, paper_rows=paper_rows)
        return {
            'task_contract': task_contract,
            'support_packets': [self._sanitize_packet(x.payload) for x in support_rows],
            'candidate_nodes': candidate_nodes,
            'query_bundle': query_bundle,
            'paper_evidence': paper_rows,
            'structure_evidence': structure_rows,
            'section_evidence': section_rows,
        }

    def _retrieve_support_packets(self, *, task: Dict[str, Any], task_contract: Dict[str, Any], top_k_nodes: int) -> List[SupportPacket]:
        topic_text = str(task_contract.get('topic_text') or '')
        queries = self._dedupe(
            [
                str(task.get('question') or ''),
                str(task.get('title') or ''),
                *expand_topic_queries(topic_text),
                *self._domain_query_expansions(topic_text),
            ]
        )
        merged: Dict[str, Tuple[SupportPacket, float]] = {}
        for q_idx, query in enumerate(queries):
            for rank, (packet, score) in enumerate(self.support_retriever.retrieve_scored(query, top_k=max(top_k_nodes, 8)), start=1):
                bonus = 1.0 / (rank + q_idx + 1)
                prev = merged.get(packet.packet_id)
                agg = float(score) + bonus
                if prev is None or agg > prev[1]:
                    merged[packet.packet_id] = (packet, agg if prev is None else prev[1] + agg)
                else:
                    merged[packet.packet_id] = (prev[0], prev[1] + agg)
        ranked = sorted(merged.values(), key=lambda item: item[1], reverse=True)
        return [packet for packet, _ in ranked[:top_k_nodes]]

    def _build_candidate_nodes(self, support_rows: List[SupportPacket]) -> List[Dict[str, Any]]:
        rows = []
        for packet in support_rows:
            payload = packet.payload
            rep_papers = list(payload.get('historical_representative_papers') or [])[:6]
            rep_ids = [str(x.get('paper_id') or '') for x in rep_papers if x.get('paper_id')]
            struct = self.structure_store.summarize(rep_ids[:4])
            rows.append(
                {
                    'packet_id': payload.get('packet_id'),
                    'node_id': payload.get('node_id'),
                    'display_name': normalize_display_name(payload.get('display_name') or ''),
                    'public_topic': public_topic_from_packet(payload),
                    'dimension_id': payload.get('dimension_id'),
                    'description': payload.get('description'),
                    'historical_stats': payload.get('historical_stats') or {},
                    'quarterly_history': self.node_history_store.summarize(
                        str(payload.get('node_id') or ''),
                        up_to_slice=str(payload.get('history_structure_slice') or ''),
                        limit=6,
                    ),
                    'historical_representative_papers': rep_papers[:5],
                    'top_limitations': struct['top_limitations'][:4],
                    'top_future_work': struct['top_future_work'][:4],
                    'top_core_ideas': struct['top_core_ideas'][:4],
                    'split_pressure': payload.get('split_pressure'),
                    'emergent_descendants': (payload.get('emergent_descendants') or [])[:6],
                    'history_structure_slice': payload.get('history_structure_slice'),
                }
            )
        return rows

    def _build_query_bundle(self, *, task: Dict[str, Any], task_contract: Dict[str, Any], candidate_nodes: List[Dict[str, Any]]) -> List[str]:
        topic_text = str(task_contract.get('topic_text') or '')
        values = [
            task['question'],
            task.get('title') or '',
            *expand_topic_queries(topic_text),
            *self._domain_query_expansions(topic_text),
        ]
        for node in candidate_nodes[:5]:
            values.extend([
                node.get('public_topic') or '',
                node.get('display_name') or '',
                node.get('description') or '',
            ])
            values.extend(x.get('display_name') or '' for x in (node.get('emergent_descendants') or [])[:3])
            values.extend(x.get('name') or '' for x in (node.get('top_limitations') or [])[:2])
            values.extend(x.get('direction') or '' for x in (node.get('top_future_work') or [])[:2])
            values.extend(x.get('name') or '' for x in (node.get('top_core_ideas') or [])[:2])
        return self._dedupe(values)[:12]

    def _retrieve_papers(self, *, task: Dict[str, Any], query_bundle: List[str], candidate_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        retriever = self.corpus.paper_retriever(task['time_cutoff'])
        rows = self._multi_query_retrieve_docs(retriever, query_bundle, top_k_per_query=8, limit=18)
        merged: Dict[str, Dict[str, Any]] = {}
        for doc, scores in rows:
            merged[doc.paper_id] = {
                'paper_id': doc.paper_id,
                'paper_title': doc.title,
                'snippet': clip_text(doc.text, 1200),
                'scores': scores,
            }
        by_id = {doc.paper_id: doc for doc in self.corpus.paper_docs(task['time_cutoff'])}
        for node in candidate_nodes[:5]:
            for row in (node.get('historical_representative_papers') or [])[:4]:
                pid = str(row.get('paper_id') or '')
                if pid and pid not in merged and pid in by_id:
                    doc = by_id[pid]
                    merged[pid] = {
                        'paper_id': doc.paper_id,
                        'paper_title': doc.title,
                        'snippet': clip_text(doc.text, 1200),
                        'scores': {'hybrid_score': 0.0, 'bm25_score': 0.0, 'tfidf_score': 0.0},
                    }
        out = list(merged.values())
        out.sort(key=lambda x: float((x.get('scores') or {}).get('combined_score') or (x.get('scores') or {}).get('hybrid_score') or 0.0), reverse=True)
        return out[:18]

    def _retrieve_structures(self, paper_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for paper in paper_rows[:12]:
            row = self.structure_store.get(paper['paper_id'])
            if not row:
                continue
            rows.append(
                {
                    'paper_id': paper['paper_id'],
                    'paper_title': row.get('title'),
                    'problem_statement': clip_text(str(row.get('problem_statement') or ''), 280),
                    'limitations': [x.get('name') for x in (row.get('explicit_limitations') or [])[:3] if x.get('name')],
                    'future_work': [x.get('direction') for x in (row.get('future_work') or [])[:3] if x.get('direction')],
                    'core_ideas': [x.get('name') for x in (row.get('core_ideas') or [])[:3] if x.get('name')],
                }
            )
        return rows[:10]

    def _retrieve_sections(self, *, task: Dict[str, Any], query_bundle: List[str], paper_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paper_ids = {row['paper_id'] for row in paper_rows[:10] if row.get('paper_id')}
        node_docs = self.corpus.node_docs(task['time_cutoff'], paper_ids=paper_ids)
        if not node_docs:
            return []
        retriever = HybridRetriever(node_docs)
        rows = self._multi_query_retrieve_docs(retriever, query_bundle, top_k_per_query=6, limit=16)
        return [
            {
                'paper_id': doc.paper_id,
                'paper_title': doc.meta.get('paper_title') or doc.title,
                'section_path': doc.meta.get('section_path'),
                'section_kind': doc.meta.get('kind'),
                'snippet': clip_text(doc.text, 1000),
                'scores': scores,
            }
            for doc, scores in rows[:16]
        ]

    @staticmethod
    def _sanitize_packet(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'packet_id': payload.get('packet_id'),
            'domain': payload.get('domain'),
            'node_id': payload.get('node_id'),
            'display_name': normalize_display_name(payload.get('display_name') or ''),
            'dimension_id': payload.get('dimension_id'),
            'description': payload.get('description'),
            'historical_stats': payload.get('historical_stats'),
            'historical_representative_papers': (payload.get('historical_representative_papers') or [])[:6],
        }

    @staticmethod
    def _dedupe(values: Iterable[str]) -> List[str]:
        out=[]; seen=set()
        for value in values:
            item=str(value or '').strip()
            if not item or item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out

    def _domain_query_expansions(self, topic_text: str) -> List[str]:
        topic = str(topic_text or '').lower()
        out: List[str] = []
        if self.domain_id == 'llm_finetuning_post_training':
            if 'parameter efficient fine tuning' in topic or 'peft' in topic:
                out.extend([
                    'reinforcement learning based fine tuning',
                    'preference alignment',
                    'RLHF',
                    'DPO',
                    'LoRA instruction tuning',
                ])
            if 'chain of thought' in topic:
                out.extend([
                    'multimodal chain of thought evaluation',
                    'reasoning error analysis',
                    'small language model chain of thought',
                ])
            if 'multimodal fine tuning evaluation' in topic:
                out.extend([
                    'vision language alignment evaluation',
                    'reasoning enhanced multimodal evaluation',
                    'multimodal reasoning alignment evaluation',
                ])
        elif self.domain_id == 'visual_generative_modeling_and_diffusion':
            if 'video generation' in topic or 'video manipulation' in topic:
                out.extend([
                    'video to audio generation',
                    'audio video generation',
                    'controllable video generation',
                    'video editing diffusion',
                    'multi-view video generation',
                ])
        elif self.domain_id == 'llm_agent':
            if 'tool augmented reasoning' in topic:
                out.extend([
                    'reinforcement learning based tool reasoning',
                    'world model tool reasoning',
                    'tool grounding',
                ])
            if 'embodied agent guidance' in topic or 'embodied agent' in topic:
                out.extend([
                    'reinforcement learning agents',
                    'multi agent reinforcement learning agents',
                    'autonomous exploration',
                ])
        return self._dedupe(out)[:8]

    def _multi_query_retrieve_docs(self, retriever: HybridRetriever, queries: Iterable[str], *, top_k_per_query: int, limit: int) -> List[Tuple[RetrievalDoc, Dict[str, float]]]:
        merged: Dict[str, Tuple[RetrievalDoc, Dict[str, float]]] = {}
        for q in self._dedupe(queries):
            rows = retriever.retrieve(q, top_k=top_k_per_query)
            for rank, (doc, scores) in enumerate(rows, start=1):
                bonus = 1.0 / (rank + 1)
                existing = merged.get(doc.doc_id)
                if existing is None:
                    merged[doc.doc_id] = (
                        doc,
                        {
                            **scores,
                            'combined_score': float(scores.get('hybrid_score') or 0.0) + bonus,
                            'matched_queries': [q],
                        },
                    )
                else:
                    _, prev = existing
                    prev['combined_score'] = float(prev.get('combined_score') or 0.0) + float(scores.get('hybrid_score') or 0.0) + bonus
                    prev['bm25_score'] = max(float(prev.get('bm25_score') or 0.0), float(scores.get('bm25_score') or 0.0))
                    prev['tfidf_score'] = max(float(prev.get('tfidf_score') or 0.0), float(scores.get('tfidf_score') or 0.0))
                    prev['hybrid_score'] = max(float(prev.get('hybrid_score') or 0.0), float(scores.get('hybrid_score') or 0.0))
                    prev_q = list(prev.get('matched_queries') or [])
                    if q not in prev_q:
                        prev_q.append(q)
                    prev['matched_queries'] = prev_q[:5]
        ranked = sorted(merged.values(), key=lambda item: float(item[1].get('combined_score') or 0.0), reverse=True)
        return ranked[:limit]


class PolicyRouter:
    def route(self, *, task_parse: Dict[str, Any], domain_id: str) -> Dict[str, str]:
        family = str(task_parse.get('family') or '')
        if family == 'direction_forecasting':
            return {'head': 'direction', 'domain_policy': 'trajectory' if domain_id != 'rag_and_retrieval_structuring' else 'rag'}
        if family == 'strategic_research_planning':
            return {'head': 'planning', 'domain_policy': 'trajectory' if domain_id != 'rag_and_retrieval_structuring' else 'rag'}
        return {'head': 'bottleneck', 'domain_policy': 'trajectory' if domain_id != 'rag_and_retrieval_structuring' else 'rag'}


class DirectionHead:
    def __init__(self, answer_client: OpenAICompatChatClient, critic_client: OpenAICompatChatClient):
        self.answer_client = answer_client
        self.critic_client = critic_client

    def run(self, *, task: Dict[str, Any], domain_id: str, evidence_bundle: Dict[str, Any], policy: Dict[str, str]) -> Dict[str, Any]:
        candidates = self._build_direction_candidates(evidence_bundle=evidence_bundle, domain_policy=policy.get('domain_policy') or 'trajectory')
        decision = self._select_candidate(task=task, candidates=candidates, evidence_bundle=evidence_bundle)
        return {
            'head': 'direction',
            'candidates': candidates,
            'decision': decision,
            'answer': decision.get('final_answer') or '',
        }

    def _build_direction_candidates(self, *, evidence_bundle: Dict[str, Any], domain_policy: str) -> List[Dict[str, Any]]:
        out = []
        for node in (evidence_bundle.get('candidate_nodes') or [])[:5]:
            hist = node.get('historical_stats') or {}
            qh = list(node.get('quarterly_history') or [])
            growth = sum(float(x.get('paper_growth') or 0.0) for x in qh[-3:])
            share = sum(float(x.get('top_conf_share') or 0.0) for x in qh[-3:]) / max(1, len(qh[-3:])) if qh else float(hist.get('top_conf_share') or 0.0)
            split_pressure = float(node.get('split_pressure') or 0.0)
            descendants = [x.get('display_name') for x in (node.get('emergent_descendants') or [])[:4] if x.get('display_name')]
            trajectory = 'steady'
            if split_pressure >= 14 or len(descendants) >= 3:
                trajectory = 'fragmenting'
            elif growth >= 20 or share >= float(hist.get('top_conf_share') or 0.0) + 0.03:
                trajectory = 'accelerating'
            elif growth <= 2 and float(hist.get('paper_count') or 0) >= 20:
                trajectory = 'cooling'
            subdirs = descendants[:2]
            if not subdirs:
                subdirs = [x.get('direction') for x in (node.get('top_future_work') or [])[:2] if x.get('direction')]
            if domain_policy == 'rag':
                retrieval_terms = []
                for text in subdirs + [x.get('name') for x in (node.get('top_core_ideas') or [])[:3] if x.get('name')]:
                    s = str(text or '').lower()
                    if any(k in s for k in ['retrieval', 'rerank', 'index', 'graph', 'chunk', 'code', 'repository', 'iterative', 'fusion']):
                        retrieval_terms.append(text)
                if retrieval_terms:
                    subdirs = retrieval_terms[:2]
            venue_shift = f"historical top venue share {float(hist.get('top_conf_share') or 0.0):.4f}; recent share {share:.4f}; split pressure {split_pressure:.2f}"
            score = 0.6 + 0.02 * abs(growth) + 0.5 * share + min(0.8, split_pressure / 16.0) + 0.04 * len(descendants)
            out.append(
                {
                    'node': node.get('display_name'),
                    'public_topic': node.get('public_topic'),
                    'trajectory_label': trajectory,
                    'subdirections': subdirs[:2],
                    'venue_or_evaluation_shift': venue_shift,
                    'main_chain': [
                        node.get('public_topic'),
                        f'growth={round(growth,2)}, share={round(share,4)}, split_pressure={round(split_pressure,2)}',
                        *(subdirs[:2] or descendants[:2]),
                    ],
                    'score': round(score, 4),
                }
            )
        out.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)
        return out[:5]

    def _select_candidate(self, *, task: Dict[str, Any], candidates: List[Dict[str, Any]], evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Select the best direction forecasting candidate and write the final answer.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Evidence bundle:
{json.dumps({'candidate_nodes': (evidence_bundle.get('candidate_nodes') or [])[:5], 'paper_evidence': (evidence_bundle.get('paper_evidence') or [])[:6], 'section_evidence': (evidence_bundle.get('section_evidence') or [])[:6]}, ensure_ascii=False, indent=2)}

Requirements:
- Work only with pre-cutoff evidence.
- Pick exactly one trajectory label.
- Prefer fragmenting when descendant pressure is strong.
- Name two concrete subdirections when possible.
- Mention one venue/evaluation shift.
- Keep the answer compact and specific.

Return JSON:
{{
  'selected_index': 0,
  'reasoning': '...',
  'final_answer': '...'
}}"""
        obj = complete_json_object(
            self.critic_client,
            [
                {'role': 'system', 'content': 'You are a precise research forecasting evaluator. Output JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            response_format={'type': 'json_object'},
            transport_retries=2,
            max_parse_attempts=3,
        )
        idx = int(obj.get('selected_index') or 0)
        idx = max(0, min(idx, max(0, len(candidates) - 1)))
        chosen = candidates[idx] if candidates else {}
        return {
            'selected_index': idx,
            'selected_candidate': chosen,
            'reasoning': obj.get('reasoning') or '',
            'final_answer': obj.get('final_answer') or '',
        }


class PlanningHead:
    def __init__(self, answer_client: OpenAICompatChatClient, critic_client: OpenAICompatChatClient):
        self.answer_client = answer_client
        self.critic_client = critic_client

    def run(self, *, task: Dict[str, Any], domain_id: str, evidence_bundle: Dict[str, Any], policy: Dict[str, str]) -> Dict[str, Any]:
        candidates = self._build_planning_candidates(evidence_bundle=evidence_bundle, domain_policy=policy.get('domain_policy') or 'trajectory')
        decision = self._select_candidate(task=task, candidates=candidates, evidence_bundle=evidence_bundle)
        return {
            'head': 'planning',
            'candidates': candidates,
            'decision': decision,
            'answer': decision.get('final_answer') or '',
        }

    def _build_planning_candidates(self, *, evidence_bundle: Dict[str, Any], domain_policy: str) -> List[Dict[str, Any]]:
        out = []
        for node in (evidence_bundle.get('candidate_nodes') or [])[:5]:
            hist = node.get('historical_stats') or {}
            qh = list(node.get('quarterly_history') or [])
            growth = sum(float(x.get('paper_growth') or 0.0) for x in qh[-3:])
            limitations = [x.get('name') for x in (node.get('top_limitations') or [])[:2] if x.get('name')]
            methods = [x.get('name') for x in (node.get('top_core_ideas') or [])[:2] if x.get('name')]
            future_work = [x.get('direction') for x in (node.get('top_future_work') or [])[:2] if x.get('direction')]
            if domain_policy == 'rag':
                methods = [m for m in methods if any(k in str(m).lower() for k in ['retrieval', 'rerank', 'graph', 'index', 'chunk', 'fusion', 'code'])] or methods
                future_work = [f for f in future_work if any(k in str(f).lower() for k in ['retrieval', 'rerank', 'graph', 'index', 'chunk', 'fusion', 'code'])] or future_work
            target_problem = limitations[:1] or [node.get('public_topic')]
            method_angle = methods[:1] or future_work[:1] or [node.get('public_topic')]
            evaluation_plan = future_work[:1] or ['evaluate on the strongest selective benchmarks and recent realistic settings']
            why_now = f"historical maturity={int(hist.get('paper_count') or 0)}, top venue share={float(hist.get('top_conf_share') or 0.0):.4f}, citation median={float(hist.get('citation_median') or 0.0):.1f}, recent growth={round(growth,2)}"
            score = 0.5 + 0.01 * float(hist.get('paper_count') or 0) + 0.35 * float(hist.get('top_conf_share') or 0.0) + 0.02 * abs(growth)
            out.append(
                {
                    'node': node.get('display_name'),
                    'public_topic': node.get('public_topic'),
                    'target_problem': target_problem[0],
                    'method_angle': method_angle[0],
                    'evaluation_plan': evaluation_plan[0],
                    'why_now': why_now,
                    'milestones': [
                        f'establish a strong baseline on {node.get("public_topic")}',
                        f'validate {method_angle[0]} against {evaluation_plan[0]}',
                    ],
                    'main_chain': [node.get('public_topic'), target_problem[0], method_angle[0], evaluation_plan[0]],
                    'score': round(score, 4),
                }
            )
        out.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)
        return out[:5]

    def _select_candidate(self, *, task: Dict[str, Any], candidates: List[Dict[str, Any]], evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Select the best planning candidate and write the final answer.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Evidence bundle:
{json.dumps({'candidate_nodes': (evidence_bundle.get('candidate_nodes') or [])[:5], 'paper_evidence': (evidence_bundle.get('paper_evidence') or [])[:6], 'structure_evidence': (evidence_bundle.get('structure_evidence') or [])[:6]}, ensure_ascii=False, indent=2)}

Requirements:
- Produce one focused agenda, not a list.
- Include target problem, method angle, evaluation plan, why-now, and milestones.
- Prefer the agenda with the strongest maturity, tractability, and specificity.

Return JSON:
{{
  'selected_index': 0,
  'reasoning': '...',
  'final_answer': '...'
}}"""
        obj = complete_json_object(
            self.critic_client,
            [
                {'role': 'system', 'content': 'You are a precise research planning evaluator. Output JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            response_format={'type': 'json_object'},
            transport_retries=2,
            max_parse_attempts=3,
        )
        idx = int(obj.get('selected_index') or 0)
        idx = max(0, min(idx, max(0, len(candidates) - 1)))
        chosen = candidates[idx] if candidates else {}
        return {
            'selected_index': idx,
            'selected_candidate': chosen,
            'reasoning': obj.get('reasoning') or '',
            'final_answer': obj.get('final_answer') or '',
        }


class BottleneckHead:
    def __init__(self, answer_client: OpenAICompatChatClient, critic_client: OpenAICompatChatClient):
        self.answer_client = answer_client
        self.critic_client = critic_client

    def run(self, *, task: Dict[str, Any], domain_id: str, evidence_bundle: Dict[str, Any], policy: Dict[str, str]) -> Dict[str, Any]:
        candidates = []
        for node in (evidence_bundle.get('candidate_nodes') or [])[:5]:
            limitations = [x.get('name') for x in (node.get('top_limitations') or [])[:2] if x.get('name')]
            opportunities = [x.get('direction') for x in (node.get('top_future_work') or [])[:2] if x.get('direction')]
            methods = [x.get('name') for x in (node.get('top_core_ideas') or [])[:2] if x.get('name')]
            if not limitations:
                limitations = [node.get('public_topic')]
            if not opportunities:
                opportunities = methods[:1] or [node.get('public_topic')]
            candidates.append(
                {
                    'node': node.get('display_name'),
                    'bottleneck': limitations[0],
                    'opportunity': opportunities[0],
                    'main_chain': [node.get('public_topic'), limitations[0], methods[0] if methods else '', opportunities[0]],
                    'score': round(0.6 + 0.03 * len(limitations) + 0.03 * len(opportunities), 4),
                }
            )
        candidates.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)
        chosen = candidates[0] if candidates else {}
        answer = ''
        if chosen:
            answer = f"A strong bottleneck-opportunity pair is the unresolved bottleneck '{chosen['bottleneck']}' and the next opening '{chosen['opportunity']}', grounded in the historical trajectory of {chosen['node']}."
        return {
            'head': 'bottleneck',
            'candidates': candidates[:5],
            'decision': {'selected_candidate': chosen, 'final_answer': answer},
            'answer': answer,
        }


class ResearchArcV2:
    def __init__(self, *, answer_client: OpenAICompatChatClient, critic_client: Optional[OpenAICompatChatClient] = None):
        self.answer_client = answer_client
        self.critic_client = critic_client or answer_client
        self.backbones: Dict[str, EvidenceBackbone] = {}
        self.router = PolicyRouter()
        self.direction_head = DirectionHead(answer_client=self.answer_client, critic_client=self.critic_client)
        self.planning_head = PlanningHead(answer_client=self.answer_client, critic_client=self.critic_client)
        self.bottleneck_head = BottleneckHead(answer_client=self.answer_client, critic_client=self.critic_client)

    def _backbone(self, domain_id: str) -> EvidenceBackbone:
        if domain_id not in self.backbones:
            self.backbones[domain_id] = EvidenceBackbone(domain_id)
        return self.backbones[domain_id]

    def parse_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        family = str(task.get('family') or '')
        subtype = 'trend_and_outlook'
        if family == 'bottleneck_opportunity_discovery':
            subtype = 'bottleneck_opportunity_linkage'
        elif family == 'direction_forecasting':
            subtype = 'trajectory_forecast'
        elif family == 'strategic_research_planning':
            subtype = 'research_agenda'
        return {
            'family': family,
            'subtype': subtype,
            'domain': task.get('domain'),
            'horizon': task.get('horizon'),
            'time_cutoff': task.get('time_cutoff'),
        }

    def plan_retrieval(self, task_parse: Dict[str, Any], policy: Dict[str, str]) -> List[Dict[str, Any]]:
        plan = [
            {'step': 'global_hybrid_recall', 'goal': 'high-recall paper retrieval before cutoff'},
            {'step': 'node_conditioned_expansion', 'goal': 'expand from likely support-packet nodes and representative papers'},
            {'step': 'section_retrieval', 'goal': 'recover method, limitation, and evaluation evidence from sections'},
            {'step': 'evidence_reranking', 'goal': 'rerank evidence using family/domain policy'},
            {'step': 'policy_head', 'goal': f"answer with {policy.get('head')} head under {policy.get('domain_policy')} domain policy"},
        ]
        return plan

    def run_task(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        task_parse = self.parse_task(task)
        policy = self.router.route(task_parse=task_parse, domain_id=domain_id)
        retrieval_plan = self.plan_retrieval(task_parse=task_parse, policy=policy)
        evidence_bundle = self._backbone(domain_id).build(task=task)
        if policy['head'] == 'direction':
            head_result = self.direction_head.run(task=task, domain_id=domain_id, evidence_bundle=evidence_bundle, policy=policy)
        elif policy['head'] == 'planning':
            head_result = self.planning_head.run(task=task, domain_id=domain_id, evidence_bundle=evidence_bundle, policy=policy)
        else:
            head_result = self.bottleneck_head.run(task=task, domain_id=domain_id, evidence_bundle=evidence_bundle, policy=policy)
        return {
            'task_parse': task_parse,
            'policy': policy,
            'retrieval_plan': retrieval_plan,
            'evidence_bundle': evidence_bundle,
            'head_result': head_result,
            'answer': head_result.get('answer') or '',
        }
