from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from researchworld.baseline_runner import tokenize
from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.research_arc_v2 import EvidenceBackbone, PolicyRouter, extract_task_contract


ROOT = Path(__file__).resolve().parents[2]

STOPWORDS = {
    'the', 'and', 'for', 'with', 'from', 'into', 'over', 'under', 'based', 'using', 'available', 'before',
    'after', 'through', 'within', 'large', 'language', 'model', 'models', 'systems', 'system', 'research',
    'trajectory', 'forecasting', 'forecast', 'strategic', 'planning', 'agenda', 'discovery', 'opportunity',
    'bottleneck', 'evaluation', 'frameworks', 'framework', 'methods', 'method', 'tasks', 'task', 'domain',
    'domains', 'specific', 'subsequent', 'period', 'available', 'literature', 'state', 'identify', 'concrete',
    'change', 'direction', 'directions', 'next', 'quarter', 'subtopics', 'post', 'training', 'agentic', 'agents',
    'retrieval', 'augmented', 'generation', 'rag', 'llm', 'llms', 'augmentation', 'based', 'knowledge',
    'published', 'scholarly', 'august', 'september', '2025', '2026', 'date', 'cutoff', 'point', 'end',
    'provide', 'ranked', 'list', 'rank', 'justify', 'justification', 'draw', 'drawn', 'existing', 'corpus',
    'publication', 'publications', 'patterns', 'pattern', 'evidence', 'evident', 'available', 'solely',
    'historical', 'relative', 'order', 'small', 'number', 'prioritized', 'priority'
}

TITLE_PATTERNS = [
    r'^Bottleneck and Opportunity Discovery in\s+',
    r'^Bottleneck and Opportunity Discovery for\s+',
    r'^Forecasting the Trajectory of\s+',
    r'^Forecasting Research Trajectory in\s+',
    r'^Forecasting Research Trajectory for\s+',
    r'^Forecasting Trajectory and Subdirections in\s+',
    r'^Forecast for\s+',
    r'^Strategic Research Planning for\s+',
    r'^Strategic Research Agenda for\s+',
]


def _clip(text: Any, limit: int = 320) -> str:
    value = str(text or '').strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + '...'


def _norm_tokens(text: Any) -> List[str]:
    return [t for t in tokenize(str(text or '').lower()) if len(t) > 2 and t not in STOPWORDS]


def _dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        item = str(value or '').strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_label(text: Any) -> str:
    return re.sub(r'\s+', ' ', str(text or '').replace('_', ' ')).strip().lower()


def _extract_focus_text(task: Dict[str, Any]) -> str:
    contract = extract_task_contract(task)
    if str(contract.get('topic_text') or '').strip():
        return str(contract.get('topic_text') or '').strip()
    title = str(task.get('title') or '').strip()
    text = title
    for pattern in TITLE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
    if text and text != title:
        return text
    question = str(task.get('question') or '').strip()
    m = re.search(r'(?:within|for|on)\s+(.+?)(?:\s+over the subsequent|\s+for the next quarter|\s+considering|\.|$)', question, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return title or question


class FocusResolver:
    def resolve(self, *, task: Dict[str, Any], evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        focus_text = _extract_focus_text(task)
        focus_terms = set(_norm_tokens(focus_text))
        ranked = []
        for idx, node in enumerate(evidence_bundle.get('candidate_nodes') or []):
            score = self._node_score(focus_terms=focus_terms, node=node, support_rank=idx)
            ranked.append({**node, 'anchor_score': round(score, 4), 'support_rank': idx + 1})
        ranked.sort(key=lambda row: float(row.get('anchor_score') or 0.0), reverse=True)
        top = float(ranked[0].get('anchor_score') or 0.0) if ranked else 0.0
        focus_nodes = [row for row in ranked[:5] if top <= 0 or float(row.get('anchor_score') or 0.0) >= max(0.18, top * 0.67)]
        if not focus_nodes and ranked:
            focus_nodes = ranked[:1]
        cluster_mode = len(focus_nodes) >= 2 and sum(1 for row in focus_nodes if float(row.get('anchor_score') or 0.0) >= max(0.18, top * 0.82)) >= 2
        return {
            'focus_text': focus_text,
            'focus_terms': sorted(focus_terms),
            'ranked_nodes': ranked[:6],
            'focus_nodes': focus_nodes[:4],
            'cluster_mode': cluster_mode,
        }

    def _node_score(self, *, focus_terms: set[str], node: Dict[str, Any], support_rank: int) -> float:
        fields = [
            (node.get('public_topic'), 1.7),
            (node.get('display_name'), 1.5),
            (node.get('description'), 0.9),
        ]
        text_score = 0.0
        for text, weight in fields:
            terms = set(_norm_tokens(text))
            if not terms:
                continue
            overlap = len(focus_terms & terms)
            text_score += weight * (overlap / max(1, len(focus_terms)))
            if focus_terms and focus_terms <= terms:
                text_score += 0.5 * weight
        for row in (node.get('historical_representative_papers') or [])[:5]:
            terms = set(_norm_tokens(row.get('title')))
            overlap = len(focus_terms & terms)
            text_score += 0.18 * overlap
        for item in (node.get('top_future_work') or [])[:4]:
            terms = set(_norm_tokens(item.get('direction')))
            text_score += 0.18 * len(focus_terms & terms)
        for item in (node.get('top_limitations') or [])[:3]:
            terms = set(_norm_tokens(item.get('name')))
            text_score += 0.12 * len(focus_terms & terms)
        return text_score + (0.3 / (support_rank + 1))


class EvidenceFormatter:
    def build(self, *, focus: Dict[str, Any], evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        focus_nodes = focus.get('focus_nodes') or []
        paper_ids = []
        for node in focus_nodes:
            for row in (node.get('historical_representative_papers') or [])[:6]:
                pid = str(row.get('paper_id') or '')
                if pid:
                    paper_ids.append(pid)
        paper_ids = _dedupe(paper_ids)
        structure_rows = [
            row for row in (evidence_bundle.get('structure_evidence') or [])
            if str(row.get('paper_id') or '') in set(paper_ids)
        ]
        section_rows = [
            row for row in (evidence_bundle.get('section_evidence') or [])
            if str(row.get('paper_id') or '') in set(paper_ids)
        ]
        if not structure_rows:
            structure_rows = list(evidence_bundle.get('structure_evidence') or [])[:8]
        if not section_rows:
            section_rows = list(evidence_bundle.get('section_evidence') or [])[:8]
        return {
            'focus_summary': {
                'focus_text': focus.get('focus_text'),
                'cluster_mode': focus.get('cluster_mode'),
                'focus_nodes': [self._node_card(node) for node in focus_nodes[:4]],
            },
            'paper_evidence': [
                row for row in (evidence_bundle.get('paper_evidence') or [])
                if str(row.get('paper_id') or '') in set(paper_ids)
            ][:10] or list(evidence_bundle.get('paper_evidence') or [])[:10],
            'structure_evidence': structure_rows[:8],
            'section_evidence': section_rows[:8],
            'subdirection_candidates': self._subdirection_candidates(focus_nodes),
            'bottleneck_candidates': self._bottleneck_candidates(focus_nodes),
            'opportunity_candidates': self._opportunity_candidates(focus_nodes),
            'title_domain_tags': self._title_domain_tags(
                focus_nodes,
                list(evidence_bundle.get('paper_evidence') or []),
                structure_rows,
            ),
        }

    def _node_card(self, node: Dict[str, Any]) -> Dict[str, Any]:
        hist = node.get('historical_stats') or {}
        qh = list(node.get('quarterly_history') or [])[-4:]
        return {
            'display_name': node.get('display_name'),
            'public_topic': node.get('public_topic'),
            'description': _clip(node.get('description'), 220),
            'anchor_score': node.get('anchor_score'),
            'historical_stats': {
                'paper_count': hist.get('paper_count'),
                'top_conf_share': hist.get('top_conf_share'),
                'citation_median': hist.get('citation_median'),
            },
            'quarterly_history': qh,
            'top_limitations': [x.get('name') for x in (node.get('top_limitations') or [])[:4] if x.get('name')],
            'top_future_work': [x.get('direction') for x in (node.get('top_future_work') or [])[:4] if x.get('direction')],
            'top_core_ideas': [x.get('name') for x in (node.get('top_core_ideas') or [])[:4] if x.get('name')],
            'recent_representative_titles': [x.get('title') for x in (node.get('historical_representative_papers') or [])[:6] if x.get('title')],
            'split_pressure': node.get('split_pressure'),
            'emergent_descendants': [
                x.get('display_name') or x.get('public_topic') or ''
                for x in (node.get('emergent_descendants') or [])[:6]
                if (x.get('display_name') or x.get('public_topic'))
            ],
        }

    def _subdirection_candidates(self, focus_nodes: List[Dict[str, Any]]) -> List[str]:
        values: List[str] = []
        for node in focus_nodes[:4]:
            values.append(str(node.get('display_name') or ''))
            values.append(str(node.get('public_topic') or ''))
            values.extend(x.get('direction') or '' for x in (node.get('top_future_work') or [])[:6])
            for row in (node.get('historical_representative_papers') or [])[:6]:
                values.append(str(row.get('title') or ''))
        return _dedupe(values)[:24]

    def _bottleneck_candidates(self, focus_nodes: List[Dict[str, Any]]) -> List[str]:
        values = []
        for node in focus_nodes[:4]:
            values.extend(x.get('name') or '' for x in (node.get('top_limitations') or [])[:6])
            values.append(str(node.get('description') or ''))
        return _dedupe(values)[:18]

    def _opportunity_candidates(self, focus_nodes: List[Dict[str, Any]]) -> List[str]:
        values = []
        for node in focus_nodes[:4]:
            values.extend(x.get('direction') or '' for x in (node.get('top_future_work') or [])[:6])
            values.extend(x.get('name') or '' for x in (node.get('top_core_ideas') or [])[:6])
        return _dedupe(values)[:18]

    def _title_domain_tags(
        self,
        focus_nodes: List[Dict[str, Any]],
        paper_rows: List[Dict[str, Any]],
        structure_rows: List[Dict[str, Any]],
    ) -> List[str]:
        texts: List[str] = []
        for node in focus_nodes[:4]:
            texts.extend(str(x.get('title') or '') for x in (node.get('historical_representative_papers') or [])[:8])
        texts.extend(str(row.get('paper_title') or '') for row in paper_rows[:12])
        texts.extend(str(row.get('paper_title') or '') for row in structure_rows[:12])
        joined = ' || '.join(texts).lower()
        mapping = {
            'legal': ['legal', 'law', 'judicial'],
            'medical': ['medical', 'clinical', 'health', 'diagnostic', 'depression'],
            'financial': ['financial', 'finance', 'stock'],
            'telecom': ['telecom', 'radio access network', 'oran', 'wireless'],
            'educational': ['education', 'educational', 'tutor', 'student'],
            'geospatial': ['geospatial', 'spatial', 'map'],
            'enterprise': ['enterprise'],
            'in-car': ['in-car', 'driving', 'vehicle', 'automotive', 'car'],
            'personalized': ['personalized', 'persona'],
            'multi-turn': ['multi-turn', 'multi round', 'conversational', 'dialogue'],
            'long-context': ['long context', 'long-context', 'very long-term'],
            'cybersecurity': ['cybersecurity', 'security'],
            'industrial': ['industrial'],
            'scientific': ['scientific'],
        }
        tags = []
        for label, keys in mapping.items():
            if any(key in joined for key in keys):
                tags.append(label)
        return tags[:12]


class StructuredGenerator:
    def __init__(self, answer_client: OpenAICompatChatClient, critic_client: OpenAICompatChatClient):
        self.answer_client = answer_client
        self.critic_client = critic_client

    def _complete_json(self, client: OpenAICompatChatClient, *, system: str, prompt: str) -> Dict[str, Any]:
        return complete_json_object(
            client,
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': prompt},
            ],
            response_format={'type': 'json_object'},
            transport_retries=2,
            max_parse_attempts=3,
        )


class DirectionHeadV3(StructuredGenerator):
    def run(self, *, task: Dict[str, Any], focus: Dict[str, Any], formatted: Dict[str, Any]) -> Dict[str, Any]:
        heuristic = self._heuristic_direction(focus, formatted)
        subdirection_pool = list(formatted.get('subdirection_candidates') or [])[:14]
        prompt = f"""Write a direction-forecast answer grounded only in the provided evidence.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted['focus_summary'], ensure_ascii=False, indent=2)}

Paper evidence:
{json.dumps(formatted['paper_evidence'], ensure_ascii=False, indent=2)}

Structure evidence:
{json.dumps(formatted['structure_evidence'], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps(formatted['section_evidence'], ensure_ascii=False, indent=2)}

Subdirection candidates:
{json.dumps(subdirection_pool, ensure_ascii=False, indent=2)}

Heuristic suggestion:
{json.dumps(heuristic, ensure_ascii=False, indent=2)}

Requirements:
- Work only with pre-cutoff evidence.
- Output exactly one label from: accelerating, steady, cooling, fragmenting.
- If the broad focus is splitting into several narrower sibling nodes or application variants, prefer fragmenting.
- Subdirections must be anchored in the evidence. Reuse wording from node names, descendant names, future-work items, or representative paper titles when possible.
- Avoid generic phrases that are not clearly visible in the evidence.
- Mention one venue/evaluation shift, preferably about concentration, benchmarking focus, or a change in topical spread.
- Keep the final answer concise and concrete.

Return JSON:
{{
  "trajectory_label": "fragmenting",
  "subdirections": ["...", "..."],
  "venue_or_evaluation_shift": "...",
  "reasoning_chain": ["...", "...", "..."]
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise research forecasting model. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the draft answer.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted['focus_summary'], ensure_ascii=False, indent=2)}

Subdirection candidates:
{json.dumps(subdirection_pool, ensure_ascii=False, indent=2)}

Heuristic suggestion:
{json.dumps(heuristic, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Rules:
- Reject generic subdirections unsupported by the evidence.
- Prefer node-specific or title-visible subdirections.
- If multiple anchored sibling variants are visible, ensure the label is fragmenting unless the evidence strongly contradicts that.
- Keep the final answer compact.

Return JSON:
{{
  "trajectory_label": "fragmenting",
  "subdirections": ["...", "..."],
  "venue_or_evaluation_shift": "..."
}}"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict research forecast evaluator. Output JSON only.',
            prompt=review_prompt,
        )
        answer = self._verbalize_direction(draft=draft, final=final, heuristic=heuristic, subdirection_pool=subdirection_pool)
        return {
            'head': 'direction',
            'heuristic': heuristic,
            'draft': draft,
            'final': final,
            'answer': answer,
        }

    def _verbalize_direction(self, *, draft: Dict[str, Any], final: Dict[str, Any], heuristic: Dict[str, Any], subdirection_pool: List[str]) -> str:
        allowed = {'accelerating', 'steady', 'cooling', 'fragmenting'}
        label = str(final.get('trajectory_label') or draft.get('trajectory_label') or heuristic.get('suggested_label') or 'steady').strip().lower()
        if label not in allowed:
            label = str(heuristic.get('suggested_label') or 'steady')
        chosen = final.get('subdirections') or draft.get('subdirections') or []
        subdirs = self._normalize_against_pool(chosen, subdirection_pool, limit=3)
        venue = str(final.get('venue_or_evaluation_shift') or draft.get('venue_or_evaluation_shift') or '').strip()
        if subdirs:
            return f"The area is most likely {label}. The next technical directions are {', '.join(subdirs[:2])}. {venue}".strip()
        return f"The area is most likely {label}. {venue}".strip()

    @staticmethod
    def _normalize_against_pool(values: Iterable[str], pool: List[str], *, limit: int) -> List[str]:
        out: List[str] = []
        lowered_pool = [(item, _normalize_label(item)) for item in pool if str(item).strip()]
        for value in values or []:
            text = str(value or '').strip()
            if not text:
                continue
            norm = _normalize_label(text)
            replacement = text
            for candidate, candidate_norm in lowered_pool:
                if norm == candidate_norm or norm in candidate_norm or candidate_norm in norm:
                    replacement = candidate
                    break
            if replacement and _normalize_label(replacement) not in {_normalize_label(x) for x in out}:
                out.append(replacement)
            if len(out) >= limit:
                break
        return out

    def _heuristic_direction(self, focus: Dict[str, Any], formatted: Dict[str, Any]) -> Dict[str, Any]:
        nodes = list(focus.get('focus_nodes') or [])
        growth = 0.0
        share_delta = 0.0
        future_work_count = 0
        core_idea_count = 0
        for node in nodes:
            qh = list(node.get('quarterly_history') or [])[-4:]
            growth += sum(float(x.get('paper_growth') or 0.0) for x in qh[-3:])
            if len(qh) >= 2:
                share_delta += float(qh[-1].get('top_conf_share') or 0.0) - float(qh[0].get('top_conf_share') or 0.0)
            future_work_count += len(node.get('top_future_work') or [])
            core_idea_count += len(node.get('top_core_ideas') or [])
        cluster_mode = bool(focus.get('cluster_mode'))
        title_domain_tags = list(formatted.get('title_domain_tags') or [])
        diversity_signal = len(set(title_domain_tags)) + min(3, future_work_count) + min(2, core_idea_count)
        if cluster_mode and len(nodes) >= 2:
            label = 'fragmenting'
        elif diversity_signal >= 4 and share_delta <= -0.03:
            label = 'fragmenting'
        elif growth >= 15 and share_delta >= -0.02:
            label = 'accelerating'
        elif growth <= 2 or share_delta <= -0.06:
            label = 'cooling'
        else:
            label = 'steady'
        return {
            'suggested_label': label,
            'cluster_mode': cluster_mode,
            'focus_node_count': len(nodes),
            'historical_diversity_signal': diversity_signal,
            'recent_growth_sum': round(growth, 4),
            'recent_top_venue_share_delta': round(share_delta, 4),
            'title_domain_tags': title_domain_tags[:8],
        }


class BottleneckHeadV3(StructuredGenerator):
    def run(self, *, task: Dict[str, Any], focus: Dict[str, Any], formatted: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Write a bottleneck-opportunity answer grounded only in the provided evidence.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted['focus_summary'], ensure_ascii=False, indent=2)}

Structure evidence:
{json.dumps(formatted['structure_evidence'], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps(formatted['section_evidence'], ensure_ascii=False, indent=2)}

Bottleneck candidates:
{json.dumps(formatted['bottleneck_candidates'], ensure_ascii=False, indent=2)}

Opportunity candidates:
{json.dumps(formatted['opportunity_candidates'], ensure_ascii=False, indent=2)}

Requirements:
- Identify one precise unresolved bottleneck, not a vague topic label.
- Link it to one concrete opportunity implied by future-work, core-idea, descendant, or recent paper evidence.
- Explain the linkage briefly.
- Keep the answer concise and benchmark-style.

Return JSON:
{{
  "bottleneck": "...",
  "opportunity": "...",
  "linkage": "...",
  "final_answer": "..."
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise research synthesis model. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the bottleneck-opportunity draft.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted['focus_summary'], ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Rules:
- The bottleneck must be specific.
- The opportunity must clearly address the bottleneck.
- Prefer wording that is visible in limitations, future-work, section snippets, or recent paper titles.

Return JSON:
{{
  "bottleneck": "...",
  "opportunity": "...",
  "linkage": "...",
  "final_answer": "..."
}}"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict benchmark evaluator. Output JSON only.',
            prompt=review_prompt,
        )
        return {
            'head': 'bottleneck',
            'draft': draft,
            'final': final,
            'answer': str(final.get('final_answer') or draft.get('final_answer') or '').strip(),
        }


class PlanningHeadV3(StructuredGenerator):
    def run(self, *, task: Dict[str, Any], focus: Dict[str, Any], formatted: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        contract = extract_task_contract(task)
        heuristic = self._planning_heuristic(focus=focus, formatted=formatted, domain_id=domain_id)
        candidate_rows = self._build_priority_candidates(focus=focus, formatted=formatted, domain_id=domain_id, contract=contract)
        prompt = f"""Rank the highest-priority research directions grounded only in the provided evidence.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Task contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted['focus_summary'], ensure_ascii=False, indent=2)}

Priority candidates:
{json.dumps(candidate_rows[:10], ensure_ascii=False, indent=2)}

Planning heuristic:
{json.dumps(heuristic, ensure_ascii=False, indent=2)}

Requirements:
- Select up to {int(contract.get('max_items') or 3)} directions.
- The direction wording should stay close to the candidate strings.
- Rank the selected directions from highest to lowest priority.
- Each direction must include a brief rationale using momentum, venue/citation signals, and technical gaps from the evidence.
- Do not collapse everything into one agenda. The task requires prioritized directions.
- Keep each rationale compact and benchmark-style.

Return JSON:
{{
  "ranked_directions": [
    {{
      "rank": 1,
      "direction": "...",
      "why_prioritized": "...",
      "evidence_titles": ["..."],
      "signal_summary": "..."
    }}
  ]
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise research planning model. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the ranked-direction draft.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Priority candidates:
{json.dumps(candidate_rows[:10], ensure_ascii=False, indent=2)}

Planning heuristic:
{json.dumps(heuristic, ensure_ascii=False, indent=2)}

Rules:
- Keep the ranked list length at or below {int(contract.get('max_items') or 3)}.
- Reject generic umbrella directions if more specific descendants or future-work directions are available.
- Prefer explicit venue/citation/momentum justification when available.
- Keep direction wording close to the candidate list.

Return JSON:
{{
  "ranked_directions": [
    {{
      "rank": 1,
      "direction": "...",
      "why_prioritized": "...",
      "evidence_titles": ["..."],
      "signal_summary": "..."
    }}
  ]
}}"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict research planning evaluator. Output JSON only.',
            prompt=review_prompt,
        )
        answer = self._verbalize_ranked_directions(
            task=task,
            contract=contract,
            candidate_rows=candidate_rows,
            draft=draft,
            final=final,
        )
        return {
            'head': 'planning',
            'contract': contract,
            'heuristic': heuristic,
            'candidate_rows': candidate_rows,
            'draft': draft,
            'final': final,
            'answer': answer,
        }

    def _planning_heuristic(self, *, focus: Dict[str, Any], formatted: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        nodes = list((formatted.get('focus_summary') or {}).get('focus_nodes') or [])
        main = nodes[0] if nodes else {}
        qh = list(main.get('quarterly_history') or [])
        share_delta = 0.0
        if len(qh) >= 2:
            share_delta = float(qh[-1].get('top_conf_share') or 0.0) - float(qh[0].get('top_conf_share') or 0.0)
        tags = list(formatted.get('title_domain_tags') or [])
        future_work = list(main.get('top_future_work') or [])
        preferred_axis = 'bottleneck_opportunity'
        rationale = 'use a concrete bottleneck-opportunity path'
        if future_work and share_delta <= -0.05:
            preferred_axis = 'descendant'
            rationale = 'venue concentration is weakening while historically visible future-work directions already indicate narrower next steps'
        elif domain_id == 'rag_and_retrieval_structuring' and tags:
            preferred_axis = 'domain_specialization'
            rationale = 'representative titles already show domain/application specialization'
        return {
            'preferred_axis': preferred_axis,
            'share_delta': round(share_delta, 4),
            'historical_future_work': future_work[:6],
            'title_domain_tags': tags[:8],
            'rationale': rationale,
        }

    def _build_priority_candidates(
        self,
        *,
        focus: Dict[str, Any],
        formatted: Dict[str, Any],
        domain_id: str,
        contract: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen = set()
        contract_terms = set(_norm_tokens(contract.get('topic_text') or ''))
        nodes = list((formatted.get('focus_summary') or {}).get('focus_nodes') or [])
        for node in nodes[:4]:
            hist = node.get('historical_stats') or {}
            qh = list(node.get('quarterly_history') or [])
            recent_growth = sum(float(x.get('paper_growth') or 0.0) for x in qh[-3:])
            share_delta = 0.0
            if len(qh) >= 2:
                share_delta = float(qh[-1].get('top_conf_share') or 0.0) - float(qh[0].get('top_conf_share') or 0.0)
            evidence_titles = list(node.get('recent_representative_titles') or [])[:3]
            for source_type, values, base in [
                ('descendant', list(node.get('emergent_descendants') or []), 1.35),
                ('future_work', list(node.get('top_future_work') or []), 1.15),
                ('core_idea', list(node.get('top_core_ideas') or []), 0.75),
            ]:
                for value in values[:5]:
                    direction = str(value or '').strip()
                    if not direction:
                        continue
                    key = _normalize_label(direction)
                    if key in seen:
                        continue
                    seen.add(key)
                    specificity = min(1.0, len(_norm_tokens(direction)) / 4.0)
                    topical_overlap = len(contract_terms & set(_norm_tokens(f"{direction} {node.get('display_name') or ''} {node.get('public_topic') or ''}")))
                    score = (
                        base
                        + min(0.9, max(0.0, recent_growth) / 25.0)
                        + min(0.55, float(hist.get('top_conf_share') or 0.0) * 1.5)
                        + min(0.45, float(hist.get('citation_median') or 0.0) / 30.0)
                        + (0.45 if source_type == 'descendant' else 0.0) * min(1.0, float(node.get('split_pressure') or 0.0) / 12.0)
                        + 0.2 * specificity
                        + 0.35 * topical_overlap
                    )
                    if contract_terms and topical_overlap == 0:
                        score -= 0.45
                    if share_delta <= -0.04 and source_type in {'descendant', 'future_work'}:
                        score += 0.2
                    if domain_id == 'rag_and_retrieval_structuring' and any(k in key for k in ['retrieval', 'rerank', 'query', 'graph', 'chunk', 'controller']):
                        score += 0.25
                    rows.append(
                        {
                            'direction': direction,
                            'source_type': source_type,
                            'source_node': node.get('display_name') or node.get('public_topic'),
                            'score': round(score, 4),
                            'recent_growth': round(recent_growth, 4),
                            'top_conf_share': hist.get('top_conf_share'),
                            'citation_median': hist.get('citation_median'),
                            'share_delta': round(share_delta, 4),
                            'split_pressure': node.get('split_pressure'),
                            'evidence_titles': evidence_titles,
                            'signal_summary': self._signal_summary(
                                source_type=source_type,
                                recent_growth=recent_growth,
                                top_conf_share=float(hist.get('top_conf_share') or 0.0),
                                citation_median=float(hist.get('citation_median') or 0.0),
                                share_delta=share_delta,
                                split_pressure=float(node.get('split_pressure') or 0.0),
                            ),
                        }
                    )
        rows.sort(key=lambda row: float(row.get('score') or 0.0), reverse=True)
        return rows[: max(8, int(contract.get('max_items') or 3) + 4)]

    @staticmethod
    def _signal_summary(
        *,
        source_type: str,
        recent_growth: float,
        top_conf_share: float,
        citation_median: float,
        share_delta: float,
        split_pressure: float,
    ) -> str:
        parts = [
            f"source={source_type}",
            f"recent_growth={round(recent_growth, 2)}",
            f"top_conf_share={round(top_conf_share, 4)}",
            f"citation_median={round(citation_median, 2)}",
        ]
        if abs(share_delta) > 1e-6:
            parts.append(f"share_delta={round(share_delta, 4)}")
        if split_pressure:
            parts.append(f"split_pressure={round(split_pressure, 2)}")
        return '; '.join(parts)

    def _verbalize_ranked_directions(
        self,
        *,
        task: Dict[str, Any],
        contract: Dict[str, Any],
        candidate_rows: List[Dict[str, Any]],
        draft: Dict[str, Any],
        final: Dict[str, Any],
    ) -> str:
        candidates_by_norm = {_normalize_label(row.get('direction')): row for row in candidate_rows}
        ranked_rows = final.get('ranked_directions') or draft.get('ranked_directions') or []
        selected: List[Dict[str, Any]] = []
        seen = set()
        max_items = int(contract.get('max_items') or 3)
        for idx, item in enumerate(ranked_rows, start=1):
            direction = str((item or {}).get('direction') or '').strip()
            if not direction:
                continue
            norm = _normalize_label(direction)
            matched = candidates_by_norm.get(norm)
            if matched is None:
                for candidate_norm, row in candidates_by_norm.items():
                    if norm in candidate_norm or candidate_norm in norm:
                        matched = row
                        break
            if matched is None:
                continue
            key = _normalize_label(matched.get('direction'))
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    'rank': len(selected) + 1,
                    'direction': matched.get('direction'),
                    'why_prioritized': str((item or {}).get('why_prioritized') or '').strip(),
                    'signal_summary': str((item or {}).get('signal_summary') or matched.get('signal_summary') or '').strip(),
                    'evidence_titles': (item or {}).get('evidence_titles') or matched.get('evidence_titles') or [],
                }
            )
            if len(selected) >= max_items:
                break
        if not selected:
            for row in candidate_rows[:max_items]:
                selected.append(
                    {
                        'rank': len(selected) + 1,
                        'direction': row.get('direction'),
                        'why_prioritized': '',
                        'signal_summary': row.get('signal_summary'),
                        'evidence_titles': row.get('evidence_titles') or [],
                    }
                )
        if len(selected) < max_items:
            existing = {_normalize_label(row.get('direction')) for row in selected}
            for row in candidate_rows:
                key = _normalize_label(row.get('direction'))
                if key in existing:
                    continue
                selected.append(
                    {
                        'rank': len(selected) + 1,
                        'direction': row.get('direction'),
                        'why_prioritized': '',
                        'signal_summary': row.get('signal_summary'),
                        'evidence_titles': row.get('evidence_titles') or [],
                    }
                )
                existing.add(key)
                if len(selected) >= max_items:
                    break
        lines = []
        for item in selected:
            reason = item.get('why_prioritized') or item.get('signal_summary') or 'historical momentum and technical readiness make this a high-priority direction'
            titles = [str(x).strip() for x in (item.get('evidence_titles') or []) if str(x).strip()][:2]
            title_suffix = f" Evidence anchors: {', '.join(titles)}." if titles else ''
            lines.append(f"{item['rank']}. {item['direction']} — {reason}.{title_suffix}".replace('..', '.'))
        return '\n'.join(lines)


class ResearchArcV3:
    def __init__(self, *, answer_client: OpenAICompatChatClient, critic_client: Optional[OpenAICompatChatClient] = None):
        self.answer_client = answer_client
        self.critic_client = critic_client or answer_client
        self.backbones: Dict[str, EvidenceBackbone] = {}
        self.router = PolicyRouter()
        self.focus_resolver = FocusResolver()
        self.formatter = EvidenceFormatter()
        self.direction_head = DirectionHeadV3(answer_client=self.answer_client, critic_client=self.critic_client)
        self.bottleneck_head = BottleneckHeadV3(answer_client=self.answer_client, critic_client=self.critic_client)
        self.planning_head = PlanningHeadV3(answer_client=self.answer_client, critic_client=self.critic_client)

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
        return [
            {'step': 'high_recall_support_packet_retrieval', 'goal': 'recover likely benchmark nodes before the cutoff'},
            {'step': 'question_anchor_alignment', 'goal': 'rank candidate nodes against the task focus instead of generic domain popularity'},
            {'step': 'representative_paper_and_section_retrieval', 'goal': 'collect limitations, future-work, and section-level evidence for anchored nodes'},
            {'step': 'anchored_evidence_synthesis', 'goal': 'compose an answer from node-specific evidence rather than generic domain-wide templates'},
            {'step': 'critic_repair', 'goal': f"repair the {policy.get('head')} answer under {policy.get('domain_policy')} policy"},
        ]

    def run_task(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        task_parse = self.parse_task(task)
        policy = self.router.route(task_parse=task_parse, domain_id=domain_id)
        retrieval_plan = self.plan_retrieval(task_parse=task_parse, policy=policy)
        evidence_bundle = self._backbone(domain_id).build(task=task)
        focus = self.focus_resolver.resolve(task=task, evidence_bundle=evidence_bundle)
        formatted = self.formatter.build(focus=focus, evidence_bundle=evidence_bundle)
        head = policy.get('head')
        if head == 'direction':
            head_result = self.direction_head.run(task=task, focus=focus, formatted=formatted)
        elif head == 'planning':
            head_result = self.planning_head.run(task=task, focus=focus, formatted=formatted, domain_id=domain_id)
        else:
            head_result = self.bottleneck_head.run(task=task, focus=focus, formatted=formatted)
        return {
            'task_parse': task_parse,
            'policy': policy,
            'retrieval_plan': retrieval_plan,
            'focus': focus,
            'formatted_evidence': formatted,
            'evidence_bundle': evidence_bundle,
            'head_result': head_result,
            'answer': head_result.get('answer') or '',
        }
