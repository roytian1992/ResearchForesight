from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.research_arc_v2 import EvidenceBackbone, PolicyRouter, extract_task_contract
from researchworld.research_arc_v3 import (
    DirectionHeadV3,
    EvidenceFormatter,
    FocusResolver,
    PlanningHeadV3,
    StructuredGenerator,
)


ROOT = Path(__file__).resolve().parents[2]


class BottleneckHeadV4(StructuredGenerator):
    RELATION_HINTS = {
        'efficiency': ['efficiency', 'cost', 'latency', 'speed', 'trade-off', 'context limit', 'context limits'],
        'structure': ['structure', 'graph', 'flat', 'topology', 'semantic'],
        'adaptivity': ['adaptive', 'non-adaptive', 'planning', 'dynamic', 'feedback', 'self-reflective', 'self reflective'],
        'relevance': ['relevance', 'lexical', 'noisy', 'sufficiency', 'query', 'retrieval'],
        'integration': ['integration', 'context', 'multi-hop', 'multi hop'],
    }

    def run(self, *, task: Dict[str, Any], focus: Dict[str, Any], formatted: Dict[str, Any]) -> Dict[str, Any]:
        candidates = self._build_candidates(formatted)
        task_contract = extract_task_contract(task)
        prompt = f"""Select and polish the strongest bottleneck-opportunity pair.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted['focus_summary'], ensure_ascii=False, indent=2)}

Candidate bottleneck-opportunity pairs:
{json.dumps(candidates[:8], ensure_ascii=False, indent=2)}

Structure evidence:
{json.dumps(formatted['structure_evidence'], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps(formatted['section_evidence'], ensure_ascii=False, indent=2)}

Paper evidence:
{json.dumps(formatted['paper_evidence'], ensure_ascii=False, indent=2)}

Requirements:
- Pick one bottleneck-opportunity pair only.
- Reuse the candidate bottleneck/opportunity wording as much as possible.
- Prefer a bottleneck that is specific and repeatedly visible in the historical evidence.
- Prefer an opportunity that directly opens because of that bottleneck, not a generic improvement direction.
- Mention 1 or 2 historical paper titles when they materially ground the claim.
- Keep the answer compact and technical.

Return JSON:
{{
  "selected_index": 0,
  "bottleneck": "...",
  "opportunity": "...",
  "evidence_titles": ["..."],
  "linkage": "...",
  "final_answer": "..."
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise benchmark-aligned research synthesis model. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the selected bottleneck-opportunity answer.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidates[:8], ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Rules:
- Reject vague bottlenecks like broad topic names.
- Reject opportunities that do not clearly address the bottleneck.
- Prefer historically grounded bottlenecks with explicit paper-title anchors.
- Prefer opportunities that reflect future-work, core ideas, or emergent descendants visible in the evidence.
- Keep the final answer compact.

Return JSON:
{{
  "selected_index": 0,
  "bottleneck": "...",
  "opportunity": "...",
  "evidence_titles": ["..."],
  "linkage": "...",
  "final_answer": "..."
}}"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict benchmark evaluator. Output JSON only.',
            prompt=review_prompt,
        )
        idx = int(final.get('selected_index') or draft.get('selected_index') or 0)
        idx = max(0, min(idx, max(0, len(candidates) - 1)))
        selected = candidates[idx] if candidates else {}
        answer = self._verbalize_bottleneck(
            draft=draft,
            final=final,
            selected=selected,
            task_contract=task_contract,
        )
        return {
            'head': 'bottleneck',
            'candidates': candidates,
            'draft': draft,
            'final': final,
            'selected_candidate': selected,
            'answer': answer,
        }

    def _verbalize_bottleneck(
        self,
        *,
        draft: Dict[str, Any],
        final: Dict[str, Any],
        selected: Dict[str, Any],
        task_contract: Dict[str, Any],
    ) -> str:
        bottleneck = str(final.get('bottleneck') or draft.get('bottleneck') or selected.get('bottleneck') or '').strip()
        opportunity = str(final.get('opportunity') or draft.get('opportunity') or selected.get('opportunity') or '').strip()
        linkage = str(final.get('linkage') or draft.get('linkage') or selected.get('support_rationale') or '').strip()
        titles = final.get('evidence_titles') or draft.get('evidence_titles') or selected.get('support_titles') or []
        title_suffix = ''
        clean_titles = [str(x).strip() for x in titles if str(x).strip()][:2]
        if clean_titles:
            title_suffix = f" Evidence anchors: {', '.join(clean_titles)}."
        if bottleneck and opportunity:
            return f"A concrete unresolved bottleneck is {bottleneck}. The associated research opportunity is {opportunity}. {linkage}.{title_suffix}".replace('..', '.')
        return str(final.get('final_answer') or draft.get('final_answer') or '').strip()

    def _build_candidates(self, formatted: Dict[str, Any]) -> List[Dict[str, Any]]:
        bottlenecks = list(formatted.get('bottleneck_candidates') or [])[:12]
        opportunities = list(formatted.get('opportunity_candidates') or [])[:12]
        title_pool = self._title_pool(formatted)
        rows: List[Dict[str, Any]] = []
        for b in bottlenecks:
            for o in opportunities:
                titles = self._support_titles(b, o, title_pool)
                score = self._pair_score(b, o)
                rows.append(
                    {
                        'bottleneck': b,
                        'opportunity': o,
                        'score': round(score, 4),
                        'support_titles': titles[:3],
                        'support_rationale': self._support_rationale(b, o),
                    }
                )
        rows.sort(key=lambda x: float(x.get('score') or 0.0), reverse=True)
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for row in rows:
            key = (str(row['bottleneck']).lower(), str(row['opportunity']).lower())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= 10:
                break
        return deduped

    def _pair_score(self, bottleneck: str, opportunity: str) -> float:
        b = str(bottleneck or '').lower()
        o = str(opportunity or '').lower()
        score = 0.0
        score += min(len(b.split()) / 10.0, 0.8)
        score += min(len(o.split()) / 10.0, 0.8)
        score += 0.25 if any(term in b for term in ['trade-off', 'non-adaptive', 'lexical', 'noisy', 'context', 'flat', 'latency', 'cost']) else 0.0
        score += 0.2 if any(term in o for term in ['adaptive', 'hybrid', 'graph', 'subgraph', 'planning', 'self-reflective', 'controller', 'retrieval']) else 0.0
        for _, hints in self.RELATION_HINTS.items():
            if any(h in b for h in hints) and any(h in o for h in hints):
                score += 0.35
        if 'trade-off' in b and any(h in o for h in ['adaptive', 'flexible', 'hybrid', 'subgraph']):
            score += 0.45
        if any(h in b for h in ['non-adaptive retrieval', 'non-adaptive retrieval queries', 'overloaded retrieval queries']) and any(h in o for h in ['adaptive', 'planning', 'self-reflective', 'retrieval as generation']):
            score += 0.55
        if any(h in b for h in ['lexical overlap', 'relevance']) and any(h in o for h in ['code-oriented', 'generation', 'selective']):
            score += 0.45
        if any(h in b for h in ['flat data', 'plain text treatment']) and any(h in o for h in ['graph', 'semantic', 'hybrid']):
            score += 0.45
        return score

    def _title_pool(self, formatted: Dict[str, Any]) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for row in (formatted.get('paper_evidence') or [])[:12]:
            rows.append({'title': str(row.get('paper_title') or ''), 'text': str(row.get('snippet') or '')})
        for row in (formatted.get('structure_evidence') or [])[:12]:
            text = ' '.join(
                [
                    str(row.get('problem_statement') or ''),
                    ' '.join(row.get('limitations') or []),
                    ' '.join(row.get('future_work') or []),
                    ' '.join(row.get('core_ideas') or []),
                ]
            )
            rows.append({'title': str(row.get('paper_title') or ''), 'text': text})
        return rows

    def _support_titles(self, bottleneck: str, opportunity: str, title_pool: List[Dict[str, str]]) -> List[str]:
        terms = [t for t in re.findall(r'[a-zA-Z][a-zA-Z\-]+', f'{bottleneck} {opportunity}'.lower()) if len(t) > 3]
        scored = []
        for row in title_pool:
            hay = f"{row['title']} {row['text']}".lower()
            overlap = sum(1 for t in terms if t in hay)
            if overlap:
                scored.append((overlap, row['title']))
        scored.sort(reverse=True)
        out = []
        seen = set()
        for _, title in scored:
            if title and title not in seen:
                seen.add(title)
                out.append(title)
            if len(out) >= 3:
                break
        return out

    def _support_rationale(self, bottleneck: str, opportunity: str) -> str:
        b = str(bottleneck or '').lower()
        o = str(opportunity or '').lower()
        if 'trade-off' in b and any(h in o for h in ['adaptive', 'flexible', 'hybrid', 'subgraph']):
            return 'addresses efficiency-relevance trade-off with adaptive retrieval control'
        if any(h in b for h in ['non-adaptive', 'overloaded retrieval queries']) and any(h in o for h in ['adaptive', 'planning', 'retrieval as generation']):
            return 'turns fixed retrieval into adaptive query planning'
        if any(h in b for h in ['lexical overlap', 'relevance']) and any(h in o for h in ['code-oriented', 'generation', 'selective']):
            return 'improves relevance capture under lexically mismatched contexts'
        if any(h in b for h in ['plain text treatment', 'flat data']) and any(h in o for h in ['graph', 'semantic', 'hybrid']):
            return 'moves from flattened text to structure-aware retrieval and reasoning'
        return 'provides a plausible direct opening from the unresolved bottleneck'


class ResearchArcV4:
    def __init__(self, *, answer_client: OpenAICompatChatClient, critic_client: Optional[OpenAICompatChatClient] = None):
        self.answer_client = answer_client
        self.critic_client = critic_client or answer_client
        self.backbones: Dict[str, EvidenceBackbone] = {}
        self.router = PolicyRouter()
        self.focus_resolver = FocusResolver()
        self.formatter = EvidenceFormatter()
        self.direction_head = DirectionHeadV3(answer_client=self.answer_client, critic_client=self.critic_client)
        self.planning_head = PlanningHeadV3(answer_client=self.answer_client, critic_client=self.critic_client)
        self.bottleneck_head = BottleneckHeadV4(answer_client=self.answer_client, critic_client=self.critic_client)

    def _fallback_answer(self, *, task: Dict[str, Any], head_result: Dict[str, Any]) -> str:
        family = str(task.get('family') or '')
        draft = head_result.get('draft') or {}
        final = head_result.get('final') or {}
        if family == 'strategic_research_planning':
            rows = final.get('ranked_directions') or draft.get('ranked_directions') or []
            lines: List[str] = []
            for idx, row in enumerate(rows[:3], start=1):
                direction = str((row or {}).get('direction') or '').strip()
                reason = str((row or {}).get('why_prioritized') or (row or {}).get('signal_summary') or '').strip()
                if not direction:
                    continue
                lines.append(f"{idx}. {direction} — {reason}".rstrip(' —'))
            return '\n'.join(lines).strip()
        if family == 'direction_forecasting':
            label = str(final.get('trajectory_label') or draft.get('trajectory_label') or '').strip().lower()
            subdirs = [str(x).strip() for x in (final.get('subdirections') or draft.get('subdirections') or []) if str(x).strip()]
            venue = str(final.get('venue_or_evaluation_shift') or draft.get('venue_or_evaluation_shift') or '').strip()
            if label and subdirs:
                return f"The area is most likely {label}. The next technical directions are {', '.join(subdirs[:3])}. {venue}".replace('..', '.').strip()
            if label:
                return f"The area is most likely {label}. {venue}".replace('..', '.').strip()
            return ''
        bottleneck = str(final.get('bottleneck') or draft.get('bottleneck') or '').strip()
        opportunity = str(final.get('opportunity') or draft.get('opportunity') or '').strip()
        linkage = str(final.get('linkage') or draft.get('linkage') or '').strip()
        if bottleneck and opportunity:
            return f"A concrete unresolved bottleneck is {bottleneck}. The associated research opportunity is {opportunity}. {linkage}".replace('..', '.').strip()
        return str(final.get('final_answer') or draft.get('final_answer') or '').strip()

    def _finalize_answer(
        self,
        *,
        task: Dict[str, Any],
        focus: Dict[str, Any],
        formatted: Dict[str, Any],
        head_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task.get('family') or '')
        fallback_answer = self._fallback_answer(task=task, head_result=head_result)
        current_answer = str(head_result.get('answer') or '').strip() or fallback_answer
        guidance = {
            'bottleneck_opportunity_discovery': [
                'State exactly one bottleneck and one directly linked opportunity.',
                'The opportunity must open because the bottleneck is addressed, not just be a neighboring topic.',
                'Keep the answer as one compact technical paragraph.',
            ],
            'direction_forecasting': [
                'Make exactly one trajectory call from accelerating, steady, cooling, fragmenting.',
                'Name one to three concrete subdirections visible in the evidence.',
                'Briefly explain the historical basis for that trajectory.',
            ],
            'strategic_research_planning': [
                'Return a short prioritized list rather than a free-form paragraph.',
                'Each item should contain a concrete direction and a compact reason.',
                'Prefer directions visible in descendants, future-work signals, or representative-paper themes.',
            ],
        }.get(family, ['Keep the answer concrete and directly aligned with the task.'])
        prompt = f"""You are refining a ResearchArc answer for a benchmark task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted.get('focus_summary') or {}, ensure_ascii=False, indent=2)}

Current structured head result:
{json.dumps({'head': head_result.get('head'), 'draft': head_result.get('draft'), 'final': head_result.get('final')}, ensure_ascii=False, indent=2)}

Current answer:
{current_answer}

Paper evidence:
{json.dumps((formatted.get('paper_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Structure evidence:
{json.dumps((formatted.get('structure_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps((formatted.get('section_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Family-specific guidance:
{json.dumps(guidance, ensure_ascii=False, indent=2)}

Rules:
- Use only the provided historical evidence.
- Preserve the same substantive position unless it is clearly under-specified.
- Improve task fulfillment, specificity, and strategic clarity.
- Remove vague filler and unsupported expansion.
- If the current answer is empty or weak, reconstruct it from the structured head result and evidence.

Return JSON:
{{
  "final_answer": "...",
  "revision_notes": ["..."]
}}"""
        revised = complete_json_object(
            self.answer_client,
            [
                {'role': 'system', 'content': 'You are a precise benchmark answer refiner. Output JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            response_format={'type': 'json_object'},
            max_tokens=1200,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
        final_answer = str(revised.get('final_answer') or '').strip() or current_answer
        if not final_answer:
            final_answer = fallback_answer
        return {
            'final_answer': final_answer.strip(),
            'revision_notes': [str(x).strip() for x in (revised.get('revision_notes') or []) if str(x).strip()],
            'pre_refine_answer': current_answer,
        }

    def _backbone(self, domain_id: str) -> EvidenceBackbone:
        if domain_id not in self.backbones:
            self.backbones[domain_id] = EvidenceBackbone(domain_id)
        return self.backbones[domain_id]

    def parse_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        family = str(task.get('family') or '')
        contract = extract_task_contract(task)
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
            'task_contract': contract,
        }

    def plan_retrieval(self, task_parse: Dict[str, Any], policy: Dict[str, str]) -> List[Dict[str, Any]]:
        return [
            {'step': 'high_recall_support_packet_retrieval', 'goal': 'recover likely benchmark nodes before the cutoff'},
            {'step': 'question_anchor_alignment', 'goal': 'rank candidate nodes against the task focus'},
            {'step': 'representative_paper_and_section_retrieval', 'goal': 'collect limitation, future-work, and section-level evidence for anchored nodes'},
            {'step': 'family_specific_synthesis', 'goal': f"compose a {policy.get('head')} answer from anchored evidence"},
            {'step': 'critic_repair', 'goal': 'repair the answer under benchmark-aligned constraints'},
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
        refinement = self._finalize_answer(task=task, focus=focus, formatted=formatted, head_result=head_result)
        evidence = {
            'papers': list(evidence_bundle.get('paper_evidence') or []),
            'structures': list(evidence_bundle.get('structure_evidence') or []),
            'pageindex': [],
            'fulltext': list(evidence_bundle.get('section_evidence') or []),
        }
        draft_answer = json.dumps(head_result.get('draft') or {}, ensure_ascii=False, sort_keys=True)
        final_answer = json.dumps(head_result.get('final') or {}, ensure_ascii=False, sort_keys=True)
        diagnostics = {
            'tool_calls': 3 + sum(1 for key in ['papers', 'structures', 'fulltext'] if evidence.get(key)),
            'reflection_steps': 2,
            'memory_updates': 0,
            'revision_rounds': 2,
            'answer_changed_after_revision': draft_answer != final_answer or str(refinement.get('pre_refine_answer') or '').strip() != str(refinement.get('final_answer') or '').strip(),
        }
        return {
            'task_parse': task_parse,
            'policy': policy,
            'retrieval_plan': retrieval_plan,
            'focus': focus,
            'formatted_evidence': formatted,
            'evidence_bundle': evidence_bundle,
            'head_result': head_result,
            'refinement': refinement,
            'retrieval_mode': 'support_packet+paper+structure+section',
            'evidence': evidence,
            'diagnostics': diagnostics,
            'answer': refinement.get('final_answer') or head_result.get('answer') or '',
        }
