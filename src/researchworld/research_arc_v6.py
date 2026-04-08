from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from researchworld.llm import OpenAICompatChatClient
from researchworld.research_arc_v2 import EvidenceBackbone, PolicyRouter, extract_task_contract
from researchworld.research_arc_v3 import (
    DirectionHeadV3,
    EvidenceFormatter,
    FocusResolver,
    PlanningHeadV3,
    StructuredGenerator,
)
from researchworld.research_arc_v4 import BottleneckHeadV4

ROOT = Path(__file__).resolve().parents[2]


class SignalAbstractorV6(StructuredGenerator):
    def run(
        self,
        *,
        task: Dict[str, Any],
        focus: Dict[str, Any],
        formatted: Dict[str, Any],
        head_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task.get('family') or '')
        contract = extract_task_contract(task)
        prior = self._build_prior_bundle(head_result=head_result)
        prompt = f"""Build a grounded historical signal map for a research-trajectory benchmark task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Task contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted.get('focus_summary') or {}, ensure_ascii=False, indent=2)}

Prior family-specific synthesis:
{json.dumps(prior, ensure_ascii=False, indent=2)}

Paper evidence:
{json.dumps((formatted.get('paper_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Structure evidence:
{json.dumps((formatted.get('structure_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps((formatted.get('section_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Candidate subdirections:
{json.dumps((formatted.get('subdirection_candidates') or [])[:18], ensure_ascii=False, indent=2)}

Candidate bottlenecks:
{json.dumps((formatted.get('bottleneck_candidates') or [])[:12], ensure_ascii=False, indent=2)}

Candidate opportunities:
{json.dumps((formatted.get('opportunity_candidates') or [])[:12], ensure_ascii=False, indent=2)}

Requirements:
- Use only pre-cutoff evidence.
- Convert raw evidence into compact, benchmark-useful historical signals.
- Prefer concrete technical observations, not broad topic names.
- Preserve evidence grounding with paper titles whenever possible.
- Surface mechanism-level structure: recurring bottlenecks, trade-offs, inflection points, and emerging directions.
- For planning tasks, include agenda axes that are concrete enough to rank.
- When a benchmark or dataset is mentioned, separate the underlying technical failure mode from the missing-resource statement whenever possible.

Return JSON:
{{
  "observations": [
    {{"signal_type": "limitation | momentum | specialization | evaluation_shift | mechanism", "statement": "...", "evidence_titles": ["..."]}}
  ],
  "tradeoffs": [
    {{"name": "...", "explanation": "...", "evidence_titles": ["..."]}}
  ],
  "recurring_bottlenecks": [
    {{"name": "...", "why_recurring": "...", "evidence_titles": ["..."]}}
  ],
  "inflection_points": [
    {{"name": "...", "why_it_matters": "...", "evidence_titles": ["..."]}}
  ],
  "emerging_directions": [
    {{"name": "...", "why_visible_now": "...", "source_type": "descendant | future_work | core_idea | representative_paper", "evidence_titles": ["..."]}}
  ],
  "agenda_axes": [
    {{"direction": "...", "why_now": "...", "dependency_or_tradeoff": "...", "evidence_titles": ["..."]}}
  ]
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise research signal abstractor. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the historical signal map.

Family:
{family}

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Rules:
- Remove generic items that are not obviously grounded.
- Prefer concrete technical bottlenecks, inflection points, and research directions.
- Keep agenda axes distinct from one another.
- Keep each field concise and evidence-anchored.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict research signal auditor. Output JSON only.',
            prompt=review_prompt,
        )
        return final or draft

    @staticmethod
    def _build_prior_bundle(*, head_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'head': head_result.get('head'),
            'heuristic': head_result.get('heuristic') or {},
            'draft': head_result.get('draft') or {},
            'final': head_result.get('final') or {},
            'candidate_rows': list(head_result.get('candidate_rows') or [])[:8],
            'candidates': list(head_result.get('candidates') or [])[:6],
            'selected_candidate': head_result.get('selected_candidate') or {},
        }


class MechanismReasonerV6(StructuredGenerator):
    def run(
        self,
        *,
        task: Dict[str, Any],
        abstraction: Dict[str, Any],
        head_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task.get('family') or '')
        if family == 'bottleneck_opportunity_discovery':
            return self._run_bottleneck(task=task, abstraction=abstraction, head_result=head_result)
        if family == 'direction_forecasting':
            return self._run_direction(task=task, abstraction=abstraction, head_result=head_result)
        return self._run_planning(task=task, abstraction=abstraction, head_result=head_result)

    def _run_bottleneck(self, *, task: Dict[str, Any], abstraction: Dict[str, Any], head_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Produce a bottleneck-opportunity answer from grounded historical signals.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Historical signal map:
{json.dumps(abstraction, ensure_ascii=False, indent=2)}

Prior synthesis:
{json.dumps({'draft': head_result.get('draft') or {}, 'final': head_result.get('final') or {}, 'candidates': list(head_result.get('candidates') or [])[:6]}, ensure_ascii=False, indent=2)}

Requirements:
- Choose exactly one concrete technical bottleneck.
- Choose exactly one opportunity that becomes more viable if that bottleneck is addressed.
- Prefer technical bottlenecks over purely missing-resource statements unless the evidence clearly makes the resource constraint itself technical.
- If a benchmark or dataset appears in the evidence, treat it as an enabler or inflection point unless the true unresolved problem is itself benchmark design or evaluation protocol.
- Explain one historical observation showing recurrence.
- Explain one mechanism-level linkage or trade-off.
- Mention one inflection point if it strengthens the why-now logic.

Return JSON:
{{
  "bottleneck": "...",
  "opportunity": "...",
  "historical_basis": "...",
  "linkage": "...",
  "inflection_point": "...",
  "evidence_titles": ["..."]
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise bottleneck reasoning model. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the bottleneck-opportunity reasoning.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Historical signal map:
{json.dumps(abstraction, ensure_ascii=False, indent=2)}

Rules:
- Reject vague bottlenecks.
- Reject answers that merely rename the absence of a benchmark when the deeper technical bottleneck is evaluability, retention failure diagnosis, retrieval precision, credit assignment, or another mechanism-level issue.
- Reject opportunities that are only adjacent topics rather than a direct opening.
- Keep the linkage causal, not merely topical.
- Preserve evidence anchoring.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict bottleneck-opportunity judge. Output JSON only.',
            prompt=review_prompt,
        )
        obj = final or draft
        return {
            'family_reasoning': obj,
            'answer': self._render_bottleneck(obj),
        }

    def _run_direction(self, *, task: Dict[str, Any], abstraction: Dict[str, Any], head_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Produce a direction-forecast answer from grounded historical signals.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Historical signal map:
{json.dumps(abstraction, ensure_ascii=False, indent=2)}

Prior synthesis:
{json.dumps({'heuristic': head_result.get('heuristic') or {}, 'draft': head_result.get('draft') or {}, 'final': head_result.get('final') or {}}, ensure_ascii=False, indent=2)}

Requirements:
- Make exactly one trajectory call from accelerating, steady, cooling, fragmenting.
- Name one primary immediate next direction and up to two supporting subdirections.
- Use one historical basis and one inflection point to justify the forecast.
- If the evidence shows narrowing into sibling variants, prefer fragmenting.
- Avoid generic trend language.

Return JSON:
{{
  "trajectory_label": "accelerating | steady | cooling | fragmenting",
  "primary_direction": "...",
  "supporting_directions": ["..."],
  "historical_basis": "...",
  "inflection_point": "...",
  "evidence_titles": ["..."]
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise direction-forecast reasoning model. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the direction forecast.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Historical signal map:
{json.dumps(abstraction, ensure_ascii=False, indent=2)}

Rules:
- The direction must be concrete and technically visible in the evidence.
- The trajectory label must match the stated historical pattern.
- The inflection point should explain why the next step follows now rather than being a generic long-term wish.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict direction forecast judge. Output JSON only.',
            prompt=review_prompt,
        )
        obj = final or draft
        return {
            'family_reasoning': obj,
            'answer': self._render_direction(obj),
        }

    def _run_planning(self, *, task: Dict[str, Any], abstraction: Dict[str, Any], head_result: Dict[str, Any]) -> Dict[str, Any]:
        contract = extract_task_contract(task)
        prompt = f"""Produce a prioritized strategic research plan from grounded historical signals.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Task contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Historical signal map:
{json.dumps(abstraction, ensure_ascii=False, indent=2)}

Prior synthesis:
{json.dumps({'heuristic': head_result.get('heuristic') or {}, 'candidate_rows': list(head_result.get('candidate_rows') or [])[:10], 'draft': head_result.get('draft') or {}, 'final': head_result.get('final') or {}}, ensure_ascii=False, indent=2)}

Requirements:
- Return a ranked list with at most {int(contract.get('max_items') or 3)} items.
- Each item must contain a concrete direction, a why-now rationale, and one dependency/trade-off statement.
- Keep directions distinct and technically specific.
- Use historically visible momentum, bottlenecks, or inflection points.
- Prefer executable near-term priorities over broad umbrellas.

Return JSON:
{{
  "ranked_directions": [
    {{
      "rank": 1,
      "direction": "...",
      "why_now": "...",
      "dependency_or_tradeoff": "...",
      "evidence_titles": ["..."]
    }}
  ]
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise strategic planning model. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the ranked plan.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Historical signal map:
{json.dumps(abstraction, ensure_ascii=False, indent=2)}

Rules:
- Keep the list ranked and concise.
- Remove umbrella categories if more specific descendants or future-work directions are available.
- Ensure each item has non-redundant why-now logic.
- Ensure each item includes one dependency or trade-off.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict strategic planning judge. Output JSON only.',
            prompt=review_prompt,
        )
        obj = final or draft
        return {
            'family_reasoning': obj,
            'answer': self._render_planning(obj),
        }

    @staticmethod
    def _render_bottleneck(obj: Dict[str, Any]) -> str:
        bottleneck = str(obj.get('bottleneck') or '').strip()
        opportunity = str(obj.get('opportunity') or '').strip()
        historical = str(obj.get('historical_basis') or '').strip()
        linkage = str(obj.get('linkage') or '').strip()
        inflection = str(obj.get('inflection_point') or '').strip()
        parts: List[str] = []
        if bottleneck:
            parts.append(f"A concrete unresolved bottleneck is {bottleneck}.")
        if historical:
            parts.append(f"Historically, {historical}.")
        if opportunity:
            parts.append(f"The associated opportunity is {opportunity}.")
        if linkage:
            parts.append(f"This linkage matters because {linkage}.")
        if inflection:
            parts.append(f"The key inflection point is {inflection}.")
        return ' '.join(parts).replace('..', '.').strip()

    @staticmethod
    def _render_direction(obj: Dict[str, Any]) -> str:
        label = str(obj.get('trajectory_label') or '').strip().lower() or 'steady'
        primary = str(obj.get('primary_direction') or '').strip()
        supporting = [str(x).strip() for x in (obj.get('supporting_directions') or []) if str(x).strip()][:2]
        historical = str(obj.get('historical_basis') or '').strip()
        inflection = str(obj.get('inflection_point') or '').strip()
        parts = [f"The trajectory is {label}."]
        if primary:
            if supporting:
                parts.append(f"The immediate next direction is {primary}, with nearby movement around {', '.join(supporting)}.")
            else:
                parts.append(f"The immediate next direction is {primary}.")
        if historical:
            parts.append(f"Historically, {historical}.")
        if inflection:
            parts.append(f"The inflection point is {inflection}.")
        return ' '.join(parts).replace('..', '.').strip()

    @staticmethod
    def _render_planning(obj: Dict[str, Any]) -> str:
        rows = list(obj.get('ranked_directions') or [])
        lines: List[str] = []
        for idx, row in enumerate(rows[:4], start=1):
            rank = int((row or {}).get('rank') or idx)
            direction = str((row or {}).get('direction') or '').strip()
            why_now = str((row or {}).get('why_now') or '').strip()
            dep = str((row or {}).get('dependency_or_tradeoff') or '').strip()
            if not direction:
                continue
            line = f"{rank}. {direction} — {why_now}"
            if dep:
                line += f" Dependency / trade-off: {dep}"
            lines.append(line.strip())
        return '\n'.join(lines).strip()


class ResearchArcV6:
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
        self.abstractor = SignalAbstractorV6(answer_client=self.answer_client, critic_client=self.critic_client)
        self.reasoner = MechanismReasonerV6(answer_client=self.answer_client, critic_client=self.critic_client)

    def _backbone(self, domain_id: str) -> EvidenceBackbone:
        if domain_id not in self.backbones:
            self.backbones[domain_id] = EvidenceBackbone(domain_id)
        return self.backbones[domain_id]

    def parse_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        family = str(task.get('family') or '')
        contract = extract_task_contract(task)
        subtype = 'trend_and_outlook'
        if family == 'bottleneck_opportunity_discovery':
            subtype = 'mechanism_bottleneck_opportunity'
        elif family == 'direction_forecasting':
            subtype = 'mechanism_inflection_forecast'
        elif family == 'strategic_research_planning':
            subtype = 'ranked_agenda_with_dependencies'
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
            {'step': 'family_specific_prior_synthesis', 'goal': f"build an initial {policy.get('head')} prior from anchored evidence"},
            {'step': 'historical_signal_abstraction', 'goal': 'compress raw evidence into observations, trade-offs, bottlenecks, inflection points, and agenda axes'},
            {'step': 'mechanism_level_reasoning', 'goal': 'derive the final answer from the abstracted signal structure rather than from raw retrieval alone'},
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
        abstraction = self.abstractor.run(task=task, focus=focus, formatted=formatted, head_result=head_result)
        reasoning = self.reasoner.run(task=task, abstraction=abstraction, head_result=head_result)
        evidence = {
            'papers': list(evidence_bundle.get('paper_evidence') or []),
            'structures': list(evidence_bundle.get('structure_evidence') or []),
            'pageindex': [],
            'fulltext': list(evidence_bundle.get('section_evidence') or []),
        }
        diagnostics = {
            'tool_calls': 5 + sum(1 for key in ['papers', 'structures', 'fulltext'] if evidence.get(key)),
            'reflection_steps': 4,
            'memory_updates': 0,
            'revision_rounds': 3,
        }
        return {
            'task_parse': task_parse,
            'policy': policy,
            'retrieval_plan': retrieval_plan,
            'focus': focus,
            'formatted_evidence': formatted,
            'evidence_bundle': evidence_bundle,
            'head_result': head_result,
            'signal_abstraction': abstraction,
            'mechanism_reasoning': reasoning.get('family_reasoning') or {},
            'retrieval_mode': 'support_packet+paper+structure+section+signal_abstraction+mechanism_reasoning',
            'evidence': evidence,
            'diagnostics': diagnostics,
            'answer': reasoning.get('answer') or head_result.get('answer') or '',
        }
