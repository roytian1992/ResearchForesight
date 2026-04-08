from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.research_arc_v2 import EvidenceBackbone, PolicyRouter, extract_task_contract
from researchworld.research_arc_v3 import DirectionHeadV3, EvidenceFormatter, FocusResolver, PlanningHeadV3
from researchworld.research_arc_v4 import BottleneckHeadV4


ROOT = Path(__file__).resolve().parents[2]


class ComparativeReasonerV5:
    def __init__(self, *, answer_client: OpenAICompatChatClient, critic_client: OpenAICompatChatClient):
        self.answer_client = answer_client
        self.critic_client = critic_client

    def run(
        self,
        *,
        task: Dict[str, Any],
        focus: Dict[str, Any],
        formatted: Dict[str, Any],
        head_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task.get('family') or '')
        if family == 'bottleneck_opportunity_discovery':
            return self._run_bottleneck(task=task, focus=focus, formatted=formatted, head_result=head_result)
        if family == 'direction_forecasting':
            return self._run_direction(task=task, focus=focus, formatted=formatted, head_result=head_result)
        return self._run_planning(task=task, focus=focus, formatted=formatted, head_result=head_result)

    def _run_bottleneck(self, *, task: Dict[str, Any], focus: Dict[str, Any], formatted: Dict[str, Any], head_result: Dict[str, Any]) -> Dict[str, Any]:
        seed_candidates = list(head_result.get('candidates') or [])[:6]
        if not seed_candidates:
            seed_candidates = [{
                'bottleneck': (head_result.get('final') or {}).get('bottleneck') or (head_result.get('draft') or {}).get('bottleneck') or '',
                'opportunity': (head_result.get('final') or {}).get('opportunity') or (head_result.get('draft') or {}).get('opportunity') or '',
                'support_rationale': (head_result.get('final') or {}).get('linkage') or (head_result.get('draft') or {}).get('linkage') or '',
                'support_titles': (head_result.get('final') or {}).get('evidence_titles') or (head_result.get('draft') or {}).get('evidence_titles') or [],
            }]
        prompt = f"""Generate 3 competing bottleneck-opportunity candidates for the task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted.get('focus_summary') or {}, ensure_ascii=False, indent=2)}

Seed bottleneck-opportunity pairs:
{json.dumps(seed_candidates, ensure_ascii=False, indent=2)}

Structure evidence:
{json.dumps((formatted.get('structure_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps((formatted.get('section_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Requirements:
- Produce exactly 3 distinct candidates.
- Each candidate must contain one bottleneck, one downstream opportunity, and one short causal linkage.
- Prefer bottlenecks that recur across multiple papers or are supported by explicit limitation/problem statements.
- Prefer opportunities that open because the bottleneck is addressed, not generic future work.
- Keep the answer compact and benchmark-style.

Return JSON:
{{
  "candidates": [
    {{
      "label": "C1",
      "bottleneck": "...",
      "opportunity": "...",
      "linkage": "...",
      "evidence_titles": ["..."],
      "final_answer": "..."
    }}
  ]
}}"""
        draft = complete_json_object(
            self.answer_client,
            [
                {'role': 'system', 'content': 'You are a precise research hypothesis generator. Output JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            response_format={'type': 'json_object'},
            max_tokens=1400,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
        candidates = list(draft.get('candidates') or [])[:3]
        judge_prompt = f"""Select the strongest bottleneck-opportunity candidate.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Selection criteria:
- bottleneck specificity
- historical grounding in the provided evidence
- causal strength of bottleneck -> opportunity linkage
- non-generic insight value
- task-family fit

Return JSON:
{{
  "selected_label": "C1",
  "reason": "...",
  "rejected": [{{"label": "C2", "reason": "..."}}],
  "final_answer": "..."
}}"""
        final = complete_json_object(
            self.critic_client,
            [
                {'role': 'system', 'content': 'You are a strict research decision judge. Output JSON only.'},
                {'role': 'user', 'content': judge_prompt},
            ],
            response_format={'type': 'json_object'},
            max_tokens=1200,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
        return {
            'candidates': candidates,
            'selection': final,
            'selected_answer': str(final.get('final_answer') or '').strip(),
        }

    def _run_direction(self, *, task: Dict[str, Any], focus: Dict[str, Any], formatted: Dict[str, Any], head_result: Dict[str, Any]) -> Dict[str, Any]:
        heuristic = head_result.get('heuristic') or {}
        seed = {
            'trajectory_label': (head_result.get('final') or {}).get('trajectory_label') or (head_result.get('draft') or {}).get('trajectory_label') or heuristic.get('suggested_label') or '',
            'subdirections': (head_result.get('final') or {}).get('subdirections') or (head_result.get('draft') or {}).get('subdirections') or [],
            'venue_or_evaluation_shift': (head_result.get('final') or {}).get('venue_or_evaluation_shift') or (head_result.get('draft') or {}).get('venue_or_evaluation_shift') or '',
        }
        prompt = f"""Generate 3 competing trajectory forecasts for the task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted.get('focus_summary') or {}, ensure_ascii=False, indent=2)}

Seed forecast:
{json.dumps(seed, ensure_ascii=False, indent=2)}

Subdirection candidates:
{json.dumps((formatted.get('subdirection_candidates') or [])[:18], ensure_ascii=False, indent=2)}

Paper evidence:
{json.dumps((formatted.get('paper_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Requirements:
- Produce exactly 3 distinct candidates.
- Each candidate must include one label from accelerating, fragmenting, steady, cooling.
- Each candidate must name 1-3 concrete subdirections visible in the evidence.
- Each candidate must contain a short rationale linking historical signals to the trajectory call.
- Avoid generic trend language.

Return JSON:
{{
  "candidates": [
    {{
      "label": "C1",
      "trajectory_label": "fragmenting",
      "subdirections": ["..."],
      "rationale": "...",
      "final_answer": "..."
    }}
  ]
}}"""
        draft = complete_json_object(
            self.answer_client,
            [
                {'role': 'system', 'content': 'You are a precise trajectory forecaster. Output JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            response_format={'type': 'json_object'},
            max_tokens=1400,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
        candidates = list(draft.get('candidates') or [])[:3]
        judge_prompt = f"""Select the strongest direction forecast.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Selection criteria:
- trajectory-call discipline
- historical-to-future reasoning quality
- specificity of subdirections
- non-generic insight
- task-family fit

Return JSON:
{{
  "selected_label": "C1",
  "reason": "...",
  "rejected": [{{"label": "C2", "reason": "..."}}],
  "final_answer": "..."
}}"""
        final = complete_json_object(
            self.critic_client,
            [
                {'role': 'system', 'content': 'You are a strict forecast selection judge. Output JSON only.'},
                {'role': 'user', 'content': judge_prompt},
            ],
            response_format={'type': 'json_object'},
            max_tokens=1200,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
        return {
            'candidates': candidates,
            'selection': final,
            'selected_answer': str(final.get('final_answer') or '').strip(),
        }

    def _run_planning(self, *, task: Dict[str, Any], focus: Dict[str, Any], formatted: Dict[str, Any], head_result: Dict[str, Any]) -> Dict[str, Any]:
        candidate_rows = list(head_result.get('candidate_rows') or [])[:12]
        heuristic = head_result.get('heuristic') or {}
        contract = head_result.get('contract') or extract_task_contract(task)
        prompt = f"""Generate 3 competing prioritized research plans for the task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Task contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Focus summary:
{json.dumps(formatted.get('focus_summary') or {}, ensure_ascii=False, indent=2)}

Priority candidates:
{json.dumps(candidate_rows, ensure_ascii=False, indent=2)}

Planning heuristic:
{json.dumps(heuristic, ensure_ascii=False, indent=2)}

Requirements:
- Produce exactly 3 distinct ranked plans.
- Each plan should contain 2 to 4 ranked directions.
- Prefer directions that are concrete, near-term tractable, and distinct from each other.
- Each direction must have a short why-now rationale.
- Avoid umbrella categories when more specific descendants/future-work directions are available.

Return JSON:
{{
  "candidates": [
    {{
      "label": "C1",
      "ranked_directions": [
        {{"rank": 1, "direction": "...", "why_prioritized": "...", "evidence_titles": ["..."]}}
      ],
      "final_answer": "..."
    }}
  ]
}}"""
        draft = complete_json_object(
            self.answer_client,
            [
                {'role': 'system', 'content': 'You are a precise research planning model. Output JSON only.'},
                {'role': 'user', 'content': prompt},
            ],
            response_format={'type': 'json_object'},
            max_tokens=1800,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
        candidates = list(draft.get('candidates') or [])[:3]
        judge_prompt = f"""Select the strongest prioritized research plan.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Candidates:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Selection criteria:
- prioritization discipline
- direction specificity
- evidence-grounded why-now logic
- tractability and distinctiveness
- task-family fit

Return JSON:
{{
  "selected_label": "C1",
  "reason": "...",
  "rejected": [{{"label": "C2", "reason": "..."}}],
  "final_answer": "..."
}}"""
        final = complete_json_object(
            self.critic_client,
            [
                {'role': 'system', 'content': 'You are a strict research planning judge. Output JSON only.'},
                {'role': 'user', 'content': judge_prompt},
            ],
            response_format={'type': 'json_object'},
            max_tokens=1400,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
        return {
            'candidates': candidates,
            'selection': final,
            'selected_answer': str(final.get('final_answer') or '').strip(),
        }


class ResearchArcV5:
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
        self.comparative = ComparativeReasonerV5(answer_client=self.answer_client, critic_client=self.critic_client)

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
            {'step': 'comparative_reasoning', 'goal': 'generate competing hypotheses and explicitly select the strongest one'},
            {'step': 'critic_repair', 'goal': 'repair the answer under benchmark-aligned constraints'},
        ]

    def _fallback_answer(self, *, task: Dict[str, Any], head_result: Dict[str, Any], decision: Optional[Dict[str, Any]] = None) -> str:
        if decision and str(decision.get('selected_answer') or '').strip():
            return str(decision.get('selected_answer') or '').strip()
        family = str(task.get('family') or '')
        draft = head_result.get('draft') or {}
        final = head_result.get('final') or {}
        if family == 'strategic_research_planning':
            rows = final.get('ranked_directions') or draft.get('ranked_directions') or []
            lines: List[str] = []
            for idx, row in enumerate(rows[:4], start=1):
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
        formatted: Dict[str, Any],
        head_result: Dict[str, Any],
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task.get('family') or '')
        current_answer = self._fallback_answer(task=task, head_result=head_result, decision=decision)
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

Current comparative decision:
{json.dumps(decision, ensure_ascii=False, indent=2)}

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
- Preserve the selected substantive position.
- Improve task fulfillment, specificity, and strategic clarity.
- Remove vague filler and unsupported expansion.
- Keep the answer concise.

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
        return {
            'final_answer': final_answer,
            'revision_notes': [str(x).strip() for x in (revised.get('revision_notes') or []) if str(x).strip()],
            'pre_refine_answer': current_answer,
        }

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
        decision = self.comparative.run(task=task, focus=focus, formatted=formatted, head_result=head_result)
        refinement = self._finalize_answer(task=task, formatted=formatted, head_result=head_result, decision=decision)
        evidence = {
            'papers': list(evidence_bundle.get('paper_evidence') or []),
            'structures': list(evidence_bundle.get('structure_evidence') or []),
            'pageindex': [],
            'fulltext': list(evidence_bundle.get('section_evidence') or []),
        }
        diagnostics = {
            'tool_calls': 4 + sum(1 for key in ['papers', 'structures', 'fulltext'] if evidence.get(key)),
            'reflection_steps': 3,
            'memory_updates': 0,
            'revision_rounds': 3,
            'answer_changed_after_revision': str(refinement.get('pre_refine_answer') or '').strip() != str(refinement.get('final_answer') or '').strip(),
        }
        return {
            'task_parse': task_parse,
            'policy': policy,
            'retrieval_plan': retrieval_plan,
            'focus': focus,
            'formatted_evidence': formatted,
            'evidence_bundle': evidence_bundle,
            'head_result': head_result,
            'comparative_reasoning': decision,
            'refinement': refinement,
            'retrieval_mode': 'support_packet+paper+structure+section+comparative_reasoning',
            'evidence': evidence,
            'diagnostics': diagnostics,
            'answer': refinement.get('final_answer') or decision.get('selected_answer') or head_result.get('answer') or '',
        }
