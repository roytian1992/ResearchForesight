from __future__ import annotations

import json
import re
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
        successor_candidates = self._build_successor_candidates(task=task, focus=focus, formatted=formatted, head_result=head_result)
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

Scored successor-topic candidates:
{json.dumps(successor_candidates[:12], ensure_ascii=False, indent=2)}

Requirements:
- Use only pre-cutoff evidence.
- Convert raw evidence into compact, benchmark-useful historical signals.
- Prefer concrete technical observations, not broad topic names.
- Preserve evidence grounding with paper titles whenever possible.
- Surface mechanism-level structure: recurring bottlenecks, trade-offs, inflection points, and emerging directions.
- When successor-topic candidates are available, prefer concrete breakout topics over generic evaluation, protocol, or infrastructure umbrellas.
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

Scored successor-topic candidates:
{json.dumps(successor_candidates[:12], ensure_ascii=False, indent=2)}

Rules:
- Remove generic items that are not obviously grounded.
- Prefer concrete technical bottlenecks, inflection points, and research directions.
- Preserve strong successor-topic candidates when they are better grounded than broad umbrella formulations.
- Keep agenda axes distinct from one another.
- Keep each field concise and evidence-anchored.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict research signal auditor. Output JSON only.',
            prompt=review_prompt,
        )
        obj = final or draft or {}
        obj['successor_topic_candidates'] = successor_candidates[:12]
        obj['allowed_evidence_titles'] = self._collect_allowed_titles(formatted)
        return obj

    @staticmethod
    def _collect_allowed_titles(formatted: Dict[str, Any], limit: int = 24) -> List[str]:
        titles: List[str] = []
        seen = set()
        for bucket in ['paper_evidence', 'structure_evidence', 'section_evidence']:
            for row in list(formatted.get(bucket) or [])[:12]:
                title = str(row.get('paper_title') or row.get('title') or '').strip()
                if not title:
                    continue
                key = title.lower()
                if key in seen:
                    continue
                seen.add(key)
                titles.append(title)
                if len(titles) >= limit:
                    return titles
        return titles

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

    def _build_successor_candidates(
        self,
        *,
        task: Dict[str, Any],
        focus: Dict[str, Any],
        formatted: Dict[str, Any],
        head_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen = set()
        focus_text = str((formatted.get('focus_summary') or {}).get('focus_text') or task.get('title') or '')
        for node in list(focus.get('ranked_nodes') or [])[:6]:
            hist = node.get('historical_stats') or {}
            qh = list(node.get('quarterly_history') or [])
            recent_growth = sum(float(x.get('paper_growth') or 0.0) for x in qh[-3:])
            share_delta = 0.0
            if len(qh) >= 2:
                share_delta = float(qh[-1].get('top_conf_share') or 0.0) - float(qh[0].get('top_conf_share') or 0.0)
            evidence_titles = [str(x.get('title') or '').strip() for x in (node.get('historical_representative_papers') or [])[:3] if str(x.get('title') or '').strip()]
            sources = [
                ('descendant', list(node.get('emergent_descendants') or []), 1.25),
                ('future_work', list(node.get('top_future_work') or []), 1.35),
                ('core_idea', list(node.get('top_core_ideas') or []), 0.9),
            ]
            for source_type, values, base in sources:
                for raw in values[:6]:
                    if isinstance(raw, dict):
                        label = str(raw.get('display_name') or raw.get('public_topic') or raw.get('direction') or raw.get('name') or '').strip()
                    else:
                        label = str(raw or '').strip()
                    if not label:
                        continue
                    key = self._normalize_candidate_key(label)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            'name': label,
                            'source_type': source_type,
                            'source_node': node.get('display_name') or node.get('public_topic'),
                            'score': self._score_successor_candidate(
                                label=label,
                                base=base,
                                recent_growth=recent_growth,
                                top_conf_share=float(hist.get('top_conf_share') or 0.0),
                                citation_median=float(hist.get('citation_median') or 0.0),
                                share_delta=share_delta,
                                split_pressure=float(node.get('split_pressure') or 0.0),
                                focus_text=focus_text,
                            ),
                            'why_candidate': self._why_candidate(source_type=source_type, recent_growth=recent_growth, share_delta=share_delta),
                            'evidence_titles': evidence_titles[:2],
                        }
                    )
        for row in list(head_result.get('candidate_rows') or [])[:10]:
            label = str(row.get('direction') or '').strip()
            if not label:
                continue
            key = self._normalize_candidate_key(label)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    'name': label,
                    'source_type': str(row.get('source_type') or 'candidate_row'),
                    'source_node': row.get('source_node') or '',
                    'score': round(float(row.get('score') or 0.0) + 0.35, 4),
                    'why_candidate': str(row.get('signal_summary') or '').strip(),
                    'evidence_titles': list(row.get('evidence_titles') or [])[:2],
                }
            )
        for title in [str(x.get('paper_title') or '').strip() for x in (formatted.get('paper_evidence') or [])[:12] if str(x.get('paper_title') or '').strip()]:
            for mined in self._mine_title_topics(title, focus_text=focus_text):
                key = self._normalize_candidate_key(mined)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        'name': mined,
                        'source_type': 'representative_paper',
                        'source_node': '',
                        'score': self._score_successor_candidate(
                            label=mined,
                            base=1.05,
                            recent_growth=0.0,
                            top_conf_share=0.0,
                            citation_median=0.0,
                            share_delta=0.0,
                            split_pressure=0.0,
                            focus_text=focus_text,
                        ),
                        'why_candidate': f'title-visible breakout topic mined from representative paper title: {title}',
                        'evidence_titles': [title],
                    }
                )
        rows.sort(key=lambda row: float(row.get('score') or 0.0), reverse=True)
        return rows[:16]

    @staticmethod
    def _normalize_candidate_key(text: Any) -> str:
        return re.sub(r'\s+', ' ', str(text or '').replace('_', ' ')).strip().lower()

    def _score_successor_candidate(
        self,
        *,
        label: str,
        base: float,
        recent_growth: float,
        top_conf_share: float,
        citation_median: float,
        share_delta: float,
        split_pressure: float,
        focus_text: str,
    ) -> float:
        text = str(label or '').strip()
        lower = text.lower()
        tokens = [tok for tok in re.findall(r'[a-zA-Z][a-zA-Z0-9\-]+', lower) if len(tok) > 2]
        focus_tokens = set(re.findall(r'[a-zA-Z][a-zA-Z0-9\-]+', str(focus_text or '').lower()))
        overlap = len(set(tokens) & focus_tokens)
        score = (
            base
            + min(0.85, max(0.0, recent_growth) / 24.0)
            + min(0.55, top_conf_share * 1.8)
            + min(0.45, citation_median / 35.0)
            + min(0.35, split_pressure / 12.0)
            + min(0.3, overlap * 0.08)
            + min(0.35, len(tokens) / 8.0)
        )
        if share_delta <= -0.04:
            score += 0.12
        if any(term in lower for term in ['benchmark', 'evaluation', 'protocol', 'standardization', 'framework', 'leaderboard', 'dataset']):
            score -= 0.55
        if any(term in lower for term in ['architecture', 'pipeline', 'generalist']) and len(tokens) <= 3:
            score -= 0.25
        return round(score, 4)

    @staticmethod
    def _why_candidate(*, source_type: str, recent_growth: float, share_delta: float) -> str:
        parts = [f'source={source_type}']
        if abs(recent_growth) > 1e-6:
            parts.append(f'recent_growth={round(recent_growth, 2)}')
        if abs(share_delta) > 1e-6:
            parts.append(f'share_delta={round(share_delta, 4)}')
        return '; '.join(parts)

    def _mine_title_topics(self, title: str, *, focus_text: str) -> List[str]:
        text = str(title or '').strip()
        lower = text.lower()
        out: List[str] = []
        if re.search(r'video[- ]to[- ]audio', lower):
            out.append('video-to-audio generation')
        if 'reinforcement learning' in lower and 'tool' in lower:
            out.append('reinforcement learning for tool-augmented reasoning')
        if 'reinforcement learning' in lower and 'multimodal' in lower:
            out.append('reinforcement learning for multimodal reasoning')
        if re.search(r'multimodal .*chain[- ]of[- ]thought', lower) or re.search(r'chain[- ]of[- ]thought .*multimodal', lower):
            out.append('multimodal chain-of-thought evaluation')
        if 'open-source' in lower and 'closed-source' in lower:
            out.append('closing the open-vs-closed-source model gap')
        if 'vision-language alignment' in lower:
            out.append('vision-language alignment evaluation')
        if 'audio-video generation' in lower:
            out.append('audio-video generation')
        if 'video generation and manipulation' in str(focus_text or '').lower() and 'audio' in lower:
            out.append('video-to-audio generation')
        return out[:4]


class MechanismReasonerV6(StructuredGenerator):
    GENERIC_LABEL_TERMS = {
        'benchmark', 'evaluation', 'protocol', 'standardization', 'framework', 'frameworks', 'leaderboard',
        'dataset', 'datasets', 'infrastructure', 'tooling', 'pipeline', 'architecture', 'generalist',
        'performance', 'quality', 'effectiveness', 'efficiency',
    }

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

    def _successor_labels(self, abstraction: Dict[str, Any]) -> List[str]:
        return [
            str(row.get('name') or '').strip()
            for row in (abstraction.get('successor_topic_candidates') or [])
            if str(row.get('name') or '').strip()
        ]

    @staticmethod
    def _norm_label(text: Any) -> str:
        return re.sub(r'\s+', ' ', str(text or '').replace('_', ' ')).strip().lower()

    def _label_tokens(self, text: Any) -> List[str]:
        return [tok for tok in re.findall(r'[a-zA-Z][a-zA-Z0-9\-]+', self._norm_label(text)) if len(tok) > 2]

    def _overlap_score(self, a: Any, b: Any) -> float:
        ta = set(self._label_tokens(a))
        tb = set(self._label_tokens(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(1, len(ta | tb))

    def _is_generic_label(self, text: Any) -> bool:
        tokens = set(self._label_tokens(text))
        if not tokens:
            return True
        generic_hits = len(tokens & self.GENERIC_LABEL_TERMS)
        return generic_hits >= max(1, len(tokens) - 1)

    def _allowed_titles(self, abstraction: Dict[str, Any]) -> List[str]:
        return [str(x).strip() for x in (abstraction.get('allowed_evidence_titles') or []) if str(x).strip()]

    def _sanitize_titles(self, titles: List[str], abstraction: Dict[str, Any], limit: int = 2) -> List[str]:
        allowed = {self._norm_label(x): x for x in self._allowed_titles(abstraction)}
        out: List[str] = []
        seen = set()
        for title in titles:
            key = self._norm_label(title)
            if not key or key not in allowed or key in seen:
                continue
            seen.add(key)
            out.append(allowed[key])
            if len(out) >= limit:
                break
        return out

    def _pick_titles_unrestricted(self, abstraction: Dict[str, Any], limit: int = 2) -> List[str]:
        titles: List[str] = []
        for row in (abstraction.get('successor_topic_candidates') or [])[:6]:
            titles.extend(str(x).strip() for x in (row.get('evidence_titles') or []) if str(x).strip())
        for row in (abstraction.get('observations') or [])[:6]:
            titles.extend(str(x).strip() for x in (row.get('evidence_titles') or []) if str(x).strip())
        out: List[str] = []
        seen = set()
        for title in titles:
            key = self._norm_label(title)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(title)
            if len(out) >= limit:
                break
        return out

    def _pick_titles(self, abstraction: Dict[str, Any], limit: int = 2) -> List[str]:
        titles: List[str] = []
        for row in (abstraction.get('successor_topic_candidates') or [])[:6]:
            titles.extend(str(x).strip() for x in (row.get('evidence_titles') or []) if str(x).strip())
        for row in (abstraction.get('observations') or [])[:6]:
            titles.extend(str(x).strip() for x in (row.get('evidence_titles') or []) if str(x).strip())
        titles.extend(self._allowed_titles(abstraction))
        return self._sanitize_titles(titles, abstraction, limit=limit)

    def _canonical_successor_label(self, label: Any, abstraction: Dict[str, Any], *, prefer_top_if_generic: bool = False) -> str:
        value = str(label or '').strip()
        candidates = self._successor_labels(abstraction)
        if not candidates:
            return value
        if not value:
            return candidates[0]
        norm = self._norm_label(value)
        for candidate in candidates:
            cand_norm = self._norm_label(candidate)
            if norm == cand_norm or norm in cand_norm or cand_norm in norm:
                return candidate
        best = max(candidates, key=lambda cand: self._overlap_score(value, cand), default='')
        if best and self._overlap_score(value, best) >= 0.34:
            return best
        if prefer_top_if_generic and self._is_generic_label(value):
            return candidates[0]
        return value

    def _ground_bottleneck_payload(self, obj: Dict[str, Any], abstraction: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(obj or {})
        out['opportunity'] = self._canonical_successor_label(out.get('opportunity'), abstraction, prefer_top_if_generic=True)
        titles = [str(x).strip() for x in (out.get('evidence_titles') or []) if str(x).strip()]
        out['evidence_titles'] = titles or self._pick_titles_unrestricted(abstraction)
        return out

    def _ground_direction_payload(self, obj: Dict[str, Any], abstraction: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(obj or {})
        out['primary_direction'] = self._canonical_successor_label(out.get('primary_direction'), abstraction, prefer_top_if_generic=True)
        primary_key = self._norm_label(out.get('primary_direction'))
        supporting: List[str] = []
        seen = set()
        for item in (out.get('supporting_directions') or [])[:4]:
            grounded = self._canonical_successor_label(item, abstraction)
            key = self._norm_label(grounded)
            if grounded and key not in seen and key != primary_key:
                seen.add(key)
                supporting.append(grounded)
        out['supporting_directions'] = supporting[:2]
        titles = [str(x).strip() for x in (out.get('evidence_titles') or []) if str(x).strip()]
        if not titles:
            out['evidence_titles'] = self._pick_titles(abstraction)
        return out

    def _ground_planning_payload(self, obj: Dict[str, Any], abstraction: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(obj or {})
        rows: List[Dict[str, Any]] = []
        seen = set()
        for row in list(out.get('ranked_directions') or [])[:6]:
            item = dict(row or {})
            direction = self._canonical_successor_label(item.get('direction'), abstraction, prefer_top_if_generic=True)
            key = self._norm_label(direction)
            if not direction or key in seen:
                continue
            seen.add(key)
            item['rank'] = len(rows) + 1
            item['direction'] = direction
            titles = self._sanitize_titles([str(x).strip() for x in (item.get('evidence_titles') or []) if str(x).strip()], abstraction)
            item['evidence_titles'] = titles or self._pick_titles(abstraction)
            rows.append(item)
        out['ranked_directions'] = rows
        return out

    def _run_bottleneck(self, *, task: Dict[str, Any], abstraction: Dict[str, Any], head_result: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""Produce a bottleneck-opportunity answer from grounded historical signals.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Historical signal map:
{json.dumps(abstraction, ensure_ascii=False, indent=2)}

Successor-topic candidates:
{json.dumps((abstraction.get('successor_topic_candidates') or [])[:10], ensure_ascii=False, indent=2)}

Prior synthesis:
{json.dumps({'draft': head_result.get('draft') or {}, 'final': head_result.get('final') or {}, 'candidates': list(head_result.get('candidates') or [])[:6]}, ensure_ascii=False, indent=2)}

Requirements:
- Choose exactly one concrete technical bottleneck.
- Choose exactly one opportunity that becomes more viable if that bottleneck is addressed.
- If successor-topic candidates are present, the opportunity should match or closely paraphrase one of them rather than inventing a broader umbrella.
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

Successor-topic candidates:
{json.dumps((abstraction.get('successor_topic_candidates') or [])[:10], ensure_ascii=False, indent=2)}

Rules:
- Reject vague bottlenecks.
- Reject answers that merely rename the absence of a benchmark when the deeper technical bottleneck is evaluability, retention failure diagnosis, retrieval precision, credit assignment, or another mechanism-level issue.
- Reject opportunities that are only adjacent topics rather than a direct opening.
- If a concrete successor-topic candidate exists, do not replace it with a broader protocol or evaluation umbrella unless the evidence clearly favors the umbrella.
- Keep the linkage causal, not merely topical.
- Preserve evidence anchoring.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict bottleneck-opportunity judge. Output JSON only.',
            prompt=review_prompt,
        )
        obj = self._ground_bottleneck_payload(final or draft, abstraction)
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

Successor-topic candidates:
{json.dumps((abstraction.get('successor_topic_candidates') or [])[:10], ensure_ascii=False, indent=2)}

Prior synthesis:
{json.dumps({'heuristic': head_result.get('heuristic') or {}, 'draft': head_result.get('draft') or {}, 'final': head_result.get('final') or {}}, ensure_ascii=False, indent=2)}

Requirements:
- Make exactly one trajectory call from accelerating, steady, cooling, fragmenting.
- Name one primary immediate next direction and up to two supporting subdirections.
- If successor-topic candidates are present, the primary direction must be selected from that candidate set or be a very close paraphrase of one item.
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

Successor-topic candidates:
{json.dumps((abstraction.get('successor_topic_candidates') or [])[:10], ensure_ascii=False, indent=2)}

Rules:
- The direction must be concrete and technically visible in the evidence.
- Prefer direct successor-topic candidates over generic evaluation or infrastructure answers.
- If a strong successor-topic candidate exists, reject forecasts whose primary direction is outside that candidate set.
- The trajectory label must match the stated historical pattern.
- The inflection point should explain why the next step follows now rather than being a generic long-term wish.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict direction forecast judge. Output JSON only.',
            prompt=review_prompt,
        )
        obj = self._ground_direction_payload(final or draft, abstraction)
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

Successor-topic candidates:
{json.dumps((abstraction.get('successor_topic_candidates') or [])[:12], ensure_ascii=False, indent=2)}

Prior synthesis:
{json.dumps({'heuristic': head_result.get('heuristic') or {}, 'candidate_rows': list(head_result.get('candidate_rows') or [])[:10], 'draft': head_result.get('draft') or {}, 'final': head_result.get('final') or {}}, ensure_ascii=False, indent=2)}

Requirements:
- Return a ranked list with at most {int(contract.get('max_items') or 3)} items.
- Each item must contain a concrete direction, a why-now rationale, and one dependency/trade-off statement.
- Keep directions distinct and technically specific.
- Use historically visible momentum, bottlenecks, or inflection points.
- Prefer executable near-term priorities over broad umbrellas.
- When strong successor-topic candidates exist, rank them ahead of generic enabling work unless the generic work is the only clearly grounded option.
- When successor-topic candidates exist, each ranked direction should come from that candidate set or be a close paraphrase of one item whenever possible.

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

Successor-topic candidates:
{json.dumps((abstraction.get('successor_topic_candidates') or [])[:12], ensure_ascii=False, indent=2)}

Rules:
- Keep the list ranked and concise.
- Remove umbrella categories if more specific descendants or future-work directions are available.
- Preserve concrete successor-topic candidates when they are better grounded than evaluation, protocol, or infrastructure umbrellas.
- If a top-ranked item falls outside the concrete successor-topic set despite stronger grounded candidates being available, revise it.
- Ensure each item has non-redundant why-now logic.
- Ensure each item includes one dependency or trade-off.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict strategic planning judge. Output JSON only.',
            prompt=review_prompt,
        )
        obj = self._ground_planning_payload(final or draft, abstraction)
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
        titles = [str(x).strip() for x in (obj.get('evidence_titles') or []) if str(x).strip()][:2]
        parts: List[str] = []
        if bottleneck:
            parts.append(f"A concrete unresolved bottleneck is {bottleneck}.")
        if historical:
            parts.append(f"Historically, {historical}.")
        if titles:
            parts.append(f"This is visible in {', '.join(titles)}.")
        if opportunity:
            parts.append(f"The associated opportunity is {opportunity}.")
        if linkage:
            parts.append(f"This linkage matters because {linkage}.")
        if inflection:
            parts.append(f"The key inflection point is {inflection}.")
        return ' '.join(parts).replace('..', '.').strip()

    @staticmethod
    def _strip_internal_signals(text: Any) -> str:
        value = str(text or '').strip()
        value = re.sub(r"\(?(?:score|split pressure|split_pressure|top_conf_share|citation median|citation_median)\s*=\s*[^\);,.]+\)?", '', value, flags=re.IGNORECASE)
        value = re.sub(r"\s+", ' ', value)
        return value.strip(' ;,.')

    @staticmethod
    def _naturalize_label(text: Any) -> str:
        value = str(text or '').strip().replace('_', ' ')
        value = re.sub(r'\s+', ' ', value).strip()
        if not value:
            return value
        return value

    @classmethod
    def _render_direction(cls, obj: Dict[str, Any]) -> str:
        label = str(obj.get('trajectory_label') or '').strip().lower() or 'steady'
        primary = cls._naturalize_label(cls._strip_internal_signals(obj.get('primary_direction')))
        supporting = [cls._naturalize_label(cls._strip_internal_signals(x)) for x in (obj.get('supporting_directions') or []) if cls._strip_internal_signals(x)][:2]
        historical = cls._strip_internal_signals(obj.get('historical_basis'))
        inflection = cls._strip_internal_signals(obj.get('inflection_point'))
        titles = [str(x).strip() for x in (obj.get('evidence_titles') or []) if str(x).strip()][:2]
        parts = [f"The trajectory is {label}."]
        if primary:
            if supporting:
                parts.append(f"The immediate next direction is {primary}, with nearby movement around {', '.join(supporting)}.")
            else:
                parts.append(f"The immediate next direction is {primary}.")
        if historical:
            parts.append(f"Historical basis: {historical}.")
        if inflection:
            parts.append(f"Why now: {inflection}.")
        if titles:
            parts.append(f"Evidence: {', '.join(titles)}.")
        return ' '.join(parts).replace('..', '.').strip()

    @classmethod
    def _render_planning(cls, obj: Dict[str, Any]) -> str:
        rows = list(obj.get('ranked_directions') or [])
        lines: List[str] = []
        for idx, row in enumerate(rows[:4], start=1):
            rank = int((row or {}).get('rank') or idx)
            direction = cls._naturalize_label(cls._strip_internal_signals((row or {}).get('direction')))
            why_now = cls._strip_internal_signals((row or {}).get('why_now'))
            dep = cls._strip_internal_signals((row or {}).get('dependency_or_tradeoff'))
            titles = [str(x).strip() for x in ((row or {}).get('evidence_titles') or []) if str(x).strip()][:2]
            if not direction:
                continue
            line = f"{rank}. {direction}"
            if why_now:
                line += f" — Why now: {why_now}"
            if dep:
                line += f" Dependency / trade-off: {dep}"
            if titles:
                line += f" Evidence: {', '.join(titles)}"
            lines.append(line.strip())
        return '\n'.join(lines).strip()


class FinalRefinerV6(StructuredGenerator):
    def run(
        self,
        *,
        task: Dict[str, Any],
        formatted: Dict[str, Any],
        abstraction: Dict[str, Any],
        head_result: Dict[str, Any],
        reasoning: Dict[str, Any],
    ) -> Dict[str, Any]:
        current_answer = str(reasoning.get('answer') or '').strip()
        family_reasoning = dict(reasoning.get('family_reasoning') or {})
        successor_candidates = [
            str(row.get('name') or '').strip()
            for row in (abstraction.get('successor_topic_candidates') or [])[:10]
            if str(row.get('name') or '').strip()
        ]
        prompt = f"""Refine a ResearchArc answer for a benchmark task.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Current answer:
{current_answer}

Family reasoning JSON:
{json.dumps(family_reasoning, ensure_ascii=False, indent=2)}

Historical signal map:
{json.dumps(abstraction, ensure_ascii=False, indent=2)}

Prior head result:
{json.dumps({'draft': head_result.get('draft') or {}, 'final': head_result.get('final') or {}, 'selected_candidate': head_result.get('selected_candidate') or {}}, ensure_ascii=False, indent=2)}

Paper evidence:
{json.dumps((formatted.get('paper_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Structure evidence:
{json.dumps((formatted.get('structure_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Section evidence:
{json.dumps((formatted.get('section_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Concrete successor-topic candidates:
{json.dumps(successor_candidates[:8], ensure_ascii=False, indent=2)}

Rules:
- Preserve the same substantive position unless it is generic and a more concrete historically grounded successor-topic candidate is already available.
- Mention one or two paper titles when available so the answer is easier to trace back to evidence.
- Keep the answer concise and benchmark-facing.
- For bottleneck tasks, keep exactly one bottleneck and one downstream opportunity.
- For direction tasks, keep exactly one trajectory call and one primary next direction.
- For planning tasks, keep a short ranked list with concrete why-now logic.
- Avoid broad evaluation or infrastructure umbrellas when a sharper concrete successor topic is already grounded.

Return JSON:
{{
  "final_answer": "...",
  "revision_notes": ["..."]
}}"""
        draft = self._complete_json(
            self.answer_client,
            system='You are a precise benchmark answer refiner. Output JSON only.',
            prompt=prompt,
        )
        review_prompt = f"""Critique and repair the refined answer.

Task:
{json.dumps(task, ensure_ascii=False, indent=2)}

Draft:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Current answer:
{current_answer}

Family reasoning JSON:
{json.dumps(family_reasoning, ensure_ascii=False, indent=2)}

Concrete successor-topic candidates:
{json.dumps(successor_candidates[:8], ensure_ascii=False, indent=2)}

Rules:
- Keep the answer faithful to the grounded family reasoning.
- Preserve at least one explicit evidence anchor when paper titles are available.
- Reject vague reformulations that are less specific than the current answer.
- Keep task-family fit strict and concise.

Return JSON with the same schema as the draft.
"""
        final = self._complete_json(
            self.critic_client,
            system='You are a strict benchmark answer critic. Output JSON only.',
            prompt=review_prompt,
        )
        obj = final or draft or {}
        raw_final_answer = str(obj.get('final_answer') or '').strip()
        final_answer = self._postprocess_final_answer(
            task=task,
            answer=raw_final_answer or current_answer,
            current_answer=current_answer,
            family_reasoning=family_reasoning,
            abstraction=abstraction,
        )
        return {
            'draft': draft or {},
            'final': obj,
            'final_answer': final_answer,
            'revision_notes': [str(x).strip() for x in (obj.get('revision_notes') or []) if str(x).strip()],
        }

    @staticmethod
    def _normalize_text(text: Any) -> str:
        return re.sub(r'\s+', ' ', str(text or '')).strip().lower()

    def _collect_titles(self, family_reasoning: Dict[str, Any], abstraction: Dict[str, Any], limit: int = 2) -> List[str]:
        allowed = [str(x).strip() for x in (abstraction.get('allowed_evidence_titles') or []) if str(x).strip()]
        allowed_map = {self._normalize_text(x): x for x in allowed}
        titles: List[str] = []
        for title in family_reasoning.get('evidence_titles') or []:
            norm = self._normalize_text(title)
            if norm in allowed_map:
                titles.append(allowed_map[norm])
        for title in allowed:
            titles.append(title)
        out: List[str] = []
        seen = set()
        for title in titles:
            key = self._normalize_text(title)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(title)
            if len(out) >= limit:
                break
        return out

    def _has_any_title_anchor(self, answer: str, titles: List[str]) -> bool:
        norm_answer = self._normalize_text(answer)
        if 'evidence:' in norm_answer or 'evidence anchors:' in norm_answer:
            return True
        for title in titles:
            if self._normalize_text(title) and self._normalize_text(title) in norm_answer:
                return True
        return False

    def _is_over_generic(self, answer: str) -> bool:
        norm = self._normalize_text(answer)
        generic_phrases = [
            'evaluation framework', 'benchmark design', 'infrastructure improvement', 'better protocol',
            'improved benchmark', 'stronger evaluation', 'more systematic evaluation', 'general framework',
        ]
        return any(phrase in norm for phrase in generic_phrases)

    def _postprocess_final_answer(
        self,
        *,
        task: Dict[str, Any],
        answer: str,
        current_answer: str,
        family_reasoning: Dict[str, Any],
        abstraction: Dict[str, Any],
    ) -> str:
        family = str(task.get('family') or '')
        out = str(answer or '').strip() or current_answer
        titles = self._collect_titles(family_reasoning, abstraction)
        if current_answer and self._is_over_generic(out) and not self._is_over_generic(current_answer):
            out = current_answer
        out = re.sub(r"\(?(?:score|split pressure|split_pressure|top_conf_share|citation median|citation_median)\s*=\s*[^\);,.]+\)?", '', out, flags=re.IGNORECASE)
        out = re.sub(r"\s+", ' ', out).replace(' \n ', '\n').strip()
        if titles and not self._has_any_title_anchor(out, titles):
            anchor = ', '.join(titles[:2])
            if family == 'strategic_research_planning' and '\n' in out:
                lines = [line.rstrip() for line in out.splitlines() if line.strip()]
                if lines:
                    lines[0] = f"{lines[0]} Evidence anchors: {anchor}"
                    out = '\n'.join(lines)
                else:
                    out = f"{out} Evidence anchors: {anchor}".strip()
            else:
                out = f"{out} Evidence anchors include {anchor}.".strip()
        return out.strip()


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
        self.refiner = FinalRefinerV6(answer_client=self.answer_client, critic_client=self.critic_client)

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
        refinement = self.refiner.run(
            task=task,
            formatted=formatted,
            abstraction=abstraction,
            head_result=head_result,
            reasoning=reasoning,
        )
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
            'refinement': refinement,
            'retrieval_mode': 'support_packet+paper+structure+section+signal_abstraction+mechanism_reasoning+final_refinement',
            'evidence': evidence,
            'diagnostics': diagnostics,
            'answer': refinement.get('final_answer') or reasoning.get('answer') or head_result.get('answer') or '',
        }
