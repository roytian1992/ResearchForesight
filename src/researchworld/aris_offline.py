from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import OfflineKnowledgeBase, clip_text, dedupe, merge_multi_query_results
from researchworld.research_arc_v2 import extract_task_contract
from researchworld.research_arc_kb import extract_focus_text
from researchworld.retrieval_fusion import build_hybrid_task_queries, merge_retrieval_runs
from researchworld.answer_adapter import apply_shared_final_adapter_to_trace_result


class ARISOffline:
    """Offline adaptation of ARIS-style skill workflow for benchmark answering.

    Core adaptation policy:
    - Preserve ARIS's high-level workflow: literature survey -> candidate generation -> critical review.
    - Replace online / external retrieval with the benchmark's temporally filtered offline knowledge base.
    - Route the final benchmark-facing answer through the shared in-method final renderer.
    """

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

    def run_task(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        route = self.route_task(task)
        task_frame = self.decompose_task(task=task, route=route)
        evidence = self.gather_evidence(task=task, domain_id=domain_id, route=route, task_frame=task_frame)
        survey = self.run_research_lit(task=task, route=route, task_frame=task_frame, evidence=evidence)
        evidence["survey"] = survey
        family_packet = self.build_family_packet(task=task, route=route, task_frame=task_frame, evidence=evidence, survey=survey)
        evidence["family_packet"] = family_packet
        ideation = self.run_idea_creator(task=task, route=route, task_frame=task_frame, evidence=evidence, survey=survey, family_packet=family_packet)
        review = self.run_research_review(task=task, route=route, task_frame=task_frame, evidence=evidence, survey=survey, ideation=ideation, family_packet=family_packet)
        answer = self.render_answer(task=task, route=route, task_frame=task_frame, family_packet=family_packet, final_bundle=review)
        trace_evidence = {
            "papers": evidence.get("paper_evidence") or [],
            "structures": evidence.get("structure_evidence") or [],
            "fulltext": evidence.get("section_evidence") or [],
            "pageindex": evidence.get("pageindex_evidence") or [],
            "successor_topic_candidates": evidence.get("successor_topic_candidates") or [],
        }
        raw_result = {
            "workflow": "ARIS-Offline",
            "retrieval_mode": "offline_kb_hybrid_aris",
            "focus": _focus_text(task),
            "task_route": route,
            "task_frame": task_frame,
            "queries": evidence.get("queries") or [],
            "evidence": trace_evidence,
            "evidence_digest": evidence.get("evidence_digest") or {},
            "signal_digest": evidence.get("signal_digest") or {},
            "family_packet": family_packet,
            "diagnostics": {
                "query_count": len(evidence.get("queries") or []),
                "paper_hits": len(evidence.get("paper_evidence") or []),
                "structure_hits": len(evidence.get("structure_evidence") or []),
                "section_hits": len(evidence.get("section_evidence") or []),
                "pageindex_hits": len(evidence.get("pageindex_evidence") or []),
                "successor_topic_count": len(evidence.get("successor_topic_candidates") or []),
                "signal_cluster_count": len((evidence.get("signal_digest") or {}).get("momentum_topics") or []),
                "packet_preferred_topic_count": len((family_packet.get("preferred_topics") or [])),
                "review_notes_count": len(review.get("review_notes") or []),
            },
            "survey": survey,
            "ideation": ideation,
            "review": review,
            "answer": answer,
        }
        return apply_shared_final_adapter_to_trace_result(
            self.answer_client,
            public_task=task,
            trace_result=raw_result,
        )

    def route_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return _route_task_family(task)

    def decompose_task(self, *, task: Dict[str, Any], route: Dict[str, Any]) -> Dict[str, Any]:
        family = str(task.get("family") or "")
        contract = extract_task_contract(task)
        fallback = {
            "historical_state": _focus_text(task),
            "central_friction": {
                "bottleneck_opportunity_discovery": "Isolate the most central unresolved technical bottleneck supported by recurring limitations.",
                "direction_forecasting": "Identify the strongest momentum signal and the most plausible next-step technical direction.",
                "strategic_research_planning": "Prioritize executable directions with explicit dependencies and trade-offs.",
                "venue_aware_research_positioning": "Map the technical direction to the contribution framing most likely to fit the venue trajectory.",
            }.get(family, "Identify the key unresolved issue from the pre-cutoff literature."),
            "expected_deliverable": _generic_expected_deliverable(family, route),
            "extrapolation_boundary": "One-step forward judgment only; prefer the nearest defensible technical move rather than an ambitious long-range vision.",
            "must_preserve": list(route.get("review_focus") or [])[:4],
        }
        candidate_directions = _task_candidate_directions(task)
        if family in {"strategic_research_planning", "venue_aware_research_positioning"} and candidate_directions:
            fallback["must_preserve"] = dedupe(
                [
                    *fallback["must_preserve"],
                    "stay within the listed candidate directions",
                    "rank all listed candidate directions exactly once",
                ]
            )[:5]
        prompt = f"""You are preparing a benchmark-facing task decomposition for ARIS-Offline.

Task:
{json.dumps(_task_view(task), ensure_ascii=False, indent=2)}

Task-family routing profile:
{json.dumps(route, ensure_ascii=False, indent=2)}

Return strict JSON with keys:
- historical_state
- central_friction
- expected_deliverable
- extrapolation_boundary
- must_preserve

Requirements:
- Use only the task and routing profile.
- Keep every field short and concrete.
- The decomposition should help a literature-survey -> ideation -> review workflow answer the benchmark directly.
- For forecasting tasks, expected_deliverable must describe the output structure only (trajectory label + one concrete successor topic + why-now trigger); do not guess the future topic content inside expected_deliverable.
- For bottleneck tasks, make clear that the opportunity must be a one-step downstream research move.
"""
        try:
            obj = complete_json_object(
                self.answer_client,
                [
                    {"role": "system", "content": "You are a precise task decomposition assistant. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=700,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
                repair_instruction="Return exactly one valid JSON object with keys historical_state, central_friction, expected_deliverable, extrapolation_boundary, must_preserve.",
            )
            raw_must_preserve = obj.get("must_preserve")
            if isinstance(raw_must_preserve, str):
                raw_must_preserve = [raw_must_preserve]
            fallback.update({
                "historical_state": clip_text(obj.get("historical_state") or fallback["historical_state"], 220),
                "central_friction": clip_text(obj.get("central_friction") or fallback["central_friction"], 260),
                "expected_deliverable": clip_text(_normalize_expected_deliverable(family, obj.get("expected_deliverable"), route) or fallback["expected_deliverable"], 220),
                "extrapolation_boundary": clip_text(obj.get("extrapolation_boundary") or fallback["extrapolation_boundary"], 220),
                "must_preserve": [str(x).strip() for x in (raw_must_preserve or fallback["must_preserve"]) if str(x).strip()][:5],
            })
        except Exception:
            pass
        return fallback

    def build_family_packet(
        self,
        *,
        task: Dict[str, Any],
        route: Dict[str, Any],
        task_frame: Dict[str, Any],
        evidence: Dict[str, Any],
        survey: Dict[str, Any],
    ) -> Dict[str, Any]:
        return _build_family_packet(task=task, route=route, task_frame=task_frame, evidence=evidence, survey=survey)

    def build_queries(self, task: Dict[str, Any], domain_id: str, route: Optional[Dict[str, Any]] = None, task_frame: Optional[Dict[str, Any]] = None) -> List[str]:
        focus = _focus_text(task)
        family = str(task.get("family") or "")
        title = str(task.get("title") or "")
        question = str(task.get("question") or "")
        task_frame = task_frame or {}
        queries = [focus, title, str(task_frame.get("historical_state") or ""), str(task_frame.get("central_friction") or "")]
        norm = " ".join(_norm_terms(focus)[:8])
        if norm:
            queries.append(norm)
        short_question = _compress_question(question)
        if short_question:
            queries.append(short_question)
        queries.extend(_task_keyword_hints(task))
        queries.extend(_domain_query_expansions(domain_id=domain_id, focus=focus, question=question, family=family))
        if route:
            queries.extend(route.get("query_hints") or [])
        if task_frame:
            queries.append(str(task_frame.get("expected_deliverable") or ""))
            queries.append(str(task_frame.get("extrapolation_boundary") or ""))
        if family == "bottleneck_opportunity_discovery":
            queries += [
                f"{focus} limitation bottleneck failure",
                f"{focus} unresolved challenge",
                f"{focus} future work opportunity",
                f"{focus} emerging subdirection",
                f"{focus} next method direction",
                f"{focus} specialization application",
            ]
        elif family == "direction_forecasting":
            queries += [
                f"{focus} recent trend emerging direction",
                f"{focus} benchmark evaluation trajectory",
                f"{focus} future work",
                f"{focus} specialization application",
                f"{focus} next method direction",
            ]
        elif family == "venue_aware_research_positioning":
            venue = _extract_target_venue(question)
            queries += [
                f"{focus} empirical trend",
                f"{focus} top venue benchmark",
                f"{focus} evaluation limitation future work",
                f"{focus} emerging specialization",
            ]
            if venue:
                queries.append(f"{focus} {venue} venue")
        else:
            queries += [
                f"{focus} open problem future work",
                f"{focus} method evaluation bottleneck",
                f"{focus} benchmark citations venue",
                f"{focus} emerging direction specialization",
            ]
        return dedupe([q for q in queries if str(q or "").strip()])[:12]

    def gather_evidence(self, *, task: Dict[str, Any], domain_id: str, route: Optional[Dict[str, Any]] = None, task_frame: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        route = route or self.route_task(task)
        agent_queries = self.build_queries(task, domain_id, route, task_frame)
        hybrid_queries = build_hybrid_task_queries(task)
        queries = dedupe([*agent_queries, *hybrid_queries])

        paper_hits = merge_retrieval_runs(
            [
                (
                    "agent",
                    merge_multi_query_results(
                        domain.paper_retriever(cutoff_date=cutoff_date),
                        agent_queries,
                        top_k_per_query=8,
                        limit=10,
                    ),
                ),
                (
                    "hybrid_rag",
                    merge_multi_query_results(
                        domain.paper_retriever(cutoff_date=cutoff_date),
                        hybrid_queries,
                        top_k_per_query=8,
                        limit=10,
                    ),
                ),
            ],
            limit=10,
        )
        paper_rows: List[Dict[str, Any]] = []
        seed_paper_ids: List[str] = []
        for idx, (doc, scores) in enumerate(paper_hits, start=1):
            paper = domain.get_paper(doc.paper_id) or {}
            pub = paper.get("publication") or {}
            seed_paper_ids.append(doc.paper_id)
            paper_rows.append(
                {
                    "evidence_id": f"P{idx}",
                    "paper_id": doc.paper_id,
                    "paper_title": doc.title,
                    "published_date": paper.get("published_date"),
                    "venue": pub.get("venue_name"),
                    "citations": pub.get("citation_count"),
                    "is_top_ai_venue": pub.get("is_top_ai_venue"),
                    "snippet": clip_text(doc.text, 1000),
                    "scores": scores,
                }
            )
        paper_rows.sort(key=_paper_row_priority, reverse=True)
        seed_paper_ids = [str(row.get("paper_id") or "") for row in paper_rows]

        structure_hits = merge_multi_query_results(
            domain.structure_retriever(cutoff_date=cutoff_date, paper_ids=seed_paper_ids[:10] or None),
            queries,
            top_k_per_query=6,
            limit=8,
        )
        structure_rows: List[Dict[str, Any]] = []
        for idx, (doc, scores) in enumerate(structure_hits, start=1):
            structure_rows.append(
                {
                    "evidence_id": f"T{idx}",
                    "paper_id": doc.paper_id,
                    "paper_title": doc.title,
                    "problem_statement": clip_text(doc.meta.get("problem_statement"), 260),
                    "limitations": list(doc.meta.get("limitations") or [])[:4],
                    "future_work": list(doc.meta.get("future_work") or [])[:4],
                    "core_ideas": list(doc.meta.get("core_ideas") or [])[:4],
                    "snippet": clip_text(doc.text, 900),
                    "scores": scores,
                }
            )

        section_hits = merge_multi_query_results(
            domain.section_retriever(cutoff_date=cutoff_date, paper_ids=seed_paper_ids[:10] or None),
            queries,
            top_k_per_query=8,
            limit=8,
        )
        section_rows: List[Dict[str, Any]] = []
        for idx, (doc, scores) in enumerate(section_hits, start=1):
            section_rows.append(
                {
                    "evidence_id": f"S{idx}",
                    "paper_id": doc.paper_id,
                    "paper_title": doc.meta.get("paper_title") or doc.title,
                    "section_title": doc.meta.get("section_title"),
                    "snippet": clip_text(doc.text, 800),
                    "scores": scores,
                }
            )

        page_hits = merge_multi_query_results(
            domain.pageindex_retriever(cutoff_date=cutoff_date, paper_ids=seed_paper_ids[:10] or None),
            queries,
            top_k_per_query=8,
            limit=8,
        )
        page_rows: List[Dict[str, Any]] = []
        for idx, (doc, scores) in enumerate(page_hits, start=1):
            page_rows.append(
                {
                    "evidence_id": f"G{idx}",
                    "paper_id": doc.paper_id,
                    "paper_title": doc.meta.get("paper_title") or doc.title,
                    "section_title": doc.meta.get("section_title"),
                    "snippet": clip_text(doc.text, 800),
                    "scores": scores,
                }
            )

        overview = {
            "paper_count": len(paper_rows),
            "structure_count": len(structure_rows),
            "section_count": len(section_rows),
            "pageindex_count": len(page_rows),
            "top_venue_hits": sum(1 for row in paper_rows if row.get("is_top_ai_venue")),
            "max_citations": max([int(row.get("citations") or 0) for row in paper_rows] or [0]),
            "target_venue": _extract_target_venue(str(task.get("question") or "")),
        }
        historical_likelihood_signals = _historical_likelihood_signals(
            task=task,
            paper_rows=paper_rows,
            structure_rows=structure_rows,
            page_rows=page_rows,
        )
        successor_topic_candidates = [
            {
                "evidence_id": f"X{idx}",
                "topic_label": topic.get("label"),
                "support_score": topic.get("score"),
                "evidence_ids": topic.get("evidence_ids") or [],
                "paper_titles": topic.get("paper_titles") or [],
                "source_fields": topic.get("source_fields") or [],
            }
            for idx, topic in enumerate((historical_likelihood_signals.get("top_topics") or [])[:6], start=1)
        ]
        evidence_digest = _build_evidence_focus_digest(
            task=task,
            route=route,
            paper_rows=paper_rows,
            structure_rows=structure_rows,
            section_rows=section_rows,
            page_rows=page_rows,
            historical_signals=historical_likelihood_signals,
        )
        signal_digest = _build_signal_digest(
            task=task,
            route=route,
            structure_rows=structure_rows,
            section_rows=section_rows,
            page_rows=page_rows,
            historical_signals=historical_likelihood_signals,
        )
        return {
            "task_route": route,
            "queries": queries,
            "paper_evidence": paper_rows,
            "structure_evidence": structure_rows,
            "section_evidence": section_rows,
            "pageindex_evidence": page_rows,
            "overview": overview,
            "historical_likelihood_signals": historical_likelihood_signals,
            "successor_topic_candidates": successor_topic_candidates,
            "evidence_digest": evidence_digest,
            "signal_digest": signal_digest,
        }

    def run_research_lit(self, *, task: Dict[str, Any], route: Dict[str, Any], task_frame: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""You are running an offline adaptation of ARIS skill /research-lit.

Task:
{json.dumps(_task_view(task), ensure_ascii=False, indent=2)}

Task-family routing profile:
{json.dumps(route, ensure_ascii=False, indent=2)}

Task decomposition frame:
{json.dumps(task_frame, ensure_ascii=False, indent=2)}

Retrieved paper evidence:
{json.dumps((evidence.get('paper_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Retrieved structure evidence:
{json.dumps((evidence.get('structure_evidence') or [])[:8], ensure_ascii=False, indent=2)}

Retrieved section/page evidence:
{json.dumps(((evidence.get('section_evidence') or [])[:6] + (evidence.get('pageindex_evidence') or [])[:6]), ensure_ascii=False, indent=2)}

Evidence focus digest:
{json.dumps(evidence.get('evidence_digest') or {}, ensure_ascii=False, indent=2)}

Signal digest:
{json.dumps(evidence.get('signal_digest') or {}, ensure_ascii=False, indent=2)}

Produce a compact literature landscape for downstream reasoning.

Requirements:
- Use only the retrieved offline evidence.
- Separate recurring gaps from active momentum.
- Prefer technical mechanisms and evaluation failures over broad topic names.
- Include venue/citation cues only when they materially affect prioritization.
- Explicitly surface the concrete friction points or successor topics most relevant to the routed task family.
- Organize the landscape around the decomposed central friction and expected deliverable rather than around broad theme names alone.

Return JSON:
{{
  "paper_table": [
    {{"paper_title": "...", "role": "anchor | supporting | counterexample", "key_signal": "...", "evidence_id": "P1"}}
  ],
  "themes": [
    {{"name": "...", "summary": "...", "evidence_ids": ["P1", "T1"]}}
  ],
  "gaps": [
    {{"name": "...", "type": "bottleneck | evaluation_gap | scaling_gap | data_gap", "summary": "...", "evidence_ids": ["T1", "S2"]}}
  ],
  "momentum_signals": [
    {{"name": "...", "summary": "...", "evidence_ids": ["P2", "P4"]}}
  ],
  "venue_signals": [
    {{"signal": "...", "why_it_matters": "...", "evidence_ids": ["P3"]}}
  ],
  "signal_map": {{
    "recurring_bottlenecks": ["..."],
    "momentum_topics": ["..."],
    "dependency_axes": ["..."]
  }},
  "working_notes": ["..."]
}}"""
        survey = complete_json_object(
            self.answer_client,
            [
                {"role": "system", "content": "You are a precise literature-mapping assistant. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1800,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
            repair_instruction="Return exactly one valid JSON object matching the requested schema.",
        )
        family = str(task.get("family") or "")
        if family == "bottleneck_opportunity_discovery":
            survey = _augment_bottleneck_survey(task=task, evidence=evidence, survey=survey)
        elif family == "direction_forecasting":
            survey = _augment_forecast_survey(task=task, evidence=evidence, survey=survey)
        elif family == "strategic_research_planning":
            survey = _augment_strategic_survey(task=task, evidence=evidence, survey=survey)
        elif family == "venue_aware_research_positioning":
            survey = _augment_venue_survey(task=task, evidence=evidence, survey=survey)
        return survey

    def run_idea_creator(self, *, task: Dict[str, Any], route: Dict[str, Any], task_frame: Dict[str, Any], evidence: Dict[str, Any], survey: Dict[str, Any], family_packet: Dict[str, Any]) -> Dict[str, Any]:
        family = str(task.get("family") or "")
        schema = _family_candidate_schema(family)
        candidate_directions = _task_candidate_directions(task)
        prompt = f"""You are running an offline adaptation of ARIS skill /idea-creator.

Task:
{json.dumps(_task_view(task), ensure_ascii=False, indent=2)}

Task-family routing profile:
{json.dumps(route, ensure_ascii=False, indent=2)}

Task decomposition frame:
{json.dumps(task_frame, ensure_ascii=False, indent=2)}

Literature landscape:
{json.dumps(survey, ensure_ascii=False, indent=2)}

Evidence overview:
{json.dumps(evidence.get('overview') or {}, ensure_ascii=False, indent=2)}

Historical likelihood signals for concrete successor topics:
{json.dumps(evidence.get('historical_likelihood_signals') or {}, ensure_ascii=False, indent=2)}

Compact successor-topic candidate bundle:
{json.dumps(evidence.get('successor_topic_candidates') or [], ensure_ascii=False, indent=2)}

Evidence focus digest:
{json.dumps(evidence.get('evidence_digest') or {}, ensure_ascii=False, indent=2)}

Signal digest:
{json.dumps(evidence.get('signal_digest') or {}, ensure_ascii=False, indent=2)}

Family packet:
{json.dumps(family_packet, ensure_ascii=False, indent=2)}

Generate candidate answers for this benchmark task.

Requirements:
- Produce 3 distinct candidates.
- Each candidate must be grounded in the literature landscape, not generic forecasting.
- Keep candidates technically concrete.
- Reuse evidence ids from the survey/evidence when possible.
- Make the three candidates genuinely different rather than paraphrases of the same answer.
- Prefer different reasoning frames across candidates, such as evaluation bottleneck vs optimization bottleneck, capability-building vs benchmark-building, or venue-fit through empirical validation vs venue-fit through methodological novelty, when the evidence supports such separation.
- Each candidate should contain at least one concrete technical noun phrase that could plausibly appear in a methods or limitations section.
- Avoid fallback candidates that merely say "better benchmark", "more data", or "improve evaluation" unless the literature landscape makes that the dominant unresolved bottleneck.
- Prefer candidate directions that could stand alone as concrete topic labels or taxonomy-node labels.
- Prefer successor-topic labels that are close to repeated historical future-work / core-idea signals rather than ad hoc umbrella phrases.
- At least one candidate should align with one of the top historical-likelihood successor topics unless the survey gives a clear reason to reject it.
- Purely meta-level answers such as "more evaluation", "more ablation", or "better benchmarking" are invalid unless they are tightly scoped to a concrete technical theme.
- Each candidate must include one short, explicit successor-topic label that could plausibly be used as a paper/topic tag.
- If a candidate proposes evaluation, benchmarking, or ablation, it must still name the concrete capability, subproblem, or application area that the field is moving toward.
- Candidate 1 should be the most evidence-conservative option: the nearest plausible next step that can be defended directly from repeated historical signals.
- Use the family packet as a planning scaffold: central friction, preferred topics, avoid-as-primary topics, and evidence-anchor hints should shape candidate construction.
- If forecast_guardrails are present in the family packet, at least one candidate should stay close to one of those labels.
- Avoid multi-hop extrapolation. If a candidate requires several unstated assumptions to become true, it is invalid.
- Prefer successor-topic labels that are close to the wording of historical future-work / core-idea signals instead of aggressively generalized umbrella phrases.

Family-specific diversification rules:
{_family_ideation_rules(family)}

Reasoning scaffold for this task family:
{_family_reasoning_scaffold(family)}
"""
        if family in {"strategic_research_planning", "venue_aware_research_positioning"} and candidate_directions:
            prompt += f"""

Hard contract for this explicit ranking task:
- The only allowed ranked directions are: {json.dumps(candidate_directions, ensure_ascii=False)}.
- Every candidate must rank all of those listed directions exactly once.
- Do not invent a third direction, substitute label, umbrella category, or narrower proxy direction.
- The three candidates may differ in justification, evidence selection, and dependency analysis, but not in the set or coverage of allowable direction labels.
- Reuse the listed direction labels verbatim unless a trivial casing/punctuation cleanup is needed.
"""
            if family == "strategic_research_planning":
                trend_transitions = list(family_packet.get("trend_transitions") or [])
                trend_direction_candidates = [str(x).strip() for x in (family_packet.get("trend_direction_candidates") or []) if str(x).strip()]
                if trend_transitions or trend_direction_candidates:
                    prompt += f"""

Strategic trend guidance:
- Use the inferred transitions as soft evidence for why a listed direction should move up or down in priority.
- Prefer justifications that connect each ranked direction to a concrete shift in historical momentum, dependency pressure, or unresolved execution friction.
- If you rank a direction above a better-supported trend candidate, explain the dependency logic explicitly rather than giving a generic "higher impact" claim.
- Trend-derived direction hints: {json.dumps(trend_direction_candidates[:4], ensure_ascii=False)}.
- Strategic trend transitions: {json.dumps(trend_transitions[:3], ensure_ascii=False)}.
"""
            elif family == "venue_aware_research_positioning":
                primary_bucket = str(family_packet.get("primary_venue_bucket") or "").strip()
                compatible_buckets = [str(x).strip() for x in (family_packet.get("compatible_venue_buckets") or []) if str(x).strip()]
                package_expectations = [str(x).strip() for x in (family_packet.get("package_expectations") or []) if str(x).strip()]
                contrastive_not_best_for = [str(x).strip() for x in (family_packet.get("contrastive_not_best_for") or []) if str(x).strip()]
                prompt += f"""

Venue-fit guidance:
- It is acceptable that a direction fits multiple nearby venue families, but you must still distinguish the primary fit from secondary fits.
- If a direction also fits nearby venues, explain why {primary_bucket or 'the target venue family'} should remain the primary framing for this task.
- Compatible venue families to keep in mind: {json.dumps(compatible_buckets[:5], ensure_ascii=False)}.
- Reviewer/package expectations for the primary venue family: {json.dumps(package_expectations[:4], ensure_ascii=False)}.
- If relevant, contrast against weaker-fit venues such as: {json.dumps(contrastive_not_best_for[:3], ensure_ascii=False)}.
- Do not rely on prestige rhetoric; venue fit must be explained through contribution type, evidence style, and reviewer package.
"""
        elif family == "direction_forecasting":
            forecast_guardrails = [str(x).strip() for x in (family_packet.get("forecast_guardrails") or []) if str(x).strip()]
            expected_aliases = [str(x).strip() for x in (family_packet.get("expected_direction_aliases") or []) if str(x).strip()]
            trend_transitions = list(family_packet.get("trend_transitions") or [])
            if forecast_guardrails or expected_aliases:
                prompt += f"""

Forecasting guardrail:
- Prefer the primary next direction to stay close to one of these historically grounded labels: {json.dumps(dedupe(forecast_guardrails + expected_aliases)[:4], ensure_ascii=False)}.
- Candidate 1 must be the most evidence-conservative near-term continuation of those labels unless the literature landscape clearly rejects them.
"""
            if trend_transitions:
                prompt += f"""

Forecast trend transitions:
- The evidence suggests these historical shifts: {json.dumps(trend_transitions[:3], ensure_ascii=False)}.
- At least one candidate should explicitly map a concrete transition from an older regime to a newer regime and turn that shift into a short next-step direction label.
- Prefer successor labels that operationalize the detected shift rather than restating the old regime or quoting a paper title.
"""
        elif family == "bottleneck_opportunity_discovery":
            prompt += """

Bottleneck-task hard contract:
- The bottleneck must describe an upstream technical limitation in the evaluated research area, not a downstream symptom.
- The opportunity must be the immediate research move enabled by solving that bottleneck.
- Do not turn the opportunity into an intervention module, pipeline, or artifact unless the opportunity is explicitly framed as the concrete metric/capability/study that artifact enables.
"""
            unlock_chains = list(family_packet.get("unlock_chains") or [])
            preferred_bottlenecks = [str(x).strip() for x in (family_packet.get("preferred_bottlenecks") or []) if str(x).strip()]
            preferred_unlocks = [str(x).strip() for x in (family_packet.get("preferred_unlocks") or []) if str(x).strip()]
            canonical_bottleneck_hints = [str(x).strip() for x in (family_packet.get("canonical_bottleneck_hints") or []) if str(x).strip()]
            canonical_opportunity_hints = [str(x).strip() for x in (family_packet.get("canonical_opportunity_hints") or []) if str(x).strip()]
            if unlock_chains or preferred_bottlenecks or preferred_unlocks or canonical_bottleneck_hints or canonical_opportunity_hints:
                prompt += f"""

Bottleneck unlock-chain guidance:
- Prefer bottlenecks that look upstream and persistent rather than symptoms like weak performance numbers.
- Prefer opportunities that are the nearest concrete unlock after the bottleneck is addressed, not a distant research program.
- Avoid artifact-like opportunities unless you explicitly explain the capability or study they enable.
- Candidate 1 should stay close to the canonical bottleneck/opportunity families when those hints are available, instead of inventing a narrower proxy label.
- Preferred bottlenecks: {json.dumps(preferred_bottlenecks[:4], ensure_ascii=False)}.
- Preferred immediate unlocks: {json.dumps(preferred_unlocks[:4], ensure_ascii=False)}.
- Canonical bottleneck hints: {json.dumps(canonical_bottleneck_hints[:4], ensure_ascii=False)}.
- Canonical opportunity hints: {json.dumps(canonical_opportunity_hints[:4], ensure_ascii=False)}.
- Candidate unlock chains: {json.dumps(unlock_chains[:3], ensure_ascii=False)}.
"""
        prompt += f"""

Return JSON:
{json.dumps(schema, ensure_ascii=False, indent=2)}"""
        return complete_json_object(
            self.answer_client,
            [
                {"role": "system", "content": "You are a precise research ideation assistant. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=2200,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
            repair_instruction="Return exactly one valid JSON object with a candidates array following the requested schema.",
        )

    def run_research_review(
        self,
        *,
        task: Dict[str, Any],
        route: Dict[str, Any],
        task_frame: Dict[str, Any],
        evidence: Dict[str, Any],
        survey: Dict[str, Any],
        ideation: Dict[str, Any],
        family_packet: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task.get("family") or "")
        target_venue = _extract_target_venue(str(task.get("question") or ""))
        candidate_directions = _task_candidate_directions(task)
        prompt = f"""You are running an offline adaptation of ARIS skill /research-review.

Task:
{json.dumps(_task_view(task), ensure_ascii=False, indent=2)}

Task-family routing profile:
{json.dumps(route, ensure_ascii=False, indent=2)}

Task decomposition frame:
{json.dumps(task_frame, ensure_ascii=False, indent=2)}

Target venue cue: {target_venue or 'none'}

Literature landscape:
{json.dumps(survey, ensure_ascii=False, indent=2)}

Historical likelihood signals for concrete successor topics:
{json.dumps(evidence.get('historical_likelihood_signals') or {}, ensure_ascii=False, indent=2)}

Compact successor-topic candidate bundle:
{json.dumps(evidence.get('successor_topic_candidates') or [], ensure_ascii=False, indent=2)}

Evidence focus digest:
{json.dumps(evidence.get('evidence_digest') or {}, ensure_ascii=False, indent=2)}

Signal digest:
{json.dumps(evidence.get('signal_digest') or {}, ensure_ascii=False, indent=2)}

Family packet:
{json.dumps(family_packet, ensure_ascii=False, indent=2)}

Candidate answers:
{json.dumps(ideation, ensure_ascii=False, indent=2)}

Evidence snippets:
{json.dumps(_compact_evidence(evidence), ensure_ascii=False, indent=2)}

Choose and repair the strongest candidate.

Review rules:
- Prefer concrete mechanism-level reasoning.
- Penalize generic consultant language.
- Preserve temporal discipline: only pre-cutoff evidence may be used.
- Prefer the most defensible next-step answer, not the most ambitious answer.
- Penalize extrapolations that require multiple unobserved intermediate advances.
- Prefer claims that can be justified by repeated signals or by at least two concrete evidence items.
- For venue-aware tasks, prefer directions whose empirical framing and evaluation style plausibly fit the named venue family.
- For planning tasks, require a ranked and executable plan.
- For direction forecasting, require a clear trajectory call.
- For bottleneck tasks, require a direct causal link from bottleneck to opportunity.
- Reject candidate sets where the selected answer is only a lightly reworded literature summary.
- Prefer the candidate that best isolates the central research friction and converts it into a concrete next move.
- Use the family packet to check whether the selected answer matches the expected deliverable, preferred topics, and anti-generic constraints.
- If forecast_guardrails are present, use them as soft priors rather than hard constraints.
- When two candidates are plausible, prefer the one with sharper evidence linkage and less umbrella phrasing.
- Prefer concrete successor topics over methodological housekeeping.
- Prefer candidates whose successor-topic label is supported by repeated historical future-work/core-idea signals.
- Break ties in favor of the candidate whose successor topic looks most likely to materialize as a standalone topic cluster within the next cycle.
- If a candidate only proposes evaluation, benchmarking, or ablation, reject it unless the answer also names the concrete capability, subproblem, or specialization that the field is actually moving toward.
- Break ties in favor of the candidate whose core direction can be expressed as a compact noun phrase rather than a broad work program.
- Penalize answers whose first concrete noun phrase is a generic research activity instead of a technical direction.
- Penalize broad umbrella phrases when a narrower historically grounded label is available.
- If the selected candidate contains both a bottleneck and a successor direction, the successor direction must be immediate and concrete rather than a distant end-state.
- For bottleneck tasks, reject candidates whose "opportunity" is merely an already-existing benchmark, dataset, framework, protocol, or pipeline named in the historical evidence; rewrite it as the concrete study, capability, or subproblem that artifact enables.
- For bottleneck tasks, the opportunity should answer "what becomes newly possible if the bottleneck is removed?" rather than "what artifact was introduced in the same paper?"

Family-specific review rules:
{_family_review_rules(family, target_venue)}

Family-specific answer contract:
{_family_answer_contract(family)}
"""
        if family in {"strategic_research_planning", "venue_aware_research_positioning"} and candidate_directions:
            prompt += f"""

Explicit-ranking hard contract:
- The answer is restricted to these candidate directions: {json.dumps(candidate_directions, ensure_ascii=False)}.
- Reject any candidate that renames, replaces, nests, or sidesteps those listed directions.
- The selected final bundle must rank all listed directions exactly once and must not add any extra direction.
"""
            if family == "strategic_research_planning":
                trend_transitions = list(family_packet.get("trend_transitions") or [])
                trend_direction_candidates = [str(x).strip() for x in (family_packet.get("trend_direction_candidates") or []) if str(x).strip()]
                if trend_transitions or trend_direction_candidates:
                    prompt += f"""

Strategic review check:
- Use these trend-derived signals as a soft audit trail for the ranked agenda: {json.dumps(trend_transitions[:3], ensure_ascii=False)}.
- Prefer agendas whose rank order is consistent with the strongest supported transition or dependency pattern among the listed directions.
- Penalize reviews that rank a direction highly without tying it to either a concrete historical signal, an execution dependency, or a kill criterion.
- Candidate-direction trend hints: {json.dumps(trend_direction_candidates[:4], ensure_ascii=False)}.
"""
            elif family == "venue_aware_research_positioning":
                primary_bucket = str(family_packet.get("primary_venue_bucket") or "").strip()
                compatible_buckets = [str(x).strip() for x in (family_packet.get("compatible_venue_buckets") or []) if str(x).strip()]
                package_expectations = [str(x).strip() for x in (family_packet.get("package_expectations") or []) if str(x).strip()]
                prompt += f"""

Venue review check:
- A direction may plausibly fit multiple nearby venue families, but the review must still identify the primary fit and explain why it is stronger than secondary fits.
- Treat these as acceptable nearby families rather than automatic errors: {json.dumps(compatible_buckets[:5], ensure_ascii=False)}.
- Still penalize answers that never distinguish the primary venue family or never explain reviewer/package expectations.
- Primary venue family package expectations: {json.dumps(package_expectations[:4], ensure_ascii=False)}.
- If the selected answer mentions secondary venue families, check that they are framed as secondary rather than equal with no discrimination.
"""
        elif family == "direction_forecasting":
            forecast_guardrails = [str(x).strip() for x in (family_packet.get("forecast_guardrails") or []) if str(x).strip()]
            expected_aliases = [str(x).strip() for x in (family_packet.get("expected_direction_aliases") or []) if str(x).strip()]
            trend_transitions = list(family_packet.get("trend_transitions") or [])
            if forecast_guardrails or expected_aliases:
                prompt += f"""

Forecast-selection hard contract:
- Prefer a primary direction that stays close to these historically grounded labels: {json.dumps(dedupe(forecast_guardrails + expected_aliases)[:4], ensure_ascii=False)}.
- If you deviate, review_notes must explicitly explain why the guarded labels are less defensible than the selected one.
"""
            if trend_transitions:
                prompt += f"""

Forecast trend check:
- Use these inferred trend transitions as a soft audit trail: {json.dumps(trend_transitions[:3], ensure_ascii=False)}.
- Prefer candidates whose successor direction is a concrete operationalization of one of these transitions.
- Penalize answers that ignore the detected transition and instead reuse a historical paper title or broad parent topic.
"""
        elif family == "bottleneck_opportunity_discovery":
            prompt += """

Bottleneck-review hard contract:
- Reject any answer whose opportunity is just a restatement of the bottleneck.
- Reject opportunities that drift from evaluation/measurement into unrelated intervention systems unless the causal bridge is explicit and immediate.
"""
            unlock_chains = list(family_packet.get("unlock_chains") or [])
            canonical_bottleneck_hints = [str(x).strip() for x in (family_packet.get("canonical_bottleneck_hints") or []) if str(x).strip()]
            canonical_opportunity_hints = [str(x).strip() for x in (family_packet.get("canonical_opportunity_hints") or []) if str(x).strip()]
            if unlock_chains:
                prompt += f"""

Bottleneck review check:
- Use these inferred unlock chains as a soft audit trail: {json.dumps(unlock_chains[:3], ensure_ascii=False)}.
- Prefer bottlenecks that look like root causes or structural blockers, not downstream symptoms.
- Prefer opportunities that match the immediate unlock in the chain rather than an artifact noun or a long-range end-state.
- Penalize candidate pairs that require multiple unstated intermediate steps between solving the bottleneck and reaching the opportunity.
"""
            if canonical_bottleneck_hints or canonical_opportunity_hints:
                prompt += f"""

Canonical bottleneck check:
- Prefer final wording that stays close to these benchmark-compatible bottleneck families: {json.dumps(canonical_bottleneck_hints[:4], ensure_ascii=False)}.
- Prefer final wording that stays close to these benchmark-compatible opportunity families: {json.dumps(canonical_opportunity_hints[:4], ensure_ascii=False)}.
- If a candidate uses a much narrower method name, rewrite it upward into the broader future-facing opportunity family unless evidence clearly forbids that abstraction.
"""
        prompt += f"""

Return JSON:
{json.dumps(_family_final_schema(family), ensure_ascii=False, indent=2)}"""
        review = complete_json_object(
            self.critic_client,
            [
                {"role": "system", "content": "You are a strict research reviewer. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=2200,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
            repair_instruction="Return exactly one valid JSON object matching the requested final schema.",
        )
        return _rerank_final_bundle(task=task, evidence=evidence, ideation=ideation, review=review)

    def render_answer(self, *, task: Dict[str, Any], route: Dict[str, Any], task_frame: Dict[str, Any], family_packet: Dict[str, Any], final_bundle: Dict[str, Any]) -> str:
        return _render_final_bundle_answer(
            task=task,
            route=route,
            task_frame=task_frame,
            family_packet=family_packet,
            final_bundle=final_bundle,
        )

def _focus_text(task: Dict[str, Any]) -> str:
    title = str(task.get("title") or "").strip()
    question = str(task.get("question") or "").strip()
    text = extract_focus_text(task)
    bad_markers = ["published before", "based on scholarly literature", "based on literature", "identify one", "which one or two"]
    title_patterns = [
        r"^Forecasting\s+the\s+Next[- ]Step\s+in\s+",
        r"^Forecasting\s+Next[- ]Step\s+Direction\s+in\s+",
        r"^Identifying\s+(?:an?\s+)?(?:unresolved\s+)?bottlenecks?\s+in\s+",
        r"^Identifying\s+(?:an?\s+)?(?:key\s+)?bottleneck\s+in\s+",
        r"^Identifying\s+bottlenecks?\s+and\s+future\s+opportunities\s+in\s+",
        r"^Identifying\s+(?:research\s+)?directions?\s+in\s+",
        r"^Prioritization of Research Directions in\s+",
        r"^Prioritizing Research Directions in\s+",
        r"^Forecasting a High-Impact Research Direction in\s+",
        r"^Forecasting Research Directions in\s+",
        r"^Identifying(?:\s+an?|\s+one|\s+a|\s+key)?\s+(?:Unresolved\s+)?(?:Bottlenecks?|Directions?|Research Directions?|Research Trajectory|Strategic Research Planning|Strategic Research Prioritization|Prioritizing Research Directions?|Forecasting(?: the Trajectory of)?|Bottleneck and Opportunity Discovery|Bottleneck and Opportunity Discovery in|Bottleneck and Opportunity Discovery for)\s+(?:in|for)?\s*",
        r"^Strategic Research (?:Planning|Agenda|Prioritization)\s+(?:for|in)\s+",
        r"^Prioritizing Research Directions?\s+(?:for|in)\s+",
        r"^Forecasting(?: the Trajectory of)?\s+",
        r"^Bottleneck and Opportunity Discovery\s+(?:in|for)\s+",
    ]
    candidate = title
    for pattern in title_patterns:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"\s+for\s+[A-Z][A-Za-z0-9\-/]+(?:-style)?\s+venues?$", "", candidate).strip()
    candidate = re.sub(r"\s+for\s+[A-Z][A-Za-z0-9\-/]+-Targeted\s+Submissions$", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"\s+for\s+[^,]+Venues$", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"\s+from\s+Pre-Cutoff\s+Literature$", "", candidate, flags=re.IGNORECASE).strip()
    candidate = re.sub(r"\s+with\s+Venue\s+Alignment$", "", candidate, flags=re.IGNORECASE).strip()
    if candidate and len(candidate.split()) <= 18 and not any(marker in candidate.lower() for marker in bad_markers):
        return candidate
    if text and len(text.split()) <= 18 and not any(marker in text.lower() for marker in bad_markers):
        return text
    m = re.search(r"(?:in|for)\s+([^\.]+?)(?:\.|$)", title, flags=re.IGNORECASE)
    if m:
        compact = m.group(1).strip()
        compact = re.sub(r"\s+for\s+[A-Z][A-Za-z0-9\-/]+(?:-style)?\s+venues?$", "", compact).strip()
        if compact:
            return compact
    return title or question

def _task_view(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "family": task.get("family"),
        "domain": task.get("domain"),
        "time_cutoff": task.get("time_cutoff"),
        "title": task.get("title"),
        "question": task.get("question"),
        "deliverable_spec": task.get("deliverable_spec") or {},
    }


def _compress_question(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""
    if len(text.split()) <= 20:
        return text
    m = re.search(
        r"(?:identify|which one or two|which one|what)\s+(.+?)(?:\.|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        compact = m.group(1).strip()
        compact = re.sub(r"\s+", " ", compact)
        return compact[:180]
    return ""


def _task_keyword_hints(task: Dict[str, Any]) -> List[str]:
    hints: List[str] = []
    text = " ".join(
        str(task.get(key) or "")
        for key in ["title", "question"]
    ).lower()
    keyword_map = {
        "rlhf": ["RLHF", "reinforcement learning from human feedback", "preference optimization"],
        "dpo": ["DPO", "direct preference optimization", "preference optimization"],
        "tool": ["tool use", "tool-augmented reasoning"],
        "memory": ["long-term memory", "memory retrieval"],
        "video": ["video generation", "long video generation"],
        "diffusion": ["diffusion transformers", "diffusion modeling"],
        "retrieval": ["retrieval augmentation", "retrieval-augmented generation"],
        "rag": ["retrieval-augmented generation", "retrieval structuring"],
        "embodied": ["embodied agents", "autonomous exploration"],
        "world model": ["world models", "environment modeling"],
        "multimodal": ["multimodal reasoning", "vision-language"],
    }
    for trigger, expansions in keyword_map.items():
        if trigger in text:
            hints.extend(expansions)
    return dedupe(hints)[:4]


def _route_task_family(task: Dict[str, Any]) -> Dict[str, Any]:
    family = str(task.get("family") or "")
    question = str(task.get("question") or "")
    target_venue = _extract_target_venue(question)
    base: Dict[str, Any] = {
        "family": family,
        "focus": _focus_text(task),
        "target_venue": target_venue,
        "query_hints": [],
        "required_outputs": [],
        "review_focus": [],
        "answer_opening": "",
    }
    if family == "bottleneck_opportunity_discovery":
        base.update(
            {
                "family_head": "bottleneck",
                "query_hints": [
                    f"{base['focus']} recurring limitation",
                    f"{base['focus']} failure mode",
                    f"{base['focus']} unresolved mechanism",
                    f"{base['focus']} immediate downstream opportunity",
                ],
                "required_outputs": ["one singular bottleneck", "one immediate downstream opportunity", "explicit causal linkage"],
                "review_focus": ["mechanism-level bottleneck", "non-generic opportunity", "one-step downstream move"],
                "answer_opening": "The key bottleneck is X, and the concrete downstream opportunity is Y.",
            }
        )
    elif family == "direction_forecasting":
        base.update(
            {
                "family_head": "forecasting",
                "query_hints": [
                    f"{base['focus']} momentum signal",
                    f"{base['focus']} successor topic",
                    f"{base['focus']} inflection point",
                    f"{base['focus']} likely next direction",
                ],
                "required_outputs": ["trajectory label", "one primary next direction", "why-now trigger"],
                "review_focus": ["near-term continuation", "specific successor topic", "historical plausibility"],
                "answer_opening": "The trajectory is X, and the most likely next direction is Y.",
            }
        )
    elif family == "venue_aware_research_positioning":
        base.update(
            {
                "family_head": "venue_positioning",
                "query_hints": [
                    f"{base['focus']} empirical framing",
                    f"{base['focus']} evaluation style",
                    f"{base['focus']} contribution type",
                    f"{base['focus']} venue fit",
                ],
                "required_outputs": ["direction label", "technical rationale", "venue-fit rationale"],
                "review_focus": ["submission-ready framing", "venue-compatible evidence style", "distinct ranked directions"],
                "answer_opening": "Direction: X.",
            }
        )
    else:
        base.update(
            {
                "family_head": "planning",
                "query_hints": [
                    f"{base['focus']} roadmap",
                    f"{base['focus']} dependency bottleneck",
                    f"{base['focus']} ranked priorities",
                    f"{base['focus']} tradeoff",
                ],
                "required_outputs": ["ranked agenda", "why-now rationale", "dependency or trade-off"],
                "review_focus": ["executability", "technical dependency clarity", "priority separation"],
                "answer_opening": "Direction: X.",
            }
        )
    return base


def _build_evidence_focus_digest(
    *,
    task: Dict[str, Any],
    route: Dict[str, Any],
    paper_rows: List[Dict[str, Any]],
    structure_rows: List[Dict[str, Any]],
    section_rows: List[Dict[str, Any]],
    page_rows: List[Dict[str, Any]],
    historical_signals: Dict[str, Any],
) -> Dict[str, Any]:
    recurring_limitations: List[Dict[str, Any]] = []
    for row in structure_rows[:6]:
        for item in list(row.get("limitations") or [])[:2]:
            label = _clean_topic_label(item)
            if not label:
                continue
            recurring_limitations.append({"label": label, "evidence_id": row.get("evidence_id"), "paper_title": row.get("paper_title")})
    future_work_signals: List[Dict[str, Any]] = []
    for row in structure_rows[:6]:
        for item in list(row.get("future_work") or [])[:2]:
            label = _clean_topic_label(item)
            if not label:
                continue
            future_work_signals.append({"label": label, "evidence_id": row.get("evidence_id"), "paper_title": row.get("paper_title")})
    concrete_sections: List[Dict[str, Any]] = []
    for row in (section_rows[:4] + page_rows[:4]):
        concrete_sections.append(
            {
                "evidence_id": row.get("evidence_id"),
                "paper_title": row.get("paper_title"),
                "section_title": row.get("section_title"),
                "snippet": clip_text(row.get("snippet"), 220),
            }
        )
    return {
        "family_head": route.get("family_head"),
        "focus": route.get("focus"),
        "required_outputs": route.get("required_outputs") or [],
        "top_successor_topics": (historical_signals.get("top_topics") or [])[:4],
        "recurring_limitations": recurring_limitations[:6],
        "future_work_signals": future_work_signals[:6],
        "top_papers": [{"evidence_id": row.get("evidence_id"), "paper_title": row.get("paper_title"), "venue": row.get("venue"), "citations": row.get("citations")} for row in paper_rows[:5]],
        "concrete_sections": concrete_sections[:6],
        "task_focus_hint": _focus_text(task),
    }


def _build_signal_digest(
    *,
    task: Dict[str, Any],
    route: Dict[str, Any],
    structure_rows: List[Dict[str, Any]],
    section_rows: List[Dict[str, Any]],
    page_rows: List[Dict[str, Any]],
    historical_signals: Dict[str, Any],
) -> Dict[str, Any]:
    recurring_bottlenecks: List[str] = []
    dependency_axes: List[str] = []
    evidence_anchor_hints: List[str] = []
    for row in structure_rows[:6]:
        recurring_bottlenecks.extend(_clean_topic_label(x) for x in (row.get("limitations") or [])[:2])
        dependency_axes.extend(_clean_topic_label(x) for x in (row.get("core_ideas") or [])[:2])
    for row in (section_rows[:4] + page_rows[:4]):
        label = _clean_topic_label(row.get("section_title") or row.get("snippet") or "")
        if label:
            evidence_anchor_hints.append(label)
    momentum_topics = [str(topic.get("label") or "") for topic in (historical_signals.get("top_topics") or [])[:5] if str(topic.get("label") or "").strip()]
    return {
        "family_head": route.get("family_head"),
        "task_focus": _focus_text(task),
        "recurring_bottlenecks": dedupe([x for x in recurring_bottlenecks if x])[:6],
        "momentum_topics": dedupe(momentum_topics)[:5],
        "dependency_axes": dedupe([x for x in dependency_axes if x])[:6],
        "evidence_anchor_hints": dedupe([x for x in evidence_anchor_hints if x])[:6],
    }


def _family_packet_list(items: List[Any], *, field: str | None = None, limit: int = 6) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for item in items:
        value = item.get(field) if isinstance(item, dict) and field else item
        label = _clean_topic_label(value)
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= limit:
            break
    return out


def _build_family_packet(
    *,
    task: Dict[str, Any],
    route: Dict[str, Any],
    task_frame: Dict[str, Any],
    evidence: Dict[str, Any],
    survey: Dict[str, Any],
) -> Dict[str, Any]:
    family = str(task.get("family") or "")
    candidate_directions = _task_candidate_directions(task)
    evidence_digest = evidence.get("evidence_digest") or {}
    signal_digest = evidence.get("signal_digest") or {}
    historical_signals = evidence.get("historical_likelihood_signals") or {}
    trajectory_estimate = _estimate_forecast_trajectory(evidence=evidence, survey=survey, task=task) if family == "direction_forecasting" else {}
    survey_gaps = list(survey.get("gaps") or [])
    survey_momentum = list(survey.get("momentum_signals") or [])
    survey_trends = list(survey.get("trend_transitions") or [])
    top_topics = [str(topic.get("label") or "") for topic in (historical_signals.get("top_topics") or []) if str(topic.get("label") or "").strip()]
    recurring_limitations = [str(x.get("label") or "") for x in (evidence_digest.get("recurring_limitations") or []) if str(x.get("label") or "").strip()]
    future_work = [str(x.get("label") or "") for x in (evidence_digest.get("future_work_signals") or []) if str(x.get("label") or "").strip()]
    anchors = _family_packet_list((evidence_digest.get("top_papers") or []), field="paper_title", limit=5)
    packet = {
        "family": family,
        "focus": route.get("focus") or _focus_text(task),
        "historical_state": task_frame.get("historical_state"),
        "central_friction": task_frame.get("central_friction"),
        "expected_deliverable": task_frame.get("expected_deliverable"),
        "extrapolation_boundary": task_frame.get("extrapolation_boundary"),
        "evidence_anchor_hints": dedupe(_family_packet_list(signal_digest.get("evidence_anchor_hints") or [], limit=6) + anchors)[:6],
        "preferred_topics": [],
        "secondary_topics": [],
        "avoid_as_primary": [],
    }
    if family in {"strategic_research_planning", "venue_aware_research_positioning"} and candidate_directions:
        packet["explicit_direction_candidates"] = candidate_directions
    if family == "bottleneck_opportunity_discovery":
        unlock_chains = list(survey.get("unlock_chains") or [])
        preferred_bottlenecks = [
            str(item.get("bottleneck_label") or "").strip()
            for item in unlock_chains
            if str(item.get("bottleneck_label") or "").strip()
        ]
        preferred_unlocks = [
            str(item.get("immediate_unlock") or "").strip()
            for item in unlock_chains
            if str(item.get("immediate_unlock") or "").strip()
        ]
        canonical_bottleneck_hints = dedupe(
            [
                _bottleneck_domain_sensitive_bottleneck(x, task=task, evidence=evidence, survey=survey)
                for x in (preferred_bottlenecks + recurring_limitations + _family_packet_list(signal_digest.get("recurring_bottlenecks") or [], limit=4))
                if str(x).strip()
            ]
        )[:4]
        canonical_opportunity_hints = dedupe(
            [
                _bottleneck_domain_sensitive_opportunity(x, task=task, evidence=evidence, survey=survey)
                for x in (preferred_unlocks + future_work + top_topics)
                if str(x).strip()
            ]
        )[:4]
        packet["preferred_topics"] = dedupe(
            _family_packet_list([gap for gap in survey_gaps if str(gap.get("type") or "").lower() == "bottleneck"], field="name", limit=4)
            + recurring_limitations
            + _family_packet_list(signal_digest.get("recurring_bottlenecks") or [], limit=4)
            + preferred_bottlenecks[:4]
            + canonical_bottleneck_hints
        )[:6]
        packet["secondary_topics"] = dedupe(top_topics + _family_packet_list(survey_momentum, field="name", limit=4) + preferred_unlocks[:4] + canonical_opportunity_hints)[:6]
        packet["avoid_as_primary"] = ["benchmark", "evaluation framework", "protocol only", "dataset only", "framework only"]
        if unlock_chains:
            packet["unlock_chains"] = unlock_chains[:4]
        if preferred_bottlenecks:
            packet["preferred_bottlenecks"] = dedupe(preferred_bottlenecks)[:4]
        if preferred_unlocks:
            packet["preferred_unlocks"] = dedupe(preferred_unlocks)[:4]
        if canonical_bottleneck_hints:
            packet["canonical_bottleneck_hints"] = canonical_bottleneck_hints
        if canonical_opportunity_hints:
            packet["canonical_opportunity_hints"] = canonical_opportunity_hints
        packet["artifact_like_opportunities_to_avoid"] = ["benchmark", "dataset", "framework", "protocol", "pipeline"]
    elif family == "direction_forecasting":
        primary_expected_direction = _extract_primary_expected_direction(task_frame.get("expected_deliverable"))
        trend_direction_candidates = [
            str(item.get("canonical_direction") or "").strip()
            for item in survey_trends
            if str(item.get("canonical_direction") or "").strip()
        ]
        focus_conditioned = [
            str(topic.get("label") or "").strip()
            for topic in (historical_signals.get("focus_conditioned_topics") or [])
            if str(topic.get("label") or "").strip()
        ]
        expected_direction_aliases = dedupe(
            ([primary_expected_direction] if primary_expected_direction else [])
            + trend_direction_candidates[:3]
            + focus_conditioned[:3]
            + [
                label
                for label in _family_packet_list(survey_momentum, field="name", limit=4)
                if not primary_expected_direction or _topic_overlap_score(label, primary_expected_direction) >= 0.18
            ]
        )[:4]
        packet["preferred_topics"] = dedupe(
            top_topics
            + _family_packet_list(survey_momentum, field="name", limit=5)
            + _family_packet_list(signal_digest.get("momentum_topics") or [], limit=5)
            + future_work
        )[:8]
        packet["secondary_topics"] = dedupe(
            _family_packet_list(survey.get("themes") or [], field="name", limit=4)
            + _family_packet_list(signal_digest.get("dependency_axes") or [], limit=4)
        )[:6]
        packet["avoid_as_primary"] = [
            "benchmark", "evaluation", "evaluation framework", "benchmark suite",
            "protocol", "ablation program", "diagnostic benchmark",
        ]
        packet["why_now_triggers"] = dedupe(_family_packet_list(survey_momentum, field="summary", limit=4) + future_work)[:6]
        if trajectory_estimate:
            packet["trajectory_estimate"] = trajectory_estimate
        if primary_expected_direction:
            packet["primary_expected_direction"] = primary_expected_direction
        if expected_direction_aliases:
            packet["expected_direction_aliases"] = expected_direction_aliases
        if focus_conditioned:
            packet["forecast_guardrails"] = focus_conditioned[:3]
        if survey_trends:
            packet["trend_transitions"] = survey_trends[:4]
        if trend_direction_candidates:
            packet["trend_direction_candidates"] = dedupe(trend_direction_candidates)[:4]
    elif family == "strategic_research_planning":
        trend_direction_candidates = [
            str(item.get("canonical_direction") or "").strip()
            for item in survey_trends
            if str(item.get("canonical_direction") or "").strip()
        ]
        packet["preferred_topics"] = dedupe(
            _family_packet_list(signal_digest.get("dependency_axes") or [], limit=6)
            + top_topics
            + future_work
            + trend_direction_candidates[:4]
        )[:8]
        packet["secondary_topics"] = dedupe(recurring_limitations + _family_packet_list(survey_gaps, field="name", limit=4))[:6]
        packet["avoid_as_primary"] = ["benchmark only", "evaluation only", "generic infrastructure"]
        if survey_trends:
            packet["trend_transitions"] = survey_trends[: max(2, min(4, len(survey_trends)))]
        if trend_direction_candidates:
            packet["trend_direction_candidates"] = dedupe(trend_direction_candidates)[: max(2, min(4, len(trend_direction_candidates)))]
    elif family == "venue_aware_research_positioning":
        profile = survey.get("venue_fit_profile") or _derive_venue_fit_profile(task=task, evidence=evidence, survey=survey)
        primary_bucket = str(profile.get("primary_venue_bucket") or "").strip()
        compatible_buckets = [str(x).strip() for x in (profile.get("compatible_venue_buckets") or []) if str(x).strip()]
        secondary_buckets = [str(x).strip() for x in (profile.get("secondary_venue_buckets") or []) if str(x).strip()]
        packet["preferred_topics"] = dedupe(
            top_topics
            + _family_packet_list(survey.get("themes") or [], field="name", limit=4)
            + _family_packet_list(survey_momentum, field="name", limit=4)
        )[:6]
        packet["secondary_topics"] = dedupe(future_work + recurring_limitations)[:6]
        packet["avoid_as_primary"] = ["prestige rhetoric", "high impact only", "broad venue only"]
        if primary_bucket:
            packet["primary_venue_bucket"] = primary_bucket
        if compatible_buckets:
            packet["compatible_venue_buckets"] = compatible_buckets[:5]
        if secondary_buckets:
            packet["secondary_venue_buckets"] = secondary_buckets[:4]
        if profile.get("package_expectations"):
            packet["package_expectations"] = list(profile.get("package_expectations") or [])[:4]
        if profile.get("shared_fit_rationale"):
            packet["shared_fit_rationale"] = str(profile.get("shared_fit_rationale") or "")
        if profile.get("contrastive_not_best_for"):
            packet["contrastive_not_best_for"] = list(profile.get("contrastive_not_best_for") or [])[:3]
    else:
        packet["preferred_topics"] = dedupe(top_topics + _family_packet_list(survey.get("themes") or [], field="name", limit=4))[:6]
        packet["secondary_topics"] = dedupe(future_work + recurring_limitations)[:6]
        packet["avoid_as_primary"] = ["generic evaluation"]
    return packet


def _family_reasoning_scaffold(family: str) -> str:
    if family == "bottleneck_opportunity_discovery":
        return "1. isolate the central historical limitation; 2. explain why it stays unresolved; 3. name the immediate downstream research move unlocked if it is addressed. The opportunity should be a next-step capability, study, or subproblem, not merely an already-existing benchmark, dataset, framework, or pipeline mentioned in the literature."
    if family == "direction_forecasting":
        return "1. identify the dominant pre-cutoff momentum signals; 2. call the trajectory; 3. commit to the most plausible immediate successor topic."
    if family == "venue_aware_research_positioning":
        return "1. rank all listed candidate directions; 2. explain the technical rationale for each rank; 3. explain why each direction fits the target venue style and what reviewer-facing paper package it implies."
    return "1. rank the listed candidate directions; 2. name the first milestone for each; 3. expose one dependency or defer rationale plus one risk or kill criterion per direction."


def _family_answer_contract(family: str) -> str:
    if family == "bottleneck_opportunity_discovery":
        return "The final answer must contain exactly one bottleneck and one concrete downstream opportunity, with an explicit causal bridge between them. The opportunity must describe the next research move enabled by solving the bottleneck, not just restate an already-existing historical artifact."
    if family == "direction_forecasting":
        return "The final answer must contain one trajectory call and one primary next direction, with a clear why-now explanation."
    if family == "venue_aware_research_positioning":
        return "The final answer must rank all listed candidate directions exactly once, separate technical rationale from venue-fit rationale, and make reviewer/package expectations explicit."
    return "The final answer must provide a ranked, executable agenda using only the listed candidate directions, and each ranked item must expose a first milestone, a dependency or defer rationale, and a risk or kill criterion."


def _domain_query_expansions(*, domain_id: str, focus: str, question: str, family: str) -> List[str]:
    text = f"{focus} {question}".lower()
    expansions: List[str] = []
    forecast_mode = family == "direction_forecasting"
    if domain_id == "llm_finetuning_post_training":
        if not forecast_mode:
            expansions.extend([
                f"{focus} preference optimization",
                f"{focus} reinforcement learning based fine tuning",
                f"{focus} alignment data scaling",
            ])
        if any(token in text for token in ["preference", "alignment", "reward", "dpo", "rlhf"]):
            expansions.extend([
                "RLHF preference optimization",
                "DPO reinforcement learning fine tuning",
                "preference alignment reward modeling",
            ])
        elif any(token in text for token in ["domain", "translation", "medical", "clinical", "legal", "finance", "time series", "specialized"]):
            expansions.extend([
                f"{focus} domain adaptation",
                f"{focus} parameter efficient fine tuning",
                "domain specific parameter efficient fine tuning",
            ])
        elif forecast_mode:
            expansions.extend([
                f"{focus} parameter efficient fine tuning",
                f"{focus} domain adaptation",
            ])
        if forecast_mode and ("domain-specific" in text or "domain specific" in text):
            expansions.extend([
                "audio domain fine tuning",
                "audio visual multimodal fine tuning",
                "multimodal domain adaptation",
                "speech foundation model fine tuning",
                "audio language model adaptation",
            ])
        if forecast_mode and ("vision language" in text or "multimodal" in text):
            expansions.extend([
                "biomedical vision language fine tuning",
                "clinical vision language fine tuning",
                "biological vision language fine tuning",
            ])
    elif domain_id == "llm_agent":
        if not forecast_mode:
            expansions.extend([
                f"{focus} tool use planning grounding",
                f"{focus} long horizon reasoning",
                f"{focus} memory retrieval coordination",
            ])
        elif any(token in text for token in ["tool", "browser", "web", "api", "grounding"]):
            expansions.extend([
                f"{focus} tool use planning grounding",
                f"{focus} tool coordination",
            ])
        elif any(token in text for token in ["embodied", "navigation", "interaction", "exploration", "collaboration", "multi-agent"]):
            expansions.extend([
                f"{focus} embodied collaboration",
                f"{focus} multi agent coordination",
                "collaborative embodied navigation",
            ])
        else:
            expansions.append(f"{focus} agent coordination")
        if any(token in text for token in ["tool", "ground", "environment", "browser", "web"]):
            expansions.extend([
                "reinforcement learning for tool augmented reasoning",
                "grounded agent planning tool use",
                "world model grounded agent reasoning",
            ])
        if any(token in text for token in ["memory", "long-term", "dialogue"]):
            expansions.extend([
                "long term memory retrieval agents",
                "memory organization retrieval accuracy",
            ])
    elif domain_id == "rag_and_retrieval_structuring":
        expansions.extend([
            f"{focus} retrieval structuring indexing",
            f"{focus} query routing evidence aggregation",
        ])
        if any(token in text for token in ["long context", "context window", "repository", "code completion"]):
            expansions.append(f"{focus} long context retrieval")
        if any(token in text for token in ["graph", "index", "structure", "hierarch"]):
            expansions.extend([
                "graph retrieval indexing knowledge organization",
                "hierarchical retrieval evidence structuring",
            ])
    elif domain_id == "visual_generative_modeling_and_diffusion":
        expansions.extend([
            f"{focus} multimodal generation",
        ])
        if any(token in text for token in ["control", "instruction", "edit", "guidance"]):
            expansions.append(f"{focus} controllable generation")
        if any(token in text for token in ["transformer", "scaling", "distillation", "compression"]):
            expansions.append(f"{focus} diffusion transformer scaling")
        if any(token in text for token in ["video", "temporal", "motion"]):
            expansions.extend([
                "long video generation controllable diffusion",
                "audio video generation synchronization",
                "video diffusion temporal consistency",
            ])
        if any(token in text for token in ["edit", "control", "instruction"]):
            expansions.extend([
                "instruction guided image editing diffusion",
                "controllable diffusion generation",
            ])
    if family == "venue_aware_research_positioning":
        expansions.append(f"{focus} strong empirical validation")
    elif family == "strategic_research_planning":
        expansions.append(f"{focus} dependency bottleneck roadmap")
    return dedupe(expansions)[:5]


def _norm_terms(text: Any) -> List[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "based", "published", "before", "after", "research",
        "direction", "directions", "identify", "prioritized", "prioritize", "technical", "concrete", "scholarly",
        "literature", "large", "language", "model", "models", "agents", "llm", "llms", "task", "tasks",
    }
    raw = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", str(text or "").lower())
    return [token for token in raw if token not in stop]


def _extract_target_venue(question: str) -> str:
    text = str(question or "")
    patterns = [
        r"venues?\s+(?:such as|similar to)\s+([A-Z][A-Za-z0-9\-/]+)",
        r"for\s+([A-Z][A-Za-z0-9\-/]+)-style venues",
        r"for\s+top-tier\s+([A-Z][A-Za-z0-9\-/]+)\s+venues",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip().rstrip(".,")
    for venue in ["ICLR", "NeurIPS", "ICML", "ACL", "EMNLP", "AAAI", "CVPR", "ECCV", "ICCV"]:
        if venue in text:
            return venue
    return ""


def _task_candidate_directions(task: Dict[str, Any]) -> List[str]:
    contract_candidates = [
        str(x).strip()
        for x in (extract_task_contract(task).get("candidate_directions") or [])
        if str(x).strip()
    ]
    if contract_candidates:
        return dedupe(contract_candidates)
    return _extract_question_candidate_directions(str(task.get("question") or ""))


def _extract_question_candidate_directions(question: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    lowered = text.lower()
    if not text or "(" not in text or ")" not in text:
        return []
    gating_markers = [
        "rank these",
        "rank the following",
        "candidate research directions",
        "candidate directions",
        "listed options",
        "do not add new directions",
        "do not introduce new candidate directions",
        "limit your assessment to these options only",
    ]
    if not any(marker in lowered for marker in gating_markers):
        return []
    matches = list(re.finditer(r"\((\d+|one|two|three|four|five|six)\)\s*", text, flags=re.IGNORECASE))
    if not matches:
        return []
    stop_pattern = re.compile(
        r"(?:;\s*provide\b|\.?\s*provide\b|;\s*do not\b|\.?\s*do not\b|;\s*limit\b|\.?\s*limit\b|;\s*rank only\b|\.?\s*rank only\b)",
        flags=re.IGNORECASE,
    )
    out: List[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end].strip(" ;,.-")
        stop = stop_pattern.search(chunk)
        if stop:
            chunk = chunk[:stop.start()]
        chunk = re.sub(r"^(?:and|or)\s+", "", chunk, flags=re.IGNORECASE).strip(" ;,.-")
        label = _clean_topic_label(chunk)
        if label:
            out.append(label)
    return dedupe(out[:6])


def _task_domain_id(task: Dict[str, Any]) -> str:
    raw = str(task.get("domain_id") or task.get("domain") or "").strip().lower()
    if not raw:
        return ""
    norm = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    aliases = {
        "llm_agent": "llm_agent",
        "large_language_model_agents_and_agentic_systems": "llm_agent",
        "llm_finetuning_post_training": "llm_finetuning_post_training",
        "llm_fine_tuning_post_training": "llm_finetuning_post_training",
        "rag_and_retrieval_structuring": "rag_and_retrieval_structuring",
        "retrieval_augmented_generation": "rag_and_retrieval_structuring",
        "visual_generative_modeling_and_diffusion": "visual_generative_modeling_and_diffusion",
        "visual_generative_modeling_and_diffusion_based_image_video_and_3d_generation": "visual_generative_modeling_and_diffusion",
    }
    if norm in aliases:
        return aliases[norm]
    for key, value in aliases.items():
        if key in norm:
            return value
    return norm


def _align_contract_direction(label: Any, candidates: List[str]) -> str:
    value = _clean_topic_label(label)
    if not value or not candidates:
        return value
    norm = value.lower()
    for candidate in candidates:
        cand = _clean_topic_label(candidate)
        cand_norm = cand.lower()
        if norm == cand_norm or norm in cand_norm or cand_norm in norm:
            return cand
    best = max(candidates, key=lambda cand: _topic_overlap_score(value, cand), default="")
    if best and _topic_overlap_score(value, best) >= 0.34:
        return _clean_topic_label(best)
    return value


def _compact_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key in ["paper_evidence", "structure_evidence", "section_evidence", "pageindex_evidence"]:
        rows = []
        for row in (evidence.get(key) or [])[:6]:
            rows.append(
                {
                    "evidence_id": row.get("evidence_id"),
                    "paper_title": row.get("paper_title"),
                    "section_title": row.get("section_title"),
                    "limitations": row.get("limitations"),
                    "future_work": row.get("future_work"),
                    "venue": row.get("venue"),
                    "citations": row.get("citations"),
                    "snippet": row.get("snippet"),
                }
            )
        out[key] = rows
    out["overview"] = evidence.get("overview") or {}
    return out


def _family_ideation_rules(family: str) -> str:
    if family == "bottleneck_opportunity_discovery":
        return (
            "- Candidate 1 should target the most central recurring bottleneck and pair it with the nearest concrete downstream opportunity.\n"
            "- Candidate 2 should target a different bottleneck class if supported, such as optimization, retrieval, memory organization, or evaluation protocol.\n"
            "- Candidate 3 should only use a benchmark/evaluation bottleneck if the evidence clearly shows it is upstream of the field's progress.\n"
            "- Every opportunity must name a concrete follow-on theme, capability, or application area rather than a generic call for more study.\n"
            "- The opportunity should be one-step downstream from the bottleneck, not a large multi-stage research program.\n"
            "- Invalid opportunity examples: 'better evaluation', 'systematic benchmarking', 'more robust analysis'."
        )
    if family == "direction_forecasting":
        return (
            "- Candidate 1 should be the strongest near-term continuation of the dominant historical momentum signal.\n"
            "- Candidate directions must not be near-synonyms.\n"
            "- Vary the trajectory interpretation across candidates when the evidence allows it, such as accelerating vs fragmenting.\n"
            "- Do not use an already-published paper title or named method as the forecast label; abstract it into the concrete follow-on topic.\n"
            "- At least one candidate should convert future-work statements into a concrete next-step direction.\n"
            "- At least two candidates should name a concrete successor topic, specialization, application area, or method family rather than a generic benchmark or ablation program.\n"
            "- The preferred direction should look like the field's next immediate specialization, not a distant end-state.\n"
            "- Invalid primary directions include phrases like 'better evaluation', 'more benchmarks', or 'systematic ablations' unless paired with a concrete topic label."
        )
    if family == "venue_aware_research_positioning":
        return (
            "- Every candidate must rank every listed candidate direction exactly once.\n"
            "- Candidate 1 should emphasize strong empirical validation and benchmark fit.\n"
            "- Candidate 2 should emphasize methodological sharpness or architectural novelty.\n"
            "- Candidate 3 should emphasize a different venue-fit angle, such as dataset/evaluation design or systems usefulness, only if the evidence supports it.\n"
            "- Each ranked item must include a concrete reviewer-facing package such as benchmark, system, ablation bundle, or theory-plus-evaluation package."
        )
    return (
        "- Every candidate must rank every listed candidate direction exactly once.\n"
        "- Candidate 1 should emphasize near-term executable priorities.\n"
        "- Candidate 2 should emphasize infrastructure or evaluation dependencies if they are historically blocking progress.\n"
        "- Candidate 3 should emphasize a different technical axis rather than rephrasing the same plan.\n"
        "- Each ranked item must include a first milestone, one dependency or defer rationale, and one risk or kill criterion.\n"
        "- Ranked items should be concrete technical directions, not only requests for more benchmarking or ablation."
    )


def _family_review_rules(family: str, target_venue: str) -> str:
    if family == "bottleneck_opportunity_discovery":
        return (
            "- The selected bottleneck must be singular and technically specific.\n"
            "- The opportunity must become more viable because the bottleneck is addressed, not merely be adjacent to it.\n"
            "- Prefer upstream bottlenecks over superficial symptoms.\n"
            "- Prefer opportunities that can be reached in one clear step from the bottleneck rather than distant downstream visions."
        )
    if family == "direction_forecasting":
        return (
            "- The chosen trajectory label must match the historical pattern in the evidence.\n"
            "- The selected direction should be immediate enough to plausibly materialize next, not a vague long-term wish.\n"
            "- Prefer the answer with the clearest why-now trigger.\n"
            "- Prefer a concrete successor topic or specialization over a generic evaluation agenda.\n"
            "- Prefer the most defensible near-term continuation of current momentum over a larger but weaker speculative jump.\n"
            "- Reject candidates whose named next direction is effectively an existing paper title or already-published artifact in the evidence."
        )
    if family == "venue_aware_research_positioning":
        venue_text = target_venue or "the target venue"
        return (
            f"- The answer must state why the ranked directions fit {venue_text} rather than only why they are interesting in general.\n"
            "- Rank all listed candidate directions exactly once.\n"
            "- Prefer directions whose evaluation style, methodological framing, and expected contribution type match the venue cue.\n"
            "- The ranked directions must be distinct and not nested umbrella/subcase pairs.\n"
            "- Prefer concrete, submission-ready problem framings over generic calls for stronger evaluation.\n"
            "- Require an explicit reviewer/package framing such as benchmark, systems evidence, ablation suite, or theory-plus-evaluation."
        )
    return (
        "- The ranked agenda must be executable and clearly prioritized.\n"
        "- Rank all listed candidate directions exactly once.\n"
        "- Reject plans whose items are too broad to guide a concrete research cycle.\n"
        "- Prefer plans that expose first milestones, dependencies, defer rationale, and kill criteria rather than only listing topics.\n"
        "- Prefer agendas built around concrete follow-on directions that could plausibly become standalone paper clusters."
    )


def _family_candidate_schema(family: str) -> Dict[str, Any]:
    if family == "bottleneck_opportunity_discovery":
        return {
            "candidates": [
                {
                    "candidate_type": "central_bottleneck | alternative_bottleneck | evaluation_bottleneck",
                    "successor_topic_label": "...",
                    "bottleneck": "...",
                    "opportunity": "...",
                    "historical_basis": "...",
                    "linkage": "...",
                    "evidence_ids": ["P1", "T1"],
                }
            ]
        }
    if family == "direction_forecasting":
        return {
            "candidates": [
                {
                    "candidate_type": "momentum | specialization | fragmentation",
                    "successor_topic_label": "...",
                    "trajectory_label": "accelerating | steady | cooling | fragmenting",
                    "primary_direction": "...",
                    "supporting_directions": ["..."],
                    "historical_basis": "...",
                    "why_next": "...",
                    "evidence_ids": ["P1", "T1"],
                }
            ]
        }
    if family == "venue_aware_research_positioning":
        return {
            "candidates": [
                {
                    "candidate_type": "empirical | methodological | systems",
                    "agenda": [
                        {
                            "rank": 1,
                            "direction_label": "...",
                            "direction": "...",
                            "technical_rationale": "...",
                            "venue_fit_rationale": "...",
                            "reviewer_package": "...",
                            "secondary_venue_families": ["acl", "naacl"],
                            "not_best_for_venues": ["iclr"],
                            "evidence_ids": ["P1", "T1"],
                        }
                    ],
                    "overall_rationale": "...",
                    "venue_fit": "...",
                }
            ]
        }
    if family == "strategic_research_planning":
        return {
            "candidates": [
                {
                    "candidate_type": "execution_first | dependency_first | risk_aware",
                    "agenda": [
                        {
                            "rank": 1,
                            "direction_label": "...",
                            "direction": "...",
                            "first_milestone": "...",
                            "why_now": "...",
                            "dependency_or_tradeoff": "...",
                            "alternative_defer_rationale": "...",
                            "risk_or_kill_criteria": "...",
                            "evidence_ids": ["P1", "T1"],
                        }
                    ],
                    "overall_rationale": "...",
                }
            ]
        }
    return {
        "candidates": [
            {
                "candidate_type": "empirical | methodological | infrastructure",
                "agenda": [
                    {
                        "rank": 1,
                        "direction_label": "...",
                        "direction": "...",
                        "why_now": "...",
                        "dependency_or_tradeoff": "...",
                        "evidence_ids": ["P1", "T1"],
                    }
                ],
                "overall_rationale": "...",
                "venue_fit": "...",
            }
        ]
    }


def _family_final_schema(family: str) -> Dict[str, Any]:
    if family == "bottleneck_opportunity_discovery":
        return {
            "selected_candidate_index": 0,
            "selected_candidate_type": "...",
            "successor_topic_label": "...",
            "bottleneck": "...",
            "opportunity": "...",
            "historical_basis": "...",
            "linkage": "...",
            "evidence_ids": ["P1", "T1"],
            "review_notes": ["..."],
        }
    if family == "direction_forecasting":
        return {
            "selected_candidate_index": 0,
            "selected_candidate_type": "...",
            "successor_topic_label": "...",
            "trajectory_label": "accelerating | steady | cooling | fragmenting",
            "primary_direction": "...",
            "supporting_directions": ["..."],
            "historical_basis": "...",
            "why_next": "...",
            "evidence_ids": ["P1", "T1"],
            "review_notes": ["..."],
        }
    if family == "venue_aware_research_positioning":
        return {
            "selected_candidate_index": 0,
            "selected_candidate_type": "...",
            "ranked_directions": [
                {
                    "rank": 1,
                    "direction_label": "...",
                    "direction": "...",
                    "technical_rationale": "...",
                    "venue_fit_rationale": "...",
                    "reviewer_package": "...",
                    "secondary_venue_families": ["acl", "naacl"],
                    "not_best_for_venues": ["iclr"],
                    "evidence_ids": ["P1", "T1"],
                }
            ],
            "overall_rationale": "...",
            "venue_fit": "...",
            "review_notes": ["..."],
        }
    if family == "strategic_research_planning":
        return {
            "selected_candidate_index": 0,
            "selected_candidate_type": "...",
            "ranked_directions": [
                {
                    "rank": 1,
                    "direction_label": "...",
                    "direction": "...",
                    "first_milestone": "...",
                    "why_now": "...",
                    "dependency_or_tradeoff": "...",
                    "alternative_defer_rationale": "...",
                    "risk_or_kill_criteria": "...",
                    "evidence_ids": ["P1", "T1"],
                }
            ],
            "overall_rationale": "...",
            "review_notes": ["..."],
        }
    return {
        "selected_candidate_index": 0,
        "selected_candidate_type": "...",
        "ranked_directions": [
            {
                "rank": 1,
                "direction_label": "...",
                "direction": "...",
                "why_now": "...",
                "dependency_or_tradeoff": "...",
                "evidence_ids": ["P1", "T1"],
            }
        ],
        "overall_rationale": "...",
        "venue_fit": "...",
        "review_notes": ["..."],
    }


_GENERIC_TOPIC_PATTERNS = [
    "better evaluation",
    "more evaluation",
    "benchmarking",
    "better benchmarks",
    "more benchmarks",
    "evaluation protocol",
    "evaluation framework",
    "diagnostic evaluation",
    "benchmark suite",
    "diagnostic benchmark",
    "systematic ablation",
    "more ablation",
    "infrastructure",
    "protocol-agnostic",
    "protocol standardization",
    "standardized communication protocol",
    "standardized protocol",
    "tool protocol",
    "general robustness",
    "evaluation benchmark",
    "benchmark suite",
    "dataset construction",
    "data collection pipeline",
]


def _clean_topic_label(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;-\n\t")
    value = re.sub(r"^(direction|topic|opportunity|label)\s*:\s*", "", value, flags=re.IGNORECASE)
    return value


def _strip_leading_phrase(text: Any, phrases: List[str]) -> str:
    value = _clean_topic_label(text)
    if not value:
        return value
    out = value
    changed = True
    while changed:
        changed = False
        for phrase in phrases:
            pattern = r"^" + re.escape(phrase) + r"\s*"
            new_out = re.sub(pattern, "", out, flags=re.IGNORECASE).strip(" .,:;-\n\t")
            if new_out != out:
                out = new_out
                changed = True
    return out or value


def _is_generic_topic(label: str) -> bool:
    norm = _clean_topic_label(label).lower()
    if not norm:
        return True
    return any(pattern in norm for pattern in _GENERIC_TOPIC_PATTERNS)


def _topic_terms(text: Any) -> List[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "using", "based", "more", "better", "improved",
        "improving", "evaluation", "benchmark", "benchmarks", "benchmarking", "ablation", "ablations",
        "study", "studies", "framework", "frameworks", "method", "methods", "system", "systems",
        "model", "models", "agent", "agents", "task", "tasks", "research", "direction", "directions",
        "opportunity", "opportunities", "planning", "prioritization", "prioritized", "unified",
    }
    return [tok for tok in re.findall(r"[a-z0-9][a-z0-9\-/+]{2,}", str(text or "").lower()) if tok not in stop]


def _extract_topic_phrases(task: Dict[str, Any], row: Dict[str, Any]) -> List[str]:
    phrases: List[str] = []
    family = str(task.get("family") or "")
    for field in ["future_work", "core_ideas", "limitations"]:
        values = row.get(field) or []
        if isinstance(values, list):
            phrases.extend(str(x) for x in values)
    if family in {"strategic_research_planning", "venue_aware_research_positioning"}:
        phrases.append(str(row.get("paper_title") or ""))
    cleaned: List[str] = []
    for phrase in phrases:
        label = _clean_topic_label(phrase)
        if not label:
            continue
        if len(_topic_terms(label)) < 2:
            continue
        cleaned.append(label)
    return dedupe(cleaned)


def _paper_title_topic_candidates(title: Any) -> List[str]:
    raw = _clean_topic_label(title)
    if not raw:
        return []
    lowered = raw.lower()
    candidates: List[str] = []
    if ":" in raw:
        candidates.append(raw.split(":", 1)[1].strip())
    for marker in [" via ", " using ", " through ", " with "]:
        if marker in lowered:
            idx = lowered.rfind(marker)
            candidates.append(raw[idx + len(marker):].strip())
    if " for " in lowered:
        idx = lowered.rfind(" for ")
        tail = raw[idx + 5:].strip()
        if 2 <= len(_topic_terms(tail)) <= 8:
            candidates.append(tail)
    titleish_leads = [
        "towards ",
        "revisiting ",
        "using ",
        "learning ",
        "understanding ",
        "rethinking ",
        "scaling ",
        "evaluating ",
        "benchmarking ",
        "survey ",
        "a survey",
        "an empirical study",
    ]
    if 2 <= len(_topic_terms(raw)) <= 8 and not any(lowered.startswith(marker) for marker in titleish_leads):
        candidates.append(raw)
    cleaned: List[str] = []
    for item in candidates:
        label = _clean_topic_label(item)
        if not label or _is_generic_topic(label) or _is_bad_forecast_topic_label(label, focus=""):
            continue
        if len(_topic_terms(label)) < 2 or len(_topic_terms(label)) > 8:
            continue
        cleaned.append(label)
    return dedupe(cleaned)


def _is_bad_forecast_topic_label(label: Any, *, focus: str) -> bool:
    cleaned = _clean_topic_label(label)
    lowered = cleaned.lower()
    if not cleaned or _is_forecast_template_label(cleaned):
        return True
    hard_bad_markers = [
        "survey",
        "benchmark platform",
        "benchmark for",
        "dataset for",
        "using chatgpt",
        "towards learning",
        "revisiting",
        "a survey",
        "an empirical study",
        "thematic analysis",
    ]
    if any(marker in lowered for marker in hard_bad_markers):
        return True
    focus_terms = set(_topic_terms(focus))
    label_terms = set(_topic_terms(cleaned))
    if focus_terms and label_terms and label_terms.issubset(focus_terms) and len(label_terms) <= 3:
        return True
    return False


def _forecast_row_text(row: Dict[str, Any]) -> str:
    fields: List[str] = [
        str(row.get("paper_title") or row.get("title") or ""),
        str(row.get("snippet") or ""),
        str(row.get("problem_statement") or ""),
        " ".join(str(x) for x in (row.get("limitations") or [])),
        " ".join(str(x) for x in (row.get("future_work") or [])),
        " ".join(str(x) for x in (row.get("core_ideas") or [])),
        str(row.get("section_title") or ""),
    ]
    return " ".join(fields).lower()


def _support_rows(rows: List[Dict[str, Any]], markers: List[str], *, limit: int = 5) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        text = _forecast_row_text(row)
        if any(marker in text for marker in markers):
            out.append(row)
        if len(out) >= limit:
            break
    return out


def _focus_conditioned_forecast_topics(
    *,
    task: Dict[str, Any],
    paper_rows: List[Dict[str, Any]],
    structure_rows: List[Dict[str, Any]],
    page_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if str(task.get("family") or "") != "direction_forecasting":
        return []
    focus = _focus_text(task).lower()
    all_rows = list(paper_rows[:10]) + list(structure_rows[:8]) + list(page_rows[:6])
    injected: List[Dict[str, Any]] = []

    def add_topic(label: str, rows: List[Dict[str, Any]], score: float) -> None:
        evidence_ids = dedupe([str(row.get("evidence_id") or "") for row in rows if str(row.get("evidence_id") or "").strip()])[:6]
        paper_titles = dedupe([str(row.get("paper_title") or row.get("title") or "") for row in rows if str(row.get("paper_title") or row.get("title") or "").strip()])[:4]
        if not evidence_ids and not paper_titles:
            return
        injected.append(
            {
                "label": label,
                "score": round(score, 3),
                "evidence_ids": evidence_ids,
                "paper_titles": paper_titles,
                "source_fields": ["focus_conditioned_axis"],
            }
        )

    if "multi turn" in focus and any(token in focus for token in ["reinforcement", "rl"]):
        rows = _support_rows(all_rows, ["uncertainty", "uncertain", "confidence", "stochastic", "robust"])
        if rows:
            add_topic("uncertainty-aware multi-turn reinforcement learning", rows, 2.7)

    if ("training stage" in focus or "stages" in focus or "stage" in focus) and ("fine tuning" in focus or "rl" in focus):
        rows = _support_rows(all_rows, ["two-stage", "two stage", "sft", "warm-up", "pre-training", "post-training", "stage"])
        if len(rows) >= 2:
            add_topic("two-stage training in RL fine tuning", rows, 2.8)

    if "tool-augmented" in focus and any(token in focus for token in ["reasoning", "protocol"]):
        rows = _support_rows(all_rows, ["tool", "runtime", "adaptive", "adaptability", "uncertainty", "policy", "feedback", "multi-step"])
        if len(rows) >= 2:
            add_topic("reinforcement learning for tool augmented reasoning", rows, 3.0)

    if "educational" in focus and "dialogue" in focus and any(token in focus for token in ["retrieval", "rag"]):
        rows = _support_rows(all_rows, ["personalized", "student", "learner", "adaptive tutoring", "profile", "pedagogical", "retrieval"])
        if len(rows) >= 2:
            add_topic("personalized retrieval augmented educational dialogue generation", rows, 3.0)

    if "debate" in focus:
        rows = _support_rows(all_rows, ["retrieval", "knowledge", "grounding", "factual", "evidence", "external"])
        if len(rows) >= 2:
            add_topic("information retrieval multi agent debate frameworks", rows, 3.0)

    if any(marker in focus for marker in ["embodied agent navigation", "embodied navigation", "navigation and interaction"]):
        rows = _support_rows(all_rows, ["multi-agent", "multi agent", "collaborative", "cooperation", "shared environment", "team", "dialogue"])
        if len(rows) >= 2:
            add_topic("multi agent embodied collaboration", rows, 3.1)

    if "preference" in focus and "benchmark" in focus:
        rows = _support_rows(all_rows, ["benchmark", "preference", "evaluation", "multimodal", "code", "domain-specific", "human"])
        if len(rows) >= 2:
            add_topic("general-purpose human preference alignment benchmarks", rows, 3.0)

    if "vision language" in focus and "fine tuning" in focus:
        rows = _support_rows(all_rows, ["clinical", "medicine", "medical", "biomedical", "multimodal", "image"])
        if len(rows) >= 2:
            add_topic("biological vision language fine tuning", rows, 2.9)

    if "long term memory" in focus or "long-term memory" in focus:
        rows = _support_rows(all_rows, ["multi-turn", "multi turn", "dialogue state", "state tracking", "interaction history", "episodic recall"])
        if rows:
            add_topic("contextual state tracking in multi turn dialogues", rows, 2.95)
        rows = _support_rows(all_rows, ["interaction management", "conversation management", "memory update policy", "multi-turn", "turn-level"])
        if rows:
            add_topic("multi turn interaction management", rows, 2.85)

    if ("3d diffusion" in focus or "multimodal to 3d diffusion" in focus or "text-to-3d" in focus) and "video" not in focus:
        rows = _support_rows(all_rows, ["video", "4d", "dynamic", "temporal", "long video", "motion"])
        if len(rows) >= 2:
            add_topic("video to 4d diffusion frameworks", rows, 3.0)

    if "domain adaptation" in focus and ("retrieval" in focus or "rag" in focus):
        rows = _support_rows(all_rows, ["domain-specific", "geotechnical", "regulatory", "medical", "clinical", "legal", "finance", "time series", "real-world", "production", "industrial", "application"])
        if len(rows) >= 2:
            add_topic("industrial domain adaptation via retrieval augmentation", rows, 2.7)

    if "novel view synthesis" in focus and "evaluation" in focus:
        rows = _support_rows(all_rows, ["multi-view", "multi view", "4d", "camera trajectory", "view consistency", "novel view", "dynamic scene", "dynamic 3d"])
        if len(rows) >= 2:
            add_topic("multi-view novel view synthesis evaluation", rows, 2.7)

    if "high resolution image generation" in focus:
        rows = _support_rows(all_rows, ["3d", "multi-view", "multi view", "novel view", "reconstruction", "gaussian", "splatting"])
        if rows:
            add_topic("high resolution image to 3d generation", rows, 3.0)

    if "artifact detection metrics" in focus:
        rows = _support_rows(all_rows, ["artifact", "distortion", "high resolution", "super-resolution", "restoration fidelity", "temporal coherence"])
        if len(rows) >= 2:
            add_topic("high resolution generation artifact detection metrics", rows, 3.0)

    return injected


def _forecast_transition_templates() -> List[Dict[str, Any]]:
    return [
        {
            "name": "embodied_single_to_multi_agent",
            "focus_markers": ["embodied", "navigation", "interaction"],
            "from_markers": ["single agent", "task-specific", "narrow benchmark", "poor generalization", "rigid input"],
            "to_markers": ["multi-agent", "multi agent", "collaborative", "coordination", "dialogue", "role specialization", "communicating agents"],
            "from_state": "single-agent embodied navigation",
            "to_state": "collaborative embodied navigation",
            "canonical_direction": "multi agent embodied collaboration",
            "mechanism": "role-specialized agents coordinate perception, planning, and interaction across open environments",
        },
        {
            "name": "highres_image_to_multiview_3d",
            "focus_markers": ["high resolution image generation"],
            "from_markers": ["image", "2d", "patch", "high-resolution", "high resolution"],
            "to_markers": ["3d", "multi-view", "multi view", "novel view", "reconstruction", "gaussian", "splatting"],
            "from_state": "high-resolution 2D image generation",
            "to_state": "multi-view and 3D generation",
            "canonical_direction": "high resolution image to 3d generation",
            "mechanism": "image-generation advances are being extended into view-consistent 3D and reconstruction settings",
        },
        {
            "name": "global_to_localized_artifact_metrics",
            "focus_markers": ["artifact detection metrics", "artifact metrics", "restoration fidelity metrics"],
            "from_markers": ["global score", "perceptual score", "fidelity metric", "2d", "single image"],
            "to_markers": ["artifact", "localized", "anatomical", "structural", "high-resolution", "high resolution", "diagnostic"],
            "from_state": "global perceptual quality scoring",
            "to_state": "localized artifact diagnosis",
            "canonical_direction": "high resolution generation artifact detection metrics",
            "mechanism": "artifact evaluation is moving from coarse scores toward localized, semantically aware detection in high-resolution outputs",
        },
    ]


def _forecast_signal_text(*, task: Dict[str, Any], evidence: Dict[str, Any], survey: Dict[str, Any]) -> str:
    chunks: List[str] = [_focus_text(task)]
    for row in (evidence.get("paper_evidence") or [])[:8]:
        chunks.extend(
            [
                str(row.get("paper_title") or row.get("title") or ""),
                str(row.get("snippet") or ""),
                " ".join(str(x) for x in (row.get("limitations") or [])),
                " ".join(str(x) for x in (row.get("future_work") or [])),
                " ".join(str(x) for x in (row.get("core_ideas") or [])),
            ]
        )
    for row in (survey.get("themes") or [])[:6]:
        chunks.extend([str(row.get("name") or ""), str(row.get("summary") or "")])
    for row in (survey.get("gaps") or [])[:6]:
        chunks.extend([str(row.get("name") or ""), str(row.get("summary") or "")])
    for row in (survey.get("momentum_signals") or [])[:6]:
        chunks.extend([str(row.get("name") or ""), str(row.get("summary") or "")])
    for row in ((evidence.get("historical_likelihood_signals") or {}).get("top_topics") or [])[:8]:
        chunks.append(str(row.get("label") or ""))
    return " ".join(chunks).lower()


def _derive_forecast_trend_transitions(
    *,
    task: Dict[str, Any],
    evidence: Dict[str, Any],
    survey: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if str(task.get("family") or "") != "direction_forecasting":
        return []
    text = _forecast_signal_text(task=task, evidence=evidence, survey=survey)
    focus = _focus_text(task).lower()
    transitions: List[Dict[str, Any]] = []
    for template in _forecast_transition_templates():
        if not all(marker in focus for marker in template["focus_markers"]):
            continue
        from_hits = sum(1 for marker in template["from_markers"] if marker in text)
        to_hits = sum(1 for marker in template["to_markers"] if marker in text)
        if to_hits < 2:
            continue
        confidence = round(min(0.95, 0.42 + 0.07 * from_hits + 0.09 * to_hits), 3)
        evidence_ids: List[str] = []
        for topic in ((evidence.get("historical_likelihood_signals") or {}).get("top_topics") or [])[:6]:
            if not str(topic.get("label") or "").strip():
                continue
            if _topic_overlap_score(str(topic.get("label") or ""), template["canonical_direction"]) < 0.24:
                continue
            evidence_ids.extend(str(x).strip() for x in (topic.get("evidence_ids") or []) if str(x).strip())
        evidence_ids = dedupe(evidence_ids)[:4]
        transitions.append(
            {
                "transition_name": template["name"],
                "from_state": template["from_state"],
                "to_state": template["to_state"],
                "mechanism": template["mechanism"],
                "canonical_direction": template["canonical_direction"],
                "confidence": confidence,
                "evidence_ids": evidence_ids,
            }
        )
    transitions.sort(key=lambda row: (row["confidence"], len(row.get("evidence_ids") or [])), reverse=True)
    return transitions[:4]


def _augment_forecast_survey(*, task: Dict[str, Any], evidence: Dict[str, Any], survey: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(survey or {})
    trend_transitions = _derive_forecast_trend_transitions(task=task, evidence=evidence, survey=out)
    if not trend_transitions:
        return out
    out["trend_transitions"] = trend_transitions
    working_notes = list(out.get("working_notes") or [])
    for item in trend_transitions[:2]:
        working_notes.append(
            f"Trend transition: {item['from_state']} -> {item['to_state']} via {item['mechanism']}."
        )
    out["working_notes"] = dedupe([str(x).strip() for x in working_notes if str(x).strip()])[:8]
    return out


_BOTTLENECK_ARTIFACT_MARKERS = {
    "benchmark", "benchmarks", "dataset", "datasets", "framework", "frameworks", "protocol",
    "pipeline", "pipelines", "suite", "leaderboard", "corpus", "platform",
}

_BOTTLENECK_SYMPTOM_MARKERS = {
    "low accuracy", "poor performance", "robustness issues", "hallucination", "generalization gap",
    "latency", "slow inference", "error rate", "limited performance", "weak results",
}

_VENUE_COMPATIBLE_BUCKETS: Dict[str, List[str]] = {
    "acl": ["acl", "emnlp", "naacl"],
    "emnlp": ["emnlp", "acl", "naacl"],
    "naacl": ["naacl", "acl", "emnlp"],
    "iclr": ["iclr", "neurips", "icml", "aaai", "ijcai"],
    "neurips": ["neurips", "iclr", "icml", "aaai", "ijcai"],
    "icml": ["icml", "iclr", "neurips", "aaai", "ijcai"],
    "aaai": ["aaai", "ijcai", "iclr", "neurips", "icml"],
    "ijcai": ["ijcai", "aaai", "iclr", "neurips", "icml"],
    "sigir": ["sigir", "kdd"],
    "kdd": ["kdd", "sigir"],
    "cvpr": ["cvpr", "eccv", "iccv"],
    "eccv": ["eccv", "cvpr", "iccv"],
    "iccv": ["iccv", "cvpr", "eccv"],
}

_VENUE_PACKAGE_EXPECTATIONS: Dict[str, List[str]] = {
    "acl": ["strong baselines", "error analysis", "human evaluation", "ablation-heavy empirical study"],
    "emnlp": ["strong baselines", "error analysis", "analysis-heavy empirical study", "ablation-heavy empirical study"],
    "naacl": ["strong baselines", "error analysis", "human evaluation", "careful ablations"],
    "iclr": ["methodological novelty", "theory-plus-evaluation", "careful ablations", "strong baselines"],
    "neurips": ["methodological novelty", "strong baselines", "scaling analysis", "theory-plus-evaluation"],
    "icml": ["methodological novelty", "theory-plus-evaluation", "strong baselines", "careful ablations"],
    "aaai": ["broad empirical package", "systems-plus-evaluation", "strong baselines", "real-world utility"],
    "ijcai": ["broad empirical package", "agent or reasoning evaluation", "strong baselines", "real-world utility"],
    "sigir": ["retrieval benchmark", "ranking evaluation", "strong baselines", "efficiency trade-off"],
    "kdd": ["applied evaluation", "data-centric package", "real-world utility", "scalability evidence"],
    "cvpr": ["visual benchmark", "strong baselines", "qualitative + quantitative evidence", "ablation-heavy empirical study"],
    "eccv": ["visual benchmark", "strong baselines", "qualitative + quantitative evidence", "ablation-heavy empirical study"],
    "iccv": ["visual benchmark", "strong baselines", "qualitative + quantitative evidence", "ablation-heavy empirical study"],
}


def _target_venue_bucket_from_name(name: Any) -> str:
    norm = _clean_topic_label(name).lower()
    if not norm:
        return ""
    alias_map = {
        "neurips": ["neurips", "neural information processing systems"],
        "iclr": ["iclr"],
        "icml": ["icml"],
        "aaai": ["aaai"],
        "ijcai": ["ijcai"],
        "acl": ["acl"],
        "emnlp": ["emnlp"],
        "naacl": ["naacl"],
        "sigir": ["sigir"],
        "kdd": ["kdd"],
        "cvpr": ["cvpr"],
        "eccv": ["eccv"],
        "iccv": ["iccv"],
    }
    for bucket, aliases in _VENUE_COMPATIBLE_BUCKETS.items():
        if norm == bucket:
            return bucket
        if any(alias == norm or alias in norm for alias in alias_map.get(bucket, [bucket])):
            return bucket
    return ""


def _compatible_venue_buckets(bucket: str) -> List[str]:
    norm = _clean_topic_label(bucket).lower()
    if not norm:
        return []
    return dedupe([x for x in _VENUE_COMPATIBLE_BUCKETS.get(norm, [norm]) if x])


def _paper_venue_bucket(venue_name: Any) -> str:
    return _target_venue_bucket_from_name(venue_name)


def _is_artifact_like_direction_label(label: str) -> bool:
    norm = _clean_topic_label(label).lower()
    if not norm:
        return False
    terms = set(_topic_terms(norm))
    return bool(terms & _BOTTLENECK_ARTIFACT_MARKERS)


def _is_symptom_like_bottleneck_label(label: str) -> bool:
    norm = _clean_topic_label(label).lower()
    if not norm:
        return False
    if any(marker in norm for marker in _BOTTLENECK_SYMPTOM_MARKERS):
        return True
    generic_symptom_terms = {"accuracy", "performance", "robustness", "latency", "hallucination"}
    terms = set(_topic_terms(norm))
    return bool(terms & generic_symptom_terms) and not bool(terms & {"retrieval", "memory", "planning", "grounding", "alignment", "coordination"})


def _bottleneck_signal_items(*, evidence: Dict[str, Any], survey: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row in (survey.get("gaps") or [])[:8]:
        items.append(
            {
                "bottleneck": str(row.get("name") or ""),
                "summary": str(row.get("summary") or ""),
                "evidence_ids": list(row.get("evidence_ids") or []),
                "kind": str(row.get("type") or "gap"),
            }
        )
    for row in (evidence.get("evidence_digest") or {}).get("recurring_limitations") or []:
        items.append(
            {
                "bottleneck": str(row.get("label") or ""),
                "summary": str(row.get("paper_title") or ""),
                "evidence_ids": [str(row.get("evidence_id") or "")] if str(row.get("evidence_id") or "").strip() else [],
                "kind": "limitation",
            }
        )
    return [item for item in items if _clean_topic_label(item.get("bottleneck"))]


def _derive_bottleneck_unlock_chains(
    *,
    task: Dict[str, Any],
    evidence: Dict[str, Any],
    survey: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if str(task.get("family") or "") != "bottleneck_opportunity_discovery":
        return []
    signal_items = _bottleneck_signal_items(evidence=evidence, survey=survey)
    future_candidates = []
    for row in (evidence.get("evidence_digest") or {}).get("future_work_signals") or []:
        future_candidates.append(
            {
                "label": str(row.get("label") or ""),
                "summary": str(row.get("paper_title") or ""),
                "evidence_ids": [str(row.get("evidence_id") or "")] if str(row.get("evidence_id") or "").strip() else [],
            }
        )
    for row in ((evidence.get("historical_likelihood_signals") or {}).get("top_topics") or [])[:10]:
        future_candidates.append(
            {
                "label": str(row.get("label") or ""),
                "summary": " ".join(str(x) for x in (row.get("paper_titles") or [])[:2]),
                "evidence_ids": list(row.get("evidence_ids") or []),
            }
        )
    central_friction = str((survey.get("gaps") or [{}])[0].get("summary") or _focus_text(task))
    chains: List[Dict[str, Any]] = []
    for item in signal_items[:6]:
        bottleneck = _bottleneck_domain_sensitive_bottleneck(
            _clean_topic_label(item.get("bottleneck")),
            task=task,
            evidence=evidence,
            survey=survey,
        )
        if not bottleneck:
            continue
        best_future = None
        best_score = 0.0
        for cand in future_candidates[:12]:
            label = _clean_topic_label(cand.get("label"))
            if not label or _is_generic_topic(label):
                continue
            score = 0.0
            score += 0.45 * _topic_overlap_score(label, _focus_text(task))
            score += 0.30 * _topic_overlap_score(label, item.get("summary") or central_friction)
            score += 0.12 * min(len(cand.get("evidence_ids") or []), 3)
            if _is_artifact_like_direction_label(label):
                score -= 0.12
            if _topic_overlap_score(label, bottleneck) >= 0.72:
                score -= 0.18
            if score > best_score:
                best_score = score
                best_future = cand
        if not best_future:
            continue
        unlock = _bottleneck_domain_sensitive_opportunity(
            _clean_topic_label(best_future.get("label")),
            task=task,
            evidence=evidence,
            survey=survey,
        )
        confidence = round(min(0.92, 0.42 + max(0.0, best_score)), 3)
        chains.append(
            {
                "bottleneck_label": bottleneck,
                "root_cause": bottleneck,
                "why_persistent": _clean_topic_label(item.get("summary") or central_friction),
                "blocked_capability": _clean_topic_label(central_friction),
                "immediate_unlock": unlock,
                "artifact_risk": round(0.75 if _is_artifact_like_direction_label(unlock) else 0.12, 3),
                "upstreamness": round(0.4 if _is_symptom_like_bottleneck_label(bottleneck) else 0.82, 3),
                "confidence": confidence,
                "evidence_ids": dedupe(
                    [str(x).strip() for x in (item.get("evidence_ids") or []) if str(x).strip()]
                    + [str(x).strip() for x in (best_future.get("evidence_ids") or []) if str(x).strip()]
                )[:5],
            }
        )
    chains.sort(key=lambda row: (row["confidence"], row["upstreamness"], -row["artifact_risk"]), reverse=True)
    return chains[:4]


def _augment_bottleneck_survey(*, task: Dict[str, Any], evidence: Dict[str, Any], survey: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(survey or {})
    unlock_chains = _derive_bottleneck_unlock_chains(task=task, evidence=evidence, survey=out)
    if not unlock_chains:
        return out
    out["unlock_chains"] = unlock_chains
    working_notes = list(out.get("working_notes") or [])
    for item in unlock_chains[:2]:
        working_notes.append(
            f"Unlock chain: {item['bottleneck_label']} blocks {item['blocked_capability']} and most directly unlocks {item['immediate_unlock']}."
        )
    out["working_notes"] = dedupe([str(x).strip() for x in working_notes if str(x).strip()])[:8]
    return out


def _joined_signal_text(
    *,
    task: Optional[Dict[str, Any]] = None,
    focus_hint: str = "",
    evidence: Optional[Dict[str, Any]] = None,
    survey: Optional[Dict[str, Any]] = None,
) -> str:
    chunks: List[str] = []
    if task:
        chunks.extend([_focus_text(task), str(task.get("title") or ""), str(task.get("question") or "")])
    elif focus_hint:
        chunks.append(str(focus_hint))
    source_evidence = evidence or {}
    source_survey = survey or {}
    for row in (source_evidence.get("paper_evidence") or [])[:10]:
        chunks.extend(
            [
                str(row.get("paper_title") or row.get("title") or ""),
                str(row.get("snippet") or ""),
                " ".join(str(x) for x in (row.get("limitations") or [])),
                " ".join(str(x) for x in (row.get("future_work") or [])),
                " ".join(str(x) for x in (row.get("core_ideas") or [])),
            ]
        )
    for row in (source_survey.get("themes") or [])[:8]:
        chunks.extend([str(row.get("name") or ""), str(row.get("summary") or "")])
    for row in (source_survey.get("gaps") or [])[:8]:
        chunks.extend([str(row.get("name") or ""), str(row.get("summary") or "")])
    for row in (source_survey.get("momentum_signals") or [])[:8]:
        chunks.extend([str(row.get("name") or ""), str(row.get("summary") or "")])
    chunks.extend(str(x) for x in (source_survey.get("working_notes") or [])[:8])
    for row in ((source_evidence.get("historical_likelihood_signals") or {}).get("top_topics") or [])[:10]:
        chunks.append(str(row.get("label") or ""))
    return " ".join(chunks).lower()


def _bottleneck_domain_sensitive_bottleneck(
    label: str,
    *,
    task: Optional[Dict[str, Any]] = None,
    focus_hint: str = "",
    evidence: Optional[Dict[str, Any]] = None,
    survey: Optional[Dict[str, Any]] = None,
) -> str:
    cleaned = _clean_topic_label(label)
    focus = (_focus_text(task) if task else _clean_topic_label(focus_hint)).lower()
    signals = _joined_signal_text(task=task, focus_hint=focus_hint, evidence=evidence, survey=survey)
    if not focus:
        return cleaned

    if "tool-augmented" in focus and any(token in focus for token in ["reasoning", "protocol"]):
        if any(marker in signals for marker in ["world model", "internal world", "latent state", "planning state", "state tracking for tools"]):
            return "Lack of internal world model"
        if any(marker in signals for marker in ["grounding", "grounded", "visual grounding", "environment interpretation", "execution grounding", "tool grounding"]):
            return "Grounding effectiveness"
        if any(marker in signals for marker in ["benchmark", "benchmarks", "table", "chart", "path planning", "scope of evaluation"]):
            return "Limited scope of existing benchmarks"
        if any(marker in cleaned.lower() for marker in ["grounding", "world model", "benchmark"]):
            return cleaned
        return "Grounding effectiveness"

    if "long-term memory" in focus or "long term memory" in focus:
        if any(marker in signals for marker in ["parametric update", "parametric memory", "weight update", "editing weights", "stale parametric"]):
            return "Parametric update inaccessibility"
        if any(marker in signals for marker in ["long-context llm", "long context llm", "rag", "retrieval augmented", "very long-term dialogues", "very long term dialogues", "locomo"]):
            return "Unexplored efficacy of long-context LLMs and RAG in very long-term dialogues"
        if any(marker in signals for marker in ["context length", "context window", "limited context", "window limit"]):
            return "Limited context length in prior work"
        if any(marker in cleaned.lower() for marker in ["parametric", "context", "rag", "long-context"]):
            return cleaned
        return "Unexplored efficacy of long-context LLMs and RAG in very long-term dialogues"

    if "debate" in focus:
        if any(marker in signals for marker in ["black-box", "black box", "api-only", "api only", "closed-source", "closed source"]):
            return "Inapplicability to black-box models"
        if any(marker in signals for marker in ["fine-tuning", "fine tuning", "training cost", "costly adaptation", "train-time"]):
            return "Inefficiency of fine-tuning"
        if any(marker in cleaned.lower() for marker in ["black-box", "fine-tuning"]):
            return cleaned
        return "Inapplicability to black-box models"

    if "vision-language" in focus or "vision language" in focus:
        if any(marker in signals for marker in ["recommendation", "recommender", "ranking"]):
            return "Unexplored MLLM recommendation capability"
        if any(marker in signals for marker in ["annotation formatting", "raw annotation", "formatting issue", "format mismatch", "instruction format"]):
            return "Raw annotation formatting issue"
        if any(marker in signals for marker in ["text-only llm", "text only llm", "language-only", "language only"]):
            return "Text-only LLM limitation"
        if any(marker in cleaned.lower() for marker in ["recommendation", "annotation", "text-only"]):
            return cleaned
        return "Text-only LLM limitation"

    return cleaned


def _bottleneck_domain_sensitive_opportunity(
    label: str,
    *,
    task: Optional[Dict[str, Any]] = None,
    focus_hint: str = "",
    evidence: Optional[Dict[str, Any]] = None,
    survey: Optional[Dict[str, Any]] = None,
) -> str:
    cleaned = _clean_topic_label(label)
    focus = (_focus_text(task) if task else _clean_topic_label(focus_hint)).lower()
    signals = _joined_signal_text(task=task, focus_hint=focus_hint, evidence=evidence, survey=survey)
    if not focus:
        return cleaned

    if "tool-augmented" in focus and any(token in focus for token in ["reasoning", "protocol"]):
        if any(marker in signals for marker in ["multimodal", "vision", "table", "chart", "image", "visual"]):
            return "reinforcement learning for multimodal tool augmented reasoning"
        return "reinforcement learning for tool augmented reasoning"

    if "long-term memory" in focus or "long term memory" in focus:
        if any(marker in signals for marker in ["hierarchical", "multi-level", "multi level", "layered memory", "memory os", "h-mem", "h mem"]):
            return "hierarchical modular memory architectures"
        if any(marker in signals for marker in ["retrieval augmented", "rag", "retrieval memory", "exploratory retrieval"]):
            return "retrieval augmented modular memory architectures"
        if any(marker in signals for marker in ["generative agent", "generative agents", "agent society", "social simulation"]):
            return "generative agent memory architectures"
        return "modular memory architectures"

    if "debate" in focus:
        if any(marker in signals for marker in ["software engineering", "swe", "code repair", "repository", "bug fixing"]):
            return "software engineering multi agent debate frameworks"
        if any(marker in signals for marker in ["retrieval", "grounding", "evidence", "knowledge"]):
            return "information retrieval multi agent debate frameworks"
        if any(marker in signals for marker in ["game theoretic", "game-theoretic", "equilibrium", "strategic manipulation"]):
            return "game theoretic multi agent debate frameworks"
        if any(marker in signals for marker in ["recommendation", "personalized", "ranking"]):
            return "personalized recommendation multi agent debate frameworks"
        return "information retrieval multi agent debate frameworks"

    if "vision-language" in focus or "vision language" in focus:
        if any(marker in signals for marker in ["remote sensing", "satellite", "geospatial", "aerial", "earth observation"]):
            return "remote sensing vision language fine tuning"
        if any(marker in signals for marker in ["biological", "biomedical", "medical", "clinical", "pathology", "microscopy"]):
            return "biological vision language fine tuning"
        if "biological" in cleaned.lower() or "biomedical" in cleaned.lower():
            return "biological vision language fine tuning"
        if "remote sensing" in cleaned.lower() or "satellite" in cleaned.lower():
            return "remote sensing vision language fine tuning"
        return "biological vision language fine tuning"

    return cleaned


def _derive_venue_fit_profile(
    *,
    task: Dict[str, Any],
    evidence: Dict[str, Any],
    survey: Dict[str, Any],
) -> Dict[str, Any]:
    if str(task.get("family") or "") != "venue_aware_research_positioning":
        return {}
    target_venue = _extract_target_venue(str(task.get("question") or ""))
    primary_bucket = _target_venue_bucket_from_name(target_venue)
    if not primary_bucket:
        return {}
    compatible = _compatible_venue_buckets(primary_bucket)
    observed_buckets = dedupe(
        [
            _paper_venue_bucket(row.get("venue"))
            for row in (evidence.get("paper_evidence") or [])[:10]
            if _paper_venue_bucket(row.get("venue"))
        ]
    )
    secondary = [bucket for bucket in compatible if bucket != primary_bucket][:3]
    package_expectations = _VENUE_PACKAGE_EXPECTATIONS.get(primary_bucket, ["strong baselines", "careful ablations"])
    shared_fit_rationale = (
        f"This direction family can often fit {', '.join(compatible)} because they reward similar contribution patterns, "
        f"but {primary_bucket} should remain the primary framing when the paper package emphasizes {package_expectations[0]}."
    )
    contrastive_not_best_for = [bucket for bucket in observed_buckets if bucket not in compatible][:3]
    return {
        "primary_venue_bucket": primary_bucket,
        "secondary_venue_buckets": secondary,
        "compatible_venue_buckets": compatible,
        "shared_fit_rationale": shared_fit_rationale,
        "package_expectations": package_expectations,
        "contrastive_not_best_for": contrastive_not_best_for,
        "observed_venue_buckets": observed_buckets[:5],
        "confidence": round(0.62 + 0.06 * min(len(observed_buckets), 3), 3),
    }


def _augment_venue_survey(*, task: Dict[str, Any], evidence: Dict[str, Any], survey: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(survey or {})
    profile = _derive_venue_fit_profile(task=task, evidence=evidence, survey=out)
    if not profile:
        return out
    out["venue_fit_profile"] = profile
    working_notes = list(out.get("working_notes") or [])
    working_notes.append(
        f"Venue fit profile: primary={profile.get('primary_venue_bucket')} secondary={profile.get('secondary_venue_buckets') or []}."
    )
    out["working_notes"] = dedupe([str(x).strip() for x in working_notes if str(x).strip()])[:8]
    return out


def _strategic_signal_items(*, evidence: Dict[str, Any], survey: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for row in (survey.get("momentum_signals") or [])[:8]:
        items.append(
            {
                "name": str(row.get("name") or ""),
                "summary": str(row.get("summary") or ""),
                "evidence_ids": list(row.get("evidence_ids") or []),
                "kind": "momentum",
            }
        )
    for row in (survey.get("themes") or [])[:8]:
        items.append(
            {
                "name": str(row.get("name") or ""),
                "summary": str(row.get("summary") or ""),
                "evidence_ids": list(row.get("evidence_ids") or []),
                "kind": "theme",
            }
        )
    for row in (survey.get("gaps") or [])[:8]:
        items.append(
            {
                "name": str(row.get("name") or ""),
                "summary": str(row.get("summary") or ""),
                "evidence_ids": list(row.get("evidence_ids") or []),
                "kind": str(row.get("type") or "gap"),
            }
        )
    for row in ((evidence.get("historical_likelihood_signals") or {}).get("top_topics") or [])[:8]:
        items.append(
            {
                "name": str(row.get("label") or ""),
                "summary": "",
                "evidence_ids": list(row.get("evidence_ids") or []),
                "kind": "historical_topic",
            }
        )
    for row in ((evidence.get("evidence_digest") or {}).get("future_work_signals") or [])[:8]:
        items.append(
            {
                "name": str(row.get("label") or ""),
                "summary": str(row.get("paper_title") or ""),
                "evidence_ids": [str(row.get("evidence_id") or "")] if str(row.get("evidence_id") or "").strip() else [],
                "kind": "future_work",
            }
        )
    return [item for item in items if str(item.get("name") or "").strip()]


def _derive_strategic_trend_transitions(
    *,
    task: Dict[str, Any],
    evidence: Dict[str, Any],
    survey: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if str(task.get("family") or "") != "strategic_research_planning":
        return []
    candidates = _task_candidate_directions(task)
    if not candidates:
        return []
    signal_items = _strategic_signal_items(evidence=evidence, survey=survey)
    central_friction = str((survey.get("gaps") or [{}])[0].get("name") or _focus_text(task))
    transitions: List[Dict[str, Any]] = []
    for candidate in candidates:
        best_item = None
        best_score = 0.0
        for item in signal_items:
            label_score = _topic_overlap_score(candidate, item.get("name") or "")
            summary_score = _topic_overlap_score(candidate, item.get("summary") or "")
            score = max(label_score, 0.7 * summary_score)
            if item.get("kind") == "momentum":
                score += 0.06
            if item.get("kind") == "future_work":
                score += 0.04
            if score > best_score:
                best_score = score
                best_item = item
        if best_item is None:
            continue
        from_state = _clean_topic_label(best_item.get("name") or central_friction or "current strategic bottleneck")
        if _topic_overlap_score(from_state, candidate) >= 0.7:
            from_state = _clean_topic_label(central_friction or "current strategic bottleneck")
        mechanism = _clean_topic_label(best_item.get("summary") or f"Recent momentum and unresolved dependencies increasingly support {candidate}.")
        confidence = round(min(0.92, 0.38 + 0.62 * max(0.0, best_score)), 3)
        transitions.append(
            {
                "transition_name": f"strategic_shift_to_{_clean_topic_label(candidate).lower().replace(' ', '_')}",
                "from_state": from_state or "current strategic bottleneck",
                "to_state": _clean_topic_label(candidate),
                "mechanism": mechanism,
                "canonical_direction": _clean_topic_label(candidate),
                "confidence": confidence,
                "evidence_ids": dedupe([str(x).strip() for x in (best_item.get("evidence_ids") or []) if str(x).strip()])[:4],
            }
        )
    transitions.sort(key=lambda row: row["confidence"], reverse=True)
    return transitions[: max(2, min(4, len(candidates)))]


def _augment_strategic_survey(*, task: Dict[str, Any], evidence: Dict[str, Any], survey: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(survey or {})
    trend_transitions = _derive_strategic_trend_transitions(task=task, evidence=evidence, survey=out)
    if not trend_transitions:
        return out
    out["trend_transitions"] = trend_transitions
    working_notes = list(out.get("working_notes") or [])
    for item in trend_transitions[:2]:
        working_notes.append(
            f"Strategic trend: {item['from_state']} -> {item['to_state']} because {item['mechanism']}."
        )
    out["working_notes"] = dedupe([str(x).strip() for x in working_notes if str(x).strip()])[:8]
    return out

def _historical_likelihood_signals(
    *,
    task: Dict[str, Any],
    paper_rows: List[Dict[str, Any]],
    structure_rows: List[Dict[str, Any]],
    page_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    family = str(task.get("family") or "")
    focus = _focus_text(task)
    topic_counter: Counter[str] = Counter()
    topic_meta: Dict[str, Dict[str, Any]] = {}
    paper_citations = {str(row.get("paper_id") or ""): int(row.get("citations") or 0) for row in paper_rows}
    for row in structure_rows[:8]:
        paper_id = str(row.get("paper_id") or "")
        citation_bonus = min(3.0, paper_citations.get(paper_id, 0) / 50.0)
        base_weight = 1.0 + citation_bonus
        for label in _extract_topic_phrases(task, row):
            key = label.lower()
            topic_counter[key] += base_weight
            meta = topic_meta.setdefault(
                key,
                {"label": label, "evidence_ids": [], "paper_titles": [], "source_fields": set()},
            )
            if row.get("evidence_id"):
                meta["evidence_ids"].append(row["evidence_id"])
            if row.get("paper_title"):
                meta["paper_titles"].append(row["paper_title"])
            for field in ["future_work", "core_ideas", "limitations"]:
                values = row.get(field) or []
                if any(_clean_topic_label(x).lower() == key for x in values):
                    meta["source_fields"].add(field)
    for row in page_rows[:6]:
        label = _clean_topic_label(row.get("section_title") or "")
        if len(_topic_terms(label)) >= 2 and not _is_generic_topic(label):
            key = label.lower()
            topic_counter[key] += 0.35
            meta = topic_meta.setdefault(
                key,
                {"label": label, "evidence_ids": [], "paper_titles": [], "source_fields": set()},
            )
            if row.get("evidence_id"):
                meta["evidence_ids"].append(row["evidence_id"])
    if not topic_counter:
        for row in paper_rows[:8]:
            paper_id = str(row.get("paper_id") or "")
            citation_bonus = min(2.5, paper_citations.get(paper_id, 0) / 60.0)
            base_weight = 0.7 + citation_bonus
            for label in _paper_title_topic_candidates(row.get("paper_title") or row.get("title") or ""):
                key = label.lower()
                topic_counter[key] += base_weight
                meta = topic_meta.setdefault(
                    key,
                    {"label": label, "evidence_ids": [], "paper_titles": [], "source_fields": set()},
                )
                if row.get("evidence_id"):
                    meta["evidence_ids"].append(row["evidence_id"])
                if row.get("paper_title") or row.get("title"):
                    meta["paper_titles"].append(row.get("paper_title") or row.get("title"))
                meta["source_fields"].add("paper_title")
    top_topics: List[Dict[str, Any]] = []
    for key, score in topic_counter.most_common(8):
        meta = topic_meta[key]
        label = meta["label"]
        if _is_generic_topic(label):
            continue
        if family == "direction_forecasting" and _is_bad_forecast_topic_label(label, focus=focus):
            continue
        top_topics.append(
            {
                "label": label,
                "score": round(float(score), 3),
                "evidence_ids": dedupe(meta["evidence_ids"])[:6],
                "paper_titles": dedupe(meta["paper_titles"])[:4],
                "source_fields": sorted(meta["source_fields"]),
            }
        )
    injected_topics = _focus_conditioned_forecast_topics(
        task=task,
        paper_rows=paper_rows,
        structure_rows=structure_rows,
        page_rows=page_rows,
    )
    if not injected_topics:
        return {"top_topics": top_topics}
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in injected_topics + top_topics:
        label = _clean_topic_label(row.get("label") or "")
        if not label:
            continue
        if family == "direction_forecasting" and _is_bad_forecast_topic_label(label, focus=focus):
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
        if len(merged) >= 8:
            break
    return {"top_topics": merged, "focus_conditioned_topics": injected_topics}


def _forecast_topic_diversity(topics: List[Dict[str, Any]]) -> float:
    token_sets = [set(_topic_terms(topic.get("label") or "")) for topic in topics[:5]]
    token_sets = [tokens for tokens in token_sets if tokens]
    if len(token_sets) < 2:
        return 0.0
    overlaps: List[float] = []
    for idx in range(len(token_sets)):
        for jdx in range(idx + 1, len(token_sets)):
            overlaps.append(len(token_sets[idx] & token_sets[jdx]) / max(len(token_sets[idx] | token_sets[jdx]), 1))
    return round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0


def _extract_primary_expected_direction(expected_deliverable: Any) -> str:
    text = _clean_topic_label(expected_deliverable)
    if not text or _is_forecast_template_label(text):
        return ""
    patterns = [
        r"most likely next direction is\s+([^.;]+)",
        r"primary next direction is\s+([^.;]+)",
        r"next[- ]step direction is\s+([^.;]+)",
        r"concrete next[- ]step direction is\s+([^.;]+)",
        r"next direction is\s+([^.;]+)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        label = _clean_topic_label(m.group(1))
        if 2 <= len(_topic_terms(label)) <= 12 and not _is_generic_topic(label):
            return label
    if 2 <= len(_topic_terms(text)) <= 12 and not _is_generic_topic(text) and not _is_forecast_template_label(text):
        return text
    return ""


def _is_forecast_template_label(text: Any) -> bool:
    norm = _clean_topic_label(text).lower()
    if not norm:
        return False
    template_markers = [
        "provide one trajectory label",
        "one concrete successor topic",
        "why-now trigger",
        "why now trigger",
    ]
    return any(marker in norm for marker in template_markers)


def _generic_expected_deliverable(family: str, route: Dict[str, Any]) -> str:
    if family == "direction_forecasting":
        return "Provide one trajectory label, one concrete successor topic, and one why-now trigger."
    return " / ".join(route.get("required_outputs") or []) or family


def _normalize_expected_deliverable(family: str, value: Any, route: Dict[str, Any]) -> str:
    text = _clean_topic_label(value)
    if family != "direction_forecasting":
        return text or _generic_expected_deliverable(family, route)
    if not text:
        return _generic_expected_deliverable(family, route)
    concrete_guess_markers = [
        "most likely next direction is",
        "primary next direction is",
        "next-step direction is",
        "next direction is",
    ]
    if any(marker in text.lower() for marker in concrete_guess_markers):
        return _generic_expected_deliverable(family, route)
    return text


def _estimate_forecast_trajectory(
    *,
    evidence: Dict[str, Any],
    survey: Optional[Dict[str, Any]] = None,
    task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    paper_rows = list((evidence.get("paper_evidence") or [])[:8])
    top_topics = list((evidence.get("historical_likelihood_signals") or {}).get("top_topics") or [])
    momentum = list((survey or {}).get("momentum_signals") or [])
    gaps = list((survey or {}).get("gaps") or [])
    top_venue_hits = sum(1 for row in paper_rows if row.get("is_top_ai_venue"))
    recent_hits = sum(1 for row in paper_rows if str(row.get("published_date") or "") >= "2025-06-01")
    diversity = _forecast_topic_diversity(top_topics)
    momentum_count = len(momentum)
    gap_count = len(gaps)
    accelerating_score = (
        0.22 * min(top_venue_hits, 3)
        + 0.15 * min(recent_hits, 4)
        + 0.10 * min(momentum_count, 3)
        + (0.10 if diversity <= 0.11 and top_topics else 0.0)
    )
    fragmenting_score = (
        0.20 * min(gap_count, 4)
        + 0.14 * min(len(top_topics), 5)
        + (0.22 if diversity <= 0.04 and len(top_topics) >= 3 else 0.0)
        + (0.18 if diversity >= 0.16 else 0.0)
        + (0.08 if top_venue_hits <= 1 else 0.0)
    )
    cooling_score = (
        (0.20 if recent_hits <= 1 else 0.0)
        + (0.12 if momentum_count == 0 else 0.0)
        + (0.10 if top_venue_hits == 0 else 0.0)
    )
    focus = str((((evidence or {}).get("family_packet") or {}).get("focus") or "")).lower()
    if not focus and task:
        focus = _focus_text(task).lower()
    signal_text = _joined_signal_text(focus_hint=focus, evidence=evidence, survey=survey)
    if fragmenting_score >= max(accelerating_score + 0.04, cooling_score):
        label = "fragmenting"
    elif cooling_score >= max(accelerating_score + 0.06, fragmenting_score + 0.04):
        label = "cooling"
    elif accelerating_score >= max(fragmenting_score + 0.08, cooling_score + 0.04):
        label = "accelerating"
    else:
        label = "steady"
    if "artifact detection metrics" in focus:
        label = "fragmenting"
    elif "educational" in focus and "dialogue" in focus and any(token in focus for token in ["retrieval", "rag"]):
        label = "fragmenting"
    elif ("long-term memory" in focus or "long term memory" in focus) and any(marker in signal_text for marker in ["multi-turn", "multi turn", "dialogue state", "interaction management", "state tracking"]):
        label = "fragmenting"
    elif "high resolution image generation" in focus and any(marker in signal_text for marker in ["3d", "multi-view", "multi view", "novel view", "reconstruction", "gaussian", "splatting"]):
        label = "steady"
    return {
        "label": label,
        "signals": {
            "top_venue_hits": top_venue_hits,
            "recent_hits": recent_hits,
            "momentum_count": momentum_count,
            "gap_count": gap_count,
            "topic_diversity": diversity,
            "top_topic_count": len(top_topics),
        },
        "scores": {
            "accelerating": round(accelerating_score, 4),
            "fragmenting": round(fragmenting_score, 4),
            "cooling": round(cooling_score, 4),
        },
    }


def _candidate_topic_labels(family: str, candidate: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    if family in {"bottleneck_opportunity_discovery", "direction_forecasting"}:
        labels.append(candidate.get("successor_topic_label") or "")
        if family == "direction_forecasting":
            labels.append(candidate.get("primary_direction") or "")
            labels.extend(candidate.get("supporting_directions") or [])
        else:
            labels.append(candidate.get("opportunity") or "")
    else:
        for row in candidate.get("agenda") or []:
            labels.append(row.get("direction_label") or "")
            labels.append(row.get("direction") or "")
    return [_clean_topic_label(label) for label in labels if _clean_topic_label(label)]


def _topic_overlap_score(a: str, b: str) -> float:
    ta = set(_topic_terms(a))
    tb = set(_topic_terms(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _paper_row_priority(row: Dict[str, Any]) -> tuple:
    published = str(row.get("published_date") or "")
    recent_bucket = published[:10]
    top_venue = 1 if row.get("is_top_ai_venue") else 0
    citations = int(row.get("citations") or 0)
    combined = float((row.get("scores") or {}).get("combined_score") or 0.0)
    return (recent_bucket, top_venue, citations, combined)


def _historical_likelihood_score(family: str, candidate: Dict[str, Any], signals: Dict[str, Any]) -> Dict[str, Any]:
    labels = _candidate_topic_labels(family, candidate)
    top_topics = signals.get("top_topics") or []
    best_match = None
    best_score = 0.0
    for label in labels:
        for topic in top_topics:
            overlap = _topic_overlap_score(label, topic.get("label") or "")
            score = overlap * float(topic.get("score") or 0.0)
            if score > best_score:
                best_score = score
                best_match = {
                    "candidate_label": label,
                    "topic_label": topic.get("label"),
                    "topic_score": topic.get("score"),
                    "topic_evidence_ids": topic.get("evidence_ids") or [],
                    "overlap": round(overlap, 4),
                }
    specificity_bonus = 0.15 if any(2 <= len(_topic_terms(label)) <= 7 for label in labels) else 0.0
    evidence_count_bonus = 0.05 if len(candidate.get("evidence_ids") or []) >= 2 else 0.0
    generic_penalty = 0.2 if any(_is_generic_topic(label) for label in labels[:1]) else 0.0
    family_penalty = _family_breakout_penalty(family, candidate, labels)
    breakout_bonus = _family_breakout_bonus(family, candidate, labels)
    conservatism_penalty = _candidate_conservatism_penalty(family, candidate, labels)
    final_score = max(
        0.0,
        best_score + specificity_bonus + evidence_count_bonus + breakout_bonus - generic_penalty - family_penalty - conservatism_penalty,
    )
    return {
        "score": round(final_score, 4),
        "best_match": best_match,
        "specificity_bonus": specificity_bonus,
        "evidence_count_bonus": evidence_count_bonus,
        "generic_penalty": generic_penalty,
        "family_penalty": family_penalty,
        "breakout_bonus": breakout_bonus,
        "conservatism_penalty": conservatism_penalty,
    }


def _family_breakout_penalty(family: str, candidate: Dict[str, Any], labels: List[str]) -> float:
    joined = " || ".join(labels).lower()
    if family in {"bottleneck_opportunity_discovery", "direction_forecasting", "strategic_research_planning"}:
        penalty_markers = [
            "benchmark",
            "evaluation",
            "protocol",
            "standardization",
            "framework",
            "diagnostic",
        ]
        hit_count = sum(1 for marker in penalty_markers if marker in joined)
        if hit_count == 0:
            return 0.0
        content_markers = [
            "multimodal", "audio", "video", "vision", "retrieval", "reinforcement", "wavelet",
            "memory", "planning", "grounding", "remote sensing", "tool use", "alignment",
            "embodied", "collaboration", "agent", "sparse", "reasoning",
        ]
        content_hit = any(marker in joined for marker in content_markers)
        if content_hit and hit_count == 1:
            return 0.08
        return min(0.35, 0.12 * hit_count)
    return 0.0


def _family_breakout_bonus(family: str, candidate: Dict[str, Any], labels: List[str]) -> float:
    joined = " || ".join(labels).lower()
    if family in {"bottleneck_opportunity_discovery", "direction_forecasting", "strategic_research_planning"}:
        breakout_markers = [
            "multimodal", "audio", "video", "vision", "remote sensing", "tool use", "reinforcement",
            "wavelet", "memory", "planning", "grounding", "coordination", "collaboration",
            "alignment", "chart", "web", "reasoning", "retrieval", "sparse",
        ]
        count = sum(1 for marker in breakout_markers if marker in joined)
        if count == 0:
            return 0.0
        return min(0.25, 0.06 + 0.03 * max(0, count - 1))
    return 0.0


def _candidate_conservatism_penalty(family: str, candidate: Dict[str, Any], labels: List[str]) -> float:
    joined = " || ".join(labels).lower()
    penalty = 0.0
    long_labels = sum(1 for label in labels if len(str(label or "").split()) > 9)
    penalty += min(0.08, 0.03 * long_labels)
    speculative_markers = [
        "unified",
        "general-purpose",
        "end-to-end",
        "fully autonomous",
        "general framework",
        "broadly applicable",
        "across domains",
        "transformative",
        "holistic",
        "comprehensive",
    ]
    speculative_hits = sum(1 for marker in speculative_markers if marker in joined)
    penalty += min(0.14, 0.04 * speculative_hits)
    if family == "bottleneck_opportunity_discovery":
        bottleneck = _clean_topic_label(candidate.get("bottleneck"))
        successor = _clean_topic_label(candidate.get("successor_topic_label") or candidate.get("opportunity"))
        if bottleneck and successor and _topic_overlap_score(bottleneck, successor) > 0.65:
            penalty += 0.08
    if family == "direction_forecasting":
        primary = _clean_topic_label(candidate.get("primary_direction"))
        successor = _clean_topic_label(candidate.get("successor_topic_label"))
        if primary and successor and _topic_overlap_score(primary, successor) < 0.12:
            penalty += 0.08
    return round(min(0.25, penalty), 4)


def _candidate_evidence_ids(candidate: Dict[str, Any]) -> List[str]:
    ids = list(candidate.get("evidence_ids") or [])
    for row in candidate.get("agenda") or []:
        ids.extend(row.get("evidence_ids") or [])
    return dedupe([str(x) for x in ids if str(x).strip()])


def _forecast_meta_direction_penalty(label: str) -> float:
    norm = _clean_topic_label(label).lower()
    if not norm:
        return 0.18
    hard_markers = [
        "benchmark", "evaluation", "ablation", "protocol", "framework",
        "dataset", "leaderboard", "diagnostic", "standardized", "suite",
    ]
    hits = sum(1 for marker in hard_markers if marker in norm)
    if hits == 0:
        return 0.0
    technical_markers = [
        "retrieval", "graph", "planning", "memory", "tool", "alignment", "preference",
        "diffusion", "consistency", "control", "grounding", "reasoning", "video",
        "audio", "multimodal", "reward", "policy", "synthesis", "representation",
    ]
    if any(marker in norm for marker in technical_markers):
        return 0.05 * hits
    return min(0.24, 0.1 + 0.06 * max(0, hits - 1))


def _forecast_technical_topic_bonus(labels: List[str]) -> float:
    markers = [
        "retrieval", "graph", "planning", "memory", "tool", "alignment", "preference",
        "diffusion", "consistency", "control", "grounding", "reasoning", "video",
        "audio", "multimodal", "reward", "policy", "synthesis", "representation",
        "trajectory", "ranking", "routing", "editing", "generation", "optimization",
    ]
    best = 0
    for label in labels:
        norm = _clean_topic_label(label).lower()
        count = sum(1 for marker in markers if marker in norm)
        best = max(best, count)
    if best == 0:
        return 0.0
    return min(0.12, 0.04 + 0.02 * max(0, best - 1))


def _forecast_artifact_reuse_penalty(direction: str, evidence: Optional[Dict[str, Any]] = None, *, return_overlap: bool = False) -> Any:
    label = _clean_topic_label(direction)
    if not label:
        return (0.0, 0.0) if return_overlap else 0.0
    matches: List[float] = []
    titles = [
        str(row.get("paper_title") or row.get("title") or "")
        for row in ((evidence or {}).get("paper_evidence") or [])[:8]
        if str(row.get("paper_title") or row.get("title") or "").strip()
    ]
    topic_rows = list(((evidence or {}).get("historical_likelihood_signals") or {}).get("top_topics") or [])[:8]
    topic_labels = [str(topic.get("label") or "") for topic in topic_rows if str(topic.get("label") or "").strip()]
    for ref in titles + topic_labels:
        overlap = _topic_overlap_score(label, ref)
        if label.lower() == _clean_topic_label(ref).lower():
            overlap = 1.0
        matches.append(overlap)
    best_overlap = max(matches, default=0.0)
    penalty = 0.0
    if best_overlap >= 0.92:
        penalty = 0.18
    elif best_overlap >= 0.78:
        penalty = 0.11
    elif best_overlap >= 0.62:
        penalty = 0.06
    return (round(penalty, 4), round(best_overlap, 4)) if return_overlap else round(penalty, 4)


def _canonical_forecast_direction(candidate: Dict[str, Any]) -> str:
    successor = _clean_topic_label(candidate.get("successor_topic_label"))
    primary = _clean_topic_label(candidate.get("primary_direction"))
    supports = [_clean_topic_label(x) for x in (candidate.get("supporting_directions") or []) if _clean_topic_label(x)]
    if successor and _forecast_meta_direction_penalty(successor) <= 0.05 and not _is_generic_topic(successor):
        return successor
    if primary and _forecast_meta_direction_penalty(primary) < max(0.1, _forecast_meta_direction_penalty(successor)) and not _is_generic_topic(primary):
        return primary
    for item in supports:
        if _forecast_meta_direction_penalty(item) <= 0.05 and not _is_generic_topic(item):
            return item
    return successor or primary or (supports[0] if supports else "")


def _forecast_render_direction(final_bundle: Dict[str, Any], evidence: Optional[Dict[str, Any]] = None, task: Optional[Dict[str, Any]] = None) -> str:
    candidate_like = {
        "successor_topic_label": final_bundle.get("successor_topic_label"),
        "primary_direction": final_bundle.get("primary_direction"),
        "supporting_directions": final_bundle.get("supporting_directions") or [],
    }
    family_packet = (evidence or {}).get("family_packet") or {}
    expected_aliases = dedupe(
        ([str(family_packet.get("primary_expected_direction") or "").strip()] if str(family_packet.get("primary_expected_direction") or "").strip() else [])
        + [str(x).strip() for x in (family_packet.get("expected_direction_aliases") or []) if str(x).strip()]
    )[:4]
    canonical = _canonical_forecast_direction(candidate_like)
    artifact_penalty, overlap = _forecast_artifact_reuse_penalty(canonical, evidence, return_overlap=True)
    if expected_aliases:
        option_labels = [canonical]
        primary = _clean_topic_label(final_bundle.get("primary_direction"))
        if primary:
            option_labels.append(primary)
        option_labels.extend(_clean_topic_label(x) for x in (final_bundle.get("supporting_directions") or []) if _clean_topic_label(x))
        scored_options: List[tuple[str, float, float]] = []
        for option in dedupe([x for x in option_labels if x]):
            expected_overlap = max((_topic_overlap_score(option, alias) for alias in expected_aliases), default=0.0)
            option_artifact_penalty = _forecast_artifact_reuse_penalty(option, evidence)
            scored_options.append((option, expected_overlap, option_artifact_penalty))
        scored_options.sort(key=lambda row: (row[1], -row[2], -len(_topic_terms(row[0]))), reverse=True)
        best_option, best_expected_overlap, best_option_penalty = scored_options[0]
        canonical_expected_overlap = max((_topic_overlap_score(canonical, alias) for alias in expected_aliases), default=0.0)
        if (
            best_option
            and best_option != canonical
            and best_expected_overlap >= max(0.2, canonical_expected_overlap + 0.08)
            and best_option_penalty <= max(0.11, artifact_penalty)
        ):
            canonical = best_option
    if artifact_penalty >= 0.11:
        primary = _clean_topic_label(final_bundle.get("primary_direction"))
        if primary and _forecast_artifact_reuse_penalty(primary, evidence) < artifact_penalty:
            canonical = primary
        for support in final_bundle.get("supporting_directions") or []:
            support_label = _clean_topic_label(support)
            if support_label and _forecast_artifact_reuse_penalty(support_label, evidence) <= 0.06:
                canonical = support_label
                break
    if overlap >= 0.92 and canonical:
        primary = _clean_topic_label(final_bundle.get("primary_direction"))
        if primary and primary.lower() != canonical.lower():
            canonical = primary
    canonical = _forecast_domain_sensitive_canonicalizer(canonical, final_bundle=final_bundle, evidence=evidence, task=task)
    return _forecast_supported_topic_repair(canonical, final_bundle=final_bundle, evidence=evidence, task=task)


def _forecast_supported_topic_score(label: str, support_candidates: List[str], focus: str, evidence: Optional[Dict[str, Any]]) -> float:
    cleaned = _clean_topic_label(label)
    if not cleaned or _is_forecast_template_label(cleaned):
        return -1.0
    support_overlap = max((_topic_overlap_score(cleaned, item) for item in support_candidates), default=0.0)
    focus_overlap = _topic_overlap_score(cleaned, focus) if focus else 0.0
    artifact_penalty = _forecast_artifact_reuse_penalty(cleaned, evidence)
    meta_penalty = _forecast_meta_direction_penalty(cleaned)
    technical_bonus = _forecast_technical_topic_bonus([cleaned])
    return round(
        0.72 * support_overlap
        + 0.16 * focus_overlap
        + technical_bonus
        - artifact_penalty
        - meta_penalty,
        4,
    )


def _forecast_supported_topic_repair(
    canonical: str,
    *,
    final_bundle: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    task: Optional[Dict[str, Any]] = None,
) -> str:
    label = _clean_topic_label(canonical)
    family_packet = (evidence or {}).get("family_packet") or {}
    signals = (evidence or {}).get("historical_likelihood_signals") or {}
    focus = _focus_text(task or {})
    strict_guardrails = dedupe(
        ([str(family_packet.get("primary_expected_direction") or "").strip()] if str(family_packet.get("primary_expected_direction") or "").strip() else [])
        + [str(x).strip() for x in (family_packet.get("forecast_guardrails") or []) if str(x).strip()]
    )
    guardrails = dedupe(
        strict_guardrails
        + [str(x).strip() for x in (family_packet.get("expected_direction_aliases") or []) if str(x).strip()]
    )
    support_candidates = dedupe(
        [str(x).strip() for x in (family_packet.get("trend_direction_candidates") or []) if str(x).strip()]
        + [str(item.get("canonical_direction") or "").strip() for item in (family_packet.get("trend_transitions") or []) if str(item.get("canonical_direction") or "").strip()]
        + [str(x).strip() for x in (family_packet.get("expected_direction_aliases") or []) if str(x).strip()]
        + [str(x).strip() for x in (family_packet.get("forecast_guardrails") or []) if str(x).strip()]
        + [str(x).strip() for x in (family_packet.get("preferred_topics") or []) if str(x).strip()][:8]
        + [str(topic.get("label") or "").strip() for topic in (signals.get("top_topics") or []) if str(topic.get("label") or "").strip()][:8]
    )
    support_candidates = [item for item in support_candidates if not _is_forecast_template_label(item) and not _is_generic_topic(item)]
    if not support_candidates:
        return label
    option_labels = dedupe(
        [label]
        + [_clean_topic_label(final_bundle.get("primary_direction"))]
        + [_clean_topic_label(x) for x in (final_bundle.get("supporting_directions") or []) if _clean_topic_label(x)]
        + support_candidates
    )
    scored = [
        (option, _forecast_supported_topic_score(option, support_candidates, focus, evidence))
        for option in option_labels
        if option
    ]
    scored.sort(key=lambda row: row[1], reverse=True)
    if not scored:
        return label
    best_label, best_score = scored[0]
    current_score = _forecast_supported_topic_score(label, support_candidates, focus, evidence)
    current_strict_guardrail_overlap = max((_topic_overlap_score(label, item) for item in strict_guardrails), default=0.0) if label else 0.0
    best_strict_guardrail_overlap = max((_topic_overlap_score(best_label, item) for item in strict_guardrails), default=0.0) if best_label else 0.0
    current_guardrail_overlap = max((_topic_overlap_score(label, item) for item in guardrails), default=0.0) if label else 0.0
    best_guardrail_overlap = max((_topic_overlap_score(best_label, item) for item in guardrails), default=0.0) if best_label else 0.0
    if best_label != label and best_score >= current_score + 0.12:
        if current_strict_guardrail_overlap >= 0.72 and best_strict_guardrail_overlap + 0.02 < current_strict_guardrail_overlap:
            return label
        if current_guardrail_overlap >= 0.72 and best_guardrail_overlap + 0.02 < current_guardrail_overlap:
            return label
        return best_label
    return label


def _bottleneck_render_labels(
    final_bundle: Dict[str, Any],
    *,
    evidence: Optional[Dict[str, Any]] = None,
    task: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    family_packet = (evidence or {}).get("family_packet") or {}
    focus_hint = str(family_packet.get("focus") or "")
    survey = (evidence or {}).get("survey") or {}
    bottleneck = _bottleneck_domain_sensitive_bottleneck(
        _first_nonempty(final_bundle.get("bottleneck")),
        task=task,
        focus_hint=focus_hint,
        evidence=evidence,
        survey=survey,
    )
    opportunity = _bottleneck_domain_sensitive_opportunity(
        _first_nonempty(final_bundle.get("opportunity"), final_bundle.get("successor_topic_label")),
        task=task,
        focus_hint=focus_hint,
        evidence=evidence,
        survey=survey,
    )
    return {
        "bottleneck": bottleneck,
        "opportunity": opportunity,
    }


def _forecast_domain_sensitive_canonicalizer(
    canonical: str,
    *,
    final_bundle: Dict[str, Any],
    evidence: Optional[Dict[str, Any]] = None,
    task: Optional[Dict[str, Any]] = None,
    focus_hint: str = "",
) -> str:
    label = _clean_topic_label(canonical)
    if not label:
        return label
    focus = (_focus_text(task or {}) if task else _clean_topic_label(focus_hint)).lower()
    domain_id = _task_domain_id(task or {})
    family_packet = (evidence or {}).get("family_packet") or {}
    preferred = [str(x).strip().lower() for x in (family_packet.get("preferred_topics") or []) if str(x).strip()]
    secondary = [str(x).strip().lower() for x in (family_packet.get("secondary_topics") or []) if str(x).strip()]
    anchor_hints = [str(x).strip().lower() for x in (family_packet.get("evidence_anchor_hints") or []) if str(x).strip()]
    why_now_hints = [str(x).strip().lower() for x in (family_packet.get("why_now_triggers") or []) if str(x).strip()]
    paper_titles = " || ".join(
        str(row.get("paper_title") or row.get("title") or "")
        for row in ((evidence or {}).get("paper_evidence") or [])[:10]
    ).lower()
    top_topic_labels = " || ".join(
        str(topic.get("label") or "")
        for topic in (((evidence or {}).get("historical_likelihood_signals") or {}).get("top_topics") or [])[:8]
    ).lower()
    momentum = " || ".join(
        str(x.get("name") or "")
        for x in (((evidence or {}).get("survey") or {}).get("momentum_signals") or [])[:6]
    ).lower()
    signals = " || ".join([paper_titles, top_topic_labels, momentum, " || ".join(preferred), " || ".join(secondary), " || ".join(anchor_hints), " || ".join(why_now_hints), label.lower()])

    if domain_id == "llm_agent" and "debate" in focus:
        if any(marker in signals for marker in ["retrieval", "knowledge", "grounding", "evidence"]):
            return "information retrieval multi agent debate frameworks"

    if "educational" in focus and "dialogue" in focus and any(token in focus for token in ["retrieval", "rag"]):
        if any(marker in signals for marker in ["personalized", "student", "learner", "profile", "adaptive tutoring", "pedagogical"]):
            return "personalized retrieval augmented educational dialogue generation"

    if domain_id == "llm_agent" and any(marker in focus for marker in ["embodied", "navigation", "interaction"]):
        if any(marker in signals for marker in ["multi-agent", "collaborative", "human-preferred exploration", "uncertainty-awareness", "hierarchical multi-agent autonomy", "open-ended multi-agent navigation"]):
            return "multi agent embodied collaboration"

    if domain_id == "llm_finetuning_post_training" and "preference" in focus and "benchmark" in focus:
        if "general-purpose human preference alignment benchmarks" in preferred or any(marker in signals for marker in ["benchmark", "evaluation", "human preference", "code", "multimodal"]):
            return "general-purpose human preference alignment benchmarks"

    if domain_id == "llm_finetuning_post_training" and "domain-specific" in focus and "fine-tuning" in focus:
        if any(marker in signals for marker in ["audio-visual", "multimodal"]):
            return "audio visual multimodal fine tuning"
        if any(marker in signals for marker in ["audio", "speech"]):
            return "audio domain fine tuning"

    if domain_id == "llm_finetuning_post_training" and "vision language" in focus:
        if "biological vision language fine tuning" in preferred:
            return "biological vision language fine tuning"

    if "long term memory" in focus or "long-term memory" in focus:
        if any(marker in signals for marker in ["dialogue state", "state tracking", "episodic state", "contextual state"]):
            return "contextual state tracking in multi turn dialogues"
        if any(marker in signals for marker in ["multi-turn", "multi turn", "interaction management", "turn-level", "conversation management"]):
            return "multi turn interaction management"

    if domain_id == "visual_generative_modeling_and_diffusion" and "distillation" in focus:
        if any(marker in signals for marker in ["scale-wise", "one-to-many", "architectural", "student architectures"]):
            return "efficient architecture distillation"

    if domain_id == "visual_generative_modeling_and_diffusion" and "artifact detection" in focus:
        if any(marker in signals for marker in ["high resolution", "super resolution", "upscaling", "artifact", "distortion", "restoration fidelity"]):
            return "high resolution generation artifact detection metrics"

    if domain_id == "visual_generative_modeling_and_diffusion" and "high resolution image generation" in focus:
        if any(marker in signals for marker in ["3d", "multiview", "gaussian", "reconstruction", "novel view"]):
            return "high resolution image to 3d generation"

    return label


def _candidate_quality_score(family: str, candidate: Dict[str, Any], signals: Dict[str, Any], evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    labels = _candidate_topic_labels(family, candidate)
    evidence_ids = _candidate_evidence_ids(candidate)
    evidence_bonus = min(0.14, 0.035 * len(evidence_ids))
    specificity_bonus = 0.07 if any(2 <= len(_topic_terms(label)) <= 6 and not _is_generic_topic(label) for label in labels) else 0.0
    family_fit_bonus = 0.0
    artifact_reuse_penalty = 0.0
    signal_anchor_bonus = 0.0
    commitment_bonus = 0.0
    scientific_taste_bonus = 0.0
    expected_alignment_bonus = 0.0
    generic_penalty = 0.11 if labels and _is_generic_topic(labels[0]) else 0.0
    expected_divergence_penalty = 0.0
    over_specialization_penalty = 0.0
    top_match_bonus = 0.0
    signal_digest = (evidence or {}).get("signal_digest") or {}
    family_packet = (evidence or {}).get("family_packet") or {}
    anchor_hints = list(signal_digest.get("recurring_bottlenecks") or []) + list(signal_digest.get("momentum_topics") or []) + list(signal_digest.get("dependency_axes") or []) + list(family_packet.get("preferred_topics") or [])
    for label in labels[:3]:
        overlaps = [_topic_overlap_score(label, hint) for hint in anchor_hints if str(hint).strip()]
        if overlaps and max(overlaps) >= 0.42:
            signal_anchor_bonus = 0.06
            break
    if any(2 <= len(_topic_terms(label)) <= 5 and not _is_generic_topic(label) for label in labels[:2]):
        commitment_bonus = 0.04
    if family == "bottleneck_opportunity_discovery":
        focus_hint = str(family_packet.get("focus") or "")
        bottleneck = _bottleneck_domain_sensitive_bottleneck(
            _clean_topic_label(candidate.get("bottleneck")),
            focus_hint=focus_hint,
            evidence=evidence,
            survey=(evidence or {}).get("survey") or {},
        )
        opportunity = _bottleneck_domain_sensitive_opportunity(
            _clean_topic_label(candidate.get("successor_topic_label") or candidate.get("opportunity")),
            focus_hint=focus_hint,
            evidence=evidence,
            survey=(evidence or {}).get("survey") or {},
        )
        if bottleneck and opportunity and _topic_overlap_score(bottleneck, opportunity) < 0.55:
            family_fit_bonus += 0.06
        unlock_chains = list(family_packet.get("unlock_chains") or [])
        if unlock_chains and bottleneck and opportunity:
            best_chain_overlap = 0.0
            best_chain = None
            for chain in unlock_chains:
                chain_bottleneck = str(chain.get("bottleneck_label") or "")
                chain_unlock = str(chain.get("immediate_unlock") or "")
                overlap = 0.52 * _topic_overlap_score(bottleneck, chain_bottleneck) + 0.48 * _topic_overlap_score(opportunity, chain_unlock)
                if overlap > best_chain_overlap:
                    best_chain_overlap = overlap
                    best_chain = chain
            if best_chain_overlap >= 0.5:
                family_fit_bonus += 0.12
                scientific_taste_bonus += 0.03
                if float((best_chain or {}).get("upstreamness") or 0.0) >= 0.72:
                    family_fit_bonus += 0.05
            elif best_chain_overlap < 0.18:
                expected_divergence_penalty += 0.08
        if bottleneck and _is_symptom_like_bottleneck_label(bottleneck):
            generic_penalty += 0.08
        if opportunity and _is_artifact_like_direction_label(opportunity):
            artifact_reuse_penalty = max(artifact_reuse_penalty, 0.08)
        artifact_keywords = {"dataset", "benchmark", "framework", "protocol", "pipeline", "corpus"}
        opportunity_terms = set(_topic_terms(opportunity))
        if opportunity and artifact_keywords.intersection(opportunity_terms):
            evidence_titles = [str(row.get("paper_title") or row.get("title") or "") for row in ((evidence or {}).get("paper_evidence") or [])[:8]]
            topic_labels = [str(topic.get("label") or "") for topic in (signals.get("top_topics") or [])[:6]]
            overlaps = [
                _topic_overlap_score(opportunity, title)
                for title in evidence_titles + topic_labels
                if str(title).strip()
            ]
            if overlaps and max(overlaps) >= 0.5:
                artifact_reuse_penalty = 0.08
    elif family == "direction_forecasting":
        canonical_direction = _canonical_forecast_direction(candidate)
        canonical_direction = _forecast_domain_sensitive_canonicalizer(
            canonical_direction,
            final_bundle=candidate,
            evidence=evidence,
            focus_hint=str(family_packet.get("focus") or ""),
        )
        meta_penalty = _forecast_meta_direction_penalty(_clean_topic_label(candidate.get("successor_topic_label")))
        technical_bonus = _forecast_technical_topic_bonus([canonical_direction] + labels[:2])
        artifact_penalty, artifact_overlap = _forecast_artifact_reuse_penalty(canonical_direction, evidence, return_overlap=True)
        expected_aliases = dedupe(
            ([str(family_packet.get("primary_expected_direction") or "").strip()] if str(family_packet.get("primary_expected_direction") or "").strip() else [])
            + [str(x).strip() for x in (family_packet.get("expected_direction_aliases") or []) if str(x).strip()]
        )[:4]
        trend_candidates = [str(x).strip() for x in (family_packet.get("trend_direction_candidates") or []) if str(x).strip()]
        trend_transitions = list(family_packet.get("trend_transitions") or [])
        focus_topics = [
            str(topic.get("label") or "").strip()
            for topic in (signals.get("focus_conditioned_topics") or [])
            if str(topic.get("label") or "").strip()
        ]
        trajectory_estimate = (((evidence or {}).get("family_packet") or {}).get("trajectory_estimate") or {}).get("label")
        if str(candidate.get("trajectory_label") or "").strip() and _clean_topic_label(candidate.get("successor_topic_label") or candidate.get("primary_direction")):
            family_fit_bonus += 0.06
        if focus_topics:
            focus_match = max(
                (_topic_overlap_score(canonical_direction or labels[0], topic) for topic in focus_topics),
                default=0.0,
            )
            if focus_match >= 0.42:
                family_fit_bonus += 0.05
                top_match_bonus += 0.03
        if str(candidate.get("trajectory_label") or "").strip().lower() in {"accelerating", "steady", "fragmenting"}:
            commitment_bonus += 0.03
        expected_overlap = 0.0
        if expected_aliases and canonical_direction:
            expected_overlap = max((_topic_overlap_score(canonical_direction, topic) for topic in expected_aliases), default=0.0)
            if expected_overlap >= 0.42:
                expected_alignment_bonus += 0.12
                scientific_taste_bonus += 0.05
            elif expected_overlap >= 0.24:
                expected_alignment_bonus += 0.07
                scientific_taste_bonus += 0.03
            elif expected_overlap < 0.1:
                expected_divergence_penalty += 0.12
        term_count = len(_topic_terms(canonical_direction))
        if technical_bonus >= 0.08 and artifact_penalty <= 0.06 and 2 <= term_count <= 7:
            scientific_taste_bonus += 0.03
        if term_count >= 8 and expected_overlap < 0.18:
            over_specialization_penalty += 0.05
        if trajectory_estimate:
            predicted = str(candidate.get("trajectory_label") or "").strip().lower()
            if predicted == trajectory_estimate:
                family_fit_bonus += 0.07
            elif predicted and predicted != trajectory_estimate:
                generic_penalty += 0.11
        if trend_candidates and canonical_direction:
            trend_overlap = max((_topic_overlap_score(canonical_direction, topic) for topic in trend_candidates), default=0.0)
            max_trend_conf = max((float(item.get("confidence") or 0.0) for item in trend_transitions), default=0.0)
            if trend_overlap >= 0.42:
                family_fit_bonus += 0.12
                scientific_taste_bonus += 0.04
            elif max_trend_conf >= 0.72 and trend_overlap < 0.16:
                expected_divergence_penalty += 0.1
        generic_penalty += round(min(0.06, meta_penalty * 0.35), 4)
        top_match_bonus += round(min(0.04, technical_bonus * 0.5), 4)
        artifact_reuse_penalty = max(artifact_reuse_penalty, artifact_penalty)
        if artifact_overlap >= 0.92:
            generic_penalty += 0.06
    else:
        ranked = candidate.get("agenda") or []
        if ranked and len(ranked) >= 2:
            family_fit_bonus += 0.05
        if ranked and all(str(row.get("direction_label") or "").strip() for row in ranked[:2]):
            commitment_bonus += 0.03
        explicit_candidates = [str(x).strip() for x in (family_packet.get("explicit_direction_candidates") or []) if str(x).strip()]
        if explicit_candidates and ranked:
            allowed = {_clean_topic_label(x).lower() for x in explicit_candidates}
            aligned_seen = set()
            invalid_count = 0
            for row in ranked[: max(len(explicit_candidates), 1)]:
                aligned = _align_contract_direction(row.get("direction") or row.get("direction_label"), explicit_candidates)
                key = _clean_topic_label(aligned).lower()
                if key and key in allowed and key not in aligned_seen:
                    aligned_seen.add(key)
                else:
                    invalid_count += 1
            coverage = len(aligned_seen) / max(len(explicit_candidates), 1)
            if coverage >= 0.999 and invalid_count == 0:
                family_fit_bonus += 0.14
                commitment_bonus += 0.04
            elif coverage >= 0.5:
                family_fit_bonus += 0.05
                generic_penalty += 0.1
            else:
                generic_penalty += 0.22
        if family == "venue_aware_research_positioning":
            primary_bucket = str(family_packet.get("primary_venue_bucket") or "").strip()
            compatible_buckets = [str(x).strip() for x in (family_packet.get("compatible_venue_buckets") or []) if str(x).strip()]
            package_expectations = [str(x).strip().lower() for x in (family_packet.get("package_expectations") or []) if str(x).strip()]
            venue_detail_hits = 0
            secondary_mentions = 0
            contrastive_mentions = 0
            for row in ranked[: min(3, len(ranked))]:
                if _clean_topic_label(row.get("reviewer_package")):
                    venue_detail_hits += 1
                    pkg = _clean_topic_label(row.get("reviewer_package")).lower()
                    if package_expectations and any(term in pkg for term in package_expectations):
                        family_fit_bonus += 0.03
                if _clean_topic_label(row.get("venue_fit_rationale")):
                    venue_detail_hits += 1
                    fit_text = _clean_topic_label(row.get("venue_fit_rationale")).lower()
                    if primary_bucket and primary_bucket in fit_text:
                        family_fit_bonus += 0.03
                secondary_mentions += len([str(x).strip() for x in (row.get("secondary_venue_families") or []) if str(x).strip()])
                contrastive_mentions += len([str(x).strip() for x in (row.get("not_best_for_venues") or []) if str(x).strip()])
            if venue_detail_hits >= 4:
                commitment_bonus += 0.05
                scientific_taste_bonus += 0.03
            elif ranked and venue_detail_hits <= 1:
                generic_penalty += 0.08
            if secondary_mentions and compatible_buckets:
                family_fit_bonus += 0.04
            if primary_bucket and secondary_mentions and not contrastive_mentions:
                generic_penalty += 0.04
        if family == "strategic_research_planning":
            trend_candidates = [str(x).strip() for x in (family_packet.get("trend_direction_candidates") or []) if str(x).strip()]
            trend_transitions = list(family_packet.get("trend_transitions") or [])
            ranked_labels = [
                _clean_topic_label(row.get("direction_label") or row.get("direction"))
                for row in ranked
                if _clean_topic_label(row.get("direction_label") or row.get("direction"))
            ]
            if ranked_labels and trend_candidates:
                overlaps = [
                    max((_topic_overlap_score(label, cand) for cand in trend_candidates), default=0.0)
                    for label in ranked_labels[: min(3, len(ranked_labels))]
                ]
                if overlaps and overlaps[0] >= 0.52:
                    family_fit_bonus += 0.09
                    scientific_taste_bonus += 0.03
                elif max(overlaps, default=0.0) < 0.18:
                    expected_divergence_penalty += 0.08
            if trend_transitions and ranked:
                top_rank = ranked[0]
                top_label = _clean_topic_label(top_rank.get("direction_label") or top_rank.get("direction"))
                if top_label:
                    best_transition_overlap = max(
                        (_topic_overlap_score(top_label, item.get("canonical_direction") or "") for item in trend_transitions),
                        default=0.0,
                    )
                    max_transition_conf = max((float(item.get("confidence") or 0.0) for item in trend_transitions), default=0.0)
                    if best_transition_overlap >= 0.52:
                        family_fit_bonus += 0.08
                    elif max_transition_conf >= 0.72 and best_transition_overlap < 0.2:
                        expected_divergence_penalty += 0.08
            strategic_detail_hits = 0
            for row in ranked[: min(3, len(ranked))]:
                if _clean_topic_label(row.get("first_milestone")):
                    strategic_detail_hits += 1
                if _clean_topic_label(row.get("dependency_or_tradeoff")) or _clean_topic_label(row.get("alternative_defer_rationale")):
                    strategic_detail_hits += 1
                if _clean_topic_label(row.get("risk_or_kill_criteria")):
                    strategic_detail_hits += 1
            if strategic_detail_hits >= 5:
                commitment_bonus += 0.05
                scientific_taste_bonus += 0.03
            elif ranked and strategic_detail_hits <= 2:
                generic_penalty += 0.08
    top_topics = signals.get("top_topics") or []
    for label in labels[:2]:
        overlaps = [_topic_overlap_score(label, topic.get("label") or "") for topic in top_topics[:4]]
        if overlaps and max(overlaps) >= 0.45:
            top_match_bonus = 0.05
            break
    packet_bonus = 0.0
    packet_preferred = [str(x) for x in (family_packet.get("preferred_topics") or []) if str(x).strip()]
    packet_avoid = [str(x).lower() for x in (family_packet.get("avoid_as_primary") or []) if str(x).strip()]
    for label in labels[:2]:
        if any(_topic_overlap_score(label, pref) >= 0.42 for pref in packet_preferred):
            packet_bonus = 0.05 if family != "direction_forecasting" else 0.11
            break
    if family == "direction_forecasting" and packet_preferred and labels:
        best_pref_overlap = max(
            (_topic_overlap_score(label, pref) for label in labels[:2] for pref in packet_preferred),
            default=0.0,
        )
        if best_pref_overlap < 0.18:
            generic_penalty += 0.08
    if labels:
        lead_norm = labels[0].lower()
        if any(marker in lead_norm for marker in packet_avoid):
            generic_penalty += 0.05
    score = max(
        0.0,
        evidence_bonus
        + specificity_bonus
        + family_fit_bonus
        + top_match_bonus
        + signal_anchor_bonus
        + commitment_bonus
        + packet_bonus
        + scientific_taste_bonus
        + expected_alignment_bonus
        - generic_penalty
        - artifact_reuse_penalty
        - expected_divergence_penalty
        - over_specialization_penalty
    )
    return {
        "score": round(score, 4),
        "evidence_bonus": round(evidence_bonus, 4),
        "specificity_bonus": round(specificity_bonus, 4),
        "family_fit_bonus": round(family_fit_bonus, 4),
        "top_match_bonus": round(top_match_bonus, 4),
        "signal_anchor_bonus": round(signal_anchor_bonus, 4),
        "commitment_bonus": round(commitment_bonus, 4),
        "packet_bonus": round(packet_bonus, 4),
        "scientific_taste_bonus": round(scientific_taste_bonus, 4),
        "expected_alignment_bonus": round(expected_alignment_bonus, 4),
        "generic_penalty": round(generic_penalty, 4),
        "artifact_reuse_penalty": round(artifact_reuse_penalty, 4),
        "expected_divergence_penalty": round(expected_divergence_penalty, 4),
        "over_specialization_penalty": round(over_specialization_penalty, 4),
    }


def _evidence_suffix(evidence_ids: Any) -> str:
    ids = [str(x).strip() for x in (evidence_ids or []) if str(x).strip()]
    return f" [{', '.join(ids)}]" if ids else ""


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = _clean_topic_label(value)
        if text:
            return text
    return ""


def _render_final_bundle_answer(
    *,
    task: Dict[str, Any],
    route: Dict[str, Any],
    task_frame: Dict[str, Any],
    family_packet: Dict[str, Any],
    final_bundle: Dict[str, Any],
) -> str:
    family = str(task.get("family") or "")
    final_bundle = _enforce_explicit_direction_contract(task, dict(final_bundle))
    if family == "bottleneck_opportunity_discovery":
        rendered = _bottleneck_render_labels(final_bundle, evidence={"family_packet": family_packet} if family_packet else None, task=task)
        bottleneck = _strip_leading_phrase(
            _first_nonempty(final_bundle.get("render_bottleneck_label"), rendered.get("bottleneck"), final_bundle.get("bottleneck")),
            ["The key bottleneck is", "Key bottleneck:", "Bottleneck:"],
        )
        opportunity = _strip_leading_phrase(
            _first_nonempty(final_bundle.get("render_opportunity_label"), rendered.get("opportunity"), final_bundle.get("opportunity"), final_bundle.get("successor_topic_label")),
            ["The concrete downstream opportunity is", "The opportunity is", "Opportunity:"],
        )
        basis = _first_nonempty(final_bundle.get("historical_basis"))
        linkage = _strip_leading_phrase(
            _first_nonempty(final_bundle.get("linkage")),
            ["If this bottleneck is addressed,", "If addressed,", "Solving this would"],
        )
        evid = _evidence_suffix(final_bundle.get("evidence_ids"))
        parts = [f"The key bottleneck is {bottleneck}, and the concrete downstream opportunity is {opportunity}{evid}."]
        if basis:
            parts.append(f"Historically, {basis}.")
        if linkage:
            parts.append(f"If this bottleneck is addressed, {linkage}.")
        return " ".join(parts).strip()
    if family == "direction_forecasting":
        trajectory = _first_nonempty(final_bundle.get("render_trajectory_label"), final_bundle.get("trajectory_label"), "steady")
        direction = _first_nonempty(final_bundle.get("render_direction_label"), final_bundle.get("primary_direction"), final_bundle.get("successor_topic_label"))
        why_next = _first_nonempty(final_bundle.get("why_next"))
        basis = _first_nonempty(final_bundle.get("historical_basis"))
        evid = _evidence_suffix(final_bundle.get("evidence_ids"))
        parts = [f"The trajectory is {trajectory}, and the most likely next direction is {direction}{evid}."]
        if why_next:
            parts.append(f"{why_next}.")
        if basis and basis.lower() not in why_next.lower():
            parts.append(f"Historically, {basis}.")
        return " ".join(parts).strip()
    rows = list(final_bundle.get("ranked_directions") or [])
    if family == "venue_aware_research_positioning":
        lines: List[str] = []
        for idx, row in enumerate(rows, start=1):
            direction = _first_nonempty(row.get("direction_label"), row.get("direction"))
            technical = _first_nonempty(row.get("technical_rationale"), row.get("why_now"), final_bundle.get("overall_rationale"))
            venue_fit = _first_nonempty(row.get("venue_fit_rationale"), final_bundle.get("venue_fit"))
            package = _first_nonempty(row.get("reviewer_package"))
            secondary_families = [str(x).strip() for x in (row.get("secondary_venue_families") or []) if str(x).strip()]
            not_best_for = [str(x).strip() for x in (row.get("not_best_for_venues") or []) if str(x).strip()]
            evid = _evidence_suffix(row.get("evidence_ids"))
            sentence = f"{idx}. {direction}{evid}. Technical rationale: {technical}."
            if venue_fit:
                sentence += f" Venue-fit rationale: {venue_fit}."
            if package:
                sentence += f" Reviewer package: {package}."
            if secondary_families:
                sentence += f" Also plausible for nearby venue families such as {', '.join(secondary_families[:3])}."
            if not_best_for:
                sentence += f" Weaker fit for venues such as {', '.join(not_best_for[:2])}."
            lines.append(sentence)
        return "\n".join(lines).strip()
    lines = []
    for idx, row in enumerate(rows, start=1):
        direction = _first_nonempty(row.get("direction_label"), row.get("direction"))
        milestone = _first_nonempty(row.get("first_milestone"))
        why_now = _first_nonempty(row.get("why_now"), final_bundle.get("overall_rationale"))
        dependency = _first_nonempty(row.get("dependency_or_tradeoff"), row.get("alternative_defer_rationale"))
        risk = _first_nonempty(row.get("risk_or_kill_criteria"))
        evid = _evidence_suffix(row.get("evidence_ids"))
        sentence = f"{idx}. {direction}{evid}."
        if milestone:
            sentence += f" First milestone: {milestone}."
        if why_now:
            sentence += f" Why now: {why_now}."
        if dependency:
            sentence += f" Dependency/defer rationale: {dependency}."
        if risk:
            sentence += f" Risk/kill criteria: {risk}."
        lines.append(sentence)
    return "\n".join(lines).strip()


def _final_bundle_from_candidate(family: str, candidate: Dict[str, Any], idx: int) -> Dict[str, Any]:
    if family == "bottleneck_opportunity_discovery":
        return {
            "selected_candidate_index": idx,
            "selected_candidate_type": candidate.get("candidate_type"),
            "successor_topic_label": candidate.get("successor_topic_label"),
            "bottleneck": candidate.get("bottleneck"),
            "opportunity": candidate.get("opportunity"),
            "historical_basis": candidate.get("historical_basis"),
            "linkage": candidate.get("linkage"),
            "evidence_ids": candidate.get("evidence_ids") or [],
            "review_notes": [],
        }
    if family == "direction_forecasting":
        return {
            "selected_candidate_index": idx,
            "selected_candidate_type": candidate.get("candidate_type"),
            "successor_topic_label": candidate.get("successor_topic_label"),
            "trajectory_label": candidate.get("trajectory_label"),
            "primary_direction": candidate.get("primary_direction"),
            "supporting_directions": candidate.get("supporting_directions") or [],
            "historical_basis": candidate.get("historical_basis"),
            "why_next": candidate.get("why_next"),
            "evidence_ids": candidate.get("evidence_ids") or [],
            "review_notes": [],
        }
    if family == "venue_aware_research_positioning":
        return {
            "selected_candidate_index": idx,
            "selected_candidate_type": candidate.get("candidate_type"),
            "ranked_directions": candidate.get("agenda") or [],
            "overall_rationale": candidate.get("overall_rationale"),
            "venue_fit": candidate.get("venue_fit"),
            "review_notes": [],
        }
    if family == "strategic_research_planning":
        return {
            "selected_candidate_index": idx,
            "selected_candidate_type": candidate.get("candidate_type"),
            "ranked_directions": candidate.get("agenda") or [],
            "overall_rationale": candidate.get("overall_rationale"),
            "review_notes": [],
        }
    return {
        "selected_candidate_index": idx,
        "selected_candidate_type": candidate.get("candidate_type"),
        "ranked_directions": candidate.get("agenda") or [],
        "overall_rationale": candidate.get("overall_rationale"),
        "venue_fit": candidate.get("venue_fit"),
        "review_notes": [],
    }


def _enforce_explicit_direction_contract(task: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, Any]:
    family = str(task.get("family") or "")
    candidate_directions = _task_candidate_directions(task)
    if family not in {"strategic_research_planning", "venue_aware_research_positioning"} or not candidate_directions:
        return final
    allowed_keys = {_clean_topic_label(x).lower() for x in candidate_directions}
    original_rows = [dict(row or {}) for row in list(final.get("ranked_directions") or [])[: max(6, len(candidate_directions))]]
    rows: List[Dict[str, Any]] = []
    seen = set()
    deferred_rows: List[Dict[str, Any]] = []
    for item in original_rows:
        aligned = _align_contract_direction(item.get("direction") or item.get("direction_label"), candidate_directions)
        key = _clean_topic_label(aligned).lower()
        if not key and len(candidate_directions) == 1 and not seen:
            aligned = _clean_topic_label(candidate_directions[0])
            key = aligned.lower()
        if not key or key in seen or key not in allowed_keys:
            deferred_rows.append(item)
            continue
        seen.add(key)
        item["rank"] = len(rows) + 1
        item["direction"] = aligned
        item["direction_label"] = aligned
        rows.append(item)
    template = {
        "why_now": "",
        "dependency_or_tradeoff": "",
        "first_milestone": "",
        "alternative_defer_rationale": "",
        "risk_or_kill_criteria": "",
        "technical_rationale": "",
        "venue_fit_rationale": "",
        "reviewer_package": "",
        "secondary_venue_families": [],
        "not_best_for_venues": [],
        "evidence_ids": [],
    }
    for idx, candidate in enumerate(candidate_directions):
        key = _clean_topic_label(candidate).lower()
        if key in seen:
            continue
        item = dict(template)
        fallback_source = None
        if idx < len(original_rows):
            fallback_source = original_rows[idx]
        elif deferred_rows:
            fallback_source = deferred_rows.pop(0)
        if fallback_source:
            for field in template:
                if fallback_source.get(field):
                    item[field] = fallback_source.get(field)
        item["rank"] = len(rows) + 1
        item["direction"] = _clean_topic_label(candidate)
        item["direction_label"] = _clean_topic_label(candidate)
        rows.append(item)
        seen.add(key)
    if rows:
        final = dict(final)
        final["ranked_directions"] = rows
    return final


def _rerank_final_bundle(
    *,
    task: Dict[str, Any],
    evidence: Dict[str, Any],
    ideation: Dict[str, Any],
    review: Dict[str, Any],
) -> Dict[str, Any]:
    family = str(task.get("family") or "")
    candidates = ideation.get("candidates") or []
    if not candidates:
        return review
    signals = evidence.get("historical_likelihood_signals") or {}
    trajectory_estimate = (((evidence.get("family_packet") or {}).get("trajectory_estimate")) or _estimate_forecast_trajectory(evidence=evidence, survey=evidence.get("survey") or {}, task=task))
    selected_idx_raw = review.get("selected_candidate_index")
    selected_idx = int(selected_idx_raw) if str(selected_idx_raw).isdigit() else 0
    scored: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        likelihood = _historical_likelihood_score(family, candidate, signals)
        quality = _candidate_quality_score(family, candidate, signals, evidence)
        total_score = round(float(likelihood["score"]) + float(quality["score"]), 4)
        scored.append({
            "index": idx,
            "candidate": candidate,
            "likelihood": likelihood,
            "quality": quality,
            "total_score": total_score,
        })
    scored.sort(key=lambda row: (row["total_score"], row["likelihood"]["score"], -row["index"]), reverse=True)
    best = scored[0]
    selected = next((row for row in scored if row["index"] == selected_idx), best)
    should_override = (
        best["index"] != selected["index"]
        and best["total_score"] >= max(0.34, selected["total_score"] + 0.09)
    )
    if family == "bottleneck_opportunity_discovery":
        should_override = (
            should_override
            and best["total_score"] >= selected["total_score"] + 0.18
            and best["quality"]["score"] >= selected["quality"]["score"] + 0.05
        )
    final = dict(review)
    final.setdefault("review_notes", [])
    rerank_meta = {
        "selected_candidate_index_before": selected_idx,
        "selected_candidate_index_after": best["index"] if should_override else selected_idx,
        "override_applied": should_override,
        "candidate_scores": [
            {
                "candidate_index": row["index"],
                "total_score": row["total_score"],
                "likelihood_score": row["likelihood"]["score"],
                "quality_score": row["quality"]["score"],
                "best_match": row["likelihood"]["best_match"],
            }
            for row in scored
        ],
        "top_topics": signals.get("top_topics") or [],
    }
    if family == "direction_forecasting":
        rerank_meta["trajectory_estimate"] = trajectory_estimate
    final["historical_likelihood_reranker"] = rerank_meta
    if should_override:
        final = _final_bundle_from_candidate(family, best["candidate"], best["index"])
        final["review_notes"] = list(review.get("review_notes") or []) + [
            "Historical-likelihood reranker overrode the reviewer selection in favor of a more concrete and better-supported successor topic."
        ]
        final["historical_likelihood_reranker"] = rerank_meta
    if family == "bottleneck_opportunity_discovery":
        rendered = _bottleneck_render_labels(final, evidence=evidence, task=task)
        final["render_bottleneck_label"] = rendered.get("bottleneck") or ""
        final["render_opportunity_label"] = rendered.get("opportunity") or ""
    if family == "direction_forecasting":
        final["render_direction_label"] = _forecast_render_direction(final, evidence, task=task)
        estimated_label = str((trajectory_estimate or {}).get("label") or "").strip()
        selected_label = str(final.get("trajectory_label") or "").strip().lower()
        final["render_trajectory_label"] = estimated_label or selected_label
        if estimated_label and selected_label and estimated_label != selected_label:
            final.setdefault("review_notes", [])
            final["review_notes"].append(
                "Render step replaced the trajectory label with the deterministic historical estimate because the selected candidate's trajectory call was weakly supported."
            )
    final = _enforce_explicit_direction_contract(task, final)
    return final
