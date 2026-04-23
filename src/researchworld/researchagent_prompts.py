from __future__ import annotations

import json
from typing import Any, Dict, List


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _join_lines(lines: List[str]) -> str:
    return "\n".join(line for line in lines if str(line).strip())


_DEFAULT_PROFILE: Dict[str, Any] = {
    "problem_identifier": (
        "You identify the single most decision-relevant research problem implied by pre-cutoff literature. "
        "Do not write a broad proposal. Prefer one unresolved issue or one next-step judgment target that a benchmark expects."
    ),
    "task_judgment_role": "You are the task-native judgment generator of an offline benchmark adaptation of ResearchAgent.",
    "task_judgment_requirements": [
        "This is the main benchmark judgment stage. It fully replaces open-ended problem ideation.",
        "Produce exactly one family-specific, benchmark-facing judgment that can later be rendered into the final answer.",
        "Stay near-term, evidence-grounded, and decisive.",
        "Avoid broad agendas, paper-writing plans, and open-ended proposal language.",
        "If the task includes explicit candidate directions, preserve them verbatim and do not introduce substitute labels.",
    ],
    "task_module_role": "You are the task-specific module of an offline benchmark adaptation of ResearchAgent.",
    "task_module_requirements": [
        "This is not method development and not experiment design.",
        "Produce one family-specific reasoning packet that the final renderer can directly use.",
        "Keep it short, concrete, and evidence-grounded.",
        "It must complement the task judgment rather than restating it.",
        "Avoid simply repeating the task judgment sentence with minor rewording.",
        "If the task includes explicit candidate directions, preserve them verbatim and do not introduce substitute labels.",
    ],
    "decision_packet_role": "You are building a decision packet for a benchmark-facing offline literature agent.",
    "decision_packet_requirements": [
        "Be benchmark-facing, concrete, and decisive.",
        "Prefer one-step defensible judgments over broad survey summaries.",
        "Use only the provided evidence and internal outputs.",
    ],
    "lite_decision_packet_role": "You are helping an offline adaptation of ResearchAgent make a better final judgment.",
    "lite_decision_packet_requirements": [
        "Stay close to the evidence and ResearchAgent outputs.",
        "Do NOT just restate the provided forward_implication; infer 3-5 plausible concrete judgments from the evidence.",
        "Make the candidate_judgments meaningfully diverse rather than minor rephrasings.",
        "Prefer compact labels that can stand on their own as a likely topic, bottleneck, or priority name.",
        "Avoid generic recommendations like better benchmarks, more evaluation, or more data unless technically grounded.",
    ],
    "family_contract": "- Produce a concrete benchmark-facing answer.",
    "family_shape": "- Start with a direct conclusion sentence.",
    "family_budget": "- Keep the answer concise but sufficiently justified to survive direct comparison.",
    "final_candidate_rules": [
        "- Prefer the most concrete, evidence-grounded, benchmark-faithful candidate.",
    ],
}


_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "bottleneck_opportunity_discovery": {
        "problem_identifier": (
            "You identify one mechanism-level bottleneck implied by pre-cutoff literature and the immediate capability it blocks. "
            "Do not generalize into benchmarking, evaluation infrastructure, or broad ecosystem advice unless the mechanism evidence clearly requires it."
        ),
        "task_judgment_requirements": [
            "For bottleneck tasks: name one unresolved mechanism bottleneck and the immediate opportunity unlocked by solving it.",
            "For bottleneck tasks: prefer a bottleneck-to-unlock chain that is visible in recurring historical failure signals and near-term future-work signals, not a broad downstream application jump.",
        ],
        "task_module_requirements": [
            "Use the packet fields this way:",
            "  - canonical_focus: the main bottleneck / forecast direction / ranked priority / venue positioning label.",
            "  - secondary_focus: the most important contrast, dependency, or nearby alternative.",
            "  - core_support: the decisive mechanism, signal chain, or reviewer-fit logic from the evidence.",
            "  - execution_hook: the immediate unlock path, first milestone, trigger, or contribution package.",
            "  - rejection_rule: the nearby but worse alternative that should not be selected.",
            "For bottleneck tasks: return the six explicit fields bottleneck_label, evidence_symptoms, root_cause_mechanism, blocked_capability, immediate_unlock, nearby_but_wrong_opportunity.",
            "For bottleneck tasks: treat the chain-stage bottleneck signals in the evidence digest as auxiliary support only; use them to preserve repeated historical limitation wording, but do not let them override a more specific bottleneck already supported by the main evidence or current task judgment.",
            "For bottleneck tasks: never use a paper title, method name, benchmark name, or dataset name as the bottleneck label or immediate unlock label; abstract them into the underlying failure mode or capability.",
            "For bottleneck tasks: the opportunity must be the immediate unlocked capability, not a broad long-range research program.",
            "For bottleneck tasks: blocked_capability and immediate_unlock should stay close to the retrieved future_work_signals or bridge_concepts; do not introduce a downstream application absent from the evidence.",
        ],
        "decision_packet_requirements": [
            "For bottleneck tasks, emphasize unresolved mechanism and unlocked opportunity.",
            "For bottleneck tasks, use recurring chain-stage limitation signals only as auxiliary corroboration; do not replace a sharper evidence-grounded bottleneck label with a broader chain summary.",
            "For bottleneck tasks, never let a paper title or named method become the bottleneck label; rewrite it as the underlying failure mode.",
            "For bottleneck tasks, keep the unlocked opportunity close to the evidence-supported future-work or bridge-concept cluster rather than inventing a new downstream application.",
            "For bottleneck tasks, if the task text does not name a deployment scenario, do not introduce one.",
            "Prefer reusing one of the focus_candidates when it fits the evidence rather than inventing a broader umbrella label.",
            "For bottleneck tasks, use one recurring bottleneck signal and one why-now trigger from the historical signal map to justify the immediate unlock.",
        ],
        "lite_decision_packet_requirements": [
            "For bottleneck_opportunity_discovery: keep blocked capability and immediate unlock near the retrieved future-work / bridge-concept cluster, and avoid inventing a new downstream application that is absent from the evidence.",
            "For bottleneck_opportunity_discovery: prefer mechanism-level bottleneck phrases over artifact titles, benchmark names, or vague ecosystem complaints.",
        ],
        "family_contract": (
            "- Explicitly state one unresolved mechanism bottleneck, the blocked capability it prevents, and one immediate unlocked opportunity that becomes newly viable if it is addressed.\n"
            "- Prefer the opportunity label to reuse a public evidence-supported research cluster from retrieved paper titles, future-work phrases, or domain-specific application families rather than inventing a new deployment scenario."
        ),
        "family_shape": "- Start with 'Bottleneck:' then name 'Blocked capability:' and 'Immediate unlock:' before the short justification. In the justification sentence, mention one or two exact retrieved paper titles when they directly support the linkage.",
        "family_budget": "- Keep the answer around 60-150 words. One bottleneck, one blocked capability, one immediate unlock, then one short sentence for why the linkage is plausible using one or two exact historical paper anchors.",
        "final_candidate_rules": [
            "- Prefer candidates that isolate one mechanism bottleneck, name the blocked capability, and state one immediate unlock that becomes newly viable once the bottleneck is fixed.",
            "- Reject answers that stay at the level of generic limitations, infrastructure wishes, or broad future agendas.",
            "- Prefer candidates whose why-now logic can be traced to a concrete historical failure pattern plus one near-term unlock trigger rather than a generic vision statement.",
            "- If two candidates are similarly grounded, choose the one with the sharper bottleneck -> blocked capability -> immediate unlock linkage.",
        ],
    },
    "direction_forecasting": {
        "problem_identifier": (
            "You identify one likely next research direction implied by pre-cutoff trajectory signals. "
            "Stay inside the task's topical scope. Do not list several futures, and do not import a method trend from another subfield just because vocabulary overlaps."
        ),
        "task_judgment_role": "You are the forecasting-specific judgment generator of an offline benchmark adaptation of ResearchAgent.",
        "task_judgment_requirements": [
            "For forecasting tasks: make one concrete next-direction call and tie it to the strongest historical trajectory signal.",
            "For forecasting tasks: stay strictly inside the topical scope named by the task and do not import a method trend from another subfield merely because it shares forecasting, retrieval, control, or evaluation vocabulary.",
            "For forecasting tasks: prefer a sharp mechanism shift, capability shift, or data/training shift over generic future-work prose.",
        ],
        "task_module_role": "You are the forecasting-specific reasoning module of an offline benchmark adaptation of ResearchAgent.",
        "task_module_requirements": [
            "Use the packet fields this way:",
            "  - canonical_focus: the main bottleneck / forecast direction / ranked priority / venue positioning label.",
            "  - secondary_focus: the most important contrast, dependency, or nearby alternative.",
            "  - core_support: the decisive mechanism, signal chain, or reviewer-fit logic from the evidence.",
            "  - execution_hook: the immediate unlock path, first milestone, trigger, or contribution package.",
            "  - rejection_rule: the nearby but worse alternative that should not be selected.",
            "For forecasting tasks: make the packet expose one primary next direction, trajectory_label, trajectory_signal, and the nearby alternative to reject.",
            "For forecasting tasks: keep the packet inside the task's topical scope and reject cross-domain method transfer when the supporting papers are only superficially similar.",
            "For forecasting tasks: if the evidence only supports localized specialization inside the current subfield, do not upscale it into a broader umbrella trend.",
        ],
        "decision_packet_requirements": [
            "For forecasting, name compact likely next directions rather than generic future work.",
            "For forecasting, keep one primary mechanism family explicit and keep nearby alternatives as rejectable contrasts rather than co-equal answers.",
            "Prefer reusing one of the focus_candidates when it fits the evidence rather than inventing a broader umbrella label.",
        ],
        "lite_decision_packet_requirements": [
            "For direction_forecasting: include alternatives across different mechanism types when plausible (e.g. training signal, data generation, uncertainty/control, tool/RL integration, modality expansion).",
        ],
        "family_contract": "- Make a concrete trajectory call and name one primary next direction that is still inside the task's topical scope.",
        "family_shape": "- Start with 'Forecast:' and include 'Trajectory:' plus a short 'Why now:' justification tied to the strongest historical signal chain.",
        "family_budget": "- Keep the answer around 80-220 words. Name one trajectory label and one primary next direction, then give enough technical justification to make the forecast credible. Avoid long multi-branch lists.",
        "final_candidate_rules": [
            "- Prefer candidates that make one primary forecast and explain why the historical evidence points there now.",
            "- Reject candidates that stitch together several loosely related direction families or drift into another subfield with overlapping vocabulary only.",
            "- If two candidates are similarly grounded, choose the one with the clearest trajectory label and strongest evidence separation from alternatives.",
        ],
    },
    "strategic_research_planning": {
        "problem_identifier": (
            "You identify the single most decision-relevant ordering judgment implied by pre-cutoff evidence. "
            "Focus on what should come first, what should be deferred, and the gating dependency behind that ordering."
        ),
        "task_judgment_requirements": [
            "For strategic planning tasks: state the ranked priority judgment and the dependency / trade-off that determines the ordering.",
            "For strategic planning tasks: make the ordering executable by grounding it in one dependency axis and one first-milestone signal rather than a broad roadmap summary.",
        ],
        "task_module_requirements": [
            "Use the packet fields this way:",
            "  - canonical_focus: the main bottleneck / forecast direction / ranked priority / venue positioning label.",
            "  - secondary_focus: the most important contrast, dependency, or nearby alternative.",
            "  - core_support: the decisive mechanism, signal chain, or reviewer-fit logic from the evidence.",
            "  - execution_hook: the immediate unlock path, first milestone, trigger, or contribution package.",
            "  - rejection_rule: the nearby but worse alternative that should not be selected.",
            "For strategic planning tasks: make the packet expose explicit ordering, first_milestone, dependency_chain, defer_rationale, and risk_or_kill_criterion.",
        ],
        "decision_packet_requirements": [
            "For planning, emphasize ordering and dependencies.",
            "For comparative strategic planning tasks with explicit candidate directions, rank only the listed candidates and keep their labels verbatim.",
            "For planning, use one agenda axis from the historical signal map to explain the ordering and one near-term milestone to make the ranking actionable.",
        ],
        "lite_decision_packet_requirements": [
            "For strategic_research_planning: output 2-3 ranked priorities with dependencies, not one isolated priority.",
            "For comparative strategic planning tasks with explicit candidate directions, stay strictly inside the listed candidates and do not rename them.",
            "For strategic_research_planning: avoid generic program-management language; the plan should read like an executable technical ordering decision.",
        ],
        "family_contract": "- Produce a short prioritized plan with explicit ordering, dependencies, and trade-offs.",
        "family_shape": "- Use 'Priority 1:' and 'Priority 2:' plus explicit 'First milestone:', 'Dependency:', 'Defer rationale:', and 'Risk/Kill criterion:'.",
        "family_budget": "- Keep the answer around 120-280 words. Give Priority 1 and Priority 2, then one concrete first milestone, one dependency, one defer rationale, and one risk/kill criterion with enough technical justification to make the ordering executable.",
        "final_candidate_rules": [
            "- Prefer candidates with a short ordered plan, explicit first milestone, dependencies, and a concrete technical trade-off judgment.",
            "- For comparative planning tasks, only the task-provided candidate direction labels are allowed; reject substitute names or relabeled abstractions.",
            "- Prefer candidates that make the ordering executable through one concrete first milestone and one explicit defer rationale, not just a high-level ranking.",
            "- If two candidates are similarly grounded, choose the one that is more contract-faithful, milestone-explicit, and dependency-explicit rather than broader.",
        ],
    },
    "venue_aware_research_positioning": {
        "problem_identifier": (
            "You identify one concrete contribution framing that best fits the venue-facing historical trajectory. "
            "Do not drift into generic paper-writing advice."
        ),
        "task_judgment_requirements": [
            "For venue-aware tasks: state the best contribution framing and why it fits the venue trajectory.",
            "For venue-aware tasks with explicit candidate directions: rank only the listed candidate directions, keep the labels verbatim, and provide a complete ordering rather than collapsing to one winner only.",
        ],
        "task_module_requirements": [
            "Use the packet fields this way:",
            "  - canonical_focus: the main bottleneck / forecast direction / ranked priority / venue positioning label.",
            "  - secondary_focus: the most important contrast, dependency, or nearby alternative.",
            "  - core_support: the decisive mechanism, signal chain, or reviewer-fit logic from the evidence.",
            "  - execution_hook: the immediate unlock path, first milestone, trigger, or contribution package.",
            "  - rejection_rule: the nearby but worse alternative that should not be selected.",
            "For venue-aware tasks: also expose contribution_package, venue_fit_signal, evaluation_signature, and nearby_but_wrong_positioning.",
            "For venue-aware tasks: use the chain-stage venue signals in the evidence digest only as auxiliary hints for package and evaluation recipe; they should not replace a more concrete positioning label already supported by the main evidence, task judgment, or task contract.",
            "For venue-aware tasks: make the packet expose one contribution framing, why it fits, what evaluation package makes it credible for that venue trajectory, and one nearby framing that is less appropriate.",
            "For venue-aware tasks: never collapse the primary positioning into abstract package labels such as new_method, empirical_comparison, analysis_or_diagnosis, or benchmark_eval; those belong in package/evaluation fields, not the main positioning label.",
            "For venue-aware tasks with explicit candidate directions: include ranked_candidates and ensure the ranking covers all listed candidate directions exactly once.",
        ],
        "decision_packet_requirements": [
            "For venue-aware tasks, prioritize concrete contribution framing over generic venue advice.",
            "For venue-aware tasks, use the chain-stage venue summary only to reinforce package/evaluation details; do not let it overwrite a more specific contribution framing that is already evidence-grounded.",
            "For venue-aware tasks with explicit candidate directions, keep the listed labels verbatim and decide a complete ranking among them rather than inventing a new umbrella label.",
            "Prefer reusing one of the focus_candidates when it fits the evidence rather than inventing a broader umbrella label.",
        ],
        "lite_decision_packet_requirements": [
            "For venue_aware_research_positioning: if the task lists candidate directions, stay strictly inside those directions and make the venue discrimination contrastive.",
            "For venue_aware_research_positioning: explicitly separate primary fit, paper package, and evaluation recipe from nearby compatible venues.",
        ],
        "family_contract": "- Explain what type of concrete contribution is most likely to fit the target venue-facing trajectory implied by the literature.",
        "family_shape": "- If the task lists candidate directions, use 'Positioning 1:', 'Positioning 2:' ... to rank all listed options exactly once, then include 'Package:', 'Why this venue:', 'Evaluation:', and 'Contrast:'. Otherwise start with 'Positioning:' and include contribution framing, package, why it fits the venue trajectory, and one contrast.",
        "family_budget": "- Keep the answer around 110-260 words. For comparative venue tasks, rank all listed options exactly once, then give a short top-fit rationale, one contribution package, one venue-fit rationale, one evaluation recipe, and one contrast.",
        "final_candidate_rules": [
            "- Prefer candidates that name one concrete contribution framing and explain why it fits the venue trajectory implied by the evidence.",
            "- For comparative venue tasks, only the task-provided candidate direction labels are allowed and the answer must rank all of them exactly once.",
            "- Reject generic venue advice, broad benchmark governance, or shallow claims that any solid paper would fit.",
            "- Prefer candidates that name the evaluation or evidence package that would make the venue fit credible.",
            "- If two candidates are similarly grounded, choose the one with the more technically specific framing.",
        ],
    },
}


def _profile(family: str) -> Dict[str, Any]:
    profile = dict(_DEFAULT_PROFILE)
    family_profile = _FAMILY_PROFILES.get(family, {})
    for key, value in family_profile.items():
        if isinstance(value, list):
            profile[key] = list(profile.get(key, [])) + list(value)
        else:
            profile[key] = value
    return profile


def problem_identifier_prompt(family: str) -> str:
    return _profile(family)["problem_identifier"]


def family_contract(family: str) -> str:
    return _profile(family)["family_contract"]


def family_shape(family: str) -> str:
    return _profile(family)["family_shape"]


def family_budget(family: str) -> str:
    return _profile(family)["family_budget"]


def final_candidate_family_rules(family: str) -> str:
    return _join_lines(_profile(family)["final_candidate_rules"])


def build_task_judgment_prompt(
    *,
    task: Dict[str, Any],
    family: str,
    task_frame: Dict[str, Any],
    digest: Dict[str, Any],
    signal_map: Dict[str, Any],
    native_bundle: Dict[str, Any],
    contract_instruction: str,
) -> str:
    profile = _profile(family)
    return f"""{profile['task_judgment_role']}

Task:
{_to_json({
    "task_id": task.get("task_id"),
    "family": family,
    "domain": task.get("domain"),
    "time_cutoff": task.get("time_cutoff"),
    "title": task.get("title"),
    "question": task.get("question"),
    "deliverable_spec": task.get("deliverable_spec") or {},
    "answer_contract": task.get("answer_contract") or {},
})}

Task frame:
{_to_json(task_frame)}

Evidence digest:
{_to_json(digest)}

Historical signal map:
{_to_json(signal_map)}

ResearchAgent-native KB bundle:
{_to_json(native_bundle)}

Return strict JSON with keys:
- task_judgment
- task_judgment_rationale

Requirements:
{_join_lines([f"- {line}" for line in profile["task_judgment_requirements"]])}
{contract_instruction}
"""


def build_task_module_prompt(
    *,
    task: Dict[str, Any],
    family: str,
    task_frame: Dict[str, Any],
    evidence_digest: Dict[str, Any],
    signal_map: Dict[str, Any],
    native_bundle: Dict[str, Any],
    top_papers: List[Dict[str, Any]],
    current_judgment: Dict[str, Any],
    contract_instruction: str,
) -> str:
    profile = _profile(family)
    return f"""{profile['task_module_role']}

Task:
{_to_json({
    "task_id": task.get("task_id"),
    "family": task.get("family"),
    "domain": task.get("domain"),
    "time_cutoff": task.get("time_cutoff"),
    "title": task.get("title"),
    "question": task.get("question"),
    "deliverable_spec": task.get("deliverable_spec") or {},
    "answer_contract": task.get("answer_contract") or {},
})}

Task frame:
{_to_json(task_frame)}

Evidence digest:
{_to_json(evidence_digest)}

Historical signal map:
{_to_json(signal_map)}

ResearchAgent-native KB bundle:
{_to_json(native_bundle)}

Retrieved top papers:
{_to_json(top_papers)}

Current selected task judgment:
{_to_json(current_judgment)}

Return strict JSON with keys:
- task_module_packet: object
- task_module_rationale

Requirements:
{_join_lines([f"- {line}" for line in profile["task_module_requirements"]])}
{contract_instruction}
"""


def build_decision_packet_prompt(
    *,
    task: Dict[str, Any],
    family: str,
    task_frame: Dict[str, Any],
    digest: Dict[str, Any],
    signal_map: Dict[str, Any],
    top_papers: List[Dict[str, Any]],
    internal_outputs: Dict[str, Any],
    contract_instruction: str,
) -> str:
    profile = _profile(family)
    return f"""{profile['decision_packet_role']}

Task:
{_to_json({
    "task_id": task.get("task_id"),
    "family": family,
    "domain": task.get("domain"),
    "time_cutoff": task.get("time_cutoff"),
    "title": task.get("title"),
    "question": task.get("question"),
    "answer_contract": task.get("answer_contract") or {},
})}

Task frame:
{_to_json(task_frame)}

Evidence digest:
{_to_json(digest)}

Historical signal map:
{_to_json(signal_map)}

Retrieved top papers:
{_to_json(top_papers)}

ResearchAgent internal outputs:
{_to_json(internal_outputs)}

Return strict JSON with keys:
- historical_baseline: short string
- unresolved_core: short string
- strongest_signals: list of 2-5 short strings
- focus_candidates: list of 2-5 compact direction / bottleneck / priority labels supported by the evidence
- candidate_judgments: list of 2-3 short concrete candidate answer claims
- preferred_judgment: one short concrete claim
- evidence_anchors: list of 2-4 short anchor strings in the form 'Paper Title: why it matters'
- family_checklist: list of 2-4 short requirements that the final answer must satisfy
- anti_patterns: list of 2-4 failure modes the final answer must avoid

Requirements:
{_join_lines([f"- {line}" for line in profile["decision_packet_requirements"]])}
{contract_instruction}
"""


def build_lite_decision_packet_prompt(
    *,
    task: Dict[str, Any],
    family: str,
    task_frame: Dict[str, Any],
    evidence_digest: Dict[str, Any],
    signal_map: Dict[str, Any],
    top_papers: List[Dict[str, Any]],
    internal_outputs: Dict[str, Any],
    contract_instruction: str,
) -> str:
    profile = _profile(family)
    return f"""{profile['lite_decision_packet_role']}

Task:
{_to_json({
    "task_id": task.get("task_id"),
    "family": family,
    "domain": task.get("domain"),
    "time_cutoff": task.get("time_cutoff"),
    "title": task.get("title"),
    "question": task.get("question"),
})}

Task frame:
{_to_json(task_frame)}

Evidence digest:
{_to_json(evidence_digest)}

Historical signal map:
{_to_json(signal_map)}

Top papers:
{_to_json(top_papers)}

ResearchAgent outputs:
{_to_json(internal_outputs)}

Return strict JSON with keys:
- historical_baseline
- unresolved_core
- strongest_signals
- focus_candidates
- candidate_judgments
- preferred_judgment
- evidence_anchors
- family_checklist
- anti_patterns

Requirements:
{_join_lines([f"- {line}" for line in profile["lite_decision_packet_requirements"]])}
{contract_instruction}
"""


def build_render_prompt(
    *,
    task: Dict[str, Any],
    family: str,
    task_frame: Dict[str, Any],
    decision_packet: Dict[str, Any],
    evidence_digest: Dict[str, Any],
    signal_map: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    internal_outputs: Dict[str, Any],
    family_packet: Dict[str, Any],
    contract_instruction: str,
) -> str:
    return f"""# Role
You are rendering benchmark-facing final answers for an offline adaptation of ResearchAgent.

# Mission
Turn the task, retrieved evidence, and ResearchAgent outputs into 3 strong candidate answers.

# Hard constraints
- Use only the task, retrieved evidence, and ResearchAgent outputs below.
- Do not add post-cutoff knowledge.
- Do not invent papers, venues, dates, or unsupported claims.
- Be concrete, decisive, and benchmark-facing rather than survey-like.
- The answer text itself must contain an explicit `Evidence:` clause that names 2 exact retrieved paper titles.
{contract_instruction}

# Public task
{_to_json({
    "task_id": task.get("task_id"),
    "family": task.get("family"),
    "domain": task.get("domain"),
    "time_cutoff": task.get("time_cutoff"),
    "title": task.get("title"),
    "question": task.get("question"),
})}

# Task frame
{_to_json(task_frame)}

# Decision packet
{_to_json(decision_packet)}

# Evidence digest
{_to_json(evidence_digest)}

# Historical signal map
{_to_json(signal_map)}

# Retrieved evidence
{_to_json(evidence)}

# ResearchAgent internal outputs
{_to_json(internal_outputs)}

# Family reasoning packet
{_to_json(family_packet)}

# Family contract
{family_contract(family)}

# Diversity requirement
- Candidate 1: most evidence-conservative and nearest-step answer.
- Candidate 2: alternative but still plausible framing supported by different evidence emphasis.
- Candidate 3: sharper or more ambitious option, but still one-step defensible.

# Style requirements
- Do not write a survey or a list of many vague possibilities.
- The first sentence must contain the core judgment.
- Prefer a technically grounded, immediate next-step judgment over institution-level or benchmark-governance recommendations unless the evidence overwhelmingly points there.
- Prefer compact direction labels, explicit bottlenecks, or explicit priorities depending on family.
- Keep the answer benchmark-facing and technically persuasive, not under-explained.
- Mention paper titles in the answer only if they materially strengthen the judgment.
- Every support_summary item must begin with the exact title of a retrieved paper.
- The final answer text must include an `Evidence:` clause that points to 2 exact retrieved paper titles used in the judgment.
- Reuse one of the focus_candidates when it is evidence-supported instead of inventing a broader umbrella term.
- Stay anchored to the family reasoning packet: keep its canonical_focus explicit, use its core_support as the main justification, and respect its rejection_rule.
- Follow the word budget and structural limits below.
- Prefer 2-4 compact, content-dense sentences over a bare label list when the task is comparative or venue-facing.
- For comparative strategic planning tasks, the answer must stay inside the listed candidate directions and must not introduce substitute labels.
- For comparative venue-aware tasks, the answer must rank all listed candidate directions exactly once, keep their labels verbatim, and avoid collapsing them into a new umbrella label.

# Family-specific answer shape
{family_shape(family)}

# Word budget and structural limits
{family_budget(family)}

# Anti-patterns to avoid
{_to_json(decision_packet.get("anti_patterns") or [])}

# Output format
Return strict JSON:
{{
  "candidates": [
    {{
      "reasoning_frame": "conservative | alternative | sharper",
      "confidence": 0.0,
      "answer": "...",
      "support_summary": ["Paper Title: what it supports", "..."]
    }}
  ]
}}
"""


def build_final_candidate_judge_prompt(
    *,
    task: Dict[str, Any],
    family: str,
    task_frame: Dict[str, Any],
    decision_packet: Dict[str, Any],
    signal_map: Dict[str, Any],
    family_packet: Dict[str, Any],
    evidence: List[Dict[str, Any]],
    scored_candidates: List[Dict[str, Any]],
    contract_instruction: str,
) -> str:
    return f"""You are the final candidate judge for a benchmark-facing offline ResearchAgent answer.

Task:
{_to_json({
    "task_id": task.get("task_id"),
    "family": family,
    "domain": task.get("domain"),
    "time_cutoff": task.get("time_cutoff"),
    "title": task.get("title"),
    "question": task.get("question"),
})}

Task frame:
{_to_json(task_frame)}

Decision packet:
{_to_json(decision_packet)}

Historical signal map:
{_to_json(signal_map)}

Family reasoning packet:
{_to_json(family_packet)}

Retrieved evidence:
{_to_json(evidence)}

Candidate answers:
{_to_json(scored_candidates)}

Selection rules:
- Choose exactly one candidate index from the list above.
- Prefer the answer with the strongest one-step judgment, the cleanest evidence linkage, and enough technical justification to make the choice credible without drifting into survey-like sprawl.
- Do not merge multiple candidates into a new hybrid answer.
- The revised answer must remain a light rewrite of the selected candidate, not a new synthesis.
- Reuse a focus_candidate when it is already supported by the evidence.
- Keep the family packet's canonical_focus explicit and do not silently replace it with a broader umbrella term.
- Use the family packet's rejection_rule to avoid selecting a nearby but worse alternative.
- Support summaries must begin with exact paper titles from the evidence.
- Forecasting: commit to one primary next direction and one trajectory label; avoid stitching together several mechanism families.
- Planning: keep to 2 or 3 priorities maximum and state dependencies explicitly.
- For comparative strategic planning tasks with explicit candidate directions, keep the listed direction labels verbatim and reject any answer that introduces substitute direction names.
- For comparative venue-aware tasks with explicit candidate directions, keep the listed direction labels verbatim, require a complete ordering, and reject answers that only nominate a single winner.
- Bottleneck: name one unresolved mechanism bottleneck, the blocked capability, and one immediate unlock; do not reward longer-range visions that are not newly enabled right away.
- Family-specific decision rules:
{final_candidate_family_rules(family)}
- Avoid these anti-patterns:
{_to_json(decision_packet.get("anti_patterns") or [])}
{contract_instruction}

Word budget:
{family_budget(family)}

Return strict JSON with keys:
- selected_candidate_index: integer
- revised_answer: short final answer
- support_summary: list of 2-3 strings in the form 'Exact Paper Title: what it supports'
- rationale: short string
"""
