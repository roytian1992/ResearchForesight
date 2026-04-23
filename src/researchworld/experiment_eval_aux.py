from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from researchworld.experiment_eval_v3 import infer_domain_id
from researchworld.experiment_eval_v4 import _collect_support_items, _render_support_snapshot
from researchworld.llm import OpenAICompatChatClient, complete_json_object

AUX_DIMENSIONS_BY_FAMILY: Dict[str, List[Dict[str, Any]]] = {
    "bottleneck_opportunity_discovery": [
        {"name": "causal_linkage", "weight": 0.55},
        {"name": "technical_plausibility", "weight": 0.45},
    ],
    "direction_forecasting": [
        {"name": "signal_grounding", "weight": 0.60},
        {"name": "forecast_discipline", "weight": 0.40},
    ],
    "strategic_research_planning": [
        {"name": "first_milestone_specificity", "weight": 0.25},
        {"name": "dependency_to_action_chain", "weight": 0.20},
        {"name": "alternative_defer_rationale", "weight": 0.20},
        {"name": "risk_and_kill_criteria", "weight": 0.20},
        {"name": "evidence_to_action_mapping", "weight": 0.15},
    ],
    "venue_aware_research_positioning": [
        {"name": "venue_specific_contribution_fit", "weight": 0.30},
        {"name": "reviewer_expectation_grounding", "weight": 0.25},
        {"name": "paper_package_specificity", "weight": 0.20},
        {"name": "contrastive_venue_discrimination", "weight": 0.25},
    ],
}

AUX_NAME_BY_FAMILY: Dict[str, str] = {
    "bottleneck_opportunity_discovery": "opportunity_grounding_score",
    "direction_forecasting": "forecast_grounding_score",
    "strategic_research_planning": "strategic_execution_grounding_score",
    "venue_aware_research_positioning": "venue_positioning_grounding_score",
}

AUX_LABEL_BY_FAMILY: Dict[str, str] = {
    "bottleneck_opportunity_discovery": "Opportunity Grounding",
    "direction_forecasting": "Forecast Grounding",
    "strategic_research_planning": "Strategic Execution Grounding",
    "venue_aware_research_positioning": "Venue Positioning Grounding",
}


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _rubric_dimension_summary(group: List[Dict[str, Any]], family: str) -> Dict[str, float]:
    dimensions = AUX_DIMENSIONS_BY_FAMILY.get(family) or []
    if not group or not dimensions:
        return {}
    dim_names = [str(item['name']) for item in dimensions]
    return {
        name: _mean([float((((row.get('family_aux_eval') or {}).get('rubric_scores') or {}).get(name) or 0.0)) for row in group])
        for name in dim_names
    }

VENUE_BUCKET_ALIASES: Dict[str, List[str]] = {
    "aaai": ["aaai", "aaai like", "aaai class", "aaai conference"],
    "acl": ["acl", "acl like", "acl class"],
    "emnlp": ["emnlp", "emnlp like", "emnlp class"],
    "iclr": ["iclr", "iclr like", "iclr class"],
    "icml": ["icml", "icml like", "icml class"],
    "ijcai": ["ijcai", "ijcai like", "ijcai class"],
    "kdd": ["kdd", "kdd like", "kdd class"],
    "naacl": ["naacl", "naacl like", "naacl class"],
    "neurips": ["neurips", "neural information processing systems", "neurips like", "neurips class"],
    "sigir": ["sigir", "sigir like", "sigir class"],
    "cvpr": ["cvpr", "cvpr like", "cvpr class"],
    "eccv": ["eccv", "eccv like", "eccv class"],
    "iccv": ["iccv", "iccv like", "iccv class"],
}

VENUE_COMPATIBLE_BUCKETS: Dict[str, List[str]] = {
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

BOTTLENECK_CAUSAL_CUES = [
    "because",
    "blocks",
    "block",
    "prevents",
    "limits",
    "constrains",
    "if addressed",
    "if solved",
    "would enable",
    "unlocks",
    "unlock",
    "becomes possible",
    "becomes viable",
    "upstream",
]

BOTTLENECK_ARTIFACT_CUES = [
    "benchmark",
    "dataset",
    "framework",
    "protocol",
    "pipeline",
    "suite",
    "leaderboard",
    "platform",
]

BOTTLENECK_SYMPTOM_CUES = [
    "low accuracy",
    "poor performance",
    "high latency",
    "hallucination",
    "generalization gap",
    "weak robustness",
    "error rate",
]

BOTTLENECK_MULTI_HOP_CUES = [
    "long term",
    "eventually",
    "broader agenda",
    "across all domains",
    "fully autonomous",
    "general-purpose",
    "end-to-end autonomy",
]

GENERIC_TARGET_TOKENS = {
    "a",
    "an",
    "and",
    "aware",
    "benchmark",
    "benchmarks",
    "comparison",
    "comparative",
    "direction",
    "directions",
    "for",
    "framework",
    "frameworks",
    "in",
    "like",
    "method",
    "methods",
    "of",
    "on",
    "or",
    "research",
    "studies",
    "study",
    "system",
    "systems",
    "task",
    "tasks",
    "the",
    "to",
    "top",
    "venue",
    "venues",
}

SOFT_GENERIC_DIRECTION_TOKENS = {
    "agent",
    "agents",
    "augmentation",
    "benchmark",
    "benchmarks",
    "evaluation",
    "framework",
    "frameworks",
    "multi",
    "planning",
    "retrieval",
    "system",
    "systems",
}

STRATEGIC_DEPENDENCY_CUES = [
    "depends on",
    "dependency",
    "dependencies",
    "prerequisite",
    "prerequisites",
    "requires",
    "require",
    "enables",
    "enable",
    "enabled",
    "unlock",
    "unlocks",
    "foundational",
    "foundation",
    "builds on",
    "relies on",
    "upstream",
    "downstream",
    "gating",
    "gate",
    "precondition",
    "structurally dependent",
]

STRATEGIC_BLOCKER_CUES = [
    "bottleneck",
    "bottlenecks",
    "constraint",
    "constraints",
    "failure mode",
    "failure modes",
    "blocked",
    "blocks",
    "blocker",
    "blockers",
    "data scarcity",
    "compute bottleneck",
    "fragile",
    "immature",
    "unresolved",
    "unreliable",
    "insufficient",
]

STRATEGIC_SEQUENCING_CUES = [
    "before",
    "first",
    "come first",
    "prioritize",
    "prioritized",
    "prioritizing",
    "precede",
    "near term",
    "next six months",
    "in the same window",
    "in isolation",
    "before broader",
    "ahead of",
]

STRATEGIC_MOMENTUM_CUES = [
    "publication momentum",
    "emerging momentum",
    "publication trend",
    "publication trends",
    "citations",
    "citation",
    "top venue",
    "top venues",
    "top conference",
    "top conferences",
    "top tier",
    "high impact",
    "mature",
    "maturity",
    "venue prominence",
    "venue signals",
]

STRATEGIC_COMPARATIVE_CUES = [
    "rather than",
    "instead of",
    "vs",
    "versus",
    "compared with",
    "compared to",
    "over the alternative",
    "more than",
    "while the alternative",
]

STRATEGIC_NEAR_TERM_DELIVERABLE_CUES = [
    "benchmark suite",
    "evaluation harness",
    "evaluation protocol",
    "infrastructure",
    "tooling",
    "prototype",
    "dataset",
    "data pipeline",
    "ablation",
    "error analysis",
    "implementation",
    "integration layer",
    "interface standard",
    "first milestone",
    "near term deliverable",
    "de risk",
    "de-risk",
]

STRATEGIC_PRIORITIZATION_CUES = [
    "highest priority",
    "prioritize",
    "prioritized",
    "prioritizing",
    "should come first",
    "should be ranked first",
    "ranked first",
    "more urgent",
    "better next step",
    "better near term bet",
    "best next six months bet",
]

STRATEGIC_MILESTONE_CUES = [
    "first milestone",
    "first step",
    "next milestone",
    "deliverable",
    "prototype",
    "pilot",
    "benchmark suite",
    "evaluation harness",
    "data pipeline",
    "instrumentation",
    "integration layer",
    "ablation plan",
    "measurement protocol",
    "small scale deployment",
    "smoke test",
]

STRATEGIC_ACTION_CUES = [
    "build",
    "implement",
    "collect",
    "run",
    "benchmark",
    "evaluate",
    "measure",
    "prototype",
    "deploy",
    "integrate",
    "audit",
    "validate",
    "de risk",
    "de-risk",
]

STRATEGIC_DEFER_CUES = [
    "defer",
    "deferred",
    "secondary",
    "later",
    "not yet",
    "after",
    "premature",
    "wait until",
    "less executable",
    "should remain secondary",
    "same window",
    "less strategically attractive",
    "diminishing returns",
]

STRATEGIC_RISK_CUES = [
    "risk",
    "risks",
    "failure mode",
    "failure modes",
    "kill criterion",
    "kill criteria",
    "stop if",
    "would invalidate",
    "contingency",
    "fallback",
    "backfire",
    "uncertain",
    "uncertainty",
    "if evidence shows",
    "if this fails",
]

STRATEGIC_DECISION_RULE_CUES = [
    "if",
    "unless",
    "until",
    "otherwise",
    "or we should stop",
    "or we stop",
    "pivot",
    "downgrade",
    "abandon",
    "terminate",
    "reconsider",
]

STRATEGIC_CONCRETE_ARTIFACT_CUES = [
    "benchmark suite",
    "evaluation harness",
    "evaluation protocol",
    "prototype",
    "dataset",
    "data pipeline",
    "integration layer",
    "instrumentation",
    "measurement protocol",
    "pilot",
    "smoke test",
    "small scale deployment",
    "ablation plan",
]

VENUE_PAPER_PACKAGE_CUES = [
    "benchmark",
    "benchmark design",
    "leaderboard",
    "dataset",
    "ablation",
    "error analysis",
    "human evaluation",
    "user study",
    "theoretical analysis",
    "theoretical guarantee",
    "proof",
    "scaling law",
    "system design",
    "latency",
    "efficiency",
    "retrieval pipeline",
    "engineering artifact",
    "demo",
]

VENUE_REVIEWER_EXPECTATION_CUES = [
    "strong baselines",
    "careful ablations",
    "error analysis",
    "human evaluation",
    "robustness analysis",
    "statistical significance",
    "theoretical guarantee",
    "proof",
    "real-world deployment",
    "efficiency trade-off",
    "systems evaluation",
    "reviewer",
    "program committee",
]

VENUE_CONTRAST_CUES = [
    "rather than",
    "instead of",
    "compared with",
    "compared to",
    "vs",
    "versus",
    "more suitable than",
    "better fit than",
    "too engineering",
    "too theoretical",
    "too benchmark-heavy",
    "too systems-oriented",
]

VENUE_PRESTIGE_RHETORIC_CUES = [
    "top tier",
    "top-tier",
    "prestigious",
    "high impact",
    "broad impact",
    "top venue",
    "top conference",
    "strong paper",
    "flagship venue",
]

VENUE_PRIMARY_SECONDARY_CUES = [
    "primary fit",
    "primary venue",
    "best fit",
    "better fit",
    "secondary fit",
    "secondary venue",
    "also fit",
    "could also fit",
    "nearby alternative",
    "closer to",
    "more suitable than",
]

VENUE_PRIOR_KNOWLEDGE: Dict[str, Dict[str, Any]] = {
    "acl": {
        "family_name": "ACL-family computational linguistics / NLP",
        "scope_signals": ["natural language processing", "computational linguistics", "multilingual", "language grounding", "information extraction", "question answering", "machine translation"],
        "preferred_contribution_signals": ["empirical study", "analysis", "resource", "dataset", "benchmark", "evaluation", "linguistic insight", "multilingual evaluation"],
        "reviewer_expectation_signals": ["strong baselines", "careful ablations", "error analysis", "human evaluation", "dataset quality", "reproducible", "limitations"],
        "nearby_buckets": ["acl", "emnlp", "naacl"],
    },
    "emnlp": {
        "family_name": "EMNLP-family empirical NLP",
        "scope_signals": ["empirical methods", "natural language processing", "generation", "information retrieval and text mining", "multilinguality", "resources and evaluation"],
        "preferred_contribution_signals": ["empirical study", "negative findings", "survey", "new resource", "position paper", "reproducibility", "efficiency"],
        "reviewer_expectation_signals": ["strong baselines", "careful ablations", "error analysis", "human evaluation", "resource release", "reproducible", "limitations"],
        "nearby_buckets": ["emnlp", "acl", "naacl"],
    },
    "naacl": {
        "family_name": "NAACL-family computational linguistics / NLP",
        "scope_signals": ["natural language processing", "computational linguistics", "multilingual", "dialogue", "generation", "resources and evaluation"],
        "preferred_contribution_signals": ["empirical study", "analysis", "resource", "dataset", "benchmark", "evaluation", "linguistic insight"],
        "reviewer_expectation_signals": ["strong baselines", "careful ablations", "error analysis", "human evaluation", "dataset quality", "reproducible", "limitations"],
        "nearby_buckets": ["naacl", "acl", "emnlp"],
    },
    "iclr": {
        "family_name": "ICLR-family representation learning / broad machine learning",
        "scope_signals": ["representation learning", "machine learning", "reinforcement learning", "generative models", "causal reasoning", "learning theory", "datasets and benchmarks"],
        "preferred_contribution_signals": ["learning method", "representation", "optimization", "theory", "benchmark", "dataset", "infrastructure", "hybrid ai systems"],
        "reviewer_expectation_signals": ["ablations", "robustness", "scaling", "theoretical justification", "benchmark evaluation", "public discussion readiness"],
        "nearby_buckets": ["iclr", "neurips", "icml", "aaai", "ijcai"],
    },
    "neurips": {
        "family_name": "NeurIPS-family broad ML / interdisciplinary ML",
        "scope_signals": ["machine learning", "deep learning", "evaluation", "infrastructure", "optimization", "probabilistic methods", "reinforcement learning", "applications"],
        "preferred_contribution_signals": ["method", "infrastructure", "evaluation methodology", "foundation models", "interdisciplinary application", "dataset", "benchmark"],
        "reviewer_expectation_signals": ["empirical comparisons", "code and data", "reproducibility", "technical depth", "broad interest", "ethics"],
        "nearby_buckets": ["neurips", "iclr", "icml", "aaai", "ijcai"],
    },
    "icml": {
        "family_name": "ICML-family rigorous machine learning",
        "scope_signals": ["machine learning", "deep learning", "evaluation", "theory", "machine learning systems", "optimization", "probabilistic methods", "reinforcement learning"],
        "preferred_contribution_signals": ["original and rigorous research", "method", "theory", "evaluation methodology", "machine learning systems", "optimization"],
        "reviewer_expectation_signals": ["rigorous experiments", "clear significance", "strong empirical comparisons", "theory or methodological depth", "replicability"],
        "nearby_buckets": ["icml", "iclr", "neurips", "aaai", "ijcai"],
    },
    "aaai": {
        "family_name": "AAAI-family broad AI",
        "scope_signals": ["artificial intelligence", "reasoning", "planning", "agents", "knowledge representation", "learning", "applications", "robotics"],
        "preferred_contribution_signals": ["ai method", "integrated system", "planning", "reasoning", "application", "benchmark", "evaluation"],
        "reviewer_expectation_signals": ["clear ai relevance", "complete empirical evaluation", "comparison to strong baselines", "broad ai interest", "practical significance"],
        "nearby_buckets": ["aaai", "ijcai", "iclr", "neurips", "icml"],
    },
    "ijcai": {
        "family_name": "IJCAI-family broad AI",
        "scope_signals": ["artificial intelligence", "reasoning", "planning", "agents", "knowledge representation", "learning", "applications", "robotics"],
        "preferred_contribution_signals": ["ai method", "integrated system", "planning", "reasoning", "application", "benchmark", "evaluation"],
        "reviewer_expectation_signals": ["clear ai relevance", "complete empirical evaluation", "comparison to strong baselines", "broad ai interest", "practical significance"],
        "nearby_buckets": ["ijcai", "aaai", "iclr", "neurips", "icml"],
    },
    "sigir": {
        "family_name": "SIGIR-family information retrieval",
        "scope_signals": ["information retrieval", "information access", "search", "ranking", "retrieval", "recommender", "evaluation", "analysis"],
        "preferred_contribution_signals": ["retrieval algorithm", "evaluation", "analysis", "application", "resource", "reproducibility", "system"],
        "reviewer_expectation_signals": ["ir evaluation", "retrieval metrics", "analysis", "resource or artifact", "reproducibility", "real search setting"],
        "nearby_buckets": ["sigir", "kdd"],
    },
    "kdd": {
        "family_name": "KDD-family data science / knowledge discovery",
        "scope_signals": ["data science", "knowledge discovery", "data mining", "practical impact", "applications", "reproducibility", "ethics"],
        "preferred_contribution_signals": ["data-driven method", "application", "practical impact", "deployment-facing evaluation", "reproducibility", "knowledge discovery"],
        "reviewer_expectation_signals": ["technical merit", "originality", "potential impact", "quality of execution", "reproducibility", "ethics"],
        "nearby_buckets": ["kdd", "sigir"],
    },
    "cvpr": {
        "family_name": "CVPR-family computer vision",
        "scope_signals": ["computer vision", "pattern recognition", "visual", "image", "video", "vision-language"],
        "preferred_contribution_signals": ["visual benchmark", "visual method", "dataset", "evaluation", "system", "analysis"],
        "reviewer_expectation_signals": ["strong visual baselines", "qualitative and quantitative evaluation", "ablations", "clear visual task gains"],
        "nearby_buckets": ["cvpr", "eccv", "iccv"],
    },
    "eccv": {
        "family_name": "ECCV-family computer vision",
        "scope_signals": ["computer vision", "pattern recognition", "visual", "image", "video", "vision-language"],
        "preferred_contribution_signals": ["visual benchmark", "visual method", "dataset", "evaluation", "system", "analysis"],
        "reviewer_expectation_signals": ["strong visual baselines", "qualitative and quantitative evaluation", "ablations", "clear visual task gains"],
        "nearby_buckets": ["eccv", "cvpr", "iccv"],
    },
    "iccv": {
        "family_name": "ICCV-family computer vision",
        "scope_signals": ["computer vision", "pattern recognition", "visual", "image", "video", "vision-language"],
        "preferred_contribution_signals": ["visual benchmark", "visual method", "dataset", "evaluation", "system", "analysis"],
        "reviewer_expectation_signals": ["strong visual baselines", "qualitative and quantitative evaluation", "ablations", "clear visual task gains"],
        "nearby_buckets": ["iccv", "cvpr", "eccv"],
    },
}


def _weighted(dimensions: List[Dict[str, Any]], raw_scores: Dict[str, Any], fallback_key: str, obj: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
    dim_names = [str(item['name']) for item in dimensions]
    scores = {name: max(0.0, min(1.0, float(raw_scores.get(name) or 0.0))) for name in dim_names}
    if any(name in raw_scores for name in dim_names):
        weighted = 0.0
        total_w = 0.0
        for item in dimensions:
            name = str(item.get('name') or '')
            weight = float(item.get('weight') or 0.0)
            total_w += weight
            weighted += weight * float(scores.get(name) or 0.0)
        overall = weighted / total_w if total_w else sum(scores.values()) / max(1, len(scores))
    else:
        overall = max(0.0, min(1.0, float(obj.get(fallback_key) or 0.0)))
    return round(float(overall), 4), {k: round(float(v), 4) for k, v in scores.items()}


def _compact_public_task(public_task: Dict[str, Any]) -> Dict[str, Any]:
    keep = ['task_id', 'family', 'domain', 'title', 'question', 'time_cutoff']
    return {k: public_task.get(k) for k in keep if k in public_task}


def _normalize_text(text: Any) -> str:
    s = str(text or '').lower().replace('_', ' ')
    s = re.sub(r'[^a-z0-9+ ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _tokenize_text(text: Any) -> List[str]:
    return [tok for tok in _normalize_text(text).split() if tok]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_component_targets(hidden_row: Dict[str, Any], component_name: str) -> List[str]:
    component_targets = hidden_row.get('component_targets') or {}
    items = []
    for component in component_targets.get('components') or []:
        if str(component.get('name') or '') != component_name:
            continue
        items.extend(str(x) for x in (component.get('canonical_values') or []) if str(x).strip())
        aliases = component.get('aliases') or {}
        for alias_values in aliases.values():
            items.extend(str(x) for x in (alias_values or []) if str(x).strip())
    return _dedupe_keep_order([x for x in items if x])


def _extract_candidate_direction_targets(public_task: Dict[str, Any], hidden_row: Dict[str, Any], family: str) -> List[str]:
    slot_targets = hidden_row.get('slot_targets') or {}
    targets: List[str] = []
    if family == 'bottleneck_opportunity_discovery':
        targets.extend(_extract_component_targets(hidden_row, 'bottleneck'))
        targets.extend(_extract_component_targets(hidden_row, 'opportunity'))
        targets.extend(str(x) for x in (slot_targets.get('core_bottleneck_labels') or []) if str(x).strip())
        targets.extend(str(x) for x in (slot_targets.get('core_opportunity_labels') or []) if str(x).strip())
        targets.extend(str(x) for x in (slot_targets.get('future_themes') or []) if str(x).strip())
    elif family == 'direction_forecasting':
        targets.extend(_extract_component_targets(hidden_row, 'next_directions'))
        targets.extend(str(x) for x in (slot_targets.get('emergent_direction_labels') or []) if str(x).strip())
        targets.extend(str(x) for x in (slot_targets.get('future_themes') or []) if str(x).strip())
    elif family == 'strategic_research_planning':
        targets.extend(_extract_component_targets(hidden_row, 'ranked_directions'))
        targets.extend(str(x) for x in (slot_targets.get('priority_direction_labels') or []) if str(x).strip())
        targets.extend(str(x) for x in (slot_targets.get('future_themes') or []) if str(x).strip())
    elif family == 'venue_aware_research_positioning':
        targets.extend(_extract_component_targets(hidden_row, 'ranked_directions'))
        targets.extend(str(x) for x in (slot_targets.get('future_themes') or []) if str(x).strip())
        targets.extend(str(x) for x in (slot_targets.get('priority_direction_labels') or []) if str(x).strip())
    else:
        targets.extend(_extract_component_targets(hidden_row, 'ranked_directions'))
        targets.extend(str(x) for x in (slot_targets.get('priority_direction_labels') or []) if str(x).strip())
        targets.extend(str(x) for x in (slot_targets.get('future_themes') or []) if str(x).strip())
    return _dedupe_keep_order([x for x in targets if x])


def _extract_target_venue_bucket(public_task: Dict[str, Any]) -> str:
    subtype = str(public_task.get('subtype') or '')
    if subtype != 'venue_targeted_planning':
        return ''
    text = f"{public_task.get('title') or ''} || {public_task.get('question') or ''}"
    norm = _normalize_text(text)
    for bucket, aliases in VENUE_BUCKET_ALIASES.items():
        for alias in aliases:
            if f"for {alias}" in norm or f"{alias} venues" in norm or f"{alias} venue" in norm:
                return bucket
    return ''


def _extract_acceptable_target_venue_buckets(public_task: Dict[str, Any], hidden_row: Dict[str, Any]) -> List[str]:
    primary = _extract_target_venue_bucket(public_task)
    if primary:
        return _dedupe_keep_order([x for x in VENUE_COMPATIBLE_BUCKETS.get(primary, [primary]) if x])
    slot_targets = hidden_row.get('slot_targets') or {}
    fallback = str(slot_targets.get('target_venue_bucket') or '').strip().lower()
    if fallback:
        return _dedupe_keep_order([x for x in VENUE_COMPATIBLE_BUCKETS.get(fallback, [fallback]) if x])
    return []


def _resolve_venue_prior_knowledge(public_task: Dict[str, Any], hidden_row: Dict[str, Any]) -> Dict[str, Any]:
    primary = _extract_target_venue_bucket(public_task)
    acceptable = _extract_acceptable_target_venue_buckets(public_task, hidden_row)
    primary_profile = VENUE_PRIOR_KNOWLEDGE.get(primary) or {}
    acceptable_profiles = {
        bucket: VENUE_PRIOR_KNOWLEDGE.get(bucket) or {}
        for bucket in acceptable
        if bucket in VENUE_PRIOR_KNOWLEDGE
    }
    return {
        "primary_bucket": primary,
        "acceptable_buckets": acceptable,
        "primary_profile": primary_profile,
        "acceptable_profiles": acceptable_profiles,
    }


def _extract_answer_bucket_mentions(answer: str) -> List[str]:
    norm = _normalize_text(answer)
    found: List[str] = []
    for bucket, aliases in VENUE_BUCKET_ALIASES.items():
        if any(alias in norm for alias in aliases):
            found.append(bucket)
    return _dedupe_keep_order(found)


def _profile_phrase_hits(answer: str, profile: Dict[str, Any], keys: List[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for key in keys:
        phrases = [str(x) for x in (profile.get(key) or []) if str(x).strip()]
        out[key] = _find_phrase_hits(answer, phrases)
    return out


def _find_mentioned_targets(answer: str, targets: List[str]) -> List[str]:
    norm = _normalize_text(answer)
    found: List[str] = []
    for target in targets:
        target_norm = _normalize_text(target)
        if target_norm and target_norm in norm:
            found.append(target)
    return _dedupe_keep_order(found)


def _target_anchor_tokens(text: str) -> List[str]:
    tokens = []
    for tok in _tokenize_text(text):
        if tok in GENERIC_TARGET_TOKENS:
            continue
        if len(tok) <= 2:
            continue
        tokens.append(tok)
    return _dedupe_keep_order(tokens)


def _distinctive_target_tokens(text: str) -> List[str]:
    return [tok for tok in _target_anchor_tokens(text) if tok not in SOFT_GENERIC_DIRECTION_TOKENS]


def _target_match_score(answer: str, target: str) -> float:
    answer_norm = _normalize_text(answer)
    target_norm = _normalize_text(target)
    if not answer_norm or not target_norm:
        return 0.0
    if target_norm in answer_norm:
        return 1.0
    target_tokens = _target_anchor_tokens(target)
    if not target_tokens:
        return 0.0
    answer_tokens_list = _tokenize_text(answer_norm)
    answer_tokens = set(answer_tokens_list)
    matched = [tok for tok in target_tokens if tok in answer_tokens]
    if not matched:
        return 0.0
    distinctive = _distinctive_target_tokens(target)
    matched_distinctive = [tok for tok in distinctive if tok in answer_tokens]
    coverage = len(matched) / len(target_tokens)
    if distinctive and not matched_distinctive:
        return round(min(0.25, 0.1 + 0.15 * coverage), 4)
    if len(target_tokens) == 1:
        return 1.0
    current = 0.0
    if len(matched) >= 2 and coverage >= 0.5:
        current = max(current, min(0.95, 0.55 + 0.45 * coverage))
    if coverage >= 0.34:
        current = max(current, min(0.7, 0.35 + 0.35 * coverage))
    current = max(current, 0.2 + 0.2 * coverage)

    # Allow immediate parent/child or slightly broader/narrower labels inside the same
    # technical cluster to count as a meaningful partial match rather than a miss.
    if distinctive:
        distinct_cov = len(matched_distinctive) / max(1, len(distinctive))
        target_bigrams = {
            f"{distinctive[idx]} {distinctive[idx + 1]}"
            for idx in range(len(distinctive) - 1)
            if distinctive[idx] and distinctive[idx + 1]
        }
        bigram_hits = [bg for bg in target_bigrams if bg in answer_norm]
        if bigram_hits:
            current = max(current, min(0.84, 0.62 + 0.08 * len(bigram_hits) + 0.12 * distinct_cov))
        elif len(matched_distinctive) >= 2:
            if distinct_cov >= 0.5:
                current = max(current, min(0.78, 0.58 + 0.2 * distinct_cov))
            else:
                current = max(current, 0.52)

    return round(float(current), 4)


def _audit_target_matches(answer: str, targets: List[str]) -> Dict[str, Any]:
    matches = []
    for target in targets:
        score = _target_match_score(answer, target)
        if score <= 0.0:
            continue
        level = 'strong' if score >= 0.75 else ('partial' if score >= 0.35 else 'weak')
        matches.append({
            'target': target,
            'score': round(score, 4),
            'level': level,
        })
    matches.sort(key=lambda item: float(item.get('score') or 0.0), reverse=True)
    strong = [m for m in matches if m['level'] == 'strong']
    partial = [m for m in matches if m['level'] in {'strong', 'partial'}]
    return {
        'matches': matches,
        'strong_match_count': len(strong),
        'partial_or_better_count': len(partial),
        'best_match_score': round(float(matches[0]['score']), 4) if matches else 0.0,
        'strong_targets': [m['target'] for m in strong],
        'partial_targets': [m['target'] for m in partial],
    }


def _has_inline_evidence_anchor(answer: str) -> bool:
    if re.search(r'\[P\d+\]', answer):
        return True
    if re.search(r'\b(according to|as shown in|as reported in|evidence from)\b', _normalize_text(answer)):
        return True
    if re.search(r'["\'][^"\']{12,120}["\']', answer):
        return True
    return False


def _phrase_present(norm_text: str, phrase: str) -> bool:
    phrase_norm = _normalize_text(phrase)
    if not norm_text or not phrase_norm:
        return False
    pattern = r'(?<![a-z0-9+])' + re.escape(phrase_norm) + r'(?![a-z0-9+])'
    return re.search(pattern, norm_text) is not None


def _find_phrase_hits(answer: str, phrases: List[str]) -> List[str]:
    norm = _normalize_text(answer)
    hits: List[str] = []
    for phrase in phrases:
        if _phrase_present(norm, phrase):
            hits.append(phrase)
    return _dedupe_keep_order(hits)


def _split_into_segments(answer: str) -> List[str]:
    raw = str(answer or "")
    parts = re.split(r'(?:\n{2,}|[\n\r]+|(?<=[.!?])\s+)', raw)
    cleaned = [segment.strip() for segment in parts if segment and segment.strip()]
    return cleaned


def _segment_contains_any(segment: str, phrases: List[str]) -> bool:
    norm = _normalize_text(segment)
    return any(_phrase_present(norm, phrase) for phrase in phrases)


def _segment_has_inline_evidence(segment: str) -> bool:
    return _has_inline_evidence_anchor(segment)


def _build_strategic_execution_segments(answer: str) -> Dict[str, List[str]]:
    segments = _split_into_segments(answer)
    milestone_segments: List[str] = []
    dependency_action_segments: List[str] = []
    defer_segments: List[str] = []
    risk_segments: List[str] = []
    evidence_action_segments: List[str] = []
    plan_shape_segments: List[str] = []

    for segment in segments:
        has_dependency = _segment_contains_any(segment, STRATEGIC_DEPENDENCY_CUES + STRATEGIC_BLOCKER_CUES)
        has_action = _segment_contains_any(segment, STRATEGIC_ACTION_CUES)
        has_priority = _segment_contains_any(segment, STRATEGIC_SEQUENCING_CUES + STRATEGIC_PRIORITIZATION_CUES)
        has_defer = _segment_contains_any(segment, STRATEGIC_DEFER_CUES + STRATEGIC_COMPARATIVE_CUES)
        has_risk = _segment_contains_any(segment, STRATEGIC_RISK_CUES)
        has_decision_rule = _segment_contains_any(segment, STRATEGIC_DECISION_RULE_CUES)
        has_artifact = _segment_contains_any(segment, STRATEGIC_CONCRETE_ARTIFACT_CUES)
        has_evidence = _segment_has_inline_evidence(segment)

        if has_priority and (has_artifact or has_action):
            plan_shape_segments.append(segment)
        if (has_priority or "first milestone" in _normalize_text(segment) or "first step" in _normalize_text(segment)) and has_artifact:
            milestone_segments.append(segment)
        if has_dependency and (has_action or has_artifact):
            dependency_action_segments.append(segment)
        if has_defer and (has_dependency or has_risk or _segment_contains_any(segment, STRATEGIC_BLOCKER_CUES + STRATEGIC_MOMENTUM_CUES)):
            defer_segments.append(segment)
        if has_risk and has_decision_rule:
            risk_segments.append(segment)
        if has_evidence and (has_action or has_artifact):
            evidence_action_segments.append(segment)

    return {
        'plan_shape_segments': _dedupe_keep_order(plan_shape_segments),
        'milestone_segments': _dedupe_keep_order(milestone_segments),
        'dependency_action_segments': _dedupe_keep_order(dependency_action_segments),
        'defer_segments': _dedupe_keep_order(defer_segments),
        'risk_segments': _dedupe_keep_order(risk_segments),
        'evidence_action_segments': _dedupe_keep_order(evidence_action_segments),
    }


def _build_strategic_execution_audit(answer: str, candidate_targets: List[str]) -> Dict[str, Any]:
    dependency_hits = _find_phrase_hits(answer, STRATEGIC_DEPENDENCY_CUES)
    blocker_hits = _find_phrase_hits(answer, STRATEGIC_BLOCKER_CUES)
    sequencing_hits = _find_phrase_hits(answer, STRATEGIC_SEQUENCING_CUES)
    momentum_hits = _find_phrase_hits(answer, STRATEGIC_MOMENTUM_CUES)
    comparative_hits = _find_phrase_hits(answer, STRATEGIC_COMPARATIVE_CUES)
    near_term_hits = _find_phrase_hits(answer, STRATEGIC_NEAR_TERM_DELIVERABLE_CUES)
    prioritization_hits = _find_phrase_hits(answer, STRATEGIC_PRIORITIZATION_CUES)
    milestone_hits = _find_phrase_hits(answer, STRATEGIC_MILESTONE_CUES)
    action_hits = _find_phrase_hits(answer, STRATEGIC_ACTION_CUES)
    defer_hits = _find_phrase_hits(answer, STRATEGIC_DEFER_CUES)
    risk_hits = _find_phrase_hits(answer, STRATEGIC_RISK_CUES)
    decision_rule_hits = _find_phrase_hits(answer, STRATEGIC_DECISION_RULE_CUES)
    target_match_audit = _audit_target_matches(answer, candidate_targets)
    strong_targets = [str(x) for x in (target_match_audit.get('strong_targets') or []) if str(x).strip()]
    partial_targets = [str(x) for x in (target_match_audit.get('partial_targets') or []) if str(x).strip()]
    segment_audit = _build_strategic_execution_segments(answer)
    plan_shape_segments = [str(x) for x in (segment_audit.get('plan_shape_segments') or []) if str(x).strip()]
    milestone_segments = [str(x) for x in (segment_audit.get('milestone_segments') or []) if str(x).strip()]
    dependency_action_segments = [str(x) for x in (segment_audit.get('dependency_action_segments') or []) if str(x).strip()]
    defer_segments = [str(x) for x in (segment_audit.get('defer_segments') or []) if str(x).strip()]
    risk_segments = [str(x) for x in (segment_audit.get('risk_segments') or []) if str(x).strip()]
    evidence_action_segments = [str(x) for x in (segment_audit.get('evidence_action_segments') or []) if str(x).strip()]
    has_plan_shape = bool(plan_shape_segments)
    has_first_milestone = bool(milestone_segments)
    has_dependency_to_action_chain = bool(dependency_action_segments)
    has_alternative_defer_rationale = bool(defer_segments)
    has_risk_or_kill_criteria = bool(risk_segments)
    has_evidence_to_action_mapping = bool(evidence_action_segments)
    momentum_dominant = len(momentum_hits) >= 2 and not has_plan_shape and not has_risk_or_kill_criteria
    essay_dominant = len(str(answer or '').split()) >= 230 and len(momentum_hits) >= 2 and not has_first_milestone and not has_evidence_to_action_mapping
    return {
        'dependency_cue_hits': dependency_hits,
        'blocker_cue_hits': blocker_hits,
        'sequencing_cue_hits': sequencing_hits,
        'momentum_cue_hits': momentum_hits,
        'comparative_cue_hits': comparative_hits,
        'near_term_deliverable_hits': near_term_hits,
        'prioritization_cue_hits': prioritization_hits,
        'milestone_cue_hits': milestone_hits,
        'action_cue_hits': action_hits,
        'defer_cue_hits': defer_hits,
        'risk_cue_hits': risk_hits,
        'decision_rule_cue_hits': decision_rule_hits,
        'strong_target_count': len(strong_targets),
        'partial_target_count': len(partial_targets),
        'plan_shape_segments': plan_shape_segments[:3],
        'milestone_segments': milestone_segments[:3],
        'dependency_action_segments': dependency_action_segments[:3],
        'defer_segments': defer_segments[:3],
        'risk_segments': risk_segments[:3],
        'evidence_action_segments': evidence_action_segments[:3],
        'has_plan_shape': has_plan_shape,
        'has_first_milestone': has_first_milestone,
        'has_dependency_to_action_chain': has_dependency_to_action_chain,
        'has_alternative_defer_rationale': has_alternative_defer_rationale,
        'has_risk_or_kill_criteria': has_risk_or_kill_criteria,
        'has_evidence_to_action_mapping': has_evidence_to_action_mapping,
        'momentum_dominant_without_execution_detail': momentum_dominant,
        'essay_dominant_without_operational_plan': essay_dominant,
    }


def _build_venue_positioning_audit(
    answer: str,
    *,
    candidate_targets: List[str],
    answer_buckets: List[str],
    target_bucket: str,
    acceptable_buckets: List[str],
    prior_knowledge: Dict[str, Any],
) -> Dict[str, Any]:
    package_hits = _find_phrase_hits(answer, VENUE_PAPER_PACKAGE_CUES)
    reviewer_hits = _find_phrase_hits(answer, VENUE_REVIEWER_EXPECTATION_CUES)
    contrast_hits = _find_phrase_hits(answer, VENUE_CONTRAST_CUES)
    prestige_hits = _find_phrase_hits(answer, VENUE_PRESTIGE_RHETORIC_CUES)
    primary_secondary_hits = _find_phrase_hits(answer, VENUE_PRIMARY_SECONDARY_CUES)
    target_match_audit = _audit_target_matches(answer, candidate_targets)
    mentioned_buckets = [str(x) for x in (answer_buckets or []) if str(x).strip()]
    acceptable_bucket_hits = [bucket for bucket in mentioned_buckets if bucket in set(acceptable_buckets or [])]
    has_bucket_contrast = len(mentioned_buckets) >= 2 or (bool(contrast_hits) and bool(target_bucket) and target_bucket in mentioned_buckets)
    has_paper_package = bool(package_hits)
    has_reviewer_grounding = bool(reviewer_hits) or bool(package_hits)
    prestige_dominant = len(prestige_hits) >= 2 and not has_paper_package and not has_bucket_contrast
    primary_profile = prior_knowledge.get('primary_profile') or {}
    prior_hits = _profile_phrase_hits(
        answer,
        primary_profile,
        ['scope_signals', 'preferred_contribution_signals', 'reviewer_expectation_signals'],
    ) if primary_profile else {'scope_signals': [], 'preferred_contribution_signals': [], 'reviewer_expectation_signals': []}
    prior_scope_hits = [str(x) for x in (prior_hits.get('scope_signals') or []) if str(x).strip()]
    prior_contribution_hits = [str(x) for x in (prior_hits.get('preferred_contribution_signals') or []) if str(x).strip()]
    prior_expectation_hits = [str(x) for x in (prior_hits.get('reviewer_expectation_signals') or []) if str(x).strip()]
    compatible_secondary_hits = [bucket for bucket in acceptable_bucket_hits if bucket != target_bucket]
    has_primary_secondary_reasoning = bool(primary_secondary_hits) or (bool(compatible_secondary_hits) and bool(contrast_hits))
    return {
        'paper_package_hits': package_hits,
        'reviewer_expectation_hits': reviewer_hits,
        'contrastive_venue_hits': contrast_hits,
        'prestige_rhetoric_hits': prestige_hits,
        'primary_secondary_fit_hits': primary_secondary_hits,
        'has_concrete_paper_package': has_paper_package,
        'has_reviewer_expectation_grounding': has_reviewer_grounding,
        'has_bucket_contrast': has_bucket_contrast,
        'acceptable_bucket_hits': acceptable_bucket_hits,
        'has_acceptable_bucket_alignment': bool(acceptable_bucket_hits),
        'compatible_secondary_bucket_hits': compatible_secondary_hits,
        'has_primary_secondary_reasoning': has_primary_secondary_reasoning,
        'primary_prior_family_name': str(primary_profile.get('family_name') or ''),
        'prior_scope_hits': prior_scope_hits,
        'prior_contribution_hits': prior_contribution_hits,
        'prior_expectation_hits': prior_expectation_hits,
        'has_prior_contribution_fit': bool(prior_contribution_hits) or bool(prior_scope_hits),
        'has_prior_expectation_fit': bool(prior_expectation_hits),
        'prestige_dominant_without_package': prestige_dominant,
        'strong_target_count': int(target_match_audit.get('strong_match_count') or 0),
        'partial_target_count': int(target_match_audit.get('partial_or_better_count') or 0),
    }


def _build_bottleneck_unlock_audit(answer: str, candidate_targets: List[str]) -> Dict[str, Any]:
    causal_hits = _find_phrase_hits(answer, BOTTLENECK_CAUSAL_CUES)
    artifact_hits = _find_phrase_hits(answer, BOTTLENECK_ARTIFACT_CUES)
    symptom_hits = _find_phrase_hits(answer, BOTTLENECK_SYMPTOM_CUES)
    multi_hop_hits = _find_phrase_hits(answer, BOTTLENECK_MULTI_HOP_CUES)
    target_match_audit = _audit_target_matches(answer, candidate_targets)
    has_unlock_linkage = bool(causal_hits) and any(phrase in _normalize_text(answer) for phrase in ["unlock", "enable", "becomes viable", "becomes possible"])
    artifact_like_opportunity = bool(artifact_hits) and not has_unlock_linkage
    symptom_like_bottleneck = bool(symptom_hits) and not bool(causal_hits)
    return {
        'causal_linkage_hits': causal_hits,
        'artifact_like_hits': artifact_hits,
        'symptom_like_hits': symptom_hits,
        'multi_hop_hits': multi_hop_hits,
        'has_explicit_unlock_linkage': has_unlock_linkage,
        'artifact_like_opportunity': artifact_like_opportunity,
        'symptom_like_bottleneck': symptom_like_bottleneck,
        'multi_hop_unlock': bool(multi_hop_hits),
        'strong_target_count': int(target_match_audit.get('strong_match_count') or 0),
        'partial_target_count': int(target_match_audit.get('partial_or_better_count') or 0),
    }


def _build_contract_audit(public_task: Dict[str, Any], hidden_row: Dict[str, Any], result_row: Dict[str, Any], family: str) -> Dict[str, Any]:
    answer = str(result_row.get('answer') or '')
    support = _collect_support_items(result_row)
    candidate_targets = _extract_candidate_direction_targets(public_task, hidden_row, family)
    exact_targets = _find_mentioned_targets(answer, candidate_targets)
    target_match_audit = _audit_target_matches(answer, candidate_targets)
    answer_bucket_mentions = _extract_answer_bucket_mentions(answer)
    target_venue_bucket = _extract_target_venue_bucket(public_task)
    acceptable_venue_buckets = _extract_acceptable_target_venue_buckets(public_task, hidden_row)
    venue_prior_knowledge = _resolve_venue_prior_knowledge(public_task, hidden_row)
    audit = {
        'candidate_direction_targets': candidate_targets,
        'mentioned_candidate_targets': exact_targets,
        'candidate_direction_count': len(candidate_targets),
        'mentioned_candidate_count': len(exact_targets),
        'target_match_audit': target_match_audit,
        'answer_bucket_mentions': answer_bucket_mentions,
        'answer_bucket_mentions_count': len(answer_bucket_mentions),
        'target_venue_bucket': target_venue_bucket,
        'acceptable_target_venue_buckets': acceptable_venue_buckets,
        'venue_prior_knowledge': venue_prior_knowledge,
        'requires_explicit_venue_bucket': family == 'venue_aware_research_positioning',
        'has_external_support_artifact': bool(support.get('has_external_support')),
        'has_inline_evidence_anchor': _has_inline_evidence_anchor(answer),
        'deliverable_requirements': list((public_task.get('deliverable_spec') or {}).get('requirements') or []),
    }
    if family == 'bottleneck_opportunity_discovery':
        audit['bottleneck_unlock_audit'] = _build_bottleneck_unlock_audit(answer, candidate_targets)
    elif family == 'strategic_research_planning':
        audit['strategic_execution_audit'] = _build_strategic_execution_audit(answer, candidate_targets)
    elif family == 'venue_aware_research_positioning':
        audit['venue_positioning_audit'] = _build_venue_positioning_audit(
            answer,
            candidate_targets=candidate_targets,
            answer_buckets=answer_bucket_mentions,
            target_bucket=target_venue_bucket,
            acceptable_buckets=acceptable_venue_buckets,
            prior_knowledge=venue_prior_knowledge,
        )
    return audit


def _apply_family_caps(
    *,
    family: str,
    public_task: Dict[str, Any],
    contract_audit: Dict[str, Any],
    overall: float,
) -> tuple[float, List[str]]:
    caps: List[float] = []
    reasons: List[str] = []
    candidate_count = int(contract_audit.get('candidate_direction_count') or 0)
    match_audit = contract_audit.get('target_match_audit') or {}
    mentioned_count = int(contract_audit.get('mentioned_candidate_count') or 0)
    strong_count = int(match_audit.get('strong_match_count') or 0)
    partial_count = int(match_audit.get('partial_or_better_count') or 0)
    best_match_score = float(match_audit.get('best_match_score') or 0.0)
    has_support = bool(contract_audit.get('has_external_support_artifact'))
    has_inline_anchor = bool(contract_audit.get('has_inline_evidence_anchor'))
    answer_buckets = [str(x) for x in (contract_audit.get('answer_bucket_mentions') or []) if str(x).strip()]
    target_bucket = str(contract_audit.get('target_venue_bucket') or '').strip()
    acceptable_buckets = [str(x) for x in (contract_audit.get('acceptable_target_venue_buckets') or []) if str(x).strip()]

    if family == 'bottleneck_opportunity_discovery':
        bottleneck_audit = contract_audit.get('bottleneck_unlock_audit') or {}
        causal_hits = [str(x) for x in (bottleneck_audit.get('causal_linkage_hits') or []) if str(x).strip()]
        artifact_hits = [str(x) for x in (bottleneck_audit.get('artifact_like_hits') or []) if str(x).strip()]
        symptom_hits = [str(x) for x in (bottleneck_audit.get('symptom_like_hits') or []) if str(x).strip()]
        multi_hop_hits = [str(x) for x in (bottleneck_audit.get('multi_hop_hits') or []) if str(x).strip()]
        has_unlock_linkage = bool(bottleneck_audit.get('has_explicit_unlock_linkage'))
        artifact_like_opportunity = bool(bottleneck_audit.get('artifact_like_opportunity'))
        symptom_like_bottleneck = bool(bottleneck_audit.get('symptom_like_bottleneck'))
        multi_hop_unlock = bool(bottleneck_audit.get('multi_hop_unlock'))
        if candidate_count and best_match_score < 0.14:
            caps.append(0.48)
            reasons.append('No task-relevant bottleneck or opportunity anchor is explicitly referenced in the answer.')
        if not has_unlock_linkage and not causal_hits:
            caps.append(0.62)
            reasons.append('Bottleneck answer does not clearly explain how solving the bottleneck unlocks the stated opportunity.')
        if artifact_like_opportunity and artifact_hits:
            caps.append(0.64)
            reasons.append('Opportunity is described more like an artifact noun than an immediate capability or study unlocked by solving the bottleneck.')
        if symptom_like_bottleneck and symptom_hits:
            caps.append(0.66)
            reasons.append('Named bottleneck looks more like a downstream symptom than an upstream technical cause.')
        if multi_hop_unlock and multi_hop_hits:
            caps.append(0.66)
            reasons.append('Opportunity appears too distant or multi-hop rather than an immediate next move after addressing the bottleneck.')
        if not has_support and not has_inline_anchor:
            caps.append(0.72)
            reasons.append('Bottleneck answer does not attach auditable support for its causal linkage.')
    elif family == 'strategic_research_planning':
        strategic_audit = contract_audit.get('strategic_execution_audit') or {}
        dependency_hits = [str(x) for x in (strategic_audit.get('dependency_cue_hits') or []) if str(x).strip()]
        comparative_hits = [str(x) for x in (strategic_audit.get('comparative_cue_hits') or []) if str(x).strip()]
        prioritization_hits = [str(x) for x in (strategic_audit.get('prioritization_cue_hits') or []) if str(x).strip()]
        milestone_hits = [str(x) for x in (strategic_audit.get('milestone_cue_hits') or []) if str(x).strip()]
        action_hits = [str(x) for x in (strategic_audit.get('action_cue_hits') or []) if str(x).strip()]
        defer_hits = [str(x) for x in (strategic_audit.get('defer_cue_hits') or []) if str(x).strip()]
        risk_hits = [str(x) for x in (strategic_audit.get('risk_cue_hits') or []) if str(x).strip()]
        decision_rule_hits = [str(x) for x in (strategic_audit.get('decision_rule_cue_hits') or []) if str(x).strip()]
        plan_shape_segments = [str(x) for x in (strategic_audit.get('plan_shape_segments') or []) if str(x).strip()]
        has_first_milestone = bool(strategic_audit.get('has_first_milestone'))
        has_plan_shape = bool(strategic_audit.get('has_plan_shape'))
        has_dependency_to_action_chain = bool(strategic_audit.get('has_dependency_to_action_chain'))
        has_alternative_defer_rationale = bool(strategic_audit.get('has_alternative_defer_rationale'))
        has_risk_or_kill_criteria = bool(strategic_audit.get('has_risk_or_kill_criteria'))
        has_evidence_to_action_mapping = bool(strategic_audit.get('has_evidence_to_action_mapping'))
        momentum_dominant = bool(strategic_audit.get('momentum_dominant_without_execution_detail'))
        essay_dominant = bool(strategic_audit.get('essay_dominant_without_operational_plan'))
        if candidate_count and best_match_score < 0.2:
            caps.append(0.45)
            reasons.append('No listed candidate direction is explicitly referenced in the answer.')
        elif candidate_count and strong_count == 0:
            caps.append(0.68)
            reasons.append('Strategic answer only weakly engages the listed candidate directions and remains loosely anchored to the task contract.')
        if candidate_count >= 2 and partial_count < 2:
            caps.append(0.68)
            reasons.append('Comparative strategic task does not visibly engage both listed candidate directions.')
        if not has_plan_shape and not plan_shape_segments:
            caps.append(0.58)
            reasons.append('Strategic answer is written mostly as a comparative essay and does not expose an operational plan-shaped first move.')
        if not has_first_milestone and not milestone_hits:
            caps.append(0.60)
            reasons.append('Strategic answer does not name a concrete first milestone, prototype, or six-month deliverable.')
        if not has_dependency_to_action_chain and not (dependency_hits and action_hits):
            caps.append(0.62)
            reasons.append('Strategic answer does not connect the stated dependency or bottleneck to a concrete action plan.')
        if candidate_count >= 2 and not has_alternative_defer_rationale and not defer_hits and not comparative_hits and not prioritization_hits:
            caps.append(0.60)
            reasons.append('Comparative strategic answer does not explicitly justify why the alternative should be deferred in the same six-month window.')
        if not has_risk_or_kill_criteria and not (risk_hits and decision_rule_hits):
            caps.append(0.55)
            reasons.append('Strategic answer does not state a concrete risk, failure trigger, or kill criterion for the proposed bet.')
        if not has_evidence_to_action_mapping:
            caps.append(0.58)
            reasons.append('Strategic answer cites evidence but does not map it clearly onto specific action steps or milestones.')
        if momentum_dominant:
            caps.append(0.48)
            reasons.append('Strategic justification leans on momentum, citation, or venue rhetoric without execution-ready detail.')
        if essay_dominant:
            caps.append(0.50)
            reasons.append('Strategic answer remains a long-form prioritization essay without enough operational planning detail for a six-month bet.')
        if not has_support and not has_inline_anchor:
            caps.append(0.70)
            reasons.append('Strategic answer does not attach auditable support for its execution plan.')
    elif family == 'venue_aware_research_positioning':
        venue_audit = contract_audit.get('venue_positioning_audit') or {}
        package_hits = [str(x) for x in (venue_audit.get('paper_package_hits') or []) if str(x).strip()]
        reviewer_hits = [str(x) for x in (venue_audit.get('reviewer_expectation_hits') or []) if str(x).strip()]
        contrast_hits = [str(x) for x in (venue_audit.get('contrastive_venue_hits') or []) if str(x).strip()]
        primary_secondary_hits = [str(x) for x in (venue_audit.get('primary_secondary_fit_hits') or []) if str(x).strip()]
        acceptable_bucket_hits = [str(x) for x in (venue_audit.get('acceptable_bucket_hits') or []) if str(x).strip()]
        compatible_secondary_hits = [str(x) for x in (venue_audit.get('compatible_secondary_bucket_hits') or []) if str(x).strip()]
        prior_scope_hits = [str(x) for x in (venue_audit.get('prior_scope_hits') or []) if str(x).strip()]
        prior_contribution_hits = [str(x) for x in (venue_audit.get('prior_contribution_hits') or []) if str(x).strip()]
        prior_expectation_hits = [str(x) for x in (venue_audit.get('prior_expectation_hits') or []) if str(x).strip()]
        has_paper_package = bool(venue_audit.get('has_concrete_paper_package'))
        has_reviewer_grounding = bool(venue_audit.get('has_reviewer_expectation_grounding'))
        has_bucket_contrast = bool(venue_audit.get('has_bucket_contrast'))
        has_acceptable_alignment = bool(venue_audit.get('has_acceptable_bucket_alignment'))
        has_primary_secondary_reasoning = bool(venue_audit.get('has_primary_secondary_reasoning'))
        has_prior_contribution_fit = bool(venue_audit.get('has_prior_contribution_fit'))
        has_prior_expectation_fit = bool(venue_audit.get('has_prior_expectation_fit'))
        prestige_dominant = bool(venue_audit.get('prestige_dominant_without_package'))
        if candidate_count and best_match_score < 0.2:
            caps.append(0.50)
            reasons.append('No future-theme or candidate direction anchor is explicitly referenced in the venue-positioning answer.')
        elif candidate_count and strong_count == 0:
            caps.append(0.70)
            reasons.append('Venue-positioning answer remains only loosely anchored to the task-specific future themes.')
        if not answer_buckets:
            caps.append(0.45)
            reasons.append('No explicit venue bucket is named in the answer.')
        elif not has_acceptable_alignment and acceptable_buckets:
            caps.append(0.60)
            reasons.append('Venue-positioning answer does not stay within the acceptable nearby venue families for this task.')
        elif target_bucket and target_bucket not in answer_buckets and acceptable_bucket_hits and str(public_task.get('subtype') or '') == 'venue_targeted_planning':
            caps.append(0.76)
            reasons.append('Answer mentions a nearby compatible venue family but does not clearly state why the named target venue bucket should be the primary fit.')
        elif target_bucket and target_bucket not in answer_buckets and str(public_task.get('subtype') or '') == 'venue_targeted_planning':
            caps.append(0.60)
            reasons.append('Venue-targeted planning answer does not explicitly stay anchored to the named target venue bucket.')
        if len(acceptable_bucket_hits) >= 2 and not has_primary_secondary_reasoning and not primary_secondary_hits:
            caps.append(0.72)
            reasons.append('Answer invokes multiple compatible venue families but does not distinguish primary fit from secondary fit.')
        if not has_bucket_contrast and not contrast_hits:
            caps.append(0.60)
            reasons.append('Venue-positioning answer does not explain why this venue family is a better fit than nearby alternatives.')
        if not has_paper_package and not package_hits:
            caps.append(0.65)
            reasons.append('Venue-positioning answer does not specify a concrete paper package such as benchmark, ablation, theory, or systems evidence.')
        if not has_reviewer_grounding and not reviewer_hits:
            caps.append(0.70)
            reasons.append('Venue-positioning answer does not make reviewer expectations or evidence expectations explicit.')
        if target_bucket and not has_prior_contribution_fit and not (prior_scope_hits or prior_contribution_hits):
            caps.append(0.72)
            reasons.append('Answer names or implies a venue family but does not align the proposed contribution package with that venue family\'s typical accepted profile.')
        if target_bucket and not has_prior_expectation_fit and not prior_expectation_hits:
            caps.append(0.74)
            reasons.append('Answer does not connect its evaluation or evidence package to the target venue family\'s typical reviewer expectations.')
        if prestige_dominant:
            caps.append(0.50)
            reasons.append('Venue-positioning answer relies on prestige rhetoric without concrete venue-fit evidence.')
        if not has_support and not has_inline_anchor:
            caps.append(0.70)
            reasons.append('Venue-positioning answer does not attach auditable support for its venue-fit reasoning.')

    if not caps:
        return overall, reasons
    return round(min(overall, min(caps)), 4), reasons


def _resolve_family(public_task: Dict[str, Any], hidden_row: Dict[str, Any]) -> str:
    public_family = str(public_task.get('family') or '')
    public_subtype = str(public_task.get('subtype') or '')
    hidden_family = str(hidden_row.get('family') or '')
    if hidden_family == 'venue_aware_research_positioning':
        return hidden_family
    if public_subtype in {'venue_aware_direction_forecast', 'venue_targeted_planning'}:
        return 'venue_aware_research_positioning'
    return public_family or hidden_family
def evaluate_family_auxiliary(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    result_row: Dict[str, Any],
) -> Dict[str, Any]:
    family = _resolve_family(public_task, hidden_row)
    answer = str(result_row.get('answer') or '')
    dimensions = AUX_DIMENSIONS_BY_FAMILY.get(family) or []
    score_key = AUX_NAME_BY_FAMILY.get(family) or 'family_aux_score'
    label = AUX_LABEL_BY_FAMILY.get(family) or 'Family Auxiliary'
    if not dimensions:
        return {
            'family_aux_metric_name': label,
            score_key: 0.0,
            'rubric_scores': {},
            'strengths': [],
            'weaknesses': ['Unsupported family for auxiliary evaluation.'],
        }
    support_snapshot = _render_support_snapshot(result_row)
    slot_targets = hidden_row.get('slot_targets') or {}
    component_targets = hidden_row.get('component_targets') or {}
    contract_audit = _build_contract_audit(public_task, hidden_row, result_row, family)

    if family == 'bottleneck_opportunity_discovery':
        rubric_text = (
            '1. causal_linkage: Does the answer explain why the identified opportunity would become viable if the bottleneck were addressed?\n'
            '2. technical_plausibility: Is the bottleneck-opportunity pair technically coherent rather than a loose thematic association?'
        )
        extra_context = {
            'topic': slot_targets.get('topic_title') or slot_targets.get('topic'),
            'historical_bottlenecks': slot_targets.get('core_bottleneck_labels') or [],
            'future_opportunities': slot_targets.get('core_opportunity_labels') or slot_targets.get('future_themes') or [],
            'bottleneck_unlock_audit': contract_audit.get('bottleneck_unlock_audit') or {},
        }
    elif family == 'direction_forecasting':
        rubric_text = (
            '1. signal_grounding: Does the forecast clearly arise from observable pre-cutoff signals, frictions, or momentum cues?\n'
            '2. forecast_discipline: Does the answer distinguish grounded expectations from speculation and keep the trajectory call calibrated?'
        )
        extra_context = {
            'topic': slot_targets.get('topic_title') or slot_targets.get('topic'),
            'history_stats': slot_targets.get('historical_stats') or {},
            'trajectory_support': slot_targets.get('trajectory_support') or {},
        }
    elif family == 'strategic_research_planning':
        candidate_count = int(contract_audit.get('candidate_direction_count') or 0)
        rubric_text = (
            'Important: task-contract fidelity is a hard requirement handled separately via caps; do not treat merely staying inside the listed options as a positive dimension.\n'
            'Treat this as a decision-ready execution metric, not a generic prioritization essay metric.\n'
            'Ranking directions and citing papers is not enough. High scores require at least one operational first move.\n'
            '1. first_milestone_specificity: Does the answer name a concrete first milestone, prototype, benchmark, data asset, instrumentation step, or other six-month deliverable?\n'
            '2. dependency_to_action_chain: Does the answer connect a real blocker or dependency to a concrete action sequence, rather than only describing why the direction matters?\n'
            '3. alternative_defer_rationale: For multi-option tasks, does the answer explicitly explain why the alternative should be deferred under the same limited time / budget window?\n'
            '4. risk_and_kill_criteria: Does the answer state what could fail, what uncertainty matters most, or what evidence would cause the team to stop, pivot, or downgrade this bet?\n'
            '5. evidence_to_action_mapping: Are specific pre-cutoff evidence items tied to specific action claims, milestones, or de-risking steps rather than serving as generic background support?'
        )
        extra_context = {
            'topic': slot_targets.get('topic_title') or slot_targets.get('topic'),
            'priority_candidates': slot_targets.get('priority_direction_labels') or slot_targets.get('future_themes') or [],
            'candidate_count': candidate_count,
            'public_subtype': public_task.get('subtype') or '',
            'strategic_execution_audit': contract_audit.get('strategic_execution_audit') or {},
        }
    else:
        rubric_text = (
            '1. venue_specific_contribution_fit: Does the answer explain why the proposed contribution type fits the target venue family\'s typical accepted profile, not just the topic area?\n'
            '2. reviewer_expectation_grounding: Does the answer make explicit what evidence package, evaluation package, or artifact reviewers in this venue family would typically expect?\n'
            '3. paper_package_specificity: Does the answer specify a concrete paper package such as benchmark design, ablation-heavy empirical study, systems artifact, theory, resource, or human evaluation?\n'
            '4. contrastive_venue_discrimination: Does the answer explain why this venue family is the primary fit relative to nearby compatible venues, especially when more than one venue family could plausibly fit?'
        )
        extra_context = {
            'topic': slot_targets.get('topic_title') or slot_targets.get('topic'),
            'future_themes': slot_targets.get('future_themes') or [],
            'target_venue_bucket': _extract_target_venue_bucket(public_task),
            'acceptable_target_venue_buckets': _extract_acceptable_target_venue_buckets(public_task, hidden_row),
            'venue_prior_knowledge': contract_audit.get('venue_prior_knowledge') or {},
            'public_subtype': public_task.get('subtype') or '',
            'venue_positioning_audit': contract_audit.get('venue_positioning_audit') or {},
        }

    prompt = f"""# Role
You are a family-specific auxiliary judge for a forward-looking research benchmark.

# Objective
Evaluate the answer on the task-family-specific reasoning quality that is NOT fully captured by generic factuality or traceability metrics.

# Rules
- All scores must be between 0.0 and 1.0.
- Judge the answer on this family-specific criterion only.
- Prefer technical coherence, causal structure, and disciplined reasoning over polished prose.
- Use the support snapshot to understand whether the answer's reasoning is connected to evidence, but do not reduce this to citation counting.
- Do NOT judge based on whether the realized future outcome actually happened; this metric is about family-specific reasoning quality.
- Enforce task contract fidelity. If the answer ignores listed candidate directions, invents substitute directions, or omits an explicitly required venue bucket, score strictly.
- For bottleneck and forecasting tasks, do NOT require exact phrase equality with hidden labels. If the answer stays within the same immediate technical cluster and is only slightly broader or narrower than the canonical label, score it as reasonably aligned rather than as a miss.
- Penalize bottleneck and forecasting answers for target mismatch only when they drift to a materially different mechanism, bottleneck family, or future-direction family.
- Do not award high scores for polished but weakly-auditable venue rhetoric.
- For strategic research planning, do NOT award high scores for generic prioritization essays. High scores require concrete execution detail: plan-shaped first move, milestone, dependency-to-action logic, defer rationale, risk / kill criteria, and evidence-to-action mapping.
- For venue positioning, do NOT award high scores for answers that only say a venue is high-impact, broad, or prestigious without specifying the contribution package and reviewer expectations.
- For venue positioning, use the provided venue prior knowledge as a soft prior about what kinds of contribution packages and reviewer expectations are typical for the target venue family.
- For venue positioning, nearby compatible venue families can still be acceptable, but high scores require the answer to distinguish primary fit from secondary fit rather than treating all nearby venues as interchangeable.
- For bottleneck tasks, do NOT award high scores if the opportunity is merely an artifact noun unless the answer explicitly explains the concrete study or capability it enables.

# Input Data
- Public Task Definition: {json.dumps(_compact_public_task(public_task), ensure_ascii=False, indent=2)}
- Family: {family}
- Family Context: {json.dumps(extra_context, ensure_ascii=False, indent=2)}
- Hidden Component Map: {json.dumps(component_targets, ensure_ascii=False, indent=2)}
- Deliverable Requirements: {json.dumps(list((public_task.get('deliverable_spec') or {}).get('requirements') or []), ensure_ascii=False)}
- Contract Audit: {json.dumps(contract_audit, ensure_ascii=False, indent=2)}
- Support Snapshot: {support_snapshot}
- Rubric Dimensions: {json.dumps(dimensions, ensure_ascii=False)}

# Candidate Answer
{answer}

# Family-Specific Rubric
{rubric_text}

# Output (Strict JSON)
{{
  "dimension_scores": {{"dimension_name": 0.0}},
  "{score_key}": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."]
}}
"""
    obj = complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict family-specific research benchmark judge. Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=1000,
        timeout=120,
        transport_retries=2,
        max_parse_attempts=3,
        repair_instruction=f"Your previous response was malformed JSON. Return exactly one valid JSON object with keys dimension_scores, {score_key}, strengths, weaknesses.",
    )
    overall, rubric_scores = _weighted(dimensions, obj.get('dimension_scores') or {}, score_key, obj)
    overall, cap_reasons = _apply_family_caps(
        family=family,
        public_task=public_task,
        contract_audit=contract_audit,
        overall=overall,
    )
    weaknesses = [str(x) for x in (obj.get('weaknesses') or []) if str(x).strip()]
    for reason in cap_reasons:
        if reason not in weaknesses:
            weaknesses.append(reason)
    return {
        'family_aux_metric_name': label,
        score_key: overall,
        'rubric_scores': rubric_scores,
        'strengths': [str(x) for x in (obj.get('strengths') or []) if str(x).strip()],
        'weaknesses': weaknesses,
        'contract_audit': contract_audit,
    }


def build_aux_result_row(
    *,
    run_id: str,
    public_task: Dict[str, Any],
    hidden_row: Dict[str, Any],
    result_row: Dict[str, Any],
    family_aux_eval: Dict[str, Any],
) -> Dict[str, Any]:
    family = _resolve_family(public_task, hidden_row)
    score_key = AUX_NAME_BY_FAMILY.get(family) or 'family_aux_score'
    return {
        'run_id': run_id,
        'task_id': public_task.get('task_id'),
        'family': family,
        'domain': infer_domain_id(result_row) or public_task.get('domain'),
        'method': str(result_row.get('agent') or result_row.get('baseline') or 'unknown'),
        'answer': str(result_row.get('answer') or ''),
        'metadata': {
            'task_title': public_task.get('title'),
            'time_cutoff': public_task.get('time_cutoff'),
            'schema_version': 'aux_v4',
        },
        'scores': {
            score_key: round(float(family_aux_eval.get(score_key) or 0.0), 4),
        },
        'family_aux_eval': family_aux_eval,
    }


def summarize_aux_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _family_score(row: Dict[str, Any], family: str) -> float:
        scores = row.get('scores') or {}
        if family == 'strategic_research_planning':
            return float(
                scores.get('strategic_execution_grounding_score')
                or scores.get('strategic_priority_grounding_score')
                or scores.get('technical_dependency_grounding_score')
                or 0.0
            )
        score_key = AUX_NAME_BY_FAMILY.get(family) or 'family_aux_score'
        return float(scores.get(score_key) or 0.0)

    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get('family') or '')].append(row)
        by_domain[str(row.get('domain') or '')].append(row)

    def _group_summary(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not group:
            return {}
        family = str(group[0].get('family') or '')
        score_key = AUX_NAME_BY_FAMILY.get(family) or 'family_aux_score'
        return {
            'count': len(group),
            'mean_family_aux_score': _mean([_family_score(row, family) for row in group]),
            'family_aux_metric_name': AUX_LABEL_BY_FAMILY.get(family) or score_key,
            'family_aux_score_key': score_key,
            'rubric_dimension_summary': _rubric_dimension_summary(group, family),
        }

    family_summary = {key: _group_summary(group) for key, group in sorted(by_family.items())}
    domain_summary = {}
    for key, group in sorted(by_domain.items()):
        domain_summary[key] = {
            'count': len(group),
        }
    family_domain_summary = {}
    for family, fam_group in sorted(by_family.items()):
        fam_by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in fam_group:
            fam_by_domain[str(row.get('domain') or '')].append(row)
        family_domain_summary[family] = {
            domain: {
                'count': len(group),
                'mean_family_aux_score': _mean([_family_score(row, family) for row in group]),
                'rubric_dimension_summary': _rubric_dimension_summary(group, family),
            }
            for domain, group in sorted(fam_by_domain.items())
        }
    return {
        'task_count': len(rows),
        'family_summary': family_summary,
        'domain_summary': domain_summary,
        'family_domain_summary': family_domain_summary,
    }


def write_aux_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get('method') or '')].append(row)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['Method', 'BottleneckOpportunityAux', 'DirectionForecastAux', 'StrategicExecutionAux', 'VenuePositioningAux'])
        writer.writeheader()
        for method, group in sorted(by_method.items()):
            fam_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for row in group:
                fam_groups[str(row.get('family') or '')].append(row)
            writer.writerow({
                'Method': method,
                'BottleneckOpportunityAux': round(sum(float((row.get('scores') or {}).get('opportunity_grounding_score') or 0.0) for row in fam_groups.get('bottleneck_opportunity_discovery', [])) / max(1, len(fam_groups.get('bottleneck_opportunity_discovery', []))), 4) if fam_groups.get('bottleneck_opportunity_discovery') else 0.0,
                'DirectionForecastAux': round(sum(float((row.get('scores') or {}).get('forecast_grounding_score') or 0.0) for row in fam_groups.get('direction_forecasting', [])) / max(1, len(fam_groups.get('direction_forecasting', []))), 4) if fam_groups.get('direction_forecasting') else 0.0,
                'StrategicExecutionAux': round(sum(float((row.get('scores') or {}).get('strategic_execution_grounding_score') or (row.get('scores') or {}).get('strategic_priority_grounding_score') or (row.get('scores') or {}).get('technical_dependency_grounding_score') or 0.0) for row in fam_groups.get('strategic_research_planning', [])) / max(1, len(fam_groups.get('strategic_research_planning', []))), 4) if fam_groups.get('strategic_research_planning') else 0.0,
                'VenuePositioningAux': round(sum(float((row.get('scores') or {}).get('venue_positioning_grounding_score') or 0.0) for row in fam_groups.get('venue_aware_research_positioning', [])) / max(1, len(fam_groups.get('venue_aware_research_positioning', []))), 4) if fam_groups.get('venue_aware_research_positioning') else 0.0,
            })


def write_rubric_breakdown_csv_aux(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_aux_results(rows)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['scope_type', 'scope_value', 'family', 'metric', 'dimension', 'mean_score', 'count'])
        writer.writeheader()
        for family, group in sorted((summary.get('family_summary') or {}).items()):
            for dimension, score in (group.get('rubric_dimension_summary') or {}).items():
                writer.writerow(
                    {
                        'scope_type': 'family',
                        'scope_value': family,
                        'family': family,
                        'metric': group.get('family_aux_metric_name') or AUX_LABEL_BY_FAMILY.get(family) or '',
                        'dimension': dimension,
                        'mean_score': score,
                        'count': int(group.get('count') or 0),
                    }
                )
        for family, domain_map in sorted((summary.get('family_domain_summary') or {}).items()):
            metric_name = (summary.get('family_summary') or {}).get(family, {}).get('family_aux_metric_name') or AUX_LABEL_BY_FAMILY.get(family) or ''
            for domain, group in sorted((domain_map or {}).items()):
                for dimension, score in (group.get('rubric_dimension_summary') or {}).items():
                    writer.writerow(
                        {
                            'scope_type': 'family_domain',
                            'scope_value': f'{family}::{domain}',
                            'family': family,
                            'metric': metric_name,
                            'dimension': dimension,
                            'mean_score': score,
                            'count': int(group.get('count') or 0),
                        }
                    )
