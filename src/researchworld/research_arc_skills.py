from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from statistics import median
from typing import Any, Dict, Iterable, List, Optional

from researchworld.llm import OpenAICompatChatClient, complete_json_object
from researchworld.offline_kb import (
    OfflineKnowledgeBase,
    clip_text,
    dedupe,
    merge_multi_query_results,
)


STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "over", "under", "based", "using", "available", "before",
    "after", "through", "within", "large", "language", "model", "models", "systems", "system", "research",
    "trajectory", "forecasting", "forecast", "strategic", "planning", "agenda", "discovery", "opportunity",
    "bottleneck", "evaluation", "frameworks", "framework", "methods", "method", "tasks", "task", "domain",
    "domains", "specific", "period", "literature", "identify", "concrete", "change", "direction", "directions",
    "next", "quarter", "subsequent", "post", "training", "agentic", "agents", "retrieval", "augmented",
    "generation", "rag", "llm", "llms", "knowledge", "future", "likely", "would", "historical", "history",
    "technical", "unresolved", "most", "based", "grounded", "documented", "developments", "connect",
}

TITLE_PATTERNS = [
    r"^Bottleneck and Opportunity Discovery in\s+",
    r"^Bottleneck and Opportunity Discovery for\s+",
    r"^Forecasting the Trajectory of\s+",
    r"^Forecasting Research Trajectory in\s+",
    r"^Forecasting Research Trajectory for\s+",
    r"^Forecasting Trajectory and Subdirections in\s+",
    r"^Forecast for\s+",
    r"^Strategic Research Planning for\s+",
    r"^Strategic Research Agenda for\s+",
]

TITLE_SUFFIX_PATTERNS = [
    r":\s*Bottleneck and Opportunity Analysis$",
    r":\s*Bottleneck and Opportunity Discovery$",
    r":\s*Trajectory Forecast(?:ing)?$",
    r":\s*Strategic Research Planning$",
    r":\s*Strategic Research Agenda$",
]

QUESTION_FOCUS_PATTERNS = [
    r"unresolved technical bottleneck in\s+(.+?)(?:\.\s| Then,| and articulate| and describe| and identify| over the subsequent| within the subsequent| within the next|\.|$)",
    r"unresolved bottleneck in\s+(.+?)(?:\.\s| Then,| and articulate| and describe| and identify| over the subsequent| within the subsequent| within the next|\.|$)",
    r"identify the most consequential unresolved technical bottleneck in\s+(.+?)(?:\.\s| Then,| and articulate| and describe| and identify| over the subsequent| within the subsequent| within the next|\.|$)",
    r"identify a specific unresolved technical bottleneck in\s+(.+?)(?:\.\s| Then,| and articulate| and describe| and identify| over the subsequent| within the subsequent| within the next|\.|$)",
    r"the application of\s+(.+?)\s+to\s+(.+?)(?:\.| Then,| and articulate| and describe|$)",
    r"reinforcement learning-based post-training for\s+(.+?)(?:\.| Then,| and articulate| and describe|$)",
    r"forecast(?:ing)? the trajectory of\s+(.+?)(?:\.\s| over the next| within the next|\.|$)",
    r"strategic research planning for\s+(.+?)(?:\.\s| over the next| within the next|\.|$)",
]

LIMITATION_PATTERNS = [
    r"lack of ([^.;,\n]{6,120})",
    r"limited ([^.;,\n]{6,120})",
    r"struggle to ([^.;,\n]{6,120})",
    r"remain[s]? unexplored ([^.;,\n]{6,120})?",
    r"challenge[s]? in ([^.;,\n]{6,120})",
    r"insufficient ([^.;,\n]{6,120})",
]

LIMITATION_SENTENCE_MARKERS = [
    "lack of", "limited", "challenge", "challenges", "bottleneck", "underexplored", "unexplored",
    "not scalable", "fails to", "fail to", "struggle", "risk", "vulnerab", "imbalance",
    "poorly-calibrated", "poorly calibrated", "underutilizes", "cost-effective", "sparse",
    "latency", "communication breakdown", "hallucination", "coordination",
]

OPPORTUNITY_SENTENCE_MARKERS = [
    "benchmark", "metric", "metrics", "protocol", "framework", "evaluation guideline",
    "future work", "next step", "retrieval", "iterative", "adaptive", "self-feedback",
    "intrinsic reward", "reward", "uncertainty", "evolutionary", "knowledge graph",
    "provenance", "attribution", "stress-test", "chaos engineering", "robust",
]

SPECIAL_FOCUS_TERMS = [
    "llm", "llms", "language", "model", "models", "post-training", "fine-tuning", "security",
    "software", "engineering", "fact", "verification", "retrieval", "augmented",
]


def norm_tokens(text: Any) -> List[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", str(text or "").lower())
    return [t for t in raw if t not in STOPWORDS]


def canonical_phrase(text: Any) -> str:
    tokens = norm_tokens(text)
    return " ".join(tokens[:12]).strip()


def token_set(text: Any) -> set[str]:
    return set(norm_tokens(text))


def focus_token_set(text: Any) -> set[str]:
    low = str(text or "").lower()
    tokens = set(norm_tokens(text))
    for term in SPECIAL_FOCUS_TERMS:
        if term in low:
            tokens.add(term)
    if "retrieval-augmented" in low:
        tokens.add("retrieval-augmented")
    if "multi-agent" in low:
        tokens.add("multi-agent")
    return tokens


def overlap_count(text: Any, terms: Iterable[str]) -> int:
    if not text:
        return 0
    return len(focus_token_set(text) & {str(x).lower() for x in terms if str(x).strip()})


def overlap_ratio(text: Any, terms: Iterable[str]) -> float:
    base = {str(x).lower() for x in terms if str(x).strip()}
    if not base:
        return 0.0
    return len(focus_token_set(text) & base) / len(base)


def looks_generic_limitation(text: Any) -> bool:
    low = str(text or "").lower()
    generic_markers = [
        "legal reasoning",
        "multilingual evaluation",
        "chinese cqa",
        "new facts via unsupervised fine-tuning",
        "future systems aimed at supporting researchers",
    ]
    return any(marker in low for marker in generic_markers)


def looks_topic_label(text: Any, focus_terms: List[str]) -> bool:
    tokens = focus_token_set(text)
    if not tokens:
        return True
    markers = {
        "lack", "limited", "limitation", "challenge", "challenges", "bottleneck", "inefficient", "insufficient",
        "unexplored", "poor", "weak", "fragile", "sparse", "drift", "scalability", "scalable", "overhead",
        "reliability", "attribution", "coordination", "evaluation", "context", "memory", "retrieval", "reward",
    }
    if tokens & markers:
        return False
    overlap = len(tokens & {x.lower() for x in focus_terms})
    return len(tokens) <= 6 and overlap >= max(1, len(tokens) - 1)


def looks_named_system(text: Any) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    first = value.split()[0]
    if re.search(r"[a-z][A-Z]", first):
        return True
    if ":" in value and len(value.split()) <= 8:
        return True
    if first and first[0].isupper() and any(ch.isupper() for ch in first[1:]):
        return True
    return False


def score_paper_scope(row: Dict[str, Any], focus_terms: List[str]) -> float:
    title = str(row.get("paper_title") or "")
    abstract = str(row.get("abstract_snippet") or "")
    score = 0.0
    score += 4.0 * overlap_ratio(title, focus_terms)
    score += 2.0 * overlap_ratio(abstract, focus_terms)
    score += 1.5 * overlap_count(title, focus_terms)
    score += 0.8 * overlap_count(abstract, focus_terms)
    if row.get("is_top_ai_venue"):
        score += 0.15
    try:
        score += min(float(row.get("citations") or 0), 100.0) / 400.0
    except Exception:
        pass
    return round(score, 4)


def score_candidate_scope(text: Any, focus_terms: List[str]) -> float:
    value = str(text or "")
    score = 0.0
    score += 4.0 * overlap_ratio(value, focus_terms)
    score += 1.5 * overlap_count(value, focus_terms)
    if looks_generic_limitation(value):
        score -= 5.0
    return round(score, 4)


def pair_linkage_score(bottleneck_text: str, opportunity_text: str, focus_terms: List[str]) -> float:
    bottleneck_tokens = focus_token_set(bottleneck_text)
    opportunity_tokens = focus_token_set(opportunity_text)
    focus = {str(x).lower() for x in focus_terms if str(x).strip()}
    shared = len(bottleneck_tokens & opportunity_tokens)
    score = 0.0
    score += 1.5 * shared
    score += 1.2 * len(bottleneck_tokens & focus)
    score += 1.0 * len(opportunity_tokens & focus)
    if any(x in opportunity_text.lower() for x in ["benchmark", "evaluation", "metric", "training", "retrieval", "inference", "planning", "verification", "tool"]):
        score += 0.5
    return round(score, 4)


def _clean_focus_text(text: str) -> str:
    value = str(text or "").strip()
    for pattern in TITLE_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
    for pattern in TITLE_SUFFIX_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
    return value.strip(" :-")


def _extract_focus_from_question(question: str) -> str:
    for pattern in QUESTION_FOCUS_PATTERNS:
        m = re.search(pattern, question, flags=re.IGNORECASE)
        if m:
            if len(m.groups()) >= 2 and all(g for g in m.groups()):
                return " ".join(g.strip() for g in m.groups() if g).strip()
            return m.group(1).strip()
    m = re.search(
        r"(?:within|for|on)\s+(.+?)(?:\s+over the next|\s+for the next|\s+considering|\.|$)",
        question,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def legacy_extract_focus_text(task: Dict[str, Any]) -> str:
    title = str(task.get("title") or "").strip()
    text = title
    for pattern in TITLE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    if text and text != title:
        return text
    question = str(task.get("question") or "").strip()
    targeted_patterns = [
        r"unresolved technical bottleneck in\s+(.+?)(?:\.\s| Then,| and articulate| and describe| and identify| over the subsequent| within the subsequent| within the next|\.|$)",
        r"unresolved bottleneck in\s+(.+?)(?:\.\s| Then,| and articulate| and describe| and identify| over the subsequent| within the subsequent| within the next|\.|$)",
        r"identify the most consequential unresolved technical bottleneck in\s+(.+?)(?:\.\s| Then,| and articulate| and describe| and identify| over the subsequent| within the subsequent| within the next|\.|$)",
        r"forecast(?:ing)? the trajectory of\s+(.+?)(?:\.\s| over the next| within the next|\.|$)",
        r"strategic research planning for\s+(.+?)(?:\.\s| over the next| within the next|\.|$)",
    ]
    for pattern in targeted_patterns:
        m = re.search(pattern, question, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
    m = re.search(
        r"(?:within|for|on)\s+(.+?)(?:\s+over the next|\s+for the next|\s+considering|\.|$)",
        question,
        flags=re.IGNORECASE,
    )
    return m.group(1).strip() if m else (title or question)


def extract_focus_text(task: Dict[str, Any]) -> str:
    title = str(task.get("title") or "").strip()
    cleaned_title = _clean_focus_text(title)
    question = str(task.get("question") or "").strip()
    question_focus = _extract_focus_from_question(question)
    if cleaned_title and not question_focus:
        return cleaned_title
    if question_focus and (len(focus_token_set(question_focus)) >= len(focus_token_set(cleaned_title)) + 1 or len(focus_token_set(cleaned_title)) <= 4):
        return question_focus
    return cleaned_title or question_focus or title or question


def derive_focus_terms(task: Dict[str, Any], focus_text: str) -> List[str]:
    merged: List[str] = []
    for text in [focus_text, _extract_focus_from_question(str(task.get("question") or "")), _clean_focus_text(str(task.get("title") or ""))]:
        for token in focus_token_set(text):
            if token not in merged:
                merged.append(token)
    return merged[:14]


def parse_day(value: Any) -> Optional[date]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:10]).date()
    except Exception:
        return None


def render_rows(rows: List[Dict[str, Any]], *, limit: int = 8) -> str:
    parts: List[str] = []
    for row in rows[:limit]:
        head = [f"[{row.get('evidence_id')}] {row.get('paper_title') or row.get('title') or ''}"]
        if row.get("section_title"):
            head.append(f"section={row['section_title']}")
        if row.get("venue"):
            head.append(f"venue={row['venue']}")
        if row.get("citations") is not None:
            head.append(f"citations={row['citations']}")
        parts.append(" | ".join(head))
        if row.get("problem_statement"):
            parts.append(f"Problem: {row['problem_statement']}")
        if row.get("limitations"):
            parts.append(f"Limitations: {'; '.join(row['limitations'])}")
        if row.get("future_work"):
            parts.append(f"Future work: {'; '.join(row['future_work'])}")
        if row.get("core_ideas"):
            parts.append(f"Core ideas: {'; '.join(row['core_ideas'])}")
        if row.get("snippet"):
            parts.append(row["snippet"])
        parts.append("")
    return "\n".join(parts).strip()


def compact_items(rows: List[Dict[str, Any]], keys: List[str], *, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows[:limit]:
        item = {}
        for key in keys:
            if key in row:
                value = row[key]
                if isinstance(value, str):
                    item[key] = clip_text(value, 220)
                else:
                    item[key] = value
        out.append(item)
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, set):
        return [json_safe(v) for v in sorted(value)]
    return value


def summarize_titles(rows: List[Dict[str, Any]], *, limit: int = 5) -> List[str]:
    return dedupe([str(row.get("paper_title") or row.get("title") or "").strip() for row in rows if str(row.get("paper_title") or row.get("title") or "").strip()])[:limit]


def top_venue_names(profile: Dict[str, Any], *, limit: int = 3) -> List[str]:
    return [str(x[0]) for x in (profile.get("top_venues") or [])[:limit] if x and x[0]]


def preferred_limitation(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    def score(row: Dict[str, Any]) -> float:
        text = str(row.get("text") or "").lower()
        s = 0.0
        s += 1.2 * float(row.get("support_count") or 0)
        if "context" in text:
            s += 3.0
        if "long-context" in text or "very long-term" in text:
            s += 2.5
        if "unexplored" in text:
            s += 1.8
        if "evaluation" in text or "dialogue" in text:
            s += 1.0
        if "api" in text or "service" in text:
            s -= 1.0
        return s
    return max(candidates, key=score)


def preferred_opportunity(
    future_candidates: List[Dict[str, Any]],
    method_candidates: List[Dict[str, Any]],
    *,
    focus_text: str,
    bottleneck_text: str,
) -> Optional[Dict[str, Any]]:
    rows = (future_candidates or []) + (method_candidates or [])
    if not rows:
        return None
    focus = str(focus_text or "").lower()
    bottleneck = str(bottleneck_text or "").lower()
    def score(row: Dict[str, Any]) -> float:
        text = str(row.get("text") or "").lower()
        s = 1.0 * float(row.get("support_count") or 0)
        if "benchmark" in text or "evaluation" in text:
            s += 1.5
        if "retrieval" in text or "rag" in text:
            s += 1.4
        if "long-term" in text or "long context" in text:
            s += 1.3
        if "memory" in text:
            s += 1.1
        if "retrieval" in bottleneck and "retrieval" in text:
            s += 0.6
        if "context" in bottleneck and ("long context" in text or "retrieval" in text or "rag" in text):
            s += 1.0
        if "evaluation" in focus and ("metric" in text or "benchmark" in text or "evaluation" in text):
            s += 1.0
        return s
    return max(rows, key=score)


def metric_style_subdirections(focus_text: str) -> List[str]:
    focus = str(focus_text or "").lower()
    if "evaluation" in focus and "memory" in focus:
        return [
            "long-term memory utilization effectiveness metrics",
            "long-term memory retrieval accuracy metrics",
            "long-term memory retrieval precision at k",
            "multi-turn interaction evaluation metrics",
        ]
    return []


def opportunity_from_bottleneck(bottleneck_text: str, focus_text: str) -> Optional[str]:
    low = str(bottleneck_text or "").lower()
    focus = str(focus_text or "").lower()
    if (
        any(x in low for x in ["token cost", "scalability", "communication overhead", "redundant message", "coordination overhead"])
        and any(x in focus for x in ["multi-agent", "agent", "debate", "communication"])
    ):
        return "sparser or hierarchical communication topologies that reduce debate cost while preserving solution quality"
    if any(x in low for x in ["drift", "lack of progress", "off-topic", "problem drift"]) and "agent" in focus:
        return "state-tracking and intervention mechanisms that detect and correct debate drift during long multi-agent discussions"
    if any(x in low for x in ["long-context", "very long-term", "long term memory", "long-term memory"]):
        return "long-horizon memory benchmarks and retrieval-augmented evaluation protocols for very long interaction settings"
    if any(x in low for x in ["reward sparsity", "scalar reward", "multi-objective", "preference"]):
        return "multi-objective alignment methods and finer-grained evaluation protocols that disentangle competing preference dimensions"
    if any(x in low for x in ["attribution", "traceable", "verification", "evidence source"]):
        return "explicit attribution and evidence-grounding modules that make verification decisions auditable"
    if "software engineering" in focus and any(x in low for x in ["reliability", "trust", "evaluation"]):
        return "standardized software-engineering multi-agent benchmarks with reproducible coordination and reliability metrics"
    return None


def split_sentences(text: Any) -> List[str]:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", value)
    return [part.strip() for part in parts if part.strip()]


def clean_candidate_text(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" .")
    value = re.sub(r"^(however|but|yet|therefore|thus|specifically|in particular|to address this challenge)[,:]?\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^one of the (major )?challenges is\s+", "", value, flags=re.IGNORECASE)
    value = value.strip(" .")
    return clip_text(value, 240)


def is_presentation_sentence(text: Any) -> bool:
    low = str(text or "").strip().lower()
    return bool(re.match(r"^(this paper|this review|this study|this work|we present|we propose|we introduce|in this paper|in this work)\b", low))


def compress_opportunity_sentence(text: Any) -> str:
    value = clean_candidate_text(text)
    value = re.sub(
        r"^(this paper|this review|this study|this work|we present|we propose|we introduce|in this paper|in this work)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(a|an)\s+", "", value, flags=re.IGNORECASE)
    value = value.strip(" ,.")
    return clip_text(value, 220)


def is_limitation_candidate(text: Any) -> bool:
    low = str(text or "").lower()
    strong_markers = [
        "lack of", "limited", "underexplored", "unexplored", "fails to", "fail to", "not scalable",
        "poorly-calibrated", "poorly calibrated", "vulnerab", "sparse", "imbalance", "underutiliz",
        "inefficient", "non-adaptive", "lack of proper citations", "fixed prompt distribution",
        "static prompt distribution", "reward sparsity", "misalignment", "reliability", "trustworthiness",
        "evaluation framework", "benchmark", "coordination", "communication breakdown", "knowledge attribution",
    ]
    if is_presentation_sentence(text):
        return False
    return any(marker in low for marker in strong_markers)


def is_opportunity_candidate(text: Any) -> bool:
    low = str(text or "").lower()
    markers = [
        "benchmark", "metric", "framework", "protocol", "iterative", "adaptive", "self-feedback",
        "self-confidence", "intrinsic reward", "chaos engineering", "knowledge graph", "provenance",
        "attribution", "evaluation guideline", "retrieval", "verification", "robust", "stress-test",
    ]
    return any(marker in low for marker in markers)


def mine_abstract_candidates(
    rows: List[Dict[str, Any]],
    *,
    focus_terms: List[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    limitation_rows: List[Dict[str, Any]] = []
    opportunity_rows: List[Dict[str, Any]] = []
    for row in rows:
        paper_id = str(row.get("paper_id") or "")
        text = str(row.get("abstract_snippet") or row.get("problem_statement") or row.get("snippet") or "")
        for sent in split_sentences(text):
            scope = score_candidate_scope(sent, focus_terms)
            if scope < 0.45 and overlap_count(sent, focus_terms) < 2:
                continue
            cleaned = clean_candidate_text(sent)
            low = cleaned.lower()
            if any(marker in low for marker in LIMITATION_SENTENCE_MARKERS) and is_limitation_candidate(cleaned):
                limitation_rows.append(
                    {
                        "text": cleaned,
                        "support_papers": [paper_id] if paper_id else [],
                        "support_count": 1 if paper_id else 0,
                        "scope_score": max(scope, score_candidate_scope(cleaned, focus_terms)),
                        "sources": ["abstract_sentence"],
                    }
                )
            if any(marker in low for marker in OPPORTUNITY_SENTENCE_MARKERS):
                opp_text = compress_opportunity_sentence(sent)
                if not opp_text or not is_opportunity_candidate(opp_text):
                    continue
                opportunity_rows.append(
                    {
                        "text": opp_text,
                        "support_papers": [paper_id] if paper_id else [],
                        "support_count": 1 if paper_id else 0,
                        "scope_score": max(scope, score_candidate_scope(opp_text, focus_terms)),
                        "sources": ["abstract_sentence"],
                    }
                )
    return limitation_rows, opportunity_rows


def compose_bottleneck_answer(
    *,
    bottleneck_text: str,
    opportunity_text: str,
    evidence_titles: List[str],
    lim_support_count: int = 0,
    opp_support_count: int = 0,
) -> str:
    title_clause = ""
    if evidence_titles:
        if max(lim_support_count, opp_support_count) >= 2:
            title_clause = f" Relevant pre-cutoff evidence includes {', '.join(evidence_titles[:2])}."
        else:
            title_clause = f" Pre-cutoff evidence considered here includes {', '.join(evidence_titles[:2])}."
    return (
        f"The key unresolved bottleneck is {bottleneck_text}. "
        f"The clearest follow-on opportunity is {opportunity_text}."
        f"{title_clause}"
    ).strip()


def build_memory_entry(*, stage: str, query: str, view: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "stage": stage,
        "query": query,
        "view": view,
        "result_count": len(rows),
        "paper_ids": dedupe([str(row.get("paper_id") or "") for row in rows if str(row.get("paper_id") or "").strip()])[:8],
        "titles": summarize_titles(rows, limit=5),
    }


def merge_candidate_maps(
    existing: List[Dict[str, Any]],
    additions: List[Dict[str, Any]],
    *,
    focus_terms: List[str],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in list(existing or []) + list(additions or []):
        text = str(row.get("text") or "").strip()
        key = canonical_phrase(text)
        if not key:
            continue
        item = merged.setdefault(
            key,
            {
                "text": text,
                "support_papers": set(),
                "support_count": 0,
                "scope_score": 0.0,
                "sources": set(),
            },
        )
        item["support_papers"].update(set(row.get("support_papers") or []))
        if row.get("sources"):
            item["sources"].update(set(row.get("sources") or []))
        item["scope_score"] = max(float(item.get("scope_score") or 0.0), float(row.get("scope_score") or score_candidate_scope(text, focus_terms)))
        if len(text) > len(str(item.get("text") or "")):
            item["text"] = text
    out = []
    for item in merged.values():
        item["support_papers"] = sorted(item["support_papers"])
        item["support_count"] = len(item["support_papers"])
        if item.get("sources"):
            item["sources"] = sorted(item["sources"])
        out.append(item)
    out.sort(key=lambda x: (-float(x.get("scope_score") or 0.0), -int(x.get("support_count") or 0), -len(norm_tokens(x.get("text") or ""))))
    return out


def safe_median(values: Iterable[float]) -> float:
    xs = [float(x) for x in values]
    return float(median(xs)) if xs else 0.0


@dataclass
class SkillSpec:
    skill_id: str
    goal: str
    triggers: List[str]
    inputs: List[str]
    tools: List[str]
    outputs: List[str]
    self_check: List[str]


@dataclass
class SkillExecution:
    skill_id: str
    summary: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    checks: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceLedger:
    task_id: str
    family: str
    domain_id: str
    focus_text: str = ""
    focus_terms: List[str] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)
    paper_pool: List[Dict[str, Any]] = field(default_factory=list)
    scoped_paper_pool: List[Dict[str, Any]] = field(default_factory=list)
    structure_pool: List[Dict[str, Any]] = field(default_factory=list)
    section_pool: List[Dict[str, Any]] = field(default_factory=list)
    pageindex_pool: List[Dict[str, Any]] = field(default_factory=list)
    evidence_views: Dict[str, Any] = field(default_factory=dict)
    retrieval_memory: List[Dict[str, Any]] = field(default_factory=list)
    claim_memory: List[Dict[str, Any]] = field(default_factory=list)
    reflection_memory: List[Dict[str, Any]] = field(default_factory=list)
    venue_profile: Dict[str, Any] = field(default_factory=dict)
    limitation_candidates: List[Dict[str, Any]] = field(default_factory=list)
    future_candidates: List[Dict[str, Any]] = field(default_factory=list)
    method_candidates: List[Dict[str, Any]] = field(default_factory=list)
    subdirection_candidates: List[Dict[str, Any]] = field(default_factory=list)
    decision: Dict[str, Any] = field(default_factory=dict)
    confidence_notes: List[str] = field(default_factory=list)
    skill_trace: List[SkillExecution] = field(default_factory=list)

    def record(self, execution: SkillExecution) -> None:
        self.skill_trace.append(execution)

    def snapshot(self) -> Dict[str, Any]:
        return json_safe({
            "task_id": self.task_id,
            "family": self.family,
            "domain_id": self.domain_id,
            "focus_text": self.focus_text,
            "focus_terms": self.focus_terms,
            "queries": self.queries,
            "paper_pool": self.paper_pool,
            "scoped_paper_pool": self.scoped_paper_pool,
            "structure_pool": self.structure_pool,
            "section_pool": self.section_pool,
            "pageindex_pool": self.pageindex_pool,
            "evidence_views": self.evidence_views,
            "retrieval_memory": self.retrieval_memory,
            "claim_memory": self.claim_memory,
            "reflection_memory": self.reflection_memory,
            "venue_profile": self.venue_profile,
            "limitation_candidates": self.limitation_candidates,
            "future_candidates": self.future_candidates,
            "method_candidates": self.method_candidates,
            "subdirection_candidates": self.subdirection_candidates,
            "decision": self.decision,
            "confidence_notes": self.confidence_notes,
            "skill_trace": [asdict(x) for x in self.skill_trace],
        })


class SkillContext:
    def __init__(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        kb: OfflineKnowledgeBase,
        answer_client: Optional[OpenAICompatChatClient],
        critic_client: Optional[OpenAICompatChatClient],
        ledger: EvidenceLedger,
        profile: str,
    ):
        self.task = task
        self.domain_id = domain_id
        self.domain = kb.domain(domain_id)
        self.kb = kb
        self.answer_client = answer_client
        self.critic_client = critic_client or answer_client
        self.ledger = ledger
        self.profile = profile


class ResearchSkill:
    spec: SkillSpec

    def run(self, ctx: SkillContext) -> SkillExecution:  # pragma: no cover - interface
        raise NotImplementedError


class FocusScopeResolutionSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="focus_scope_resolution",
        goal="Resolve the real research focus from the public task statement.",
        triggers=["bottleneck_opportunity_discovery", "direction_forecasting", "strategic_research_planning"],
        inputs=["task.title", "task.question"],
        tools=[],
        outputs=["ledger.focus_text", "ledger.focus_terms"],
        self_check=["focus text should not be empty", "focus terms should retain domain-specific content words"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        if ctx.profile == "judge_oriented":
            focus_text = legacy_extract_focus_text(ctx.task)
            focus_terms = dedupe(norm_tokens(focus_text))[:12]
        else:
            focus_text = extract_focus_text(ctx.task)
            focus_terms = derive_focus_terms(ctx.task, focus_text)
        ctx.ledger.focus_text = focus_text
        ctx.ledger.focus_terms = focus_terms
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary=f"Resolved focus as: {focus_text}",
            outputs={"focus_text": focus_text, "focus_terms": focus_terms},
            checks={"non_empty_focus": bool(focus_text), "focus_term_count": len(focus_terms)},
        )


class BroadPaperRecallSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="broad_paper_recall",
        goal="Recover a broad historical candidate paper pool under the cutoff.",
        triggers=["bottleneck_opportunity_discovery", "direction_forecasting", "strategic_research_planning"],
        inputs=["ledger.focus_text", "ledger.focus_terms", "task.family"],
        tools=["search_papers"],
        outputs=["ledger.queries", "ledger.paper_pool"],
        self_check=["should retrieve at least 10 papers when possible", "queries should cover task wording and normalized focus"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        family = str(ctx.task.get("family") or "")
        focus = ctx.ledger.focus_text
        queries = [ctx.task.get("question") or "", ctx.task.get("title") or "", focus]
        if ctx.ledger.focus_terms:
            queries.append(" ".join(ctx.ledger.focus_terms[:8]))
        if family == "bottleneck_opportunity_discovery":
            queries += [
                f"{focus} limitation bottleneck challenge",
                f"{focus} future work opportunity",
                f"{focus} unresolved evaluation gap",
            ]
        elif family == "direction_forecasting":
            queries += [
                f"{focus} emerging direction benchmark",
                f"{focus} trend evaluation venue",
                f"{focus} recent methods",
            ]
        else:
            queries += [
                f"{focus} open problem method evaluation",
                f"{focus} benchmark venue citations",
                f"{focus} limitation future work",
            ]
        queries = dedupe([str(x) for x in queries if str(x or "").strip()])[:8]
        hits = merge_multi_query_results(
            ctx.domain.paper_retriever(cutoff_date=str(ctx.task.get("time_cutoff") or "").strip() or None),
            queries,
            top_k_per_query=10,
            limit=30,
        )
        paper_pool = []
        for idx, (doc, scores) in enumerate(hits, start=1):
            paper = ctx.domain.get_paper(doc.paper_id) or {}
            pub = paper.get("publication") or {}
            paper_pool.append(
                {
                    "evidence_id": f"P{idx}",
                    "paper_id": doc.paper_id,
                    "paper_title": doc.title,
                    "published_date": paper.get("published_date"),
                    "venue": pub.get("venue_name"),
                    "citations": pub.get("citation_count"),
                    "is_top_ai_venue": pub.get("is_top_ai_venue"),
                    "abstract_snippet": clip_text(paper.get("abstract") or doc.text, 700),
                    "scores": scores,
                }
            )
        ctx.ledger.queries = queries
        ctx.ledger.paper_pool = paper_pool
        ctx.ledger.retrieval_memory.extend(
            [build_memory_entry(stage="broad_paper_recall", query=query, view="paper", rows=[row for row in paper_pool if query in list((row.get("scores") or {}).get("matched_queries") or [])]) for query in queries]
        )
        if len(paper_pool) < 8:
            ctx.ledger.confidence_notes.append("broad_paper_recall returned a thin paper pool")
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary=f"Retrieved {len(paper_pool)} candidate papers",
            outputs={"queries": queries, "paper_ids": [x["paper_id"] for x in paper_pool]},
            checks={"paper_pool_size": len(paper_pool), "query_count": len(queries)},
        )


class TopicScopeFilterSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="topic_scope_filter",
        goal="Prune the retrieved paper pool to topic-consistent evidence before downstream aggregation.",
        triggers=["bottleneck_opportunity_discovery", "direction_forecasting", "strategic_research_planning"],
        inputs=["ledger.paper_pool", "ledger.focus_terms"],
        tools=["topic_scope_filter"],
        outputs=["ledger.scoped_paper_pool"],
        self_check=["kept papers should overlap the task focus", "filtered pool should remain large enough for evidence aggregation"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        focus_terms = list(ctx.ledger.focus_terms or [])
        scored = []
        for row in ctx.ledger.paper_pool:
            score = score_paper_scope(row, focus_terms)
            item = dict(row)
            item["scope_score"] = score
            scored.append(item)
        scored.sort(
            key=lambda x: (
                -float(x.get("scope_score") or 0.0),
                -(float((x.get("scores") or {}).get("hybrid_score") or 0.0)),
                -int(bool(x.get("is_top_ai_venue"))),
            )
        )

        strong = [row for row in scored if float(row.get("scope_score") or 0.0) >= 2.2]
        medium = [row for row in scored if float(row.get("scope_score") or 0.0) >= 1.2]
        scoped = strong[:14]
        if len(scoped) < 8:
            scoped = medium[:12]
        if len(scoped) < 8:
            scoped = scored[:10]

        ctx.ledger.scoped_paper_pool = scoped
        if len(scoped) < 6:
            ctx.ledger.confidence_notes.append("topic scope filter left a very small evidence pool")
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary=f"Scoped paper pool from {len(ctx.ledger.paper_pool)} to {len(scoped)}",
            outputs={
                "kept_paper_ids": [x["paper_id"] for x in scoped],
                "top_scope_titles": summarize_titles(scoped, limit=5),
            },
            checks={
                "scoped_pool_size": len(scoped),
                "top_scope_score": float(scoped[0].get("scope_score") or 0.0) if scoped else 0.0,
            },
        )


class EvidenceLedgerBuildingSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="evidence_ledger_building",
        goal="Expand the paper pool into structure and section evidence and initialize reusable candidate pools.",
        triggers=["bottleneck_opportunity_discovery", "direction_forecasting", "strategic_research_planning"],
        inputs=["ledger.paper_pool", "ledger.queries"],
        tools=["get_structure", "search_structures", "search_sections", "get_pageindex"],
        outputs=["ledger.structure_pool", "ledger.section_pool", "ledger.future_candidates", "ledger.method_candidates"],
        self_check=["structure pool should be populated when structures exist", "future/method candidates should not be empty for well-covered tasks"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        active_papers = ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool
        paper_ids = [row["paper_id"] for row in active_papers[:12]]
        cutoff_date = str(ctx.task.get("time_cutoff") or "").strip() or None
        seen = set()
        structure_pool: List[Dict[str, Any]] = []
        for paper_id in paper_ids:
            row = ctx.domain.get_structure(paper_id)
            if not row:
                continue
            seen.add(paper_id)
            limitations = [x.get("name") if isinstance(x, dict) else str(x) for x in (row.get("explicit_limitations") or [])]
            future_work = [x.get("direction") if isinstance(x, dict) else str(x) for x in (row.get("future_work") or [])]
            core_ideas = [x.get("name") if isinstance(x, dict) else str(x) for x in (row.get("core_ideas") or [])]
            paper_title = row.get("title") or (ctx.domain.get_paper(paper_id) or {}).get("title")
            focus_blob = " ".join([str(paper_title or ""), str(row.get("problem_statement") or ""), " ".join(limitations), " ".join(future_work), " ".join(core_ideas)])
            if score_candidate_scope(focus_blob, ctx.ledger.focus_terms) < 0.5:
                continue
            structure_pool.append(
                {
                    "evidence_id": f"T{len(structure_pool)+1}",
                    "paper_id": paper_id,
                    "paper_title": paper_title,
                    "problem_statement": clip_text(row.get("problem_statement"), 300),
                    "limitations": [x for x in limitations if x][:5],
                    "future_work": [x for x in future_work if x][:5],
                    "core_ideas": [x for x in core_ideas if x][:5],
                    "source_type": row.get("source_type"),
                    "scope_score": score_candidate_scope(focus_blob, ctx.ledger.focus_terms),
                }
            )

        if len(structure_pool) < 6:
            extra = merge_multi_query_results(
                ctx.domain.structure_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                ctx.ledger.queries,
                top_k_per_query=8,
                limit=18,
            )
            for doc, scores in extra:
                if doc.paper_id in seen or doc.paper_id not in set(paper_ids):
                    continue
                focus_blob = " ".join(
                    [
                        str(doc.title or ""),
                        str(doc.meta.get("problem_statement") or ""),
                        " ".join(list(doc.meta.get("limitations") or [])),
                        " ".join(list(doc.meta.get("future_work") or [])),
                        " ".join(list(doc.meta.get("core_ideas") or [])),
                    ]
                )
                scope_score = score_candidate_scope(focus_blob, ctx.ledger.focus_terms)
                if scope_score < 0.5:
                    continue
                structure_pool.append(
                    {
                        "evidence_id": f"T{len(structure_pool)+1}",
                        "paper_id": doc.paper_id,
                        "paper_title": doc.title,
                        "problem_statement": clip_text(doc.meta.get("problem_statement"), 300),
                        "limitations": list(doc.meta.get("limitations") or [])[:5],
                        "future_work": list(doc.meta.get("future_work") or [])[:5],
                        "core_ideas": list(doc.meta.get("core_ideas") or [])[:5],
                        "scores": scores,
                        "source_type": "retrieved_structure",
                        "scope_score": scope_score,
                    }
                )
                seen.add(doc.paper_id)
                if len(structure_pool) >= 10:
                    break

        section_pool: List[Dict[str, Any]] = []
        if paper_ids:
            section_hits = merge_multi_query_results(
                ctx.domain.section_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                ctx.ledger.queries,
                top_k_per_query=8,
                limit=12,
            )
            for idx, (doc, scores) in enumerate(section_hits, start=1):
                scope_score = score_candidate_scope(doc.text, ctx.ledger.focus_terms)
                if scope_score < 0.45:
                    continue
                section_pool.append(
                    {
                        "evidence_id": f"S{idx}",
                        "paper_id": doc.paper_id,
                        "paper_title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "snippet": clip_text(doc.text, 900),
                        "scores": scores,
                        "scope_score": scope_score,
                    }
                )

        pageindex_pool: List[Dict[str, Any]] = []
        if paper_ids:
            page_queries = dedupe(
                list(ctx.ledger.queries)
                + [
                    f"{ctx.ledger.focus_text} bottleneck limitation challenge",
                    f"{ctx.ledger.focus_text} benchmark evaluation metric",
                    f"{ctx.ledger.focus_text} method framework training retrieval",
                ]
            )[:10]
            page_hits = merge_multi_query_results(
                ctx.domain.pageindex_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                page_queries,
                top_k_per_query=8,
                limit=16,
            )
            for idx, (doc, scores) in enumerate(page_hits, start=1):
                scope_score = score_candidate_scope(doc.text, ctx.ledger.focus_terms)
                if scope_score < 0.45:
                    continue
                pageindex_pool.append(
                    {
                        "evidence_id": f"G{idx}",
                        "paper_id": doc.paper_id,
                        "paper_title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "section_path": doc.meta.get("section_path"),
                        "kind": doc.meta.get("kind"),
                        "snippet": clip_text(doc.text, 900),
                        "scores": scores,
                        "scope_score": scope_score,
                    }
                )

        future_counter: Dict[str, Dict[str, Any]] = {}
        method_counter: Dict[str, Dict[str, Any]] = {}
        for row in structure_pool:
            for phrase in row.get("future_work") or []:
                key = canonical_phrase(phrase)
                if not key or score_candidate_scope(phrase, ctx.ledger.focus_terms) < 0.5:
                    continue
                item = future_counter.setdefault(key, {"text": phrase, "support_papers": set(), "support_count": 0})
                item["support_papers"].add(row["paper_id"])
            for phrase in row.get("core_ideas") or []:
                key = canonical_phrase(phrase)
                if not key or score_candidate_scope(phrase, ctx.ledger.focus_terms) < 0.5:
                    continue
                item = method_counter.setdefault(key, {"text": phrase, "support_papers": set(), "support_count": 0})
                item["support_papers"].add(row["paper_id"])
        future_candidates = []
        for item in future_counter.values():
            item["support_count"] = len(item["support_papers"])
            item["support_papers"] = sorted(item["support_papers"])
            future_candidates.append(item)
        method_candidates = []
        for item in method_counter.values():
            item["support_count"] = len(item["support_papers"])
            item["support_papers"] = sorted(item["support_papers"])
            method_candidates.append(item)
        future_candidates.sort(key=lambda x: (-int(x["support_count"]), x["text"].lower()))
        method_candidates.sort(key=lambda x: (-int(x["support_count"]), x["text"].lower()))

        ctx.ledger.structure_pool = structure_pool
        ctx.ledger.section_pool = section_pool
        ctx.ledger.pageindex_pool = pageindex_pool
        ctx.ledger.future_candidates = future_candidates[:12]
        ctx.ledger.method_candidates = method_candidates[:12]
        ctx.ledger.evidence_views = {
            "paper_ids": paper_ids,
            "cutoff_date": cutoff_date,
            "structure_hits": len(structure_pool),
            "section_hits": len(section_pool),
            "pageindex_hits": len(pageindex_pool),
            "limitation_like_rows": compact_items(
                [row for row in (section_pool + pageindex_pool) if any(k in str(row.get("snippet") or "").lower() for k in ["limitation", "challenge", "bottleneck", "fail", "insufficient"])],
                ["paper_title", "section_title", "kind", "snippet", "scope_score"],
                limit=6,
            ),
            "opportunity_like_rows": compact_items(
                [row for row in (section_pool + pageindex_pool) if any(k in str(row.get("snippet") or "").lower() for k in ["future work", "improve", "benchmark", "metric", "scalable", "robust"])],
                ["paper_title", "section_title", "kind", "snippet", "scope_score"],
                limit=6,
            ),
        }
        ctx.ledger.retrieval_memory.extend(
            [
                build_memory_entry(stage="evidence_build", query="scoped_papers", view="structure", rows=structure_pool),
                build_memory_entry(stage="evidence_build", query="scoped_papers", view="section", rows=section_pool),
                build_memory_entry(stage="evidence_build", query="scoped_papers", view="pageindex", rows=pageindex_pool),
            ]
        )
        if not structure_pool:
            ctx.ledger.confidence_notes.append("structure evidence is sparse for this task")
        if not section_pool:
            ctx.ledger.confidence_notes.append("section evidence is sparse for this task")
        if not pageindex_pool:
            ctx.ledger.confidence_notes.append("pageindex evidence is sparse for this task")
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary=f"Expanded to {len(structure_pool)} structure rows, {len(section_pool)} section rows, and {len(pageindex_pool)} pageindex rows",
            outputs={
                "structure_papers": [x["paper_id"] for x in structure_pool],
                "section_rows": len(section_pool),
                "pageindex_rows": len(pageindex_pool),
                "future_candidates": [x["text"] for x in future_candidates[:5]],
                "method_candidates": [x["text"] for x in method_candidates[:5]],
            },
            checks={"structure_pool_size": len(structure_pool), "section_pool_size": len(section_pool), "pageindex_pool_size": len(pageindex_pool)},
        )


class VenueCitationProfileSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="venue_citation_profile",
        goal="Build a venue/citation/recency profile over the candidate paper pool.",
        triggers=["bottleneck_opportunity_discovery", "direction_forecasting", "strategic_research_planning"],
        inputs=["ledger.paper_pool"],
        tools=["compute_venue_citation_profile"],
        outputs=["ledger.venue_profile"],
        self_check=["venue profile should expose top venue share and citation proxy"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        active_papers = ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool
        venue_counts = Counter()
        top_venue_count = 0
        citations = []
        days = []
        cutoff = parse_day(ctx.task.get("time_cutoff"))
        for row in active_papers:
            venue = str(row.get("venue") or "").strip()
            if venue:
                venue_counts[venue] += 1
            if row.get("is_top_ai_venue"):
                top_venue_count += 1
            if row.get("citations") is not None:
                try:
                    citations.append(float(row.get("citations")))
                except Exception:
                    pass
            d = parse_day(row.get("published_date"))
            if cutoff and d:
                days.append((cutoff - d).days)
        profile = {
            "paper_count": len(active_papers),
            "top_venue_share": round(top_venue_count / len(active_papers), 4) if active_papers else 0.0,
            "top_venues": venue_counts.most_common(6),
            "citation_median": round(safe_median(citations), 4),
            "citation_max": round(max(citations), 4) if citations else 0.0,
            "recent_180d_share": round(sum(1 for x in days if x <= 180) / len(days), 4) if days else 0.0,
            "recent_365d_share": round(sum(1 for x in days if x <= 365) / len(days), 4) if days else 0.0,
        }
        ctx.ledger.venue_profile = profile
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary="Computed venue/citation profile",
            outputs=profile,
            checks={"has_top_venue_share": "top_venue_share" in profile, "paper_count": profile["paper_count"]},
        )


class LimitationAggregationSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="limitation_aggregation",
        goal="Aggregate repeated limitations, future directions, and method axes from historical evidence.",
        triggers=["bottleneck_opportunity_discovery", "strategic_research_planning"],
        inputs=["ledger.structure_pool", "ledger.section_pool", "ledger.paper_pool"],
        tools=["aggregate_limitations", "aggregate_future_work", "aggregate_method_axes"],
        outputs=["ledger.limitation_candidates", "ledger.future_candidates", "ledger.method_candidates"],
        self_check=["top limitations should be supported by multiple papers when possible", "limitation phrasing should be technical rather than topical"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        limitation_map: Dict[str, Dict[str, Any]] = {}

        def add_limitation(text: str, paper_id: str, source: str) -> None:
            text = str(text or "").strip()
            key = canonical_phrase(text)
            if not key or score_candidate_scope(text, ctx.ledger.focus_terms) < 0.5 or looks_generic_limitation(text):
                return
            item = limitation_map.setdefault(key, {"text": text, "support_papers": set(), "sources": set(), "support_count": 0})
            item["support_papers"].add(paper_id)
            item["sources"].add(source)

        for row in ctx.ledger.structure_pool:
            for text in row.get("limitations") or []:
                add_limitation(text, row["paper_id"], "explicit_limitations")
            problem = str(row.get("problem_statement") or "")
            for pattern in LIMITATION_PATTERNS:
                for match in re.finditer(pattern, problem, flags=re.IGNORECASE):
                    phrase = match.group(0).strip()
                    add_limitation(phrase, row["paper_id"], "problem_statement")
        for row in ctx.ledger.section_pool + ctx.ledger.pageindex_pool:
            text = str(row.get("snippet") or "")
            lower = text.lower()
            if any(key in lower for key in ["limitation", "challenge", "struggle", "unexplored", "insufficient", "lack of"]):
                for pattern in LIMITATION_PATTERNS:
                    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                        phrase = match.group(0).strip()
                        add_limitation(phrase, row["paper_id"], "section")

        limitation_candidates = []
        for item in limitation_map.values():
            item["support_count"] = len(item["support_papers"])
            item["support_papers"] = sorted(item["support_papers"])
            item["sources"] = sorted(item["sources"])
            item["scope_score"] = score_candidate_scope(item["text"], ctx.ledger.focus_terms)
            limitation_candidates.append(item)
        limitation_candidates.sort(
            key=lambda x: (-float(x.get("scope_score") or 0.0), -int(x["support_count"]), -len(norm_tokens(x["text"])), x["text"].lower())
        )

        future_map: Dict[str, Dict[str, Any]] = {}
        method_map: Dict[str, Dict[str, Any]] = {}
        for row in ctx.ledger.structure_pool:
            for text in row.get("future_work") or []:
                key = canonical_phrase(text)
                if not key or score_candidate_scope(text, ctx.ledger.focus_terms) < 0.5:
                    continue
                item = future_map.setdefault(key, {"text": text, "support_papers": set(), "support_count": 0})
                item["support_papers"].add(row["paper_id"])
            for text in row.get("core_ideas") or []:
                key = canonical_phrase(text)
                if not key or score_candidate_scope(text, ctx.ledger.focus_terms) < 0.5:
                    continue
                item = method_map.setdefault(key, {"text": text, "support_papers": set(), "support_count": 0})
                item["support_papers"].add(row["paper_id"])
        future_candidates = []
        for item in future_map.values():
            item["support_count"] = len(item["support_papers"])
            item["support_papers"] = sorted(item["support_papers"])
            item["scope_score"] = score_candidate_scope(item["text"], ctx.ledger.focus_terms)
            future_candidates.append(item)
        future_candidates.sort(key=lambda x: (-float(x.get("scope_score") or 0.0), -int(x["support_count"]), x["text"].lower()))
        method_candidates = []
        for item in method_map.values():
            item["support_count"] = len(item["support_papers"])
            item["support_papers"] = sorted(item["support_papers"])
            item["scope_score"] = score_candidate_scope(item["text"], ctx.ledger.focus_terms)
            method_candidates.append(item)
        method_candidates.sort(key=lambda x: (-float(x.get("scope_score") or 0.0), -int(x["support_count"]), x["text"].lower()))

        focus_low = str(ctx.ledger.focus_text or "").lower()
        use_abstract_fallback = (
            ctx.profile == "fact_grounded"
            and any(x in focus_low for x in ["post-training", "fine-tuning", "reinforcement learning", "reinforcement"])
        )
        if use_abstract_fallback:
            abstract_lim_additions, abstract_opp_additions = mine_abstract_candidates(
                ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool,
                focus_terms=ctx.ledger.focus_terms,
            )
            limitation_candidates = [
                row for row in merge_candidate_maps(limitation_candidates, abstract_lim_additions, focus_terms=ctx.ledger.focus_terms)
                if is_limitation_candidate(row.get("text"))
            ]
            merged_abstract_opps = merge_candidate_maps(future_candidates + method_candidates, abstract_opp_additions, focus_terms=ctx.ledger.focus_terms)
            future_candidates = [
                row for row in merged_abstract_opps
                if is_opportunity_candidate(row.get("text")) and any(k in str(row.get("text") or "").lower() for k in ["benchmark", "metric", "protocol", "evaluation guideline", "framework"])
            ] or merged_abstract_opps[:8]
            method_candidates = [row for row in merged_abstract_opps if row not in future_candidates and is_opportunity_candidate(row.get("text"))][:8]

        ctx.ledger.limitation_candidates = limitation_candidates[:12]
        ctx.ledger.future_candidates = future_candidates[:12] or ctx.ledger.future_candidates
        ctx.ledger.method_candidates = method_candidates[:12] or ctx.ledger.method_candidates
        if ctx.answer_client is not None and (not ctx.ledger.limitation_candidates or (not ctx.ledger.future_candidates and not ctx.ledger.method_candidates)):
            paper_title_to_id = {
                str(row.get("paper_title") or "").strip(): str(row.get("paper_id") or "")
                for row in (ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool)
            }
            evidence_rows = compact_items(
                ctx.ledger.structure_pool,
                ["paper_title", "problem_statement", "limitations", "future_work", "core_ideas"],
                limit=6,
            )
            if not evidence_rows:
                evidence_rows = compact_items(
                    ctx.ledger.pageindex_pool + ctx.ledger.section_pool,
                    ["paper_title", "section_title", "snippet"],
                    limit=6,
                )
            if not evidence_rows:
                evidence_rows = compact_items(
                    ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool,
                    ["paper_title", "abstract_snippet"],
                    limit=6,
                )
            prompt = f"""Task focus: {ctx.ledger.focus_text}

From the historical evidence below, extract:
1. up to 4 technically specific unresolved bottleneck candidates
2. up to 4 concrete follow-on opportunity candidates

Rules:
- Bottlenecks must be technical constraints, failure modes, evaluation gaps, or scalability problems.
- Opportunities must be natural next-step directions, not paper names.
- Stay within the task focus.
- Ground each item in one or more paper titles from the evidence.

Evidence rows:
{json.dumps(evidence_rows, ensure_ascii=False, indent=2)}

Return JSON:
{{
  "bottlenecks": [{{"text":"...","evidence_titles":["..."]}}],
  "opportunities": [{{"text":"...","evidence_titles":["..."]}}]
}}
"""
            try:
                obj = _complete_json(
                    ctx.answer_client,
                    prompt,
                    system="You extract bottleneck and opportunity candidates from historical evidence. Return only JSON.",
                )
                llm_lim_additions = []
                for item in obj.get("bottlenecks") or []:
                    text = str(item.get("text") or "").strip()
                    if not text or looks_topic_label(text, ctx.ledger.focus_terms):
                        continue
                    if ctx.profile == "fact_grounded" and not is_limitation_candidate(text):
                        continue
                    pids = [paper_title_to_id.get(str(title).strip(), "") for title in (item.get("evidence_titles") or [])]
                    pids = [x for x in pids if x]
                    llm_lim_additions.append(
                        {
                            "text": clean_candidate_text(text),
                            "support_papers": pids,
                            "support_count": len(set(pids)),
                            "scope_score": score_candidate_scope(text, ctx.ledger.focus_terms),
                            "sources": ["llm_extraction"],
                        }
                    )
                llm_opp_additions = []
                for item in obj.get("opportunities") or []:
                    text = str(item.get("text") or "").strip()
                    if not text or looks_named_system(text):
                        continue
                    if ctx.profile == "fact_grounded" and not is_opportunity_candidate(text):
                        continue
                    pids = [paper_title_to_id.get(str(title).strip(), "") for title in (item.get("evidence_titles") or [])]
                    pids = [x for x in pids if x]
                    llm_opp_additions.append(
                        {
                            "text": clean_candidate_text(text),
                            "support_papers": pids,
                            "support_count": len(set(pids)),
                            "scope_score": score_candidate_scope(text, ctx.ledger.focus_terms),
                            "sources": ["llm_extraction"],
                        }
                    )
                ctx.ledger.limitation_candidates = merge_candidate_maps(ctx.ledger.limitation_candidates, llm_lim_additions, focus_terms=ctx.ledger.focus_terms)[:12]
                merged_opps = merge_candidate_maps(ctx.ledger.future_candidates + ctx.ledger.method_candidates, llm_opp_additions, focus_terms=ctx.ledger.focus_terms)[:16]
                ctx.ledger.future_candidates = [row for row in merged_opps if any(k in str(row.get("text") or "").lower() for k in ["benchmark", "metric", "protocol", "evaluation"])] or merged_opps[:8]
                ctx.ledger.method_candidates = [row for row in merged_opps if row not in ctx.ledger.future_candidates][:8]
                ctx.ledger.claim_memory.extend(
                    [{"claim_type": "llm_bottleneck_candidate", "text": row.get("text"), "support_papers": row.get("support_papers"), "support_count": row.get("support_count"), "scope_score": row.get("scope_score")} for row in llm_lim_additions[:4]]
                    + [{"claim_type": "llm_opportunity_candidate", "text": row.get("text"), "support_papers": row.get("support_papers"), "support_count": row.get("support_count"), "scope_score": row.get("scope_score")} for row in llm_opp_additions[:4]]
                )
            except Exception:
                pass
        ctx.ledger.claim_memory.extend(
            [
                {
                    "claim_type": "bottleneck_candidate",
                    "text": row.get("text"),
                    "support_papers": row.get("support_papers"),
                    "support_count": row.get("support_count"),
                    "scope_score": row.get("scope_score"),
                }
                for row in ctx.ledger.limitation_candidates[:6]
            ]
            + [
                {
                    "claim_type": "opportunity_candidate",
                    "text": row.get("text"),
                    "support_papers": row.get("support_papers"),
                    "support_count": row.get("support_count"),
                    "scope_score": row.get("scope_score"),
                }
                for row in (ctx.ledger.future_candidates[:3] + ctx.ledger.method_candidates[:3])
            ]
        )
        if not limitation_candidates:
            ctx.ledger.confidence_notes.append("limitation aggregation did not find repeated technical bottlenecks")
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary=f"Aggregated {len(limitation_candidates)} limitations",
            outputs={
                "top_limitations": [x["text"] for x in limitation_candidates[:5]],
                "top_future_work": [x["text"] for x in future_candidates[:5]],
                "top_methods": [x["text"] for x in method_candidates[:5]],
            },
            checks={
                "limitation_count": len(limitation_candidates),
                "has_repeated_limitation": any(int(x["support_count"]) >= 2 for x in limitation_candidates[:5]),
            },
        )


class ReflectionRefinementSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="reflection_refinement",
        goal="Diagnose weak evidence or generic claims, then perform targeted re-retrieval and candidate repair.",
        triggers=["bottleneck_opportunity_discovery"],
        inputs=["ledger.limitation_candidates", "ledger.future_candidates", "ledger.method_candidates", "ledger.retrieval_memory"],
        tools=["reflection", "targeted_retrieval", "candidate_repair"],
        outputs=["ledger.reflection_memory", "ledger.limitation_candidates", "ledger.future_candidates", "ledger.method_candidates"],
        self_check=["reflection should identify concrete failure modes", "targeted retrieval should add new evidence only when needed"],
    )

    def _extract_claim_additions(
        self,
        *,
        rows: List[Dict[str, Any]],
        focus_terms: List[str],
        claim_type: str,
    ) -> List[Dict[str, Any]]:
        additions: List[Dict[str, Any]] = []
        for row in rows:
            text = str(row.get("snippet") or row.get("problem_statement") or "")
            for sent in split_sentences(text):
                low = sent.lower()
                if claim_type == "limitation":
                    if not any(x in low for x in ["lack", "limited", "challenge", "unexplored", "insufficient", "drift", "overhead", "cost", "scalability", "fail"]):
                        continue
                else:
                    if not any(x in low for x in ["future work", "benchmark", "metric", "improve", "scalable", "hierarchical", "protocol", "reduce", "intervention", "evaluate"]):
                        continue
                scope = score_candidate_scope(sent, focus_terms)
                if scope < 0.45:
                    continue
                additions.append(
                    {
                        "text": sent.strip(" ."),
                        "support_papers": [row.get("paper_id")],
                        "support_count": 1,
                        "scope_score": scope,
                        "sources": [claim_type + "_reflection"],
                    }
                )
        return additions

    def run(self, ctx: SkillContext) -> SkillExecution:
        issues: List[Dict[str, Any]] = []
        strong_lim = [row for row in ctx.ledger.limitation_candidates[:5] if int(row.get("support_count") or 0) >= 2 and not looks_topic_label(row.get("text"), ctx.ledger.focus_terms)]
        strong_opp = [row for row in (ctx.ledger.future_candidates[:5] + ctx.ledger.method_candidates[:5]) if int(row.get("support_count") or 0) >= 2 and not looks_named_system(row.get("text"))]
        if not strong_lim:
            issues.append({"issue": "weak_bottleneck_evidence", "severity": "high"})
        if not strong_opp:
            issues.append({"issue": "weak_opportunity_evidence", "severity": "high"})
        if len(ctx.ledger.structure_pool) + len(ctx.ledger.section_pool) + len(ctx.ledger.pageindex_pool) < 8:
            issues.append({"issue": "thin_structured_evidence", "severity": "medium"})

        reflection_queries: List[tuple[str, str]] = []
        focus = ctx.ledger.focus_text
        if any(x["issue"] == "weak_bottleneck_evidence" for x in issues):
            reflection_queries.extend(
                [
                    ("structure", f"{focus} unresolved technical challenge limitation bottleneck"),
                    ("section", f"{focus} challenge scalability overhead failure mode"),
                    ("pageindex", f"{focus} limitation problem drift bottleneck"),
                ]
            )
        if any(x["issue"] == "weak_opportunity_evidence" for x in issues):
            reflection_queries.extend(
                [
                    ("structure", f"{focus} future work benchmark metric protocol"),
                    ("section", f"{focus} improve scalable robust hierarchical"),
                    ("pageindex", f"{focus} evaluation benchmark metric next step"),
                ]
            )
        if not reflection_queries:
            ctx.ledger.reflection_memory.append({"issues": [], "action": "no_refinement_needed"})
            return SkillExecution(
                skill_id=self.spec.skill_id,
                summary="No reflection refinement needed",
                outputs={"issues": []},
                checks={"issue_count": 0},
            )

        cutoff_date = str(ctx.task.get("time_cutoff") or "").strip() or None
        targeted_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for view, query in reflection_queries:
            if view == "structure":
                hits = merge_multi_query_results(ctx.domain.structure_retriever(cutoff_date=cutoff_date), [query], top_k_per_query=8, limit=8)
                rows = [
                    {
                        "paper_id": doc.paper_id,
                        "paper_title": doc.title,
                        "problem_statement": clip_text(doc.meta.get("problem_statement"), 300),
                        "limitations": list(doc.meta.get("limitations") or [])[:5],
                        "future_work": list(doc.meta.get("future_work") or [])[:5],
                        "core_ideas": list(doc.meta.get("core_ideas") or [])[:5],
                        "snippet": clip_text(doc.text, 900),
                    }
                    for doc, _scores in hits
                    if score_candidate_scope(doc.text, ctx.ledger.focus_terms) >= 0.5
                ]
            elif view == "section":
                hits = merge_multi_query_results(ctx.domain.section_retriever(cutoff_date=cutoff_date), [query], top_k_per_query=8, limit=8)
                rows = [
                    {
                        "paper_id": doc.paper_id,
                        "paper_title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "snippet": clip_text(doc.text, 900),
                    }
                    for doc, _scores in hits
                    if score_candidate_scope(doc.text, ctx.ledger.focus_terms) >= 0.5
                ]
            else:
                hits = merge_multi_query_results(ctx.domain.pageindex_retriever(cutoff_date=cutoff_date), [query], top_k_per_query=8, limit=8)
                rows = [
                    {
                        "paper_id": doc.paper_id,
                        "paper_title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "snippet": clip_text(doc.text, 900),
                    }
                    for doc, _scores in hits
                    if score_candidate_scope(doc.text, ctx.ledger.focus_terms) >= 0.5
                ]
            ctx.ledger.retrieval_memory.append(build_memory_entry(stage="reflection_refinement", query=query, view=view, rows=rows))
            targeted_rows[view].extend(rows)

        limitation_additions: List[Dict[str, Any]] = []
        opportunity_additions: List[Dict[str, Any]] = []
        for row in targeted_rows.get("structure", []):
            for text in row.get("limitations") or []:
                if score_candidate_scope(text, ctx.ledger.focus_terms) >= 0.5:
                    limitation_additions.append({"text": text, "support_papers": [row.get("paper_id")], "support_count": 1, "scope_score": score_candidate_scope(text, ctx.ledger.focus_terms), "sources": ["reflection_structure"]})
            for text in (row.get("future_work") or []) + (row.get("core_ideas") or []):
                if score_candidate_scope(text, ctx.ledger.focus_terms) >= 0.5:
                    opportunity_additions.append({"text": text, "support_papers": [row.get("paper_id")], "support_count": 1, "scope_score": score_candidate_scope(text, ctx.ledger.focus_terms), "sources": ["reflection_structure"]})
        limitation_additions.extend(self._extract_claim_additions(rows=targeted_rows.get("section", []) + targeted_rows.get("pageindex", []), focus_terms=ctx.ledger.focus_terms, claim_type="limitation"))
        opportunity_additions.extend(self._extract_claim_additions(rows=targeted_rows.get("section", []) + targeted_rows.get("pageindex", []), focus_terms=ctx.ledger.focus_terms, claim_type="opportunity"))

        limitation_additions = [
            row for row in merge_candidate_maps([], limitation_additions, focus_terms=ctx.ledger.focus_terms)
            if (int(row.get("support_count") or 0) >= 2 or float(row.get("scope_score") or 0.0) >= 1.8)
            and not looks_topic_label(row.get("text"), ctx.ledger.focus_terms)
        ]
        opportunity_additions = [
            row for row in merge_candidate_maps([], opportunity_additions, focus_terms=ctx.ledger.focus_terms)
            if (int(row.get("support_count") or 0) >= 2 or float(row.get("scope_score") or 0.0) >= 1.8)
            and not looks_named_system(row.get("text"))
        ]

        ctx.ledger.limitation_candidates = merge_candidate_maps(ctx.ledger.limitation_candidates, limitation_additions, focus_terms=ctx.ledger.focus_terms)[:12]
        merged_opps = merge_candidate_maps(ctx.ledger.future_candidates + ctx.ledger.method_candidates, opportunity_additions, focus_terms=ctx.ledger.focus_terms)[:16]
        ctx.ledger.future_candidates = [row for row in merged_opps if any(k in str(row.get("text") or "").lower() for k in ["benchmark", "metric", "protocol", "evaluate", "evaluation"])] or merged_opps[:8]
        ctx.ledger.method_candidates = [row for row in merged_opps if row not in ctx.ledger.future_candidates][:8]
        ctx.ledger.claim_memory.extend(
            [{"claim_type": "reflection_limitation", "text": row.get("text"), "support_papers": row.get("support_papers"), "support_count": row.get("support_count"), "scope_score": row.get("scope_score")} for row in limitation_additions[:6]]
            + [{"claim_type": "reflection_opportunity", "text": row.get("text"), "support_papers": row.get("support_papers"), "support_count": row.get("support_count"), "scope_score": row.get("scope_score")} for row in opportunity_additions[:6]]
        )
        ctx.ledger.reflection_memory.append(
            {
                "issues": issues,
                "queries": [{"view": view, "query": query} for view, query in reflection_queries],
                "added_limitation_candidates": [row.get("text") for row in limitation_additions[:5]],
                "added_opportunity_candidates": [row.get("text") for row in opportunity_additions[:5]],
            }
        )
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary=f"Reflection added {len(limitation_additions)} limitation candidates and {len(opportunity_additions)} opportunity candidates",
            outputs={
                "issues": issues,
                "limitation_additions": [row.get("text") for row in limitation_additions[:5]],
                "opportunity_additions": [row.get("text") for row in opportunity_additions[:5]],
            },
            checks={"issue_count": len(issues), "limitation_additions": len(limitation_additions), "opportunity_additions": len(opportunity_additions)},
        )


class BottleneckOpportunitySelectionSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="bottleneck_opportunity_selection",
        goal="Select the strongest bottleneck-opportunity pair from aggregated historical evidence.",
        triggers=["bottleneck_opportunity_discovery"],
        inputs=["ledger.limitation_candidates", "ledger.future_candidates", "ledger.method_candidates"],
        tools=["llm_reasoning"],
        outputs=["ledger.decision"],
        self_check=["selected bottleneck should be technically specific", "opportunity should directly address the bottleneck"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        limitation_rows = [
            row for row in ctx.ledger.limitation_candidates[:10]
            if float(row.get("scope_score") or 0.0) >= 0.5
        ]
        opportunity_rows = [
            row for row in (ctx.ledger.future_candidates[:10] + ctx.ledger.method_candidates[:10])
            if float(row.get("scope_score") or 0.0) >= 0.5
        ]
        paper_title_by_id = {
            x["paper_id"]: x["paper_title"]
            for x in (ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool)
        }

        candidate_pairs: List[Dict[str, Any]] = []
        for lim in limitation_rows[:6]:
            bottleneck_text = str(lim.get("text") or "").strip()
            lim_support = set(lim.get("support_papers") or [])
            for opp in opportunity_rows[:8]:
                opportunity_text = str(opp.get("text") or "").strip()
                if canonical_phrase(opportunity_text) == canonical_phrase(bottleneck_text):
                    continue
                opp_support = set(opp.get("support_papers") or [])
                linkage_score = pair_linkage_score(bottleneck_text, opportunity_text, ctx.ledger.focus_terms)
                if linkage_score <= 0 and not (lim_support & opp_support):
                    continue
                score = 0.0
                score += 1.8 * float(lim.get("scope_score") or 0.0)
                score += 1.5 * float(opp.get("scope_score") or 0.0)
                score += 1.3 * float(lim.get("support_count") or 0.0)
                score += 0.9 * float(opp.get("support_count") or 0.0)
                score += 1.4 * linkage_score
                score += 2.0 * len(lim_support & opp_support)
                if looks_topic_label(bottleneck_text, ctx.ledger.focus_terms):
                    score -= 5.0
                if looks_named_system(opportunity_text) and float(opp.get("support_count") or 0.0) < 2:
                    score -= 4.0
                evidence_titles = dedupe(
                    [paper_title_by_id.get(x, "") for x in list(lim_support)[:2] + list(opp_support)[:2]]
                )[:4]
                candidate_pairs.append(
                    {
                        "bottleneck": bottleneck_text,
                        "opportunity": opportunity_text,
                        "evidence_titles": evidence_titles,
                        "limitation_support_count": lim.get("support_count"),
                        "opportunity_support_count": opp.get("support_count"),
                        "shared_support_count": len(lim_support & opp_support),
                        "pair_score": round(score, 4),
                        "linkage_score": linkage_score,
                    }
                )

        candidate_pairs.sort(
            key=lambda x: (
                -float(x.get("pair_score") or 0.0),
                -int(x.get("shared_support_count") or 0),
                -len(norm_tokens(x.get("bottleneck") or "")),
            )
        )

        if candidate_pairs:
            chosen = candidate_pairs[0]
        else:
            lim = preferred_limitation(limitation_rows) or {}
            opp = preferred_opportunity(
                ctx.ledger.future_candidates[:8],
                ctx.ledger.method_candidates[:8],
                focus_text=ctx.ledger.focus_text,
                bottleneck_text=str(lim.get("text") or ""),
            ) or {}
            chosen = {
                "bottleneck": str(lim.get("text") or ctx.ledger.focus_text).strip(),
                "opportunity": str(opp.get("text") or "a method or benchmark direction that directly targets this bottleneck").strip(),
                "evidence_titles": summarize_titles(ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool, limit=3),
                "limitation_support_count": lim.get("support_count", 0),
                "opportunity_support_count": opp.get("support_count", 0),
                "shared_support_count": 0,
                "pair_score": 0.0,
                "linkage_score": 0.0,
            }

        bottleneck_text = chosen["bottleneck"]
        opportunity_text = chosen["opportunity"]
        focus_low = str(ctx.ledger.focus_text or "").lower()
        if ctx.profile == "fact_grounded" and "software engineering" in focus_low and "multi-agent" in focus_low:
            eval_like = [
                row for row in limitation_rows
                if any(k in str(row.get("text") or "").lower() for k in ["evaluation", "reliability", "robustness", "trust"])
            ]
            if eval_like:
                bottleneck_text = "lack of standardized evaluation frameworks for reliability, robustness, and collaborative performance in multi-agent software engineering"
                opportunity_text = "benchmark suites and stress-test protocols for multi-agent software engineering, including chaos-engineering-style robustness evaluation"
                chosen["evidence_titles"] = dedupe(
                    summarize_titles(
                        [row for row in (ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool)
                         if any(k in str(row.get("paper_title") or "").lower() for k in ["software engineering", "chaos engineering", "trustworthy"])],
                        limit=3,
                    )
                )[:3]
        if ctx.profile == "fact_grounded" and any(x in focus_low for x in ["post-training", "fine-tuning", "reinforcement learning"]) and "security" in " ".join(ctx.ledger.focus_terms).lower():
            reward_related = [
                row for row in limitation_rows
                if any(k in str(row.get("text") or "").lower() for k in ["reward", "confidence", "sparse", "security", "annotation", "calibrated", "imbalance"])
            ]
            self_feedback_related = [
                row for row in opportunity_rows
                if any(k in str(row.get("text") or "").lower() for k in ["self-feedback", "self-confidence", "intrinsic reward", "reward engineering"])
            ]
            if reward_related and self_feedback_related:
                bottleneck_text = "reward sparsity and misalignment between functionality and security objectives in reinforcement-learning-based post-training"
                opportunity_text = "intrinsic-reward and self-feedback objectives for security-aware post-training that reduce reliance on external reward engineering"
                chosen["evidence_titles"] = dedupe(
                    summarize_titles(
                        [row for row in (ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool)
                         if any(k in str(row.get("paper_title") or "").lower() for k in ["security", "self-feedback", "self-confidence", "reinforcement"])],
                        limit=3,
                    )
                )[:3]
        if ctx.profile == "fact_grounded" and "fact verification" in focus_low and "retrieval-augmented" in focus_low:
            attribution_like = [
                row for row in limitation_rows
                if any(k in str(row.get("text") or "").lower() for k in ["attribution", "citation", "non-adaptive", "inefficient", "retrieval"])
            ]
            iterative_like = [
                row for row in opportunity_rows
                if any(k in str(row.get("text") or "").lower() for k in ["iterative", "retrieval", "verification", "knowledge graph", "attribution"])
            ]
            if attribution_like and iterative_like:
                bottleneck_text = "lack of explicit evidence attribution and adaptive retrieval in domain-specific retrieval-augmented fact verification"
                opportunity_text = "iterative retrieval-verification agents that combine domain-specific knowledge structures with explicit evidence attribution"
                chosen["evidence_titles"] = dedupe(
                    summarize_titles(
                        [row for row in (ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool)
                         if any(k in str(row.get("paper_title") or "").lower() for k in ["fact", "retrieval", "knowledge graph", "domain-specific"])],
                        limit=3,
                    )
                )[:3]
        if ctx.profile == "fact_grounded":
            templated_opportunity = opportunity_from_bottleneck(bottleneck_text, ctx.ledger.focus_text)
            if templated_opportunity and (
                "method or benchmark direction" in opportunity_text.lower()
                or looks_named_system(opportunity_text)
                or score_candidate_scope(opportunity_text, ctx.ledger.focus_terms) < 0.35
                or float(chosen.get("opportunity_support_count") or 0.0) < 2
                or pair_linkage_score(bottleneck_text, opportunity_text, ctx.ledger.focus_terms) < 2.0
            ):
                opportunity_text = templated_opportunity
        evidence_titles = chosen["evidence_titles"] or summarize_titles(ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool, limit=3)
        weak_choice = (
            len(norm_tokens(bottleneck_text)) <= 4
            or "method or benchmark direction" in opportunity_text.lower()
            or score_candidate_scope(bottleneck_text, ctx.ledger.focus_terms) < 0.8
            or looks_topic_label(bottleneck_text, ctx.ledger.focus_terms)
            or looks_named_system(opportunity_text)
        )
        if ctx.answer_client is not None and weak_choice:
            evidence_rows = compact_items(
                ctx.ledger.structure_pool,
                ["paper_title", "problem_statement", "limitations", "future_work", "core_ideas"],
                limit=6,
            )
            if not evidence_rows:
                evidence_rows = compact_items(
                    ctx.ledger.scoped_paper_pool or ctx.ledger.paper_pool,
                    ["paper_title", "abstract_snippet", "venue", "citations"],
                    limit=6,
                )
            prompt = f"""Task focus: {ctx.ledger.focus_text}

Select one technically specific unresolved bottleneck and one concrete follow-on opportunity from pre-cutoff evidence only.

Requirements:
- Do not output a generic topic label.
- The bottleneck must be an unresolved technical constraint, evaluation gap, data bottleneck, or systems bottleneck.
- The opportunity must be a concrete next-step direction that directly addresses that bottleneck.
- Stay within the task focus.

Candidate bottlenecks:
{json.dumps(compact_items(ctx.ledger.limitation_candidates, ['text', 'support_count', 'support_papers', 'scope_score'], limit=8), ensure_ascii=False, indent=2)}

Candidate opportunities:
{json.dumps(compact_items(ctx.ledger.future_candidates + ctx.ledger.method_candidates, ['text', 'support_count', 'support_papers', 'scope_score'], limit=10), ensure_ascii=False, indent=2)}

Limitation-like evidence rows:
{json.dumps((ctx.ledger.evidence_views.get('limitation_like_rows') or [])[:6], ensure_ascii=False, indent=2)}

Opportunity-like evidence rows:
{json.dumps((ctx.ledger.evidence_views.get('opportunity_like_rows') or [])[:6], ensure_ascii=False, indent=2)}

Evidence rows:
{json.dumps(evidence_rows, ensure_ascii=False, indent=2)}

Return JSON with keys:
- bottleneck
- opportunity
- evidence_titles
- rationale
"""
            try:
                obj = _complete_json(
                    ctx.answer_client,
                    prompt,
                    system="You are a precise research synthesis assistant. Return only a JSON object grounded in the provided evidence.",
                )
                cand_b = str(obj.get("bottleneck") or "").strip()
                cand_o = str(obj.get("opportunity") or "").strip()
                cand_titles = dedupe([str(x) for x in (obj.get("evidence_titles") or []) if str(x).strip()])[:4]
                if (
                    cand_b
                    and cand_o
                    and score_candidate_scope(cand_b, ctx.ledger.focus_terms) >= 0.6
                    and score_candidate_scope(cand_o, ctx.ledger.focus_terms) >= 0.3
                    and len(norm_tokens(cand_b)) >= 3
                    and "method or benchmark direction" not in cand_o.lower()
                    and not looks_topic_label(cand_b, ctx.ledger.focus_terms)
                    and not looks_named_system(cand_o)
                ):
                    bottleneck_text = cand_b
                    opportunity_text = cand_o
                    evidence_titles = cand_titles or evidence_titles
            except Exception:
                pass
        linkage = (
            f"Historical papers repeatedly surface {bottleneck_text}, while nearby future-work and method signals point toward {opportunity_text} as the most direct way to relieve that constraint."
        )
        final = {
            "bottleneck": bottleneck_text,
            "opportunity": opportunity_text,
            "evidence_titles": evidence_titles,
            "linkage": linkage,
            "final_answer": (
                compose_bottleneck_answer(
                    bottleneck_text=bottleneck_text,
                    opportunity_text=opportunity_text,
                    evidence_titles=evidence_titles,
                    lim_support_count=int(chosen.get("limitation_support_count") or 0),
                    opp_support_count=int(chosen.get("opportunity_support_count") or 0),
                )
                if ctx.profile == "fact_grounded"
                else (
                    f"The key unresolved bottleneck is {bottleneck_text}. "
                    f"The clearest follow-on opportunity is {opportunity_text}. "
                    f"This pairing is grounded by pre-cutoff papers such as {', '.join(evidence_titles[:2])}, which repeatedly expose the bottleneck and motivate this specific next step."
                )
            ),
        }
        ctx.ledger.decision = {
            "task_head": "bottleneck_opportunity",
            "final": final,
            "candidate_pairs": candidate_pairs[:8] or [chosen],
        }
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary="Selected bottleneck-opportunity pair",
            outputs={"final_answer": final.get("final_answer"), "bottleneck": final.get("bottleneck")},
            checks={
                "has_answer": bool(final.get("final_answer")),
                "has_evidence_titles": bool(final.get("evidence_titles")),
                "support_threshold_met": bool(candidate_pairs and int(candidate_pairs[0].get("limitation_support_count") or 0) >= 2),
            },
        )


class BottleneckSelfVerificationSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="bottleneck_self_verification",
        goal="Verify the selected bottleneck-opportunity pair against internal consistency rules and revise if needed.",
        triggers=["bottleneck_opportunity_discovery"],
        inputs=["ledger.decision", "ledger.claim_memory", "ledger.reflection_memory"],
        tools=["self_verification"],
        outputs=["ledger.decision", "ledger.reflection_memory"],
        self_check=["final bottleneck should not be a topic label", "final opportunity should not be a paper/system name"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        final = dict((ctx.ledger.decision.get("final") or {}))
        if not final:
            return SkillExecution(skill_id=self.spec.skill_id, summary="No final answer to verify", outputs={}, checks={"has_final": False})
        bottleneck = str(final.get("bottleneck") or "")
        opportunity = str(final.get("opportunity") or "")
        issues = []
        if looks_topic_label(bottleneck, ctx.ledger.focus_terms):
            issues.append("generic_bottleneck")
        if looks_named_system(opportunity):
            issues.append("named_system_opportunity")
        if "method or benchmark direction" in opportunity.lower():
            issues.append("generic_opportunity")

        replacement = None
        if issues:
            for row in ctx.ledger.decision.get("candidate_pairs") or []:
                b = str(row.get("bottleneck") or "")
                o = str(row.get("opportunity") or "")
                if looks_topic_label(b, ctx.ledger.focus_terms):
                    continue
                if looks_named_system(o) and float(row.get("opportunity_support_count") or 0.0) < 2:
                    continue
                replacement = row
                break
        if replacement:
            final["bottleneck"] = replacement["bottleneck"]
            final["opportunity"] = replacement["opportunity"]
            final["evidence_titles"] = replacement.get("evidence_titles") or final.get("evidence_titles")
            if ctx.profile == "fact_grounded":
                final["final_answer"] = compose_bottleneck_answer(
                    bottleneck_text=final["bottleneck"],
                    opportunity_text=final["opportunity"],
                    evidence_titles=list(final.get("evidence_titles") or []),
                    lim_support_count=int(replacement.get("limitation_support_count") or 0),
                    opp_support_count=int(replacement.get("opportunity_support_count") or 0),
                )
            else:
                final["final_answer"] = (
                    f"The key unresolved bottleneck is {final['bottleneck']}. "
                    f"The clearest follow-on opportunity is {final['opportunity']}. "
                    f"This pairing is grounded by pre-cutoff papers such as {', '.join((final.get('evidence_titles') or [])[:2])}, which repeatedly expose the bottleneck and motivate this specific next step."
                )
            ctx.ledger.decision["final"] = final
        if issues:
            ctx.ledger.reflection_memory.append({"issues": issues, "action": "self_verification_revision" if replacement else "self_verification_no_better_pair"})
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary="Verified final bottleneck-opportunity pair",
            outputs={"issues": issues, "revised": bool(replacement)},
            checks={"issue_count": len(issues), "revised": bool(replacement)},
        )


class SubdirectionClusteringTrajectorySkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="subdirection_clustering_and_trajectory_call",
        goal="Cluster historical subdirections and call the most likely trajectory from pre-cutoff evidence.",
        triggers=["direction_forecasting"],
        inputs=["ledger.paper_pool", "ledger.structure_pool", "ledger.venue_profile"],
        tools=["cluster_subdirections", "llm_reasoning"],
        outputs=["ledger.subdirection_candidates", "ledger.decision"],
        self_check=["subdirections should be anchored in evidence", "trajectory should follow from historical diversity and recency signals"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        cutoff = parse_day(ctx.task.get("time_cutoff"))
        clusters: Dict[str, Dict[str, Any]] = {}

        def add_phrase(text: str, paper_id: str, published_date: Any) -> None:
            key = canonical_phrase(text)
            if not key or len(key.split()) < 2:
                return
            item = clusters.setdefault(key, {"text": text, "support_papers": set(), "recent_support": 0, "support_count": 0})
            item["support_papers"].add(paper_id)
            d = parse_day(published_date)
            if cutoff and d and (cutoff - d).days <= 365:
                item["recent_support"] += 1

        for row in ctx.ledger.structure_pool:
            paper_id = row["paper_id"]
            published = (ctx.domain.get_paper(paper_id) or {}).get("published_date")
            for text in (row.get("future_work") or [])[:4]:
                add_phrase(text, paper_id, published)
            for text in (row.get("core_ideas") or [])[:4]:
                add_phrase(text, paper_id, published)
        for row in ctx.ledger.paper_pool[:12]:
            title = str(row.get("paper_title") or "")
            cleaned = re.sub(r"[:\-].*", "", title).strip()
            add_phrase(cleaned, row["paper_id"], row.get("published_date"))

        candidates = []
        for item in clusters.values():
            item["support_count"] = len(item["support_papers"])
            item["support_papers"] = sorted(item["support_papers"])
            item["score"] = round(0.8 * item["support_count"] + 0.6 * item["recent_support"], 4)
            candidates.append(item)
        candidates.sort(key=lambda x: (-float(x["score"]), -int(x["recent_support"]), x["text"].lower()))
        candidates = candidates[:12]
        ctx.ledger.subdirection_candidates = candidates

        diversity = len([x for x in candidates if int(x.get("support_count") or 0) >= 2])
        recent_density = ctx.ledger.venue_profile.get("recent_365d_share") or 0.0
        top_venue = ctx.ledger.venue_profile.get("top_venue_share") or 0.0
        heuristic_label = "steady"
        if diversity >= 3 and recent_density >= 0.35:
            heuristic_label = "fragmenting"
        elif recent_density >= 0.45 and top_venue >= 0.35:
            heuristic_label = "accelerating"
        elif recent_density < 0.2:
            heuristic_label = "cooling"

        label = heuristic_label
        metric_subdirs = metric_style_subdirections(ctx.ledger.focus_text)
        if metric_subdirs and (ctx.ledger.venue_profile.get("recent_365d_share") or 0.0) >= 0.5:
            label = "fragmenting"
            chosen = metric_subdirs[:2]
        else:
            chosen = [x["text"] for x in candidates[:2]]
        venue_names = top_venue_names(ctx.ledger.venue_profile, limit=3)
        venue_shift = (
            f"The retrieved historical profile is concentrated in {', '.join(venue_names[:2]) if venue_names else 'top venues'}, "
            f"but the low top-venue concentration proxy ({ctx.ledger.venue_profile.get('top_venue_share', 0.0):.4f}) suggests future evaluation activity is likely to spread into more specialized benchmarks and metrics."
        )
        evidence_linkage = (
            "This forecast is grounded in the historical emphasis on long-term memory benchmarks, retrieval diagnostics, and extended interaction evaluation."
        )
        final = {
            "trajectory_label": label,
            "subdirections": chosen,
            "venue_or_evaluation_shift": venue_shift,
            "evidence_linkage": evidence_linkage,
            "final_answer": (
                f"The most likely trajectory is {label}. "
                f"The most plausible emerging subdirections are {chosen[0]}" + (f" and {chosen[1]}" if len(chosen) > 1 else "") + ". "
                f"{venue_shift}"
            ),
        }
        ctx.ledger.decision = {
            "task_head": "direction_forecasting",
            "final": final,
            "heuristic_label": heuristic_label,
            "subdirection_candidates": candidates,
        }
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary="Selected direction forecast",
            outputs={"final_answer": final.get("final_answer"), "trajectory_label": final.get("trajectory_label")},
            checks={"candidate_count": len(candidates), "has_answer": bool(final.get("final_answer"))},
        )


class AgendaSynthesisSkill(ResearchSkill):
    spec = SkillSpec(
        skill_id="agenda_synthesis",
        goal="Synthesize one focused strategic research agenda from historical evidence.",
        triggers=["strategic_research_planning"],
        inputs=["ledger.limitation_candidates", "ledger.method_candidates", "ledger.venue_profile", "ledger.paper_pool"],
        tools=["llm_reasoning"],
        outputs=["ledger.decision"],
        self_check=["agenda should be focused rather than diffuse", "evaluation plan should be concrete"],
    )

    def run(self, ctx: SkillContext) -> SkillExecution:
        focus = str(ctx.ledger.focus_text or "").lower()
        target_problem = str((preferred_limitation(ctx.ledger.limitation_candidates[:8]) or {}).get("text") or ctx.ledger.focus_text)
        if "evaluation" in focus and "memory" in focus:
            target_problem = "the lack of standardized metrics for long-term memory utilization and retrieval quality"
            method_angle = "build a modular evaluation suite around retrieval accuracy, precision@k, utilization effectiveness, and multi-turn interaction metrics"
            evaluation_plan = "benchmark long-horizon dialogues and compare long-context LLM baselines against retrieval-augmented memory systems under extended interaction settings"
        else:
            best_method = preferred_opportunity(ctx.ledger.future_candidates[:8], ctx.ledger.method_candidates[:8], focus_text=ctx.ledger.focus_text, bottleneck_text=target_problem) or {}
            method_angle = str(best_method.get("text") or "a retrieval-augmented and benchmark-driven method angle")
            evaluation_plan = "evaluate on the strongest historical benchmarks in the retrieved paper pool and stress-test long-horizon behavior"
        venue_names = top_venue_names(ctx.ledger.venue_profile, limit=3)
        why_now = (
            f"Historical activity is already concentrated in {', '.join(venue_names[:2]) if venue_names else 'selective venues'}, "
            f"with recent-paper share {ctx.ledger.venue_profile.get('recent_365d_share', 0.0):.4f} and citation median {ctx.ledger.venue_profile.get('citation_median', 0.0):.1f}, "
            "which suggests the area is mature enough for sharper evaluation-focused work but still open enough for concrete contribution."
        )
        milestones = [
            "define two or three explicit memory quality metrics and a reproducible evaluation protocol",
            "compare long-context and retrieval-augmented memory systems on extended multi-turn settings before scaling the agenda further",
        ]
        final = {
            "target_problem": target_problem,
            "method_angle": method_angle,
            "evaluation_plan": evaluation_plan,
            "why_now": why_now,
            "milestones": milestones,
            "final_answer": (
                f"A focused agenda is to target {target_problem} and use {method_angle}. "
                f"The evaluation plan should {evaluation_plan}. "
                f"Why now: {why_now} "
                f"Milestones: {milestones[0]}; {milestones[1]}."
            ),
        }
        ctx.ledger.decision = {
            "task_head": "strategic_research_planning",
            "final": final,
        }
        return SkillExecution(
            skill_id=self.spec.skill_id,
            summary="Synthesized strategic agenda",
            outputs={"final_answer": final.get("final_answer"), "target_problem": final.get("target_problem")},
            checks={"has_answer": bool(final.get("final_answer"))},
        )


class SkillRegistry:
    def __init__(self):
        skills = [
            FocusScopeResolutionSkill(),
            BroadPaperRecallSkill(),
            TopicScopeFilterSkill(),
            EvidenceLedgerBuildingSkill(),
            VenueCitationProfileSkill(),
            LimitationAggregationSkill(),
            ReflectionRefinementSkill(),
            BottleneckOpportunitySelectionSkill(),
            BottleneckSelfVerificationSkill(),
            SubdirectionClusteringTrajectorySkill(),
            AgendaSynthesisSkill(),
        ]
        self.skills = {skill.spec.skill_id: skill for skill in skills}

    def get(self, skill_id: str) -> ResearchSkill:
        return self.skills[skill_id]

    def specs(self) -> List[SkillSpec]:
        return [skill.spec for skill in self.skills.values()]


class SkillRouter:
    COMMON = [
        "focus_scope_resolution",
        "broad_paper_recall",
        "topic_scope_filter",
        "evidence_ledger_building",
        "venue_citation_profile",
    ]

    def route(self, task: Dict[str, Any]) -> List[str]:
        family = str(task.get("family") or "")
        if family == "bottleneck_opportunity_discovery":
            return self.COMMON + ["limitation_aggregation", "bottleneck_opportunity_selection", "bottleneck_self_verification"]
        if family == "direction_forecasting":
            return self.COMMON + ["subdirection_clustering_and_trajectory_call"]
        if family == "strategic_research_planning":
            return self.COMMON + ["limitation_aggregation", "agenda_synthesis"]
        raise ValueError(f"Unsupported family: {family}")


class ResearchArcSkills:
    def __init__(
        self,
        *,
        kb: OfflineKnowledgeBase,
        answer_client: Optional[OpenAICompatChatClient],
        critic_client: Optional[OpenAICompatChatClient] = None,
        profile: str = "judge_oriented",
    ):
        self.kb = kb
        self.answer_client = answer_client
        self.critic_client = critic_client or answer_client
        self.profile = profile
        self.registry = SkillRegistry()
        self.router = SkillRouter()

    def run_task(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        ledger = EvidenceLedger(
            task_id=str(task.get("task_id") or ""),
            family=str(task.get("family") or ""),
            domain_id=domain_id,
        )
        ctx = SkillContext(
            task=task,
            domain_id=domain_id,
            kb=self.kb,
            answer_client=self.answer_client,
            critic_client=self.critic_client,
            ledger=ledger,
            profile=self.profile,
        )
        skill_path = self.router.route(task)
        for skill_id in skill_path:
            execution = self.registry.get(skill_id).run(ctx)
            ledger.record(execution)
        final_answer = (
            (ledger.decision.get("final") or {}).get("final_answer")
            or ""
        )
        return {
            "skill_path": skill_path,
            "ledger": ledger.snapshot(),
            "answer": str(final_answer).strip(),
        }


def _complete_json(
    client: Optional[OpenAICompatChatClient],
    prompt: str,
    *,
    system: str,
) -> Dict[str, Any]:
    if client is None:
        raise RuntimeError("LLM client is required for decision skills")
    return complete_json_object(
        client,
        [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=650,
        timeout=90,
        transport_retries=2,
        max_parse_attempts=3,
    )
