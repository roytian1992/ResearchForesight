from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.llm import OpenAICompatChatClient, complete_json_object, load_openai_compat_config
from researchworld.refined_release import load_task_refined_public_by_id

JUDGE_PROFILES = [
    "standard",
    "idea_arena",
    "structured_idea_arena",
    "structured_idea_arena_evidence",
    "structured_idea_arena_evidence_light",
    "structured_idea_arena_linkage_light",
]
CITATION_SPAN_PATTERN = re.compile(r"\[(?:(?:P|F|T)\d+(?:\s*,\s*(?:P|F|T)\d+)*)\]|\((?:(?:P|F|T)\d+(?:\s*,\s*(?:P|F|T)\d+)*)\)")
CITATION_ID_PATTERN = re.compile(r"(?:P|F|T)\d+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct best-of-k pairwise judging on benchmark v3 answers.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-llm-config", default="configs/llm/qwen3_235b_8002.local.yaml")
    parser.add_argument("--fallback-judge-llm-config", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260404)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-rounds", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--escalate-conf-threshold", type=float, default=0.65)
    parser.add_argument("--job-retries", type=int, default=3)
    parser.add_argument("--judge-profile", choices=JUDGE_PROFILES, default="standard")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Method spec in form method_id=path/to/results.jsonl",
    )
    return parser.parse_args()


def load_public_tasks(release_dir: Path) -> Dict[str, Dict[str, Any]]:
    return load_task_refined_public_by_id(release_dir)


def load_method_results(specs: Iterable[str]) -> Tuple[List[str], Dict[str, Dict[str, Dict[str, Any]]]]:
    methods: List[str] = []
    results_by_method: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --input spec: {spec}")
        method, path_str = spec.split("=", 1)
        method = method.strip()
        path = Path(path_str.strip())
        if not method or not path.exists():
            raise ValueError(f"Invalid method/path: {spec}")
        methods.append(method)
        results_by_method[method] = {str(row["task_id"]): row for row in iter_jsonl(path)}
    return methods, results_by_method


def family_dimensions(family: str) -> List[str]:
    dims = ["task_fit", "insightfulness", "specificity", "clarity", "strategic_value"]
    if family == "bottleneck_opportunity_discovery":
        dims.append("bottleneck_opportunity_linkage")
    elif family == "direction_forecasting":
        dims.append("trajectory_plausibility")
    elif family == "strategic_research_planning":
        dims.append("agenda_quality")
    return dims


def idea_arena_dimensions(*, include_evidence_anchoring: bool = False, include_linkage_quality: bool = False) -> List[str]:
    dims = ["novelty", "significance"]
    if include_linkage_quality:
        dims.append("linkage_quality")
    dims.extend(["clarity", "feasibility", "expected_effectiveness"])
    if include_evidence_anchoring:
        dims.append("evidence_anchoring")
    return dims


def judge_dimensions(profile: str, family: str) -> List[str]:
    if profile == "structured_idea_arena_evidence":
        return idea_arena_dimensions(include_evidence_anchoring=True)
    if profile == "structured_idea_arena_evidence_light":
        return idea_arena_dimensions()
    if profile == "structured_idea_arena_linkage_light":
        return idea_arena_dimensions(include_linkage_quality=True)
    if profile in {"idea_arena", "structured_idea_arena"}:
        return idea_arena_dimensions()
    return family_dimensions(family)


def normalize_answer_for_judging(answer: str, *, max_sentences: int = 8, max_chars: int = 1800) -> str:
    text = str(answer or "")
    text = CITATION_SPAN_PATTERN.sub("", text)
    text = re.sub(r"[*_`#>-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "(empty answer)"
    sentences = [part.strip(" ;") for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        sentences = [text]
    kept: List[str] = []
    used = 0
    for sentence in sentences:
        clipped = sentence[:280].strip()
        if not clipped:
            continue
        extra = len(clipped) + (1 if kept else 0)
        if kept and (len(kept) >= max_sentences or used + extra > max_chars):
            break
        if not kept and len(clipped) > max_chars:
            kept.append(clipped[:max_chars].strip())
            break
        kept.append(clipped)
        used += extra
    return "\n".join(f"- {sentence}" for sentence in kept)


def split_answer_sentences(answer: str, *, keep_citations: bool) -> List[str]:
    text = str(answer or "")
    if not keep_citations:
        text = CITATION_SPAN_PATTERN.sub("", text)
    text = re.sub(r"[*_`#>-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = [part.strip(" ;") for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    if not sentences:
        sentences = [text]
    return sentences


def extract_evidence_anchor(answer: str, *, include_source_ids: bool = True, max_chunks: int = 2) -> str:
    raw_text = str(answer or "")
    if not raw_text.strip():
        return "(no explicit evidence anchors)"

    evidence_chunks: List[str] = []
    citation_ids = []
    for span in CITATION_SPAN_PATTERN.findall(raw_text):
        for cited in CITATION_ID_PATTERN.findall(span):
            if cited not in citation_ids:
                citation_ids.append(cited)
            if len(citation_ids) >= 4:
                break
        if len(citation_ids) >= 4:
            break
    raw_sentences = split_answer_sentences(raw_text, keep_citations=True)
    clean_sentences = split_answer_sentences(raw_text, keep_citations=False)
    explicit_patterns = [
        r"\bEvidence\s*:\s*([^\n]+)",
        r"\bEvidence Anchors?\s*:\s*([^\n]+)",
        r"\bSupport(?:ing Evidence)?\s*:\s*([^\n]+)",
        r"\bGround(?:ing|ed By)?\s*:\s*([^\n]+)",
        r"\bRetrieved Evidence\s*:\s*([^\n]+)",
    ]
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE):
            chunk = re.sub(r"\s+", " ", match.group(1)).strip(" ;,.")
            if chunk and not re.fullmatch(r"(?:(?:P|F|T)\d+(?:\s*,\s*(?:P|F|T)\d+)*)", chunk):
                evidence_chunks.append(chunk[:260])

    citation_sentences = []
    for idx, (raw_sentence, clean_sentence) in enumerate(zip(raw_sentences, clean_sentences)):
        if re.search(r"\b(?:Evidence(?: Anchors?)?|Support(?:ing Evidence)?|Ground(?:ing|ed By)?|Retrieved Evidence)\s*:", raw_sentence, flags=re.IGNORECASE):
            if idx > 0:
                citation_sentences.append(clean_sentences[idx - 1][:260].strip())
            if not re.search(r"^\s*Evidence(?: Anchors?)?\s*:\s*(?:(?:P|F|T)\d+(?:\s*,\s*(?:P|F|T)\d+)*)\.?\s*$", clean_sentence, flags=re.IGNORECASE):
                citation_sentences.append(clean_sentence[:260].strip())
        elif CITATION_SPAN_PATTERN.search(raw_sentence):
            citation_sentences.append(clean_sentence[:260].strip())

    picked: List[str] = []
    seen = set()
    for chunk in evidence_chunks + citation_sentences:
        key = chunk.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(chunk)
        if len(picked) >= max_chunks:
            break

    if include_source_ids and citation_ids:
        picked.insert(0, f"retrieved_sources: {', '.join(citation_ids)}")

    if not picked:
        return "(no explicit evidence anchors)"
    return " | ".join(picked)[:420]


def build_structured_idea_view(
    answer: str,
    *,
    include_evidence_anchor: bool = False,
    evidence_include_source_ids: bool = True,
    evidence_max_chunks: int = 2,
    include_linkage_summary: bool = False,
) -> str:
    sentences = split_answer_sentences(answer, keep_citations=False)
    if not sentences:
        sentences = ["(empty answer)"]

    def take(indices: List[int], fallback_slice: slice) -> str:
        selected = []
        for idx in indices:
            if 0 <= idx < len(sentences):
                selected.append(sentences[idx][:260].strip())
        if not selected:
            selected = [s[:260].strip() for s in sentences[fallback_slice] if s.strip()]
        joined = " ".join(part for part in selected if part)
        return joined[:420].strip() if joined else "(not clearly stated)"

    slots = {
        "core_bottleneck": take([0], slice(0, 1)),
        "mechanism_or_causal_claim": take([1, 2], slice(1, 3)),
        "enabled_opportunity": take([3, 4], slice(3, 5)),
        "why_it_matters": take([2, 3], slice(2, 4)),
        "scope_and_generality": take([0, 4, 5], slice(0, 6)),
    }
    if include_linkage_summary:
        slots["linkage_summary"] = take([0, 1, 3], slice(0, 4))
    if include_evidence_anchor:
        slots["evidence_anchor"] = extract_evidence_anchor(
            answer,
            include_source_ids=evidence_include_source_ids,
            max_chunks=evidence_max_chunks,
        )
    return "\n".join(f"- {key}: {value}" for key, value in slots.items())


def build_prompt(
    profile: str,
    *,
    public_task: Dict[str, Any],
    family: str,
    dims: List[str],
    answer_a: str,
    answer_b: str,
) -> str:
    if profile == "idea_arena":
        return f"""# Role
You are evaluating two competing research-intelligence outputs in a blind pairwise comparison inspired by Idea Arena.

# Task Context
- Task ID: {public_task.get("task_id")}
- Family: {family}
- Domain: {public_task.get("domain")}
- Time Cutoff: {public_task.get("time_cutoff")}
- Research Question: {public_task.get("question")}
- Deliverable Spec: {json.dumps(public_task.get("deliverable_spec") or {}, ensure_ascii=False)}
- Answer Contract: {json.dumps(public_task.get("answer_contract") or {}, ensure_ascii=False)}

# Evaluation Dimensions
{json.dumps(dims, ensure_ascii=False)}

# Dimension Guidance
- novelty: Which answer offers the more non-obvious, original, or differentiating research idea or framing relative to the pre-cutoff literature?
- significance: Which answer would matter more if a research team followed it?
- clarity: Which answer is easier to understand and evaluate after normalization?
- feasibility: Which answer is more realistically actionable under the stated cutoff and available evidence?
- expected_effectiveness: Which answer is more likely to help a research team make the right next move on this task?

# Bias Controls
1. Do not reward answer length, citation count, formatting quality, markdown polish, or namedropping by themselves.
2. The answers have been normalized into the same presentation format. Judge content, not style.
3. Penalize hidden hindsight or future leakage beyond the cutoff.
4. Avoid tie unless the two answers are genuinely hard to separate after considering all five dimensions.

# Normalized Answer A
{answer_a}

# Normalized Answer B
{answer_b}

# Output Format (Strict JSON)
{{
  "winner": "A | B | tie",
  "confidence": 0.0,
  "reason": "State the decisive difference in research value. Do not mention formatting, length, or citation density as positives by themselves.",
  "dimension_votes": {{
    "novelty": "A | B | tie",
    "significance": "A | B | tie",
    "clarity": "A | B | tie",
    "feasibility": "A | B | tie",
    "expected_effectiveness": "A | B | tie"
  }}
}}
"""
    if profile == "structured_idea_arena":
        return f"""# Role
You are evaluating two competing research-intelligence outputs in a blind pairwise comparison inspired by Idea Arena.

# Task Context
- Task ID: {public_task.get("task_id")}
- Family: {family}
- Domain: {public_task.get("domain")}
- Time Cutoff: {public_task.get("time_cutoff")}
- Research Question: {public_task.get("question")}
- Deliverable Spec: {json.dumps(public_task.get("deliverable_spec") or {}, ensure_ascii=False)}
- Answer Contract: {json.dumps(public_task.get("answer_contract") or {}, ensure_ascii=False)}

# Evaluation Dimensions
{json.dumps(dims, ensure_ascii=False)}

# Structured Comparison Protocol
Each answer has been converted into the same five slots:
- core_bottleneck
- mechanism_or_causal_claim
- enabled_opportunity
- why_it_matters
- scope_and_generality

Judge only the research content represented by these slots.

# Bias Controls
1. Do not reward original prose quality, citation count, formatting polish, or verbosity.
2. The slot extraction may omit detail; prefer the answer whose slots imply the stronger research idea, not the more polished wording.
3. Penalize hindsight or future leakage beyond the cutoff.
4. Avoid tie unless the two structured ideas are genuinely comparable.

# Dimension Guidance
- novelty: Which structured idea is more original or less obvious relative to pre-cutoff literature?
- significance: Which idea would matter more if pursued?
- clarity: Which structured idea is easier to interpret after conversion into the same slots?
- feasibility: Which idea is more realistically actionable from the pre-cutoff state of the field?
- expected_effectiveness: Which idea is more likely to help a research team choose the right next direction?

# Structured Answer A
{answer_a}

# Structured Answer B
{answer_b}

# Output Format (Strict JSON)
{{
  "winner": "A | B | tie",
  "confidence": 0.0,
  "reason": "State the decisive difference in idea quality using the structured slots. Do not mention length, citations, or style.",
  "dimension_votes": {{
    "novelty": "A | B | tie",
    "significance": "A | B | tie",
    "clarity": "A | B | tie",
    "feasibility": "A | B | tie",
    "expected_effectiveness": "A | B | tie"
  }}
}}
"""
    if profile == "structured_idea_arena_evidence":
        return f"""# Role
You are evaluating two competing research-intelligence outputs in a blind pairwise comparison inspired by Idea Arena.

# Task Context
- Task ID: {public_task.get("task_id")}
- Family: {family}
- Domain: {public_task.get("domain")}
- Time Cutoff: {public_task.get("time_cutoff")}
- Research Question: {public_task.get("question")}
- Deliverable Spec: {json.dumps(public_task.get("deliverable_spec") or {}, ensure_ascii=False)}
- Answer Contract: {json.dumps(public_task.get("answer_contract") or {}, ensure_ascii=False)}

# Evaluation Dimensions
{json.dumps(dims, ensure_ascii=False)}

# Structured Comparison Protocol
Each answer has been converted into the same six slots:
- core_bottleneck
- mechanism_or_causal_claim
- enabled_opportunity
- why_it_matters
- scope_and_generality
- evidence_anchor

Judge the research content represented by these slots, plus how well the key claims are anchored to specific pre-cutoff evidence.

# Bias Controls
1. Do not reward citation count, formatting polish, verbosity, or namedropping by themselves.
2. Give modest credit when the core claim is anchored to specific, relevant, auditable pre-cutoff evidence signals such as retrieved papers, benchmarks, datasets, explicit evidence clauses, or concrete prior systems.
3. Do not reward decorative citations or irrelevant evidence that does not materially support the main claim.
4. Penalize hindsight or future leakage beyond the cutoff.
5. Avoid tie unless the two structured ideas are genuinely comparable after considering both idea quality and evidence anchoring.

# Dimension Guidance
- novelty: Which structured idea is more original or less obvious relative to pre-cutoff literature?
- significance: Which idea would matter more if pursued?
- clarity: Which structured idea is easier to interpret after conversion into the same slots?
- feasibility: Which idea is more realistically actionable from the pre-cutoff state of the field?
- expected_effectiveness: Which idea is more likely to help a research team choose the right next direction?
- evidence_anchoring: Which answer better grounds its key claims in specific, relevant, and auditable pre-cutoff evidence rather than unsupported assertion?

# Structured Answer A
{answer_a}

# Structured Answer B
{answer_b}

# Output Format (Strict JSON)
{{
  "winner": "A | B | tie",
  "confidence": 0.0,
  "reason": "State the decisive difference in idea quality or evidence anchoring using the structured slots. Do not treat citation density alone as a positive.",
  "dimension_votes": {{
    "novelty": "A | B | tie",
    "significance": "A | B | tie",
    "clarity": "A | B | tie",
    "feasibility": "A | B | tie",
    "expected_effectiveness": "A | B | tie",
    "evidence_anchoring": "A | B | tie"
  }}
}}
"""
    if profile == "structured_idea_arena_evidence_light":
        return f"""# Role
You are evaluating two competing research-intelligence outputs in a blind pairwise comparison inspired by Idea Arena.

# Task Context
- Task ID: {public_task.get("task_id")}
- Family: {family}
- Domain: {public_task.get("domain")}
- Time Cutoff: {public_task.get("time_cutoff")}
- Research Question: {public_task.get("question")}
- Deliverable Spec: {json.dumps(public_task.get("deliverable_spec") or {}, ensure_ascii=False)}
- Answer Contract: {json.dumps(public_task.get("answer_contract") or {}, ensure_ascii=False)}

# Evaluation Dimensions
{json.dumps(dims, ensure_ascii=False)}

# Structured Comparison Protocol
Each answer has been converted into the same six slots:
- core_bottleneck
- mechanism_or_causal_claim
- enabled_opportunity
- why_it_matters
- scope_and_generality
- evidence_anchor

Primary judgment should still come from idea quality. Treat `evidence_anchor` only as a weak calibration signal that matters mainly when two ideas are otherwise very close, or when one answer makes claims that look clearly under-supported relative to its confidence.

# Bias Controls
1. Do not reward citation count, formatting polish, verbosity, or namedropping by themselves.
2. Use evidence anchoring very lightly: it should mostly break close calls or penalize overconfident unsupported claims, and it should never outweigh a clearly better idea.
3. Give some credit when key claims are anchored to specific, relevant, auditable pre-cutoff evidence signals such as retrieved papers, benchmarks, datasets, explicit evidence clauses, or concrete prior systems.
4. Do not reward decorative citations, repeated references to essentially the same benchmark, or irrelevant evidence that does not materially support the main claim.
5. A narrow or weak benchmark signal should not by itself justify a broad bottleneck, forecast, or opportunity claim; prefer evidence that directly supports the central mechanism or the bottleneck-to-opportunity link being proposed.
6. Penalize hindsight or future leakage beyond the cutoff.
7. If one answer has a clearly stronger bottleneck-to-opportunity logic, better task fit, or better executional reasoning, do not let lighter evidence differences reverse that judgment.
8. Avoid tie unless the two structured ideas are genuinely comparable after considering both idea quality and weak evidence anchoring.

# Dimension Guidance
- novelty: Which structured idea is more original or less obvious relative to pre-cutoff literature?
- significance: Which idea would matter more if pursued?
- clarity: Which structured idea is easier to interpret after conversion into the same slots?
- feasibility: Which idea is more realistically actionable from the pre-cutoff state of the field? Use evidence anchoring only as a weak credibility check here, and prefer evidence that actually supports the proposed mechanism rather than merely showing a nearby failure case.
- expected_effectiveness: Which idea is more likely to help a research team choose the right next direction? Prefer answers whose key claims are at least somewhat auditable when the core ideas are otherwise similar, but do not let repetitive, weak, or slightly better evidence outweigh better research judgment.

# Structured Answer A
{answer_a}

# Structured Answer B
{answer_b}

# Output Format (Strict JSON)
{{
  "winner": "A | B | tie",
  "confidence": 0.0,
  "reason": "State the decisive difference in idea quality. Mention evidence anchoring only if it was a weak tie-breaker or exposed a clearly unsupported claim. Do not treat citation density alone as a positive.",
  "dimension_votes": {{
    "novelty": "A | B | tie",
    "significance": "A | B | tie",
    "clarity": "A | B | tie",
    "feasibility": "A | B | tie",
    "expected_effectiveness": "A | B | tie"
  }}
}}
"""
    if profile == "structured_idea_arena_linkage_light":
        return f"""# Role
You are evaluating two competing research-intelligence outputs in a blind pairwise comparison inspired by Idea Arena.

# Task Context
- Task ID: {public_task.get("task_id")}
- Family: {family}
- Domain: {public_task.get("domain")}
- Time Cutoff: {public_task.get("time_cutoff")}
- Research Question: {public_task.get("question")}
- Deliverable Spec: {json.dumps(public_task.get("deliverable_spec") or {}, ensure_ascii=False)}
- Answer Contract: {json.dumps(public_task.get("answer_contract") or {}, ensure_ascii=False)}

# Evaluation Dimensions
{json.dumps(dims, ensure_ascii=False)}

# Structured Comparison Protocol
Each answer has been converted into the same seven slots:
- core_bottleneck
- mechanism_or_causal_claim
- enabled_opportunity
- why_it_matters
- scope_and_generality
- linkage_summary
- evidence_anchor

Primary judgment should come from research-intelligence quality, especially whether the answer identifies the right bottleneck and explains a concrete causal path from that bottleneck to the proposed opportunity, forecast, agenda, or venue positioning. Treat `evidence_anchor` only as a secondary calibration signal.

# Bias Controls
1. Do not reward citation count, formatting polish, verbosity, or namedropping by themselves.
2. Prefer answers with a tighter bottleneck-to-opportunity linkage even if the other answer has more visible retrieval traces.
3. Do not over-reward answers that mainly diagnose a benchmark failure or local paper limitation without clearly explaining why that failure implies the proposed next move.
4. Use evidence anchoring lightly: it should break close calls or penalize overconfident unsupported claims, not outweigh a clearly better mechanism-level idea.
5. A narrow benchmark signal should not by itself justify a broad bottleneck, forecast, or opportunity claim; prefer evidence that directly supports the central mechanism or the bottleneck-to-opportunity link being proposed.
6. Penalize hindsight or future leakage beyond the cutoff.
7. Avoid tie unless the two structured ideas are genuinely comparable after considering idea quality, linkage quality, and light evidence anchoring.

# Dimension Guidance
- novelty: Which structured idea is more original or less obvious relative to pre-cutoff literature?
- significance: Which idea would matter more if pursued?
- linkage_quality: Which answer better explains why the identified bottleneck causally leads to the proposed opportunity, forecast, agenda, or venue recommendation?
- clarity: Which structured idea is easier to interpret after conversion into the same slots?
- feasibility: Which idea is more realistically actionable from the pre-cutoff state of the field? Prefer mechanisms that are internally coherent and do not rely on unsupported jumps.
- expected_effectiveness: Which idea is more likely to help a research team choose the right next direction? Prefer answers with stronger mechanistic closure and only use evidence anchoring as a secondary check.

# Structured Answer A
{answer_a}

# Structured Answer B
{answer_b}

# Output Format (Strict JSON)
{{
  "winner": "A | B | tie",
  "confidence": 0.0,
  "reason": "State the decisive difference in idea quality, especially linkage quality when relevant. Do not treat citation density alone as a positive.",
  "dimension_votes": {{
    "novelty": "A | B | tie",
    "significance": "A | B | tie",
    "linkage_quality": "A | B | tie",
    "clarity": "A | B | tie",
    "feasibility": "A | B | tie",
    "expected_effectiveness": "A | B | tie"
  }}
}}
"""
    return f"""# Role
You are a Lead Research Auditor conducting a blind peer review of two competing technical responses. Your goal is to identify which answer provides superior "Research Intelligence" and "Strategic Utility."

# Core Evaluation Philosophy
- Substance over Surface: Prefer a concise, high-signal answer over a verbose, generic one.
- The "So What?" Test: Does the answer identify critical bottlenecks or non-obvious trade-offs?
- Temporal Integrity: Strictly penalize any future leakage (knowledge beyond the {public_task.get("time_cutoff")}) or hindsight bias.
- Task Alignment: Evaluate against the specific requirements of the "{family}" task family.

# Input Context
- Task ID: {public_task.get("task_id")}
- Domain: {public_task.get("domain")}
- Time Cutoff: {public_task.get("time_cutoff")}
- Strategic Question: {public_task.get("question")}

# Dimensions to Judge
{json.dumps(dims, ensure_ascii=False)}

# Rules of Engagement
1. Strictly Evidence-Based: Compare Answer A and Answer B only on their internal logic and specificity.
2. Anti-Generic Bias: Heavily penalize consultant-speak (vague, universally true statements that lack domain-specific friction).
3. Winner Selection: Avoid tie unless the answers are structurally and qualitatively indistinguishable. A tie is a failure of discrimination.

# Evaluation Data
- Answer A: {answer_a}
- Answer B: {answer_b}

# Output Format (Strict JSON)
{{
  "winner": "A | B | tie",
  "confidence": 0.0,
  "reason": "Identify the inflection point that separated the winner from the loser. Be specific about logic, temporal discipline, or strategic depth.",
  "dimension_votes": {{
    "dimension_name": "A | B | tie"
  }}
}}
"""


def base_orientation(seed: int, task_id: str, method_x: str, method_y: str) -> Tuple[str, str]:
    key = f"{seed}:{task_id}:{method_x}:{method_y}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return (method_x, method_y) if int(digest, 16) % 2 == 0 else (method_y, method_x)


def round_orientation(seed: int, task_id: str, method_x: str, method_y: str, round_index: int) -> Tuple[str, str]:
    first, second = base_orientation(seed, task_id, method_x, method_y)
    if round_index % 2 == 1:
        return first, second
    return second, first


def judge_round(
    client: OpenAICompatChatClient,
    fallback_client: OpenAICompatChatClient | None,
    *,
    judge_profile: str,
    public_task: Dict[str, Any],
    family: str,
    method_a: str,
    method_b: str,
    row_a: Dict[str, Any],
    row_b: Dict[str, Any],
) -> Dict[str, Any]:
    dims = judge_dimensions(judge_profile, family)
    answer_a = str(row_a.get("answer") or "")
    answer_b = str(row_b.get("answer") or "")
    if judge_profile == "idea_arena":
        answer_a = normalize_answer_for_judging(answer_a)
        answer_b = normalize_answer_for_judging(answer_b)
    elif judge_profile == "structured_idea_arena":
        answer_a = build_structured_idea_view(answer_a)
        answer_b = build_structured_idea_view(answer_b)
    elif judge_profile == "structured_idea_arena_evidence":
        answer_a = build_structured_idea_view(answer_a, include_evidence_anchor=True)
        answer_b = build_structured_idea_view(answer_b, include_evidence_anchor=True)
    elif judge_profile == "structured_idea_arena_evidence_light":
        answer_a = build_structured_idea_view(
            answer_a,
            include_evidence_anchor=True,
            evidence_include_source_ids=False,
            evidence_max_chunks=1,
        )
        answer_b = build_structured_idea_view(
            answer_b,
            include_evidence_anchor=True,
            evidence_include_source_ids=False,
            evidence_max_chunks=1,
        )
    elif judge_profile == "structured_idea_arena_linkage_light":
        answer_a = build_structured_idea_view(
            answer_a,
            include_evidence_anchor=True,
            evidence_include_source_ids=False,
            include_linkage_summary=True,
        )
        answer_b = build_structured_idea_view(
            answer_b,
            include_evidence_anchor=True,
            evidence_include_source_ids=False,
            include_linkage_summary=True,
        )
    prompt = build_prompt(
        judge_profile,
        public_task=public_task,
        family=family,
        dims=dims,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    messages = [
        {"role": "system", "content": "You are a strict pairwise benchmark judge. Return JSON only."},
        {"role": "user", "content": prompt},
    ]
    try:
        obj = complete_json_object(
            client,
            messages,
            response_format={"type": "json_object"},
            max_tokens=900,
            timeout=120,
            transport_retries=2,
            max_parse_attempts=3,
        )
    except Exception as exc:
        if fallback_client is None:
            raise
        err = str(exc).lower()
        fallback_triggers = ["httperror", "urlerror", "timeout", "rate limit", "429", "connection reset", "temporarily unavailable"]
        if not any(token in err for token in fallback_triggers):
            raise
        obj = complete_json_object(
            fallback_client,
            messages,
            response_format={"type": "json_object"},
            max_tokens=900,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
    winner = str(obj.get("winner") or "tie").strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    winner_method = "tie"
    if winner == "A":
        winner_method = method_a
    elif winner == "B":
        winner_method = method_b
    dim_votes_raw = obj.get("dimension_votes") or {}
    dim_votes = {}
    for dim in dims:
        value = str(dim_votes_raw.get(dim) or "tie").strip()
        dim_votes[dim] = value if value in {"A", "B", "tie"} else "tie"
    return {
        "winner_label": winner,
        "winner_method": winner_method,
        "confidence": round(float(obj.get("confidence") or 0.0), 4),
        "reason": str(obj.get("reason") or "").strip(),
        "dimension_votes": dim_votes,
        "judge_profile": judge_profile,
    }


def needs_escalation(round_rows: List[Dict[str, Any]], *, min_rounds: int, conf_threshold: float) -> bool:
    if len(round_rows) < min_rounds:
        return True
    winners = [str(row.get("winner_method") or "tie") for row in round_rows]
    non_ties = [w for w in winners if w != "tie"]
    counts = {w: non_ties.count(w) for w in set(non_ties)}
    majority = len(round_rows) // 2 + 1
    majority_found = any(v >= majority for v in counts.values())
    mean_conf = sum(float(row.get("confidence") or 0.0) for row in round_rows) / max(len(round_rows), 1)
    mirror_non_ties = [winners[0], winners[1]] if len(winners) >= 2 else winners
    mirror_non_ties = [w for w in mirror_non_ties if w != "tie"]
    mirror_consistent = len(set(mirror_non_ties)) <= 1
    tie_count = winners.count("tie")
    return (not majority_found) or (mean_conf < conf_threshold) or (not mirror_consistent) or (tie_count >= 1)


def decide_final(round_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    winners = [str(row.get("winner_method") or "tie") for row in round_rows]
    non_ties = [w for w in winners if w != "tie"]
    counts = {w: non_ties.count(w) for w in set(non_ties)}
    majority = len(round_rows) // 2 + 1
    final_winner = "tie"
    for method, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if count >= majority:
            final_winner = method
            break
    mean_conf = sum(float(row.get("confidence") or 0.0) for row in round_rows) / max(len(round_rows), 1)
    first_two = winners[:2]
    mirror_consistent = len(set([w for w in first_two if w != "tie"])) <= 1
    return {
        "winner_method": final_winner,
        "wins": {k: int(v) for k, v in counts.items()} | {"tie": int(winners.count("tie"))},
        "rounds_run": len(round_rows),
        "mean_confidence": round(mean_conf, 4),
        "mirror_consistent": mirror_consistent,
        "unstable": final_winner == "tie" or not mirror_consistent,
    }


def run_comparison(
    client: OpenAICompatChatClient,
    fallback_client: OpenAICompatChatClient | None,
    *,
    judge_profile: str,
    seed: int,
    public_task: Dict[str, Any],
    method_x: str,
    method_y: str,
    row_x: Dict[str, Any],
    row_y: Dict[str, Any],
    min_rounds: int,
    max_rounds: int,
    escalate_conf_threshold: float,
) -> Dict[str, Any]:
    task_id = str(public_task["task_id"])
    family = str(public_task.get("family") or "")
    comparison_key = f"{task_id}::{min(method_x, method_y)}__vs__{max(method_x, method_y)}"
    raw_rounds = []
    for round_index in range(1, max_rounds + 1):
        method_a, method_b = round_orientation(seed, task_id, method_x, method_y, round_index)
        row_a = row_x if method_a == method_x else row_y
        row_b = row_y if method_b == method_y else row_x
        judged = judge_round(
            client,
            fallback_client,
            judge_profile=judge_profile,
            public_task=public_task,
            family=family,
            method_a=method_a,
            method_b=method_b,
            row_a=row_a,
            row_b=row_b,
        )
        raw_rounds.append(
            {
                "round_index": round_index,
                "orientation": f"round_{round_index}",
                "method_a": method_a,
                "method_b": method_b,
                **judged,
            }
        )
        if round_index >= min_rounds and not needs_escalation(
            raw_rounds,
            min_rounds=min_rounds,
            conf_threshold=escalate_conf_threshold,
        ):
            break
    final = decide_final(raw_rounds)
    return {
        "comparison_key": comparison_key,
        "task_id": task_id,
        "family": family,
        "domain": str(public_task.get("domain") or ""),
        "methods": [method_x, method_y],
        "judge_profile": judge_profile,
        "rounds": raw_rounds,
        **final,
    }


def run_comparison_with_retries(
    client: OpenAICompatChatClient,
    fallback_client: OpenAICompatChatClient | None,
    *,
    judge_profile: str,
    seed: int,
    public_task: Dict[str, Any],
    method_x: str,
    method_y: str,
    row_x: Dict[str, Any],
    row_y: Dict[str, Any],
    min_rounds: int,
    max_rounds: int,
    escalate_conf_threshold: float,
    job_retries: int,
) -> Dict[str, Any]:
    last_error = ""
    for attempt in range(job_retries + 1):
        try:
            return run_comparison(
                client,
                fallback_client,
                judge_profile=judge_profile,
                seed=seed,
                public_task=public_task,
                method_x=method_x,
                method_y=method_y,
                row_x=row_x,
                row_y=row_y,
                min_rounds=min_rounds,
                max_rounds=max_rounds,
                escalate_conf_threshold=escalate_conf_threshold,
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < job_retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        f"run_comparison failed after retries: task={public_task.get('task_id')} pair={method_x}__vs__{method_y}; error={last_error}"
    )


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_jsonl = output_dir / "pairwise_bestofk_results.jsonl"
    raw_jsonl = output_dir / "pairwise_bestofk_rounds.jsonl"
    error_jsonl = output_dir / "pairwise_bestofk_errors.jsonl"

    public_by_id = load_public_tasks(release_dir)
    methods, results_by_method = load_method_results(args.input)
    allowed_task_ids = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    task_ids = sorted(set.intersection(*(set(results_by_method[m].keys()) for m in methods)))
    if allowed_task_ids is not None:
        task_ids = [task_id for task_id in task_ids if task_id in allowed_task_ids]
    if args.task_limit is not None:
        task_ids = task_ids[: args.task_limit]

    jobs = []
    for task_id in task_ids:
        public_task = public_by_id.get(task_id)
        if not public_task:
            continue
        for method_x, method_y in combinations(methods, 2):
            jobs.append(
                {
                    "comparison_key": f"{task_id}::{min(method_x, method_y)}__vs__{max(method_x, method_y)}",
                    "public_task": public_task,
                    "method_x": method_x,
                    "method_y": method_y,
                    "row_x": results_by_method[method_x][task_id],
                    "row_y": results_by_method[method_y][task_id],
                }
            )

    completed = set()
    if args.resume and final_jsonl.exists():
        completed = {str(row.get("comparison_key") or "") for row in iter_jsonl(final_jsonl)}
    if completed:
        jobs = [job for job in jobs if job["comparison_key"] not in completed]

    judge_client = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))
    fallback_client = None
    fallback_path = Path(args.fallback_judge_llm_config)
    if fallback_path.exists():
        fallback_client = OpenAICompatChatClient(load_openai_compat_config(fallback_path))
    final_mode = "a" if args.resume and final_jsonl.exists() else "w"
    raw_mode = "a" if args.resume and raw_jsonl.exists() else "w"
    err_mode = "a" if args.resume and error_jsonl.exists() else "w"
    total = len(jobs)
    with (
        final_jsonl.open(final_mode, encoding="utf-8") as final_handle,
        raw_jsonl.open(raw_mode, encoding="utf-8") as raw_handle,
        error_jsonl.open(err_mode, encoding="utf-8") as err_handle,
    ):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    run_comparison_with_retries,
                    judge_client,
                    fallback_client,
                    judge_profile=args.judge_profile,
                    seed=args.seed,
                    public_task=job["public_task"],
                    method_x=job["method_x"],
                    method_y=job["method_y"],
                    row_x=job["row_x"],
                    row_y=job["row_y"],
                    min_rounds=args.min_rounds,
                    max_rounds=args.max_rounds,
                    escalate_conf_threshold=args.escalate_conf_threshold,
                    job_retries=args.job_retries,
                ): job
                for job in jobs
            }
            for idx, future in enumerate(as_completed(future_map), start=1):
                job = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:
                    err_row = {
                        "comparison_key": job["comparison_key"],
                        "task_id": job["public_task"].get("task_id"),
                        "family": job["public_task"].get("family"),
                        "domain": job["public_task"].get("domain"),
                        "methods": [job["method_x"], job["method_y"]],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    err_handle.write(json.dumps(err_row, ensure_ascii=False) + "\n")
                    err_handle.flush()
                    print(f"[pairwise-bestofk][error] {idx}/{total} {job['comparison_key']} {err_row['error']}", flush=True)
                    continue
                for round_row in row["rounds"]:
                    raw_handle.write(
                        json.dumps(
                            {
                                "comparison_key": row["comparison_key"],
                                "task_id": row["task_id"],
                                "family": row["family"],
                                "domain": row["domain"],
                                "methods": row["methods"],
                                "judge_profile": args.judge_profile,
                                **round_row,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                raw_handle.flush()
                final_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                final_handle.flush()
                print(
                    f"[pairwise-bestofk] {idx}/{total} {row['comparison_key']} winner={row['winner_method']} rounds={row['rounds_run']}",
                    flush=True,
                )

    summary = {
        "task_count": len(task_ids),
        "method_count": len(methods),
        "methods": methods,
        "comparison_count": sum(1 for _ in iter_jsonl(final_jsonl)),
        "rounds_path": str(raw_jsonl),
        "results_path": str(final_jsonl),
        "errors_path": str(error_jsonl),
        "min_rounds": args.min_rounds,
        "max_rounds": args.max_rounds,
        "job_retries": args.job_retries,
        "judge_profile": args.judge_profile,
        "fallback_judge_llm_config": str(fallback_path) if fallback_client is not None else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
