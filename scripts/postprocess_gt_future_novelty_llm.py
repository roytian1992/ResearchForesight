from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.eval_v3 import build_hidden_eval_v3_row
from researchworld.eval_v3_1 import build_hidden_eval_v3_1_row
from researchworld.llm import OpenAICompatChatClient, complete_json_object, load_openai_compat_config


STRING_KEYS = ("display_name", "name", "direction", "topic", "topic_title", "title")
STRICT_RULES = {
    "bottleneck_opportunity_discovery": "future_descendants",
    "direction_forecasting": "emergent_descendants",
    "strategic_research_planning": "direction_records",
    "venue_aware_research_positioning": "direction_records",
}
PIPELINE_VERSION = "candidate_llm_vote_v3_qwen8002"
VALID_REASON_CODES = {
    "keep_novel_descendant",
    "keep_novel_specialization",
    "keep_novel_reframing",
    "keep_novel_dependency",
    "keep_rule_clear",
    "remove_duplicate_history",
    "remove_duplicate_future",
    "remove_too_generic",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "via",
    "with",
}
GENERIC_MODIFIERS = {
    "analysis",
    "analyses",
    "approach",
    "approaches",
    "architecture",
    "architectures",
    "benchmark",
    "benchmarks",
    "driven",
    "evaluation",
    "evaluations",
    "framework",
    "frameworks",
    "guided",
    "improved",
    "improving",
    "method",
    "methods",
    "model",
    "models",
    "novel",
    "protocol",
    "protocols",
    "robust",
    "scalable",
    "study",
    "studies",
    "system",
    "systems",
}
ABBREVIATION_EXPANSIONS = {
    "rl": "reinforcement learning",
    "rag": "retrieval augmented generation",
    "llm": "large language model",
    "vlm": "vision language model",
    "mllm": "multimodal large language model",
    "sft": "supervised fine tuning",
    "dpo": "direct preference optimization",
    "ppo": "proximal policy optimization",
    "grpo": "group relative policy optimization",
}
SOFT_HISTORY_CANDIDATE_SCORE = 0.58
SOFT_FUTURE_CANDIDATE_SCORE = 0.62
DEFAULT_VOTE_TEMPERATURES = (0.2, 0.5, 0.8)
DEFAULT_VOTE_MAX_TOKENS = 12000


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_hidden_v3_manifest(release_dir: Path, rows: Sequence[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    family_counts = Counter()
    domain_counts = Counter()
    claim_type_counts = Counter()
    slot_field_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row.get("family") or "")
        domain = str(row.get("domain") or "")
        family_counts[family] += 1
        domain_counts[domain] += 1
        for claim in row.get("claim_bank") or []:
            claim_type_counts[str(claim.get("claim_type") or "")] += 1
        for key in (row.get("slot_targets") or {}).keys():
            slot_field_counts[family][str(key)] += 1
    return {
        "release_dir": str(release_dir),
        "output": str(output_path),
        "task_count": len(rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "claim_type_counts": dict(claim_type_counts),
        "slot_field_coverage": {family: dict(counter) for family, counter in slot_field_counts.items()},
        "notes": [
            "v3 hidden eval adds slot_targets, claim_bank, and judge_profile on top of hidden eval.",
            "This manifest was regenerated after LLM-based future novelty post-processing.",
        ],
    }


def build_hidden_v31_manifest(release_dir: Path, rows: Sequence[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    family_counts = Counter()
    domain_counts = Counter()
    component_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    future_unit_counts = Counter()
    for row in rows:
        family = str(row.get("family") or "")
        domain = str(row.get("domain") or "")
        family_counts[family] += 1
        domain_counts[domain] += 1
        for component in ((row.get("component_targets") or {}).get("components") or []):
            component_counts[family][str(component.get("name") or "")] += 1
        future_unit_counts[family] += len(((row.get("future_alignment_targets") or {}).get("alignment_units") or []))
    return {
        "release_dir": str(release_dir),
        "output": str(output_path),
        "task_count": len(rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "component_coverage": {family: dict(counter) for family, counter in component_counts.items()},
        "future_unit_counts": dict(future_unit_counts),
        "notes": [
            "v3.1 adds component_targets and future_alignment_targets on top of v3 hidden eval.",
            "This manifest was regenerated after LLM-based future novelty post-processing.",
        ],
    }


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("-", " ").split()).strip()


def expand_abbreviations(text: Any) -> str:
    normalized = normalize_ws(text).lower()
    if not normalized:
        return ""
    for short, long_form in ABBREVIATION_EXPANSIONS.items():
        normalized = re.sub(rf"\b{re.escape(short)}\b", long_form, normalized)
    return normalized


def ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def is_synthetic_comparative_topic(public: Dict[str, Any]) -> bool:
    variant = normalize_ws(public.get("task_variant")).lower()
    topic = normalize_ws(public.get("topic"))
    topic_title = normalize_ws(public.get("topic_title"))
    if variant == "comparative_opportunity_prioritization":
        return True
    text = f"{topic} || {topic_title}".lower()
    return " vs " in text or " versus " in text


def label_from_item(item: Any) -> str:
    if isinstance(item, str):
        return normalize_ws(item)
    if isinstance(item, dict):
        for key in STRING_KEYS:
            value = normalize_ws(item.get(key))
            if value:
                return value
    return ""


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        norm = normalize_ws(value)
        lowered = norm.lower()
        if not norm or lowered in seen:
            continue
        seen.add(lowered)
        out.append(norm)
    return out


def extract_history_labels(hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    public = ensure_dict(hidden_row.get("public_metadata") or trace_row.get("public_metadata"))
    gt = ensure_dict(trace_row.get("ground_truth"))
    support = ensure_dict(trace_row.get("support_context"))
    if not is_synthetic_comparative_topic(public):
        for key in ("topic", "topic_title"):
            value = normalize_ws(public.get(key))
            if value:
                labels.append(value)
    # candidate_directions are future-side planning options for strategic/venue tasks,
    # so they must not be treated as historical context.
    for key in ("top_limitations", "top_future_work", "history_chain"):
        for item in support.get(key) or []:
            label = label_from_item(item)
            if label:
                labels.append(label)
    for key in ("historical_limitation_signals", "historical_future_work_signals"):
        for item in gt.get(key) or []:
            label = label_from_item(item)
            if label:
                labels.append(label)
    return dedupe_keep_order(labels)


def extract_future_labels(hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    public = ensure_dict(hidden_row.get("public_metadata") or trace_row.get("public_metadata"))
    gt = ensure_dict(trace_row.get("ground_truth"))
    for key in ("future_descendants", "emergent_descendants", "realized_opportunity_directions", "direction_records"):
        for item in gt.get(key) or []:
            label = label_from_item(item)
            if label:
                labels.append(label)
    for item in public.get("future_themes") or []:
        label = label_from_item(item)
        if label:
            labels.append(label)
    return dedupe_keep_order(labels)


def tokenize(text: Any) -> List[str]:
    return [stem_token(token.lower()) for token in TOKEN_RE.findall(expand_abbreviations(text))]


def stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def full_tokens(text: Any) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def reduced_tokens(text: Any) -> List[str]:
    return [token for token in full_tokens(text) if token not in GENERIC_MODIFIERS]


def token_set(text: Any, *, reduced: bool) -> set[str]:
    values = reduced_tokens(text) if reduced else full_tokens(text)
    return set(values)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def similarity(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_duplicate_like(candidate: str, anchor: str) -> Tuple[bool, float, str]:
    cand = normalize_ws(candidate)
    base = normalize_ws(anchor)
    if not cand or not base:
        return False, 0.0, ""
    if cand.lower() == base.lower():
        return True, 1.0, "exact_match"
    reduced_c = token_set(cand, reduced=True)
    reduced_b = token_set(base, reduced=True)
    full_c = token_set(cand, reduced=False)
    full_b = token_set(base, reduced=False)
    if reduced_c and reduced_b and reduced_c == reduced_b:
        return True, 0.96, "same_reduced_tokens"
    jac_reduced = jaccard(reduced_c, reduced_b)
    cont_reduced = containment(reduced_c, reduced_b)
    jac_full = jaccard(full_c, full_b)
    if jac_reduced >= 0.82:
        return True, jac_reduced, "high_reduced_jaccard"
    if cont_reduced >= 0.88 and max(len(reduced_c), len(reduced_b)) - min(len(reduced_c), len(reduced_b)) <= 1:
        return True, cont_reduced, "near_subset_reduced"
    if jac_full >= 0.9:
        return True, jac_full, "high_full_jaccard"
    return False, max(jac_reduced, cont_reduced, jac_full), ""


def best_match(candidate: str, bank: Sequence[str]) -> Tuple[str, float, str]:
    best_label = ""
    best_score = 0.0
    best_reason = ""
    for anchor in bank:
        dup, score, reason = is_duplicate_like(candidate, anchor)
        if score > best_score:
            best_label = anchor
            best_score = score
            best_reason = reason if dup else ""
    return best_label, best_score, best_reason


def heuristic_candidate_for_label(label: str, history_labels: Sequence[str], kept_labels: Sequence[str]) -> Optional[Dict[str, Any]]:
    history_match, history_score, history_rule = best_match(label, history_labels)
    if history_rule:
        return {
            "candidate_kind": "history",
            "matched_label": history_match,
            "match_score": round(history_score, 4),
            "match_rule": history_rule,
        }
    if history_match and history_score >= SOFT_HISTORY_CANDIDATE_SCORE:
        return {
            "candidate_kind": "history",
            "matched_label": history_match,
            "match_score": round(history_score, 4),
            "match_rule": "soft_similarity",
        }
    future_match, future_score, future_rule = best_match(label, kept_labels)
    if future_rule and future_score >= 0.86:
        return {
            "candidate_kind": "future_peer",
            "matched_label": future_match,
            "match_score": round(future_score, 4),
            "match_rule": future_rule,
        }
    if future_match and future_score >= SOFT_FUTURE_CANDIDATE_SCORE:
        return {
            "candidate_kind": "future_peer",
            "matched_label": future_match,
            "match_score": round(future_score, 4),
            "match_rule": "soft_similarity",
        }
    return None


def filter_labeled_list(items: Any, kept_labels: set[str]) -> Any:
    if not isinstance(items, list):
        return items
    out = []
    for item in items:
        label = label_from_item(item)
        if label and label in kept_labels:
            out.append(item)
    return out


def filter_future_papers_against_kept_labels(items: Any, kept_labels: Sequence[str]) -> Tuple[Any, Dict[str, Any]]:
    if not isinstance(items, list):
        return items, {"removed_count": 0}
    if not kept_labels:
        removed = [{"title": label_from_item(item), "future_score": 0.0} for item in items[:5]]
        return [], {"removed_count": len(items), "removed_examples": removed, "selection_policy": "no_kept_future_labels"}
    kept = []
    removed = []
    for item in items:
        title = label_from_item(item)
        best_future_score = max((similarity(title, label) for label in kept_labels), default=0.0)
        if best_future_score >= 0.12:
            kept.append(item)
        else:
            removed.append({"title": title, "future_score": round(best_future_score, 4)})
    return kept, {
        "removed_count": len(removed),
        "removed_examples": removed[:5],
        "selection_policy": "title_to_kept_future_label_similarity_only",
    }


def strict_keep(internal_row: Dict[str, Any]) -> bool:
    family = str(internal_row.get("family") or "")
    key = STRICT_RULES.get(family)
    if not key:
        return False
    ground_truth = internal_row.get("ground_truth") or {}
    return bool(ground_truth.get(key))


class CascadeJSONJudge:
    def __init__(self, config_paths: Sequence[Path], *, timeout: int, transport_retries: int, max_parse_attempts: int) -> None:
        self.models: List[Tuple[str, OpenAICompatChatClient]] = []
        for path in config_paths:
            self.models.append((path.name, OpenAICompatChatClient(load_openai_compat_config(path))))
        self.timeout = timeout
        self.transport_retries = transport_retries
        self.max_parse_attempts = max_parse_attempts

    def complete(self, messages: List[Dict[str, str]], *, max_tokens: int, start_index: int = 0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        errors: List[Dict[str, str]] = []
        for idx in range(start_index, len(self.models)):
            try:
                return self.complete_one_model(messages, max_tokens=max_tokens, model_index=idx)
            except Exception as exc:
                name, _ = self.models[idx]
                errors.append({"model_name": name, "error": str(exc)})
        raise RuntimeError(f"all LLMs failed: {json.dumps(errors, ensure_ascii=False)}")

    def complete_one_model(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        model_index: int,
        temperature: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        name, client = self.models[model_index]
        obj = complete_json_object(
            client,
            messages,
            timeout=self.timeout,
            transport_retries=self.transport_retries,
            max_parse_attempts=self.max_parse_attempts,
            max_tokens=max_tokens,
            temperature=client.config.temperature if temperature is None else temperature,
        )
        return obj, {"model_name": name, "cascade_index": model_index}


def candidate_review_prompt(
    task_id: str,
    family: str,
    topic: str,
    label: str,
    heuristic_candidate: Dict[str, Any],
) -> List[Dict[str, str]]:
    system = (
        "You are reviewing one future ground-truth label for a research benchmark. "
        "Be conservative about removal. Remove only if the candidate label is materially the same idea as the matched label. "
        "Do not remove just because there is lexical overlap, a shared parent topic, or a candidate-direction list. "
        "Keep genuinely new descendants, specializations, downstream applications, venue reframings, and new technical combinations. "
        "Return exactly one JSON object."
    )
    suspect_kind = str(heuristic_candidate.get("candidate_kind") or "")
    matched_label = normalize_ws(heuristic_candidate.get("matched_label"))
    user_lines = [
        f"task_id: {task_id}",
        f"family: {family}",
        f"topic: {topic}",
        f"candidate_future_label: {label}",
        f"heuristic_suspect_type: {suspect_kind}",
        f"matched_label: {matched_label}",
        f"heuristic_match_rule: {heuristic_candidate.get('match_rule')}",
        f"heuristic_match_score: {heuristic_candidate.get('match_score')}",
        "remove_only_if: materially the same idea",
        "keep_if: new specialization, descendant, downstream application, venue framing, dependency shift, or objective shift",
        "exact abbreviation-vs-full-form duplicates may be removed",
        "return_json_schema: {\"decision\":\"keep|remove\",\"reason_code\":\"keep_novel_descendant|keep_novel_specialization|keep_novel_reframing|keep_novel_dependency|keep_rule_clear|remove_duplicate_history|remove_duplicate_future|remove_too_generic\",\"matched_label\":\"...\",\"confidence\":0.0,\"rationale\":\"...\"}",
    ]
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(user_lines)}]


def normalize_reason_code(value: Any) -> str:
    normalized = normalize_ws(value).lower().replace(" ", "_").replace("-", "_")
    return normalized if normalized in VALID_REASON_CODES else normalized


def build_vote_schema_retry_message(error: Exception) -> str:
    return (
        "Your previous answer did not match the required JSON schema. "
        f"Schema validation error: {error}. "
        "Retry now and return exactly one JSON object with keys "
        "`decision`, `reason_code`, `matched_label`, `confidence`, and `rationale`."
    )


def normalize_vote_obj(obj: Dict[str, Any], heuristic_candidate: Dict[str, Any]) -> Dict[str, Any]:
    decision = str(obj.get("decision") or "").strip().lower()
    if decision not in {"keep", "remove"}:
        raise ValueError(f"invalid decision: {decision}")
    reason_code = normalize_reason_code(obj.get("reason_code"))
    if reason_code not in VALID_REASON_CODES:
        if decision == "remove":
            reason_code = (
                "remove_duplicate_history"
                if heuristic_candidate.get("candidate_kind") == "history"
                else "remove_duplicate_future"
            )
        else:
            reason_code = "keep_rule_clear"
    matched_label = normalize_ws(
        obj.get("matched_label")
        or obj.get("matched_history_label")
        or obj.get("matched_future_label")
        or heuristic_candidate.get("matched_label")
    )
    return {
        "decision": decision,
        "reason_code": reason_code,
        "matched_label": matched_label,
        "confidence": float(obj.get("confidence") or 0.0),
        "rationale": normalize_ws(obj.get("rationale")),
    }


def review_candidate_label(
    judge: CascadeJSONJudge,
    task_id: str,
    family: str,
    topic: str,
    label: str,
    heuristic_candidate: Dict[str, Any],
) -> Dict[str, Any]:
    messages = candidate_review_prompt(task_id, family, topic, label, heuristic_candidate)
    vote_trace: List[Dict[str, Any]] = []
    votes: List[Dict[str, Any]] = []

    for model_index in range(len(judge.models)):
        vote_temperature = DEFAULT_VOTE_TEMPERATURES[min(model_index, len(DEFAULT_VOTE_TEMPERATURES) - 1)]
        model_messages = list(messages)
        normalized_vote: Optional[Dict[str, Any]] = None
        for schema_retry in range(3):
            try:
                obj, meta = judge.complete_one_model(
                    model_messages,
                    max_tokens=DEFAULT_VOTE_MAX_TOKENS,
                    model_index=model_index,
                    temperature=vote_temperature,
                )
            except Exception as exc:
                vote_trace.append(
                    {
                        "label": label,
                        "model_name": judge.models[model_index][0],
                        "cascade_index": model_index,
                        "temperature": vote_temperature,
                        "status": "transport_or_parse_failed",
                        "error": str(exc),
                    }
                )
                break
            try:
                normalized_vote = normalize_vote_obj(obj, heuristic_candidate)
            except Exception as exc:
                vote_trace.append(
                    {
                        "label": label,
                        "model_name": meta["model_name"],
                        "cascade_index": meta["cascade_index"],
                        "temperature": vote_temperature,
                        "status": "schema_invalid",
                        "schema_retry": schema_retry,
                        "error": str(exc),
                        "raw_keys": sorted(obj.keys()) if isinstance(obj, dict) else [],
                    }
                )
                if schema_retry >= 2:
                    break
                model_messages = model_messages + [
                    {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                    {"role": "user", "content": build_vote_schema_retry_message(exc)},
                ]
                continue
            vote = {
                "label": label,
                "model_name": meta["model_name"],
                "cascade_index": meta["cascade_index"],
                "temperature": vote_temperature,
                **normalized_vote,
            }
            votes.append(vote)
            vote_trace.append(
                {
                    "label": label,
                    "model_name": meta["model_name"],
                    "cascade_index": meta["cascade_index"],
                    "temperature": vote_temperature,
                    "status": "ok",
                    "decision": normalized_vote["decision"],
                    "reason_code": normalized_vote["reason_code"],
                    "confidence": normalized_vote["confidence"],
                }
            )
            break

    remove_votes = [vote for vote in votes if vote["decision"] == "remove"]
    keep_votes = [vote for vote in votes if vote["decision"] == "keep"]
    if len(votes) < 2:
        majority_decision = "keep"
        winner_votes = keep_votes
        winning_reason_code = "keep_rule_clear"
        winning_rationale = "Insufficient valid model votes; defaulting to conservative keep."
        winning_match = ""
        majority_confidence = 0.0
    elif len(remove_votes) > len(keep_votes):
        majority_decision = "remove"
        winner_votes = remove_votes
        exemplar = max(remove_votes, key=lambda row: row["confidence"])
        winning_reason_code = exemplar["reason_code"]
        winning_rationale = exemplar["rationale"]
        winning_match = exemplar["matched_label"]
        majority_confidence = sum(vote["confidence"] for vote in remove_votes) / len(remove_votes)
    else:
        majority_decision = "keep"
        winner_votes = keep_votes
        exemplar = max(keep_votes, key=lambda row: row["confidence"]) if keep_votes else None
        winning_reason_code = exemplar["reason_code"] if exemplar else "keep_rule_clear"
        winning_rationale = exemplar["rationale"] if exemplar else "Majority keep."
        winning_match = exemplar["matched_label"] if exemplar else ""
        majority_confidence = (
            sum(vote["confidence"] for vote in keep_votes) / len(keep_votes) if keep_votes else 0.0
        )
    return {
        "label": label,
        "heuristic_candidate": heuristic_candidate,
        "votes": votes,
        "vote_trace": vote_trace,
        "decision": majority_decision,
        "reason_code": winning_reason_code,
        "matched_label": winning_match,
        "confidence": round(majority_confidence, 4),
        "rationale": winning_rationale,
        "valid_vote_count": len(votes),
        "majority_vote_count": len(winner_votes),
    }


def judge_task(
    judge: CascadeJSONJudge,
    task_id: str,
    family: str,
    topic: str,
    history_labels: Sequence[str],
    future_labels: Sequence[str],
) -> Dict[str, Any]:
    if not future_labels:
        return {
            "judge_pipeline_version": PIPELINE_VERSION,
            "task_id": task_id,
            "family": family,
            "topic": topic,
            "history_labels": list(history_labels),
            "future_labels": [],
            "overall_confidence": 1.0,
            "judge_trace": [],
            "label_decisions": [],
            "kept_future_labels": [],
            "removed_future_labels": [],
            "heuristic_candidate_count": 0,
            "llm_reviewed_label_count": 0,
            "auto_kept_label_count": 0,
        }
    task_trace: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    kept: List[str] = []
    removed: List[Dict[str, Any]] = []
    heuristic_candidate_count = 0
    llm_reviewed_label_count = 0
    auto_kept_label_count = 0
    for label in future_labels:
        heuristic_candidate = heuristic_candidate_for_label(label, history_labels, kept)
        if heuristic_candidate is None:
            decisions.append(
                {
                    "label": label,
                    "decision": "keep",
                    "reason_code": "keep_rule_clear",
                    "matched_history_label": "",
                    "matched_future_label": "",
                    "confidence": 1.0,
                    "rationale": "No rule-based duplicate candidate was triggered against history or already-kept future labels.",
                }
            )
            kept.append(label)
            auto_kept_label_count += 1
            continue
        heuristic_candidate_count += 1
        llm_reviewed_label_count += 1
        review = review_candidate_label(judge, task_id, family, topic, label, heuristic_candidate)
        task_trace.append(review)
        if review["decision"] == "keep":
            decisions.append(
                {
                    "label": label,
                    "decision": "keep",
                    "reason_code": review["reason_code"],
                    "matched_history_label": review["matched_label"] if heuristic_candidate["candidate_kind"] == "history" else "",
                    "matched_future_label": review["matched_label"] if heuristic_candidate["candidate_kind"] == "future_peer" else "",
                    "confidence": review["confidence"],
                    "rationale": review["rationale"],
                }
            )
            kept.append(label)
        else:
            decision_row = {
                "label": label,
                "decision": "remove",
                "reason_code": review["reason_code"],
                "matched_history_label": review["matched_label"] if heuristic_candidate["candidate_kind"] == "history" else "",
                "matched_future_label": review["matched_label"] if heuristic_candidate["candidate_kind"] == "future_peer" else "",
                "confidence": review["confidence"],
                "rationale": review["rationale"],
            }
            decisions.append(decision_row)
            removed.append(
                {
                    "label": label,
                    "reason_code": review["reason_code"],
                    "matched_history_label": decision_row["matched_history_label"],
                    "matched_future_label": decision_row["matched_future_label"],
                    "confidence": round(review["confidence"], 4),
                    "rationale": review["rationale"],
                    "heuristic_match_rule": heuristic_candidate["match_rule"],
                    "heuristic_match_score": heuristic_candidate["match_score"],
                }
            )
    overall_confidence = sum(row["confidence"] for row in decisions) / len(decisions) if decisions else 1.0
    return {
        "judge_pipeline_version": PIPELINE_VERSION,
        "task_id": task_id,
        "family": family,
        "topic": topic,
        "history_labels": list(history_labels),
        "future_labels": list(future_labels),
        "overall_confidence": round(overall_confidence, 4),
        "judge_trace": task_trace,
        "label_decisions": decisions,
        "kept_future_labels": kept,
        "removed_future_labels": removed,
        "heuristic_candidate_count": heuristic_candidate_count,
        "llm_reviewed_label_count": llm_reviewed_label_count,
        "auto_kept_label_count": auto_kept_label_count,
    }


def patch_rows(
    hidden_row: Dict[str, Any],
    trace_row: Dict[str, Any],
    internal_row: Optional[Dict[str, Any]],
    decision_report: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    hidden = json.loads(json.dumps(hidden_row, ensure_ascii=False))
    trace = json.loads(json.dumps(trace_row, ensure_ascii=False))
    internal = json.loads(json.dumps(internal_row, ensure_ascii=False)) if internal_row is not None else None

    kept_labels = set(decision_report["kept_future_labels"])
    history_labels = decision_report["history_labels"]

    hidden_public = ensure_dict(hidden.get("public_metadata"))
    trace_public = ensure_dict(trace.get("public_metadata"))
    hidden["public_metadata"] = hidden_public
    trace["public_metadata"] = trace_public
    hidden_public["future_themes"] = list(decision_report["kept_future_labels"])
    trace_public["future_themes"] = list(decision_report["kept_future_labels"])
    if internal is not None:
        internal_public = ensure_dict(internal.get("public_metadata"))
        internal["public_metadata"] = internal_public
        internal_public["future_themes"] = list(decision_report["kept_future_labels"])

    for row in [hidden, trace, internal]:
        if row is None:
            continue
        gt = ensure_dict(row.get("ground_truth"))
        for key in ("future_descendants", "emergent_descendants", "realized_opportunity_directions", "direction_records"):
            if key in gt:
                gt[key] = filter_labeled_list(gt.get(key), kept_labels)
        ref = gt.get("reference_papers") or {}
        if isinstance(ref, dict):
            for key in ("future_q4", "future_q1"):
                if key in ref:
                    filtered, _ = filter_future_papers_against_kept_labels(ref.get(key), decision_report["kept_future_labels"])
                    ref[key] = filtered
        row["ground_truth"] = gt

    support = ensure_dict(trace.get("support_context"))
    support_report: Dict[str, Any] = {}
    if "future_validation_set" in support:
        filtered, support_report = filter_future_papers_against_kept_labels(
            support.get("future_validation_set"),
            decision_report["kept_future_labels"],
        )
        support["future_validation_set"] = filtered
    trace["support_context"] = support
    if internal is not None and isinstance(internal.get("support_context"), dict) and "future_validation_set" in internal["support_context"]:
        filtered, _ = filter_future_papers_against_kept_labels(
            internal["support_context"].get("future_validation_set"),
            decision_report["kept_future_labels"],
        )
        internal["support_context"]["future_validation_set"] = filtered

    report = {
        "task_id": hidden.get("task_id"),
        "family": hidden.get("family"),
        "topic": hidden_public.get("topic") or trace_public.get("topic") or "",
        "history_label_count": len(history_labels),
        "original_future_labels": decision_report["future_labels"],
        "kept_future_labels": decision_report["kept_future_labels"],
        "removed_future_labels": decision_report["removed_future_labels"],
        "overall_confidence": decision_report["overall_confidence"],
        "judge_trace": decision_report["judge_trace"],
        "support_future_validation_update": support_report,
    }
    return hidden, trace, internal, report


def copy_release_file(src_dir: Path, dst_dir: Path, name: str) -> None:
    src = src_dir / name
    if src.exists():
        shutil.copy2(src, dst_dir / name)


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based future novelty cleanup for benchmark GT.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--primary-llm-config", required=True)
    parser.add_argument("--secondary-llm-config", required=True)
    parser.add_argument("--tertiary-llm-config", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--max-parse-attempts", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_by_id = {row["task_id"]: row for row in iter_jsonl(release_dir / "tasks_hidden_eval.jsonl")}
    trace_by_id = {row["task_id"]: row for row in iter_jsonl(release_dir / "tasks_build_trace.jsonl")}
    internal_by_id = {}
    internal_path = release_dir / "tasks_internal_full.jsonl"
    if internal_path.exists():
        internal_by_id = {row["task_id"]: row for row in iter_jsonl(internal_path)}

    ordered_task_ids = sorted(hidden_by_id)
    if args.task_limit and args.task_limit > 0:
        ordered_task_ids = ordered_task_ids[: args.task_limit]
        hidden_by_id = {task_id: hidden_by_id[task_id] for task_id in ordered_task_ids}
        trace_by_id = {task_id: trace_by_id[task_id] for task_id in ordered_task_ids}
        internal_by_id = {task_id: row for task_id, row in internal_by_id.items() if task_id in hidden_by_id}

    judge = CascadeJSONJudge(
        [
            Path(args.primary_llm_config),
            Path(args.secondary_llm_config),
            Path(args.tertiary_llm_config),
        ],
        timeout=args.timeout,
        transport_retries=args.transport_retries,
        max_parse_attempts=args.max_parse_attempts,
    )

    cache_path = output_dir / "future_novelty_llm_decisions.jsonl"
    cached: Dict[str, Dict[str, Any]] = {}
    resume_cache_compatible = False
    if args.resume and cache_path.exists():
        cached_rows = list(iter_jsonl(cache_path))
        compatible_rows = [row for row in cached_rows if row.get("judge_pipeline_version") == PIPELINE_VERSION]
        if compatible_rows and len(compatible_rows) == len(cached_rows):
            cached = {row["task_id"]: row for row in compatible_rows}
            resume_cache_compatible = True

    tasks_for_llm: List[Tuple[str, str, str, List[str], List[str]]] = []
    for task_id in sorted(hidden_by_id):
        hidden = hidden_by_id[task_id]
        trace = trace_by_id[task_id]
        history_labels = extract_history_labels(hidden, trace)
        future_labels = extract_future_labels(hidden, trace)
        topic = normalize_ws((hidden.get("public_metadata") or {}).get("topic_title") or (hidden.get("public_metadata") or {}).get("topic"))
        if task_id in cached:
            prior = cached[task_id]
            if prior.get("future_labels") == future_labels and prior.get("history_labels") == history_labels:
                continue
        tasks_for_llm.append((task_id, str(hidden.get("family") or ""), topic, history_labels, future_labels))

    if tasks_for_llm:
        cache_mode = "a" if resume_cache_compatible else "w"
        with cache_path.open(cache_mode, encoding="utf-8") as cache_handle:
            if cache_mode == "w" and cached:
                for report in cached.values():
                    cache_handle.write(json.dumps(report, ensure_ascii=False) + "\n")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(judge_task, judge, task_id, family, topic, history_labels, future_labels): task_id
                    for task_id, family, topic, history_labels, future_labels in tasks_for_llm
                }
                done = 0
                for future in concurrent.futures.as_completed(futures):
                    report = future.result()
                    cached[report["task_id"]] = report
                    cache_handle.write(json.dumps(report, ensure_ascii=False) + "\n")
                    cache_handle.flush()
                    done += 1
                    print(f"[future_novelty_llm] judged {done}/{len(tasks_for_llm)} {report['task_id']}", flush=True)

    patched_hidden: List[Dict[str, Any]] = []
    patched_trace: List[Dict[str, Any]] = []
    patched_internal: List[Dict[str, Any]] = []
    hidden_v3_rows: List[Dict[str, Any]] = []
    hidden_v31_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    summary = Counter()
    strict_task_ids: List[str] = []

    for task_id in sorted(hidden_by_id):
        hidden = hidden_by_id[task_id]
        trace = trace_by_id[task_id]
        internal = internal_by_id.get(task_id)
        decision_report = cached[task_id]
        patched_h, patched_t, patched_i, report = patch_rows(hidden, trace, internal, decision_report)
        patched_hidden.append(patched_h)
        patched_trace.append(patched_t)
        if patched_i is not None:
            patched_internal.append(patched_i)
            if strict_keep(patched_i):
                strict_task_ids.append(task_id)
        hidden_v3 = build_hidden_eval_v3_row(patched_h, patched_t)
        hidden_v31 = build_hidden_eval_v3_1_row(hidden_v3, patched_t)
        hidden_v3_rows.append(hidden_v3)
        hidden_v31_rows.append(hidden_v31)
        report_rows.append(report)

        summary["task_count"] += 1
        summary["kept_future_labels"] += len(report["kept_future_labels"])
        summary["removed_future_labels"] += len(report["removed_future_labels"])
        summary["heuristic_candidate_labels"] += int(decision_report.get("heuristic_candidate_count") or 0)
        summary["llm_reviewed_label_count"] += int(decision_report.get("llm_reviewed_label_count") or 0)
        summary["auto_kept_label_count"] += int(decision_report.get("auto_kept_label_count") or 0)
        if report["removed_future_labels"]:
            summary["tasks_touched"] += 1
        if decision_report.get("llm_reviewed_label_count"):
            summary["tasks_using_llm"] += 1
        if not report["kept_future_labels"] and report["original_future_labels"]:
            summary["tasks_cleared_to_empty"] += 1
        if (report.get("support_future_validation_update") or {}).get("removed_count"):
            summary["tasks_with_future_paper_prune"] += 1

    copy_release_file(release_dir, output_dir, "tasks.jsonl")
    copy_release_file(release_dir, output_dir, "task_ids.txt")
    copy_release_file(release_dir, output_dir, "manifest.json")
    dump_jsonl(output_dir / "tasks_hidden_eval.jsonl", patched_hidden)
    dump_jsonl(output_dir / "tasks_build_trace.jsonl", patched_trace)
    if patched_internal:
        dump_jsonl(output_dir / "tasks_internal_full.jsonl", patched_internal)
    dump_jsonl(output_dir / "tasks_hidden_eval_v3.jsonl", hidden_v3_rows)
    dump_jsonl(output_dir / "tasks_hidden_eval_v3_1.jsonl", hidden_v31_rows)
    dump_json(output_dir / "tasks_hidden_eval_v3_manifest.json", build_hidden_v3_manifest(release_dir, hidden_v3_rows, output_dir / "tasks_hidden_eval_v3.jsonl"))
    dump_json(output_dir / "tasks_hidden_eval_v3_1_manifest.json", build_hidden_v31_manifest(release_dir, hidden_v31_rows, output_dir / "tasks_hidden_eval_v3_1.jsonl"))
    (output_dir / "strict_task_ids.txt").write_text("\n".join(strict_task_ids) + ("\n" if strict_task_ids else ""), encoding="utf-8")
    dump_json(
        output_dir / "strict_summary.json",
        {
            "release_dir": str(output_dir),
            "strict_task_count": len(strict_task_ids),
            "strict_rules": STRICT_RULES,
            "notes": ["Strict count is recomputed after LLM-based future novelty cleanup using metrics-corresponding GT fields only."],
        },
    )

    for name in ("kb", "future_kb"):
        src = release_dir / name
        if src.exists():
            ensure_symlink(src, output_dir / name)

    manifest = {}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest["release_name"] = output_dir.name
    manifest["source_release"] = str(release_dir)
    manifest["future_novelty_postprocess"] = {
        "script": Path(__file__).name,
        "judge_pipeline_version": PIPELINE_VERSION,
        "task_count": summary["task_count"],
        "tasks_touched": summary["tasks_touched"],
        "removed_future_labels": summary["removed_future_labels"],
        "tasks_cleared_to_empty": summary["tasks_cleared_to_empty"],
        "heuristic_candidate_labels": summary["heuristic_candidate_labels"],
        "llm_reviewed_label_count": summary["llm_reviewed_label_count"],
        "auto_kept_label_count": summary["auto_kept_label_count"],
        "tasks_using_llm": summary["tasks_using_llm"],
        "tasks_with_future_paper_prune": summary["tasks_with_future_paper_prune"],
        "llm_cascade": [
            str(Path(args.primary_llm_config).name),
            str(Path(args.secondary_llm_config).name),
            str(Path(args.tertiary_llm_config).name),
        ],
        "notes": [
            "This pass uses rule-based duplicate-like hits only as LLM review candidates, instead of asking the LLM to judge every future label.",
            "candidate_directions are excluded from historical context to avoid contaminating strategic and venue future labels.",
            "Each candidate label is reviewed independently with concise prompts and three-model majority vote.",
            "Future paper pruning is driven by surviving future labels, not by direct history-overlap removal.",
        ],
    }
    manifest["strict_task_count"] = len(strict_task_ids)
    dump_json(manifest_path, manifest)

    report = {
        "release_dir": str(release_dir),
        "output_dir": str(output_dir),
        **{key: int(value) for key, value in summary.items()},
        "strict_task_count_after_cleanup": len(strict_task_ids),
        "examples": report_rows[:12],
    }
    dump_json(output_dir / "future_novelty_dedup_report.json", report)
    dump_jsonl(output_dir / "future_novelty_dedup_rows.jsonl", report_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
