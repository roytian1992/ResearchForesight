from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.eval_v3 import build_hidden_eval_v3_row
from researchworld.eval_v3_1 import build_hidden_eval_v3_1_row


SCRIPT_VERSION = "evidence_chain_expansion_v1"

DOMAIN_CODE_TO_DISPLAY = {
    "llm_agent": "LLM agents",
    "llm_finetuning_post_training": "LLM fine-tuning and post-training",
    "rag_and_retrieval_structuring": "RAG and retrieval structuring",
    "visual_generative_modeling_and_diffusion": "Visual generative modeling and diffusion",
}
DOMAIN_DISPLAY_TO_CODE = {value: key for key, value in DOMAIN_CODE_TO_DISPLAY.items()}
DOMAIN_ORDER = [
    "llm_agent",
    "llm_finetuning_post_training",
    "rag_and_retrieval_structuring",
    "visual_generative_modeling_and_diffusion",
]

STRICT_RULES = {
    "bottleneck_opportunity_discovery": "future_descendants",
    "direction_forecasting": "emergent_descendants",
    "strategic_research_planning": "direction_records",
    "venue_aware_research_positioning": "direction_records",
}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_copy_or_link(src_dir: Path, dst_dir: Path, name: str) -> None:
    src = src_dir / name
    dst = dst_dir / name
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if src.is_dir():
        dst.symlink_to(src, target_is_directory=True)
    else:
        shutil.copy2(src, dst)


def normalize_domain_code(value: Any) -> str:
    raw = str(value or "").strip()
    return DOMAIN_DISPLAY_TO_CODE.get(raw, raw)


def domain_display(value: Any) -> str:
    code = normalize_domain_code(value)
    return DOMAIN_CODE_TO_DISPLAY.get(code, str(value or code))


def title_case(text: Any) -> str:
    raw = str(text or "").replace("_", " ").strip()
    return raw[:1].upper() + raw[1:] if raw else ""


def candidate_quality_judge(row: Dict[str, Any]) -> Dict[str, Any]:
    return (row.get("judge") or row.get("candidate_quality_judge") or {})


def judge_score(row: Dict[str, Any]) -> float:
    judge = candidate_quality_judge(row)
    if judge.get("overall_score") is not None:
        return float(judge.get("overall_score") or 0.0)
    if judge.get("avg_score") is not None:
        return float(judge.get("avg_score") or 0.0)
    scores = judge.get("scores") or {}
    if scores:
        values = [float(v or 0.0) for v in scores.values()]
        if values:
            return sum(values) / len(values)
    return 0.0


def strict_keep(row: Dict[str, Any]) -> bool:
    family = str(row.get("family") or "")
    gt_key = STRICT_RULES.get(family)
    if not gt_key:
        return False
    ground_truth = row.get("ground_truth") or {}
    return bool(ground_truth.get(gt_key))


def load_release_rows(release_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    public_rows = list(iter_jsonl(release_dir / "tasks.jsonl"))
    hidden_rows = {row["task_id"]: row for row in iter_jsonl(release_dir / "tasks_hidden_eval.jsonl")}
    trace_rows = {row["task_id"]: row for row in iter_jsonl(release_dir / "tasks_build_trace.jsonl")}
    internal_rows = {row["task_id"]: row for row in iter_jsonl(release_dir / "tasks_internal_full.jsonl")}
    return public_rows, hidden_rows, trace_rows, internal_rows


def merge_source_row(public_row: Dict[str, Any], hidden_row: Dict[str, Any], trace_row: Dict[str, Any], internal_row: Dict[str, Any]) -> Dict[str, Any]:
    domain_code = normalize_domain_code(hidden_row.get("domain") or trace_row.get("domain") or internal_row.get("domain") or public_row.get("domain"))
    return {
        "task_id": public_row.get("task_id"),
        "internal_task_id": hidden_row.get("internal_task_id") or trace_row.get("internal_task_id") or internal_row.get("internal_task_id"),
        "family": public_row.get("family") or hidden_row.get("family") or trace_row.get("family"),
        "subtype": public_row.get("subtype"),
        "domain": domain_code,
        "horizon": public_row.get("horizon", "half_year"),
        "title": public_row.get("title") or hidden_row.get("title") or internal_row.get("title"),
        "question": public_row.get("question"),
        "time_cutoff": public_row.get("time_cutoff"),
        "deliverable_spec": public_row.get("deliverable_spec"),
        "gold_answer": hidden_row.get("gold_answer") or internal_row.get("gold_answer"),
        "expected_answer_points": hidden_row.get("expected_answer_points") or internal_row.get("expected_answer_points") or [],
        "ground_truth": trace_row.get("ground_truth") or hidden_row.get("ground_truth") or internal_row.get("ground_truth") or {},
        "support_context": trace_row.get("support_context") or {},
        "time_context": trace_row.get("time_context") or {"history_end": public_row.get("time_cutoff") or "2025-08-31"},
        "seed": trace_row.get("seed") or {},
        "public_metadata": hidden_row.get("public_metadata") or trace_row.get("public_metadata") or internal_row.get("public_metadata") or {},
        "judge": hidden_row.get("judge") or trace_row.get("judge") or internal_row.get("judge") or {},
        "quality_signals": trace_row.get("quality_signals") or {},
        "evaluation_rubric": hidden_row.get("evaluation_rubric"),
    }


def public_deliverable_spec(family: str, subtype: str | None = None) -> Dict[str, Any]:
    base = {
        "format": "free_form_research_analysis",
        "requirements": [
            "Use only evidence available up to the cutoff unless the benchmark setting explicitly provides an offline knowledge base.",
            "State a concrete conclusion rather than vague trend language.",
            "Support the conclusion with literature-based reasoning.",
        ],
    }
    if family == "direction_forecasting" or subtype == "venue_aware_direction_forecast":
        base["requirements"] += [
            "Name one specific next-step direction and characterize the trajectory.",
            "Identify one likely top-tier venue bucket for that direction.",
        ]
    elif family == "strategic_research_planning" and subtype == "comparative_opportunity_prioritization":
        base["requirements"] += [
            "Choose one direction over the alternative rather than hedging.",
            "Justify the comparative priority with evidence-based reasoning.",
            "Explain the trade-off that makes the other option less strategically attractive in the same window.",
        ]
    else:
        base["requirements"] += [
            "Select and justify a small ranked set of priority directions, or identify a focused bottleneck-opportunity argument when the task calls for it.",
            "Make the reasoning structure explicit rather than giving an unstructured list.",
        ]
    return base


def likely_bucket_and_venue(stats: Dict[str, Any]) -> Tuple[str, str, Dict[str, int]]:
    top_buckets = dict(stats.get("top_venue_buckets") or {})
    conf_buckets = {
        str(k): int(v)
        for k, v in top_buckets.items()
        if str(k) not in {"other", "unknown"} and int(v or 0) > 0
    }
    likely_bucket = ""
    if conf_buckets:
        likely_bucket = max(conf_buckets.items(), key=lambda kv: (kv[1], kv[0]))[0]
    top_venues = dict(stats.get("top_venues") or {})
    likely_venue = ""
    if likely_bucket:
        venue_rows = [
            (name, int(count or 0))
            for name, count in top_venues.items()
            if str(name) not in {"arXiv.org", "unknown"} and int(count or 0) > 0
        ]
        if venue_rows:
            likely_venue = sorted(venue_rows, key=lambda kv: (-kv[1], kv[0]))[0][0]
    return likely_bucket, likely_venue, conf_buckets


def family_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    gt = row.get("ground_truth") or {}
    family = str(row.get("family") or "")
    if family in {"direction_forecasting", "bottleneck_opportunity_discovery"}:
        return gt.get("future_half_stats") or {}
    if family == "strategic_research_planning":
        return gt.get("target_window_stats") or {}
    return {}


def selection_rank(row: Dict[str, Any]) -> Tuple[float, int, float, int, int, str]:
    stats = family_stats(row)
    hist = (row.get("support_context") or {}).get("historical_stats") or {}
    return (
        judge_score(row),
        int(stats.get("top_conf_count") or 0),
        float(stats.get("top_conf_share") or 0.0),
        int(stats.get("paper_count") or 0),
        int(hist.get("paper_count") or 0),
        str(row.get("title") or ""),
    )


def venue_direction_question(topic_title: str) -> str:
    return (
        f"Based on scholarly literature available before September 1, 2025, identify one concrete next-step direction within {topic_title} "
        f"that is most likely to gain traction in top-tier AI venues during the subsequent six-month period. Also identify the most likely venue bucket "
        f"(for example AAAI-like, EMNLP-like, ICLR-like, or similar top-tier venues) where that traction would appear. "
        f"Your answer must be justified only with pre-cutoff evidence."
    )


def venue_planning_question(topic_title: str, bucket: str) -> str:
    bucket_label = f"{bucket}-like" if bucket else "top-tier"
    return (
        f"A research team wants to maximize its relevance for {bucket_label} venues in the next submission cycle. "
        f"Based only on literature available before September 1, 2025, which one or two next-step directions in {topic_title} should be prioritized, "
        f"and what evidence-based rationale supports that ranking?"
    )


def make_hidden_row(
    *,
    task_id: str,
    internal_task_id: str,
    family: str,
    domain_code: str,
    title: str,
    gold_answer: str,
    expected_answer_points: Sequence[str],
    ground_truth: Dict[str, Any],
    public_metadata: Dict[str, Any],
    judge: Dict[str, Any],
    evaluation_rubric: Any = None,
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "internal_task_id": internal_task_id,
        "family": family,
        "domain": domain_code,
        "title": title,
        "gold_answer": gold_answer,
        "expected_answer_points": list(expected_answer_points),
        "ground_truth": ground_truth,
        "public_metadata": public_metadata,
        "judge": judge,
        "evaluation_rubric": evaluation_rubric,
    }


def make_trace_row(
    *,
    task_id: str,
    internal_task_id: str,
    family: str,
    domain_code: str,
    ground_truth: Dict[str, Any],
    public_metadata: Dict[str, Any],
    judge: Dict[str, Any],
    quality_signals: Dict[str, Any],
    seed: Dict[str, Any],
    support_context: Dict[str, Any],
    time_context: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "internal_task_id": internal_task_id,
        "family": family,
        "domain": domain_code,
        "ground_truth": ground_truth,
        "judge": judge,
        "public_metadata": public_metadata,
        "quality_signals": quality_signals,
        "rewrite": None,
        "rewrite_leakage_check": None,
        "rewrite_surface_check": None,
        "seed": seed,
        "support_context": support_context,
        "time_context": time_context,
    }


def make_new_venue_task_from_source(source_row: Dict[str, Any], task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    family = str(source_row.get("family") or "")
    domain_code = normalize_domain_code(source_row.get("domain"))
    topic_title = str((source_row.get("public_metadata") or {}).get("topic_title") or (source_row.get("public_metadata") or {}).get("topic") or "this topic")
    gt = json.loads(json.dumps(source_row.get("ground_truth") or {}, ensure_ascii=False))
    support_context = json.loads(json.dumps(source_row.get("support_context") or {}, ensure_ascii=False))
    time_context = json.loads(json.dumps(source_row.get("time_context") or {}, ensure_ascii=False))
    source_judge = json.loads(json.dumps(source_row.get("judge") or {}, ensure_ascii=False))
    source_quality = json.loads(json.dumps(source_row.get("quality_signals") or {}, ensure_ascii=False))

    subtype = "venue_aware_direction_forecast" if family == "direction_forecasting" else "venue_targeted_planning"
    likely_bucket = ""
    likely_venue = ""
    if family == "direction_forecasting":
        stats = gt.get("future_half_stats") or {}
        likely_bucket, likely_venue, top_conf_buckets = likely_bucket_and_venue(stats)
        emergent_descendants = list(gt.get("emergent_descendants") or [])
        predicted_direction = title_case(
            (gt.get("future_terminal") or {}).get("display_name")
            or (((source_row.get("public_metadata") or {}).get("future_themes") or [""])[0])
        )
        direction_records = []
        for row in emergent_descendants[:3]:
            direction_records.append(
                {
                    "display_name": title_case((row or {}).get("display_name") or ""),
                    "future_paper_count": int((row or {}).get("future_paper_count") or 0),
                    "top_conf_count": int((row or {}).get("top_conf_count") or 0),
                    "top_conf_share": float((row or {}).get("top_conf_share") or 0.0),
                    "source_node_id": (row or {}).get("node_id"),
                }
            )
        if not direction_records and predicted_direction:
            direction_records = [
                {
                    "display_name": predicted_direction,
                    "future_paper_count": int(stats.get("paper_count") or 0),
                    "top_conf_count": int(stats.get("top_conf_count") or 0),
                    "top_conf_share": float(stats.get("top_conf_share") or 0.0),
                }
            ]
        gt["direction_records"] = direction_records
        gt["venue_forecast"] = {
            "likely_bucket": likely_bucket,
            "likely_venue": likely_venue,
            "future_top_conf_count": int(stats.get("top_conf_count") or 0),
            "future_top_conf_share": float(stats.get("top_conf_share") or 0.0),
            "top_venue_buckets": top_conf_buckets,
        }
        title = f"Forecasting Top-Venue Traction in {topic_title}"
        question = venue_direction_question(topic_title)
        gold_answer = (
            f"The strongest venue-aware forecast is {predicted_direction}, and the most likely top-tier venue bucket is {likely_bucket}. "
            f"This follows from the pre-cutoff record: the topic already showed enough methodological maturity to support a concrete next step, "
            f"its follow-on trajectory was active rather than flat, and its evaluation or application profile aligned with the style of {likely_bucket}-like venues."
        )
        expected = [
            "Identifies one concrete next-step direction rather than a broad topic area.",
            "Names one likely top-tier venue bucket and links it to the direction with evidence-based reasoning.",
            "Justifies the forecast using pre-cutoff signals such as historical maturity, methodological branching, evaluation emphasis, or venue profile.",
        ]
        public_metadata = {
            **(source_row.get("public_metadata") or {}),
            "task_variant": "venue_aware_research_positioning",
            "future_themes": list((source_row.get("public_metadata") or {}).get("future_themes") or []),
        }
        support_context["venue_forecast"] = gt["venue_forecast"]
    else:
        stats = gt.get("target_window_stats") or {}
        likely_bucket, likely_venue, top_conf_buckets = likely_bucket_and_venue(stats)
        direction_records = list(gt.get("direction_records") or [])
        direction_records.sort(key=lambda row: int((row or {}).get("future_paper_count") or 0), reverse=True)
        top_directions = [
            title_case((row or {}).get("display_name") or "")
            for row in direction_records
            if title_case((row or {}).get("display_name") or "")
        ]
        if not top_directions:
            top_directions = [
                title_case(x) for x in ((source_row.get("public_metadata") or {}).get("future_themes") or []) if title_case(x)
            ]
        top_directions = top_directions[:2]
        rank_block = "; then ".join(top_directions) if len(top_directions) > 1 else (top_directions[0] if top_directions else "the strongest emerging directions")
        gt["target_venue_bucket"] = likely_bucket
        gt["target_venue_name"] = likely_venue
        gt["venue_forecast"] = {
            "likely_bucket": likely_bucket,
            "likely_venue": likely_venue,
            "future_top_conf_count": int(stats.get("top_conf_count") or 0),
            "future_top_conf_share": float(stats.get("top_conf_share") or 0.0),
            "top_venue_buckets": top_conf_buckets,
        }
        title = f"Venue-Targeted Prioritization of Research Directions in {topic_title}"
        question = venue_planning_question(topic_title, likely_bucket)
        gold_answer = (
            f"For a team targeting {likely_bucket}-like venues, the highest-priority directions should be {rank_block}. "
            f"These directions best match the pre-cutoff evidence: they sit closest to the strongest emergent descendants, "
            f"they align with the topic's future-work and evaluation signals, and they are the directions most compatible with the realized {likely_bucket}-weighted venue mix."
        )
        expected = [
            "Produces a ranked research plan rather than an unstructured list.",
            "Targets the venue bucket named in the question and links the ranking to that venue style.",
            "Uses pre-cutoff evidence such as emergent descendants, historical future-work signals, and venue profile to justify the ranking.",
        ]
        public_metadata = {
            **(source_row.get("public_metadata") or {}),
            "task_variant": "venue_aware_research_positioning",
            "future_themes": top_directions or list((source_row.get("public_metadata") or {}).get("future_themes") or []),
        }
        support_context["target_venue_bucket"] = likely_bucket
        support_context["target_venue_name"] = likely_venue

    public_row = {
        "task_id": task_id,
        "family": "venue_aware_research_positioning",
        "subtype": subtype,
        "domain": domain_display(domain_code),
        "horizon": source_row.get("horizon", "half_year"),
        "title": title,
        "question": question,
        "time_cutoff": (time_context or {}).get("history_end") or "2025-08-31",
        "deliverable_spec": public_deliverable_spec("venue_aware_research_positioning", subtype),
    }
    hidden_internal = make_hidden_row(
        task_id=task_id,
        internal_task_id=f"derived::{subtype}::{source_row.get('task_id')}",
        family="venue_aware_research_positioning",
        domain_code=domain_code,
        title=title,
        gold_answer=gold_answer,
        expected_answer_points=expected,
        ground_truth=gt,
        public_metadata=public_metadata,
        judge={
            "decision": "accept",
            "overall_score": judge_score(source_row),
            "source_task_id": source_row.get("task_id"),
            "source_judge": source_judge,
            "derivation_method": subtype,
        },
        evaluation_rubric=source_row.get("evaluation_rubric"),
    )
    trace_row = make_trace_row(
        task_id=task_id,
        internal_task_id=hidden_internal["internal_task_id"],
        family="venue_aware_research_positioning",
        domain_code=domain_code,
        ground_truth=gt,
        public_metadata=public_metadata,
        judge=hidden_internal["judge"],
        quality_signals={
            **source_quality,
            "derivation_method": subtype,
            "derived_from_task_id": source_row.get("task_id"),
            "source_judge_score": judge_score(source_row),
        },
        seed={
            **(source_row.get("seed") or {}),
            "derived_from_task_id": source_row.get("task_id"),
        },
        support_context=support_context,
        time_context=time_context or {"history_end": "2025-08-31", "future_window": "2025-09-01_to_2026-02-28"},
    )
    return public_row, hidden_internal, trace_row, hidden_internal


def planning_score_from_row(row: Dict[str, Any]) -> float:
    stats = (row.get("ground_truth") or {}).get("target_window_stats") or {}
    hist = (row.get("support_context") or {}).get("historical_stats") or {}
    return round(
        float(stats.get("planning_priority_score") or row.get("planning_priority_score") or 0.0)
        + 2.0 * float(stats.get("trend_signal") or row.get("trend_signal") or 0.0)
        + 2.0 * float(stats.get("top_conf_share") or 0.0)
        + min(3.0, float(stats.get("paper_count") or 0) / 30.0)
        + min(2.0, float(hist.get("paper_count") or 0) / 150.0),
        4,
    )


def path_lcp(a: str, b: str) -> int:
    sa = str(a or "").split("/")
    sb = str(b or "").split("/")
    n = 0
    for x, y in zip(sa, sb):
        if x != y:
            break
        n += 1
    return n


def topic_display_name(row: Dict[str, Any]) -> str:
    meta = row.get("public_metadata") or {}
    text = meta.get("topic_title") or meta.get("topic") or row.get("title") or ""
    return title_case(text)


def compact_candidate_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    hist = (row.get("support_context") or {}).get("historical_stats") or {}
    fut = (row.get("ground_truth") or {}).get("target_window_stats") or {}
    return {
        "historical_paper_count": int(hist.get("paper_count") or 0),
        "historical_top_conf_share": float(hist.get("top_conf_share") or 0.0),
        "future_paper_count": int(fut.get("paper_count") or 0),
        "future_top_conf_count": int(fut.get("top_conf_count") or 0),
        "future_top_conf_share": float(fut.get("top_conf_share") or 0.0),
        "planning_score": planning_score_from_row(row),
    }


def build_planning_node_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "domain": normalize_domain_code(row.get("domain")),
        "node_id": str((row.get("seed") or {}).get("node_id") or ""),
        "packet_id": str((row.get("seed") or {}).get("packet_id") or ""),
        "display_name": topic_display_name(row),
        "description": str(((row.get("support_context") or {}).get("node_description")) or ((row.get("public_metadata") or {}).get("topic_title")) or ""),
        "stats": compact_candidate_stats(row),
        "source_row": row,
        "score": planning_score_from_row(row),
    }


def rank_comparative_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    diff = abs(float(a["score"]) - float(b["score"]))
    min_score = min(float(a["score"]), float(b["score"]))
    max_score = max(float(a["score"]), float(b["score"]))
    common = path_lcp(a["node_id"], b["node_id"])
    diff_pref = -abs(diff - 1.25)
    future_sum = float(a["stats"]["future_paper_count"]) + float(b["stats"]["future_paper_count"])
    return (float(common), min_score, diff_pref, max_score, future_sum)


def select_comparative_pairs(
    node_records: List[Dict[str, Any]],
    *,
    min_common_prefix: int,
    min_score_gap: float,
    max_occurrence_per_node: int,
    target_count: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pairs: List[Tuple[Tuple[float, float, float, float, float], Dict[str, Any], Dict[str, Any]]] = []
    for i in range(len(node_records)):
        for j in range(i + 1, len(node_records)):
            a, b = node_records[i], node_records[j]
            common = path_lcp(a["node_id"], b["node_id"])
            if common < min_common_prefix:
                continue
            if abs(float(a["score"]) - float(b["score"])) < min_score_gap:
                continue
            pairs.append((rank_comparative_pair(a, b), a, b))
    pairs.sort(key=lambda item: item[0], reverse=True)
    chosen: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    usage: Counter[str] = Counter()
    seen = set()
    for _, a, b in pairs:
        key = tuple(sorted((a["node_id"], b["node_id"])))
        if key in seen:
            continue
        if usage[a["node_id"]] >= max_occurrence_per_node or usage[b["node_id"]] >= max_occurrence_per_node:
            continue
        chosen.append((a, b))
        seen.add(key)
        usage[a["node_id"]] += 1
        usage[b["node_id"]] += 1
        if len(chosen) >= target_count:
            break
    return chosen


def weighted_share(numerators: Sequence[Tuple[int, float]]) -> float:
    total = sum(int(count or 0) for count, _ in numerators)
    if total <= 0:
        return 0.0
    weighted = sum(int(count or 0) * float(share or 0.0) for count, share in numerators)
    return float(weighted) / float(total)


def make_comparative_task(a: Dict[str, Any], b: Dict[str, Any], task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    winner, loser = (a, b) if float(a["score"]) >= float(b["score"]) else (b, a)
    winner_name = winner["display_name"]
    loser_name = loser["display_name"]
    domain_code = normalize_domain_code(a["domain"])
    domain_label = domain_display(domain_code)
    question = (
        f"Based on scholarly literature available before September 1, 2025, which research direction should be prioritized over the next six months in the {domain_label} domain: "
        f"{a['display_name']} or {b['display_name']}? "
        f"Your answer must justify the choice using only pre-cutoff evidence, including technical maturity, unresolved bottlenecks, emerging momentum, and likely downstream leverage."
    )
    gold_answer = (
        f"The stronger priority is {winner_name}. Relative to {loser_name}, it entered the cutoff with a stronger combination of historical maturity, "
        f"post-cutoff trajectory strength, and research leverage. Historically, {winner_name} had {winner['stats']['historical_paper_count']} papers before the cutoff, compared with "
        f"{loser['stats']['historical_paper_count']} for {loser_name}. In the subsequent six-month window, {winner_name} was associated with "
        f"{winner['stats']['future_paper_count']} papers versus {loser['stats']['future_paper_count']} for {loser_name}. "
        f"A strong answer should therefore prioritize {winner_name} while explicitly explaining the trade-off that makes {loser_name} less strategically attractive under the same cutoff-bound evidence."
    )
    public_metadata = {
        "topic": f"{winner_name} vs {loser_name}",
        "topic_title": f"{winner_name} vs {loser_name}",
        "future_themes": [winner_name, loser_name],
        "task_variant": "comparative_opportunity_prioritization",
    }
    time_context = {
        "history_end": "2025-08-31",
        "future_window": "2025-09-01_to_2026-02-28",
    }
    winner_source = winner["source_row"]
    loser_source = loser["source_row"]
    winner_gt = winner_source.get("ground_truth") or {}
    loser_gt = loser_source.get("ground_truth") or {}
    winner_tw = winner_gt.get("target_window_stats") or {}
    loser_tw = loser_gt.get("target_window_stats") or {}
    direction_records = [
        {
            "display_name": winner_name,
            "future_paper_count": winner["stats"]["future_paper_count"],
            "top_conf_count": winner["stats"]["future_top_conf_count"],
            "top_conf_share": winner["stats"]["future_top_conf_share"],
            "planning_score": float(winner["score"]),
            "source_node_id": winner["node_id"],
            "source_task_id": winner_source.get("task_id"),
            "comparative_role": "winner",
        },
        {
            "display_name": loser_name,
            "future_paper_count": loser["stats"]["future_paper_count"],
            "top_conf_count": loser["stats"]["future_top_conf_count"],
            "top_conf_share": loser["stats"]["future_top_conf_share"],
            "planning_score": float(loser["score"]),
            "source_node_id": loser["node_id"],
            "source_task_id": loser_source.get("task_id"),
            "comparative_role": "loser",
        },
    ]
    ground_truth = {
        "direction_records": direction_records,
        "emergent_descendants": [
            {"display_name": winner_name, "source_node_id": winner["node_id"]},
            {"display_name": loser_name, "source_node_id": loser["node_id"]},
        ],
        "winner_node_id": winner["node_id"],
        "winner_display_name": winner_name,
        "loser_node_id": loser["node_id"],
        "loser_display_name": loser_name,
        "winner_score": float(winner["score"]),
        "loser_score": float(loser["score"]),
        "winner_stats": winner["stats"],
        "loser_stats": loser["stats"],
        "target_window_stats": {
            "paper_count": int(winner["stats"]["future_paper_count"]) + int(loser["stats"]["future_paper_count"]),
            "top_conf_count": int(winner["stats"]["future_top_conf_count"]) + int(loser["stats"]["future_top_conf_count"]),
            "top_conf_share": weighted_share(
                [
                    (int(winner["stats"]["future_paper_count"]), float(winner["stats"]["future_top_conf_share"])),
                    (int(loser["stats"]["future_paper_count"]), float(loser["stats"]["future_top_conf_share"])),
                ]
            ),
            "planning_priority_score": float(winner["score"]),
            "trend_signal": max(float(winner_tw.get("trend_signal") or 0.0), float(loser_tw.get("trend_signal") or 0.0)),
        },
    }
    public_row = {
        "task_id": task_id,
        "family": "strategic_research_planning",
        "subtype": "comparative_opportunity_prioritization",
        "domain": domain_label,
        "horizon": "half_year",
        "title": f"Comparative Prioritization: {winner_name} vs. {loser_name}",
        "question": question,
        "time_cutoff": "2025-08-31",
        "deliverable_spec": public_deliverable_spec("strategic_research_planning", "comparative_opportunity_prioritization"),
    }
    hidden_internal = make_hidden_row(
        task_id=task_id,
        internal_task_id=f"derived::comparative_opportunity_prioritization::{winner_source.get('task_id')}::{loser_source.get('task_id')}",
        family="strategic_research_planning",
        domain_code=domain_code,
        title=public_row["title"],
        gold_answer=gold_answer,
        expected_answer_points=[
            f"Chooses one direction rather than hedging, and makes the comparative priority explicit between {a['display_name']} and {b['display_name']}.",
            "Justifies the choice using pre-cutoff evidence about technical maturity, bottlenecks, research momentum, or venue/impact trajectory.",
            "Explains the trade-off: why the deprioritized option is less strategically attractive in the same time window rather than merely describing the winner in isolation.",
        ],
        ground_truth=ground_truth,
        public_metadata=public_metadata,
        judge={
            "decision": "accept",
            "overall_score": round((judge_score(winner_source) + judge_score(loser_source)) / 2.0, 4),
            "source_task_ids": [winner_source.get("task_id"), loser_source.get("task_id")],
            "derivation_method": "comparative_opportunity_prioritization",
        },
        evaluation_rubric=winner_source.get("evaluation_rubric"),
    )
    trace_row = make_trace_row(
        task_id=task_id,
        internal_task_id=hidden_internal["internal_task_id"],
        family="strategic_research_planning",
        domain_code=domain_code,
        ground_truth=ground_truth,
        public_metadata=public_metadata,
        judge=hidden_internal["judge"],
        quality_signals={
            "derivation_method": "comparative_opportunity_prioritization",
            "pair_strength": abs(float(a["score"]) - float(b["score"])),
            "winner_source_task_id": winner_source.get("task_id"),
            "loser_source_task_id": loser_source.get("task_id"),
        },
        seed={
            "pair_node_ids": [a["node_id"], b["node_id"]],
            "pair_packet_ids": [a["packet_id"], b["packet_id"]],
        },
        support_context={
            "candidate_a": {
                "display_name": a["display_name"],
                "description": a["description"],
                "stats": a["stats"],
                "support_context": a["source_row"].get("support_context") or {},
            },
            "candidate_b": {
                "display_name": b["display_name"],
                "description": b["description"],
                "stats": b["stats"],
                "support_context": b["source_row"].get("support_context") or {},
            },
        },
        time_context=time_context,
    )
    return public_row, hidden_internal, trace_row, hidden_internal


def build_hidden_v3_manifest(release_dir: Path, rows: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    family_counts = Counter()
    domain_counts = Counter()
    for row in rows:
        family_counts[str(row.get("family") or "")] += 1
        domain_counts[str(row.get("domain") or "")] += 1
    return {
        "release_dir": str(release_dir),
        "output": str(output_path),
        "task_count": len(rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "notes": [f"This manifest was regenerated by {SCRIPT_VERSION}."],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand a strict release via evidence-chain reuse.")
    parser.add_argument(
        "--source-release",
        default=str(ROOT / "data" / "releases" / "benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002"),
    )
    parser.add_argument(
        "--output-release",
        default=str(ROOT / "data" / "releases" / "benchmark_full_curated422_plus_supplements106_candidate_llmfuture_vote_v3_qwen8002_evidenceexp_v1"),
    )
    parser.add_argument("--comparative-min-common-prefix", type=int, default=2)
    parser.add_argument("--comparative-min-score-gap", type=float, default=0.05)
    parser.add_argument("--comparative-max-occurrence-per-node", type=int, default=3)
    parser.add_argument("--comparative-max-per-domain", type=int, default=8)
    parser.add_argument("--forecast-venue-max-per-domain", type=int, default=99)
    parser.add_argument("--planning-venue-max-per-domain", type=int, default=99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_release = Path(args.source_release)
    output_release = Path(args.output_release)
    output_release.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads((source_release / "manifest.json").read_text(encoding="utf-8"))
    public_rows, hidden_by_id, trace_by_id, internal_by_id = load_release_rows(source_release)
    merged_rows: List[Dict[str, Any]] = []
    for public_row in public_rows:
        task_id = str(public_row["task_id"])
        merged_rows.append(merge_source_row(public_row, hidden_by_id[task_id], trace_by_id[task_id], internal_by_id[task_id]))

    strict_rows = [row for row in merged_rows if strict_keep(row)]
    existing_ids = {str(row["task_id"]) for row in public_rows}

    next_counters = {"ECV": 1, "ECS": 1}

    def new_task_id(prefix: str) -> str:
        while True:
            task_id = f"RTLv3-{prefix}-{next_counters[prefix]:04d}"
            next_counters[prefix] += 1
            if task_id not in existing_ids:
                existing_ids.add(task_id)
                return task_id

    venue_used_nodes: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in strict_rows:
        if str(row.get("family") or "") != "venue_aware_research_positioning":
            continue
        domain_code = normalize_domain_code(row.get("domain"))
        subtype = str(row.get("subtype") or "")
        node_id = str((row.get("seed") or {}).get("node_id") or "")
        if node_id:
            venue_used_nodes[domain_code][subtype].add(node_id)

    added_public: List[Dict[str, Any]] = []
    added_hidden: List[Dict[str, Any]] = []
    added_trace: List[Dict[str, Any]] = []
    added_internal: List[Dict[str, Any]] = []
    venue_additions: List[Dict[str, Any]] = []
    comparative_additions: List[Dict[str, Any]] = []

    forecast_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    planning_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    planning_node_pool: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for row in strict_rows:
        domain_code = normalize_domain_code(row.get("domain"))
        family = str(row.get("family") or "")
        node_id = str((row.get("seed") or {}).get("node_id") or "")
        if family == "direction_forecasting":
            stats = row.get("ground_truth", {}).get("future_half_stats") or {}
            bucket, _, _ = likely_bucket_and_venue(stats)
            if bucket and int(stats.get("top_conf_count") or 0) > 0 and node_id not in venue_used_nodes[domain_code]["venue_aware_direction_forecast"]:
                forecast_candidates[domain_code].append(row)
        elif family == "strategic_research_planning":
            stats = row.get("ground_truth", {}).get("target_window_stats") or {}
            bucket, _, _ = likely_bucket_and_venue(stats)
            if bucket and int(stats.get("top_conf_count") or 0) > 0 and node_id not in venue_used_nodes[domain_code]["venue_targeted_planning"]:
                planning_candidates[domain_code].append(row)
            if node_id:
                rec = build_planning_node_record(row)
                prev = planning_node_pool[domain_code].get(node_id)
                if prev is None or float(rec["score"]) > float(prev["score"]):
                    planning_node_pool[domain_code][node_id] = rec

    for domain_code in DOMAIN_ORDER:
        pool = sorted(forecast_candidates[domain_code], key=selection_rank, reverse=True)
        chosen_nodes: set[str] = set()
        for row in pool:
            if len(chosen_nodes) >= args.forecast_venue_max_per_domain:
                break
            node_id = str((row.get("seed") or {}).get("node_id") or "")
            if not node_id or node_id in chosen_nodes or node_id in venue_used_nodes[domain_code]["venue_aware_direction_forecast"]:
                continue
            public_row, hidden_row, trace_row, internal_row = make_new_venue_task_from_source(row, new_task_id("ECV"))
            added_public.append(public_row)
            added_hidden.append(hidden_row)
            added_trace.append(trace_row)
            added_internal.append(internal_row)
            venue_additions.append(public_row)
            chosen_nodes.add(node_id)

    for domain_code in DOMAIN_ORDER:
        pool = sorted(planning_candidates[domain_code], key=selection_rank, reverse=True)
        chosen_nodes: set[str] = set()
        for row in pool:
            if len(chosen_nodes) >= args.planning_venue_max_per_domain:
                break
            node_id = str((row.get("seed") or {}).get("node_id") or "")
            if not node_id or node_id in chosen_nodes or node_id in venue_used_nodes[domain_code]["venue_targeted_planning"]:
                continue
            public_row, hidden_row, trace_row, internal_row = make_new_venue_task_from_source(row, new_task_id("ECV"))
            added_public.append(public_row)
            added_hidden.append(hidden_row)
            added_trace.append(trace_row)
            added_internal.append(internal_row)
            venue_additions.append(public_row)
            chosen_nodes.add(node_id)

    for domain_code in DOMAIN_ORDER:
        records = list(planning_node_pool[domain_code].values())
        chosen_pairs = select_comparative_pairs(
            records,
            min_common_prefix=args.comparative_min_common_prefix,
            min_score_gap=args.comparative_min_score_gap,
            max_occurrence_per_node=args.comparative_max_occurrence_per_node,
            target_count=args.comparative_max_per_domain,
        )
        for a, b in chosen_pairs:
            public_row, hidden_row, trace_row, internal_row = make_comparative_task(a, b, new_task_id("ECS"))
            added_public.append(public_row)
            added_hidden.append(hidden_row)
            added_trace.append(trace_row)
            added_internal.append(internal_row)
            comparative_additions.append(public_row)

    merged_public = public_rows + added_public
    merged_hidden = [hidden_by_id[row["task_id"]] for row in public_rows] + added_hidden
    merged_trace = [trace_by_id[row["task_id"]] for row in public_rows] + added_trace
    merged_internal = [internal_by_id[row["task_id"]] for row in public_rows] + added_internal

    hidden_v3_rows = [build_hidden_eval_v3_row(hidden, trace) for hidden, trace in zip(merged_hidden, merged_trace)]
    hidden_v31_rows = [build_hidden_eval_v3_1_row(hidden_v3, trace) for hidden_v3, trace in zip(hidden_v3_rows, merged_trace)]

    strict_ids = [row["task_id"] for row in merged_hidden if strict_keep(row)]
    strict_family_counts = Counter(row["family"] for row in merged_hidden if strict_keep(row))
    family_counts = Counter(row["family"] for row in merged_public)
    domain_counts = Counter(row["domain"] for row in merged_public)
    subtype_counts = Counter((row["family"], row.get("subtype") or "") for row in merged_public)

    dump_jsonl(output_release / "tasks.jsonl", merged_public)
    dump_jsonl(output_release / "tasks_hidden_eval.jsonl", merged_hidden)
    dump_jsonl(output_release / "tasks_build_trace.jsonl", merged_trace)
    dump_jsonl(output_release / "tasks_internal_full.jsonl", merged_internal)
    dump_jsonl(output_release / "tasks_hidden_eval_v3.jsonl", hidden_v3_rows)
    dump_jsonl(output_release / "tasks_hidden_eval_v3_1.jsonl", hidden_v31_rows)
    dump_json(output_release / "tasks_hidden_eval_v3_manifest.json", build_hidden_v3_manifest(source_release, hidden_v3_rows, output_release / "tasks_hidden_eval_v3.jsonl"))
    dump_json(output_release / "tasks_hidden_eval_v3_1_manifest.json", build_hidden_v3_manifest(source_release, hidden_v31_rows, output_release / "tasks_hidden_eval_v3_1.jsonl"))
    (output_release / "task_ids.txt").write_text("\n".join(row["task_id"] for row in merged_public) + "\n", encoding="utf-8")
    (output_release / "strict_task_ids.txt").write_text("\n".join(strict_ids) + "\n", encoding="utf-8")
    dump_jsonl(output_release / "added_tasks_public.jsonl", added_public)
    dump_jsonl(output_release / "added_tasks_hidden_eval.jsonl", added_hidden)
    dump_jsonl(output_release / "added_tasks_build_trace.jsonl", added_trace)
    dump_jsonl(output_release / "added_tasks_internal_full.jsonl", added_internal)

    strict_summary = {
        "strict_task_count": len(strict_ids),
        "strict_family_counts": dict(strict_family_counts),
        "base_strict_task_count": int(source_manifest.get("strict_task_count") or 0),
        "strict_added_task_count": len(strict_ids) - int(source_manifest.get("strict_task_count") or 0),
        "notes": [
            "Strict counting uses metrics-corresponding GT fields only.",
            "Comparative strategic additions include direction_records so they survive strict filtering.",
        ],
    }
    dump_json(output_release / "strict_summary.json", strict_summary)

    manifest = {
        "release_name": output_release.name,
        "source_release": str(source_release),
        "script": str(Path(__file__).resolve()),
        "script_version": SCRIPT_VERSION,
        "task_count": len(merged_public),
        "base_task_count": len(public_rows),
        "added_task_count": len(added_public),
        "strict_task_count": len(strict_ids),
        "base_strict_task_count": int(source_manifest.get("strict_task_count") or 0),
        "strict_added_task_count": len(strict_ids) - int(source_manifest.get("strict_task_count") or 0),
        "family_counts": dict(family_counts),
        "strict_family_counts": dict(strict_family_counts),
        "domain_counts": dict(domain_counts),
        "subtype_counts": {f"{family}::{subtype}": count for (family, subtype), count in sorted(subtype_counts.items())},
        "future_novelty_postprocess": source_manifest.get("future_novelty_postprocess"),
        "evidence_chain_expansion": {
            "comparative_min_common_prefix": args.comparative_min_common_prefix,
            "comparative_min_score_gap": args.comparative_min_score_gap,
            "comparative_max_occurrence_per_node": args.comparative_max_occurrence_per_node,
            "forecast_venue_additions": len([row for row in venue_additions if row.get("subtype") == "venue_aware_direction_forecast"]),
            "planning_venue_additions": len([row for row in venue_additions if row.get("subtype") == "venue_targeted_planning"]),
            "comparative_additions": len(comparative_additions),
        },
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "tasks_hidden_eval_v3": "tasks_hidden_eval_v3.jsonl",
            "tasks_hidden_eval_v3_1": "tasks_hidden_eval_v3_1.jsonl",
            "task_ids": "task_ids.txt",
            "strict_task_ids": "strict_task_ids.txt",
            "strict_summary": "strict_summary.json",
            "added_tasks_public": "added_tasks_public.jsonl",
        },
        "notes": [
            "This release expands the strict source via evidence-chain reuse only; it does not run new seed-packet generation.",
            "Venue additions are derived from strict forecasting/planning rows with usable venue-bucket evidence.",
            "Comparative strategic additions are built from strict planning rows and remain strict-compatible via direction_records.",
        ],
    }
    dump_json(output_release / "manifest.json", manifest)

    for name in ("kb", "future_kb"):
        maybe_copy_or_link(source_release, output_release, name)

    summary = {
        "source_release": str(source_release),
        "output_release": str(output_release),
        "base_task_count": len(public_rows),
        "added_task_count": len(added_public),
        "final_task_count": len(merged_public),
        "base_strict_task_count": int(source_manifest.get("strict_task_count") or 0),
        "final_strict_task_count": len(strict_ids),
        "strict_added_task_count": len(strict_ids) - int(source_manifest.get("strict_task_count") or 0),
        "venue_additions": Counter(row["subtype"] for row in venue_additions),
        "comparative_additions": len(comparative_additions),
        "strict_family_counts": dict(strict_family_counts),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
