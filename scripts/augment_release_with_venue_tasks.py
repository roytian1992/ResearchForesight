from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.eval_v3 import build_hidden_eval_v3_row


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_label(text: Any) -> str:
    return str(text or "").replace("_", " ").strip()


def title_case(text: Any) -> str:
    raw = normalize_label(text)
    return raw[:1].upper() + raw[1:] if raw else ""


def likely_bucket_and_venue(stats: Dict[str, Any]) -> Tuple[str, str, Dict[str, int]]:
    top_buckets = dict(stats.get("top_venue_buckets") or {})
    conf_buckets = {str(k): int(v) for k, v in top_buckets.items() if str(k) not in {"other", "unknown"} and int(v or 0) > 0}
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


def public_deliverable_spec(family: str) -> Dict[str, Any]:
    base = {
        "format": "free_form_research_analysis",
        "requirements": [
            "Use only evidence available up to the cutoff unless the benchmark setting explicitly provides an offline knowledge base.",
            "State a concrete conclusion rather than vague trend language.",
            "Support the conclusion with literature-based reasoning.",
        ],
    }
    if family == "direction_forecasting":
        base["requirements"] += [
            "Name one specific next-step direction and characterize the trajectory.",
            "Identify one likely top-tier venue bucket for that direction.",
        ]
    elif family == "strategic_research_planning":
        base["requirements"] += [
            "Select and justify a small ranked set of priority directions.",
            "Make the ranking explicit for the target venue bucket named in the question.",
        ]
    return base


def direction_title(topic_title: str) -> str:
    return f"Forecasting Top-Venue Traction in {topic_title}"


def direction_question(topic_title: str, history_end: str) -> str:
    return (
        f"Based on scholarly literature available before September 1, 2025, identify one concrete next-step direction within {topic_title} "
        f"that is most likely to gain traction in top-tier AI venues during the subsequent six-month period. Also identify the most likely venue bucket "
        f"(for example AAAI-like, EMNLP-like, ICLR-like, or similar top-tier venues) where that traction would appear. "
        f"Your answer must be justified only with pre-cutoff evidence and should not rely on post-{history_end} developments."
    )


def planning_title(topic_title: str) -> str:
    return f"Venue-Targeted Prioritization of Research Directions in {topic_title}"


def planning_question(topic_title: str, bucket: str) -> str:
    bucket_label = f"{bucket}-like" if bucket else "top-tier"
    return (
        f"Suppose a research team wants to maximize its relevance for {bucket_label} venues in the next submission cycle. "
        f"Based only on literature available before September 1, 2025, which one or two next-step directions in {topic_title} should be prioritized, "
        f"and what evidence-based rationale supports that ranking?"
    )


def build_direction_variant(
    *,
    source_public: Dict[str, Any],
    source_hidden: Dict[str, Any],
    source_trace: Dict[str, Any],
    source_internal: Dict[str, Any],
    internal_task_id: str,
    public_task_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    gt = json.loads(json.dumps(source_hidden.get("ground_truth") or {}, ensure_ascii=False))
    trace_gt = json.loads(json.dumps(source_trace.get("ground_truth") or {}, ensure_ascii=False))
    future_stats = gt.get("future_half_stats") or {}
    likely_bucket, likely_venue, top_conf_buckets = likely_bucket_and_venue(future_stats)
    topic_title = str((source_hidden.get("public_metadata") or {}).get("topic_title") or (source_hidden.get("public_metadata") or {}).get("topic") or source_trace.get("public_metadata", {}).get("topic_title") or "")
    future_terminal = gt.get("future_terminal") or {}
    trajectory = (gt.get("trajectory") or {}).get("trajectory_label") or "fragmenting"
    predicted_direction = title_case(future_terminal.get("display_name") or ((source_hidden.get("public_metadata") or {}).get("future_themes") or [""])[0])

    venue_forecast = {
        "likely_bucket": likely_bucket,
        "likely_venue": likely_venue,
        "future_top_conf_count": int(future_stats.get("top_conf_count") or 0),
        "future_top_conf_share": float(future_stats.get("top_conf_share") or 0.0),
        "top_venue_buckets": top_conf_buckets,
    }
    gt["venue_forecast"] = venue_forecast
    trace_gt["venue_forecast"] = venue_forecast

    question = direction_question(topic_title=topic_title, history_end=str(source_public.get("time_cutoff") or "2025-08-31"))
    gold_answer = (
        f"The strongest venue-aware forecast is {predicted_direction}, and the most likely top-tier venue bucket is {likely_bucket}. "
        f"This forecast is supported by the pre-cutoff record: the area had already begun to branch into more specialized subdirections, "
        f"its historical literature showed enough technical maturity to sustain follow-on work, and the most plausible continuation aligned with the evaluation and application style typically favored in {likely_bucket}-like venues. "
        f"The realized post-cutoff window later concentrated {int(future_stats.get('top_conf_count') or 0)} top-tier papers in this venue bucket structure, which validates that forecast."
    )
    expected_points = [
        "Identifies one concrete next-step direction rather than a broad topic area.",
        "Names one likely top-tier venue bucket and links it to the direction with evidence-based reasoning.",
        "Justifies the forecast using pre-cutoff signals such as historical maturity, methodological branching, evaluation emphasis, or venue/citation profile.",
    ]

    hidden = {
        "task_id": public_task_id,
        "internal_task_id": internal_task_id,
        "family": "direction_forecasting",
        "domain": source_hidden.get("domain"),
        "title": direction_title(topic_title),
        "gold_answer": gold_answer,
        "expected_answer_points": expected_points,
        "evaluation_rubric": source_hidden.get("evaluation_rubric"),
        "judge": source_hidden.get("judge"),
        "ground_truth": gt,
        "public_metadata": {
            **(source_hidden.get("public_metadata") or {}),
            "task_variant": "venue_aware",
        },
    }
    trace = {
        "task_id": public_task_id,
        "internal_task_id": internal_task_id,
        "family": "direction_forecasting",
        "domain": source_trace.get("domain"),
        "seed": source_trace.get("seed"),
        "time_context": source_trace.get("time_context"),
        "support_context": {
            **(source_trace.get("support_context") or {}),
            "venue_forecast": venue_forecast,
        },
        "ground_truth": trace_gt,
        "quality_signals": {
            **(source_trace.get("quality_signals") or {}),
            "derived_variant": "venue_aware",
        },
        "rewrite": source_trace.get("rewrite"),
        "rewrite_leakage_check": source_trace.get("rewrite_leakage_check"),
        "rewrite_surface_check": source_trace.get("rewrite_surface_check"),
        "judge": source_trace.get("judge"),
        "public_metadata": hidden["public_metadata"],
    }
    public = {
        "task_id": public_task_id,
        "family": "direction_forecasting",
        "subtype": "venue_aware_direction_forecast",
        "domain": source_public.get("domain"),
        "horizon": source_public.get("horizon"),
        "title": hidden["title"],
        "question": question,
        "time_cutoff": source_public.get("time_cutoff"),
        "deliverable_spec": public_deliverable_spec("direction_forecasting"),
    }
    internal = {
        **source_internal,
        "task_id": internal_task_id,
        "subtype": "venue_aware_direction_forecast",
        "title": hidden["title"],
        "question": question,
        "draft_question": question,
        "gold_answer": gold_answer,
        "draft_reference_answer": gold_answer,
        "expected_answer_points": expected_points,
        "ground_truth": gt,
        "public_metadata": hidden["public_metadata"],
        "support_context": trace["support_context"],
        "quality_signals": trace["quality_signals"],
        "derived_from": source_hidden.get("task_id"),
    }
    return public, hidden, trace, internal


def build_planning_variant(
    *,
    source_public: Dict[str, Any],
    source_hidden: Dict[str, Any],
    source_trace: Dict[str, Any],
    source_internal: Dict[str, Any],
    internal_task_id: str,
    public_task_id: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    gt = json.loads(json.dumps(source_hidden.get("ground_truth") or {}, ensure_ascii=False))
    trace_gt = json.loads(json.dumps(source_trace.get("ground_truth") or {}, ensure_ascii=False))
    target_stats = gt.get("target_window_stats") or {}
    likely_bucket, likely_venue, top_conf_buckets = likely_bucket_and_venue(target_stats)
    topic_title = str((source_hidden.get("public_metadata") or {}).get("topic_title") or (source_hidden.get("public_metadata") or {}).get("topic") or "")
    direction_records = list(gt.get("direction_records") or [])
    direction_records.sort(key=lambda row: int((row or {}).get("future_paper_count") or 0), reverse=True)
    top_directions = [title_case((row or {}).get("display_name") or "") for row in direction_records if title_case((row or {}).get("display_name") or "")]
    if not top_directions:
        top_directions = [title_case(x) for x in ((source_hidden.get("public_metadata") or {}).get("future_themes") or []) if title_case(x)]
    top_directions = top_directions[:2]
    rank_block = "; then ".join(top_directions) if len(top_directions) > 1 else (top_directions[0] if top_directions else "the strongest emerging directions")

    gt["target_venue_bucket"] = likely_bucket
    gt["target_venue_name"] = likely_venue
    gt["venue_forecast"] = {
        "likely_bucket": likely_bucket,
        "likely_venue": likely_venue,
        "future_top_conf_count": int(target_stats.get("top_conf_count") or 0),
        "future_top_conf_share": float(target_stats.get("top_conf_share") or 0.0),
        "top_venue_buckets": top_conf_buckets,
    }
    trace_gt.update({k: gt[k] for k in ["target_venue_bucket", "target_venue_name", "venue_forecast"]})

    question = planning_question(topic_title=topic_title, bucket=likely_bucket)
    gold_answer = (
        f"For a team targeting {likely_bucket}-like venues, the highest-priority directions should be {rank_block}. "
        f"These directions best match the historical evidence: they sit closest to the strongest emergent descendants, they align with the topic's pre-cutoff future-work and evaluation signals, "
        f"and they are the directions that later showed the clearest concentration in the realized {likely_bucket}-weighted top-tier venue mix. "
        f"In the realized target window, this topic produced {int(target_stats.get('top_conf_count') or 0)} top-tier papers with a top-venue share of {float(target_stats.get('top_conf_share') or 0.0):.4f}."
    )
    expected_points = [
        "Produces a ranked research plan rather than an unstructured list.",
        "Targets the venue bucket named in the question and links the ranking to that venue style.",
        "Uses pre-cutoff evidence such as emergent descendants, historical future-work signals, and venue/citation profile to justify the ranking.",
    ]

    hidden = {
        "task_id": public_task_id,
        "internal_task_id": internal_task_id,
        "family": "strategic_research_planning",
        "domain": source_hidden.get("domain"),
        "title": planning_title(topic_title),
        "gold_answer": gold_answer,
        "expected_answer_points": expected_points,
        "evaluation_rubric": source_hidden.get("evaluation_rubric"),
        "judge": source_hidden.get("judge"),
        "ground_truth": gt,
        "public_metadata": {
            **(source_hidden.get("public_metadata") or {}),
            "task_variant": "venue_targeted",
        },
    }
    trace = {
        "task_id": public_task_id,
        "internal_task_id": internal_task_id,
        "family": "strategic_research_planning",
        "domain": source_trace.get("domain"),
        "seed": source_trace.get("seed"),
        "time_context": source_trace.get("time_context"),
        "support_context": {
            **(source_trace.get("support_context") or {}),
            "target_venue_bucket": likely_bucket,
            "target_venue_name": likely_venue,
        },
        "ground_truth": trace_gt,
        "quality_signals": {
            **(source_trace.get("quality_signals") or {}),
            "derived_variant": "venue_targeted",
        },
        "rewrite": source_trace.get("rewrite"),
        "rewrite_leakage_check": source_trace.get("rewrite_leakage_check"),
        "rewrite_surface_check": source_trace.get("rewrite_surface_check"),
        "judge": source_trace.get("judge"),
        "public_metadata": hidden["public_metadata"],
    }
    public = {
        "task_id": public_task_id,
        "family": "strategic_research_planning",
        "subtype": "venue_targeted_planning",
        "domain": source_public.get("domain"),
        "horizon": source_public.get("horizon"),
        "title": hidden["title"],
        "question": question,
        "time_cutoff": source_public.get("time_cutoff"),
        "deliverable_spec": public_deliverable_spec("strategic_research_planning"),
    }
    internal = {
        **source_internal,
        "task_id": internal_task_id,
        "subtype": "venue_targeted_planning",
        "title": hidden["title"],
        "question": question,
        "draft_question": question,
        "gold_answer": gold_answer,
        "draft_reference_answer": gold_answer,
        "expected_answer_points": expected_points,
        "ground_truth": gt,
        "public_metadata": hidden["public_metadata"],
        "support_context": trace["support_context"],
        "quality_signals": trace["quality_signals"],
        "derived_from": source_hidden.get("task_id"),
    }
    return public, hidden, trace, internal


def next_public_task_id(rows: List[Dict[str, Any]]) -> int:
    mx = 0
    for row in rows:
        text = str(row.get("task_id") or "")
        if text.startswith("RTLv3-"):
            try:
                mx = max(mx, int(text.split("-")[-1]))
            except Exception:
                pass
    return mx + 1


def select_direction_sources(hidden_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    for row in hidden_rows:
        if row.get("family") != "direction_forecasting":
            continue
        gt = row.get("ground_truth") or {}
        stats = gt.get("future_half_stats") or {}
        likely_bucket, _, conf_buckets = likely_bucket_and_venue(stats)
        if int(stats.get("top_conf_count") or 0) <= 0 or not likely_bucket:
            continue
        candidates.append((row, int(stats.get("top_conf_count") or 0), float(stats.get("top_conf_share") or 0.0), sum(conf_buckets.values())))
    by_domain: Dict[str, List[Tuple[Dict[str, Any], int, float, int]]] = defaultdict(list)
    for item in candidates:
        by_domain[str(item[0].get("domain"))].append(item)
    selected: List[Dict[str, Any]] = []
    for domain, items in by_domain.items():
        items.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        seen_topics = set()
        count = 0
        for row, *_ in items:
            topic = str((row.get("public_metadata") or {}).get("topic_title") or (row.get("public_metadata") or {}).get("topic") or "")
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            selected.append(row)
            count += 1
            if count >= 2:
                break
    selected.sort(key=lambda row: str(row.get("task_id")))
    return selected


def select_planning_sources(hidden_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    for row in hidden_rows:
        if row.get("family") != "strategic_research_planning":
            continue
        gt = row.get("ground_truth") or {}
        stats = gt.get("target_window_stats") or {}
        likely_bucket, _, conf_buckets = likely_bucket_and_venue(stats)
        if int(stats.get("top_conf_count") or 0) <= 0 or not likely_bucket:
            continue
        candidates.append((row, int(stats.get("top_conf_count") or 0), float(stats.get("top_conf_share") or 0.0), sum(conf_buckets.values())))
    by_domain: Dict[str, List[Tuple[Dict[str, Any], int, float, int]]] = defaultdict(list)
    for item in candidates:
        by_domain[str(item[0].get("domain"))].append(item)
    selected: List[Dict[str, Any]] = []
    for domain, items in by_domain.items():
        items.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        seen_topics = set()
        count = 0
        for row, *_ in items:
            topic = str((row.get("public_metadata") or {}).get("topic_title") or (row.get("public_metadata") or {}).get("topic") or "")
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            selected.append(row)
            count += 1
            if count >= 2:
                break
    selected.sort(key=lambda row: str(row.get("task_id")))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment a release bundle with venue-aware direction/planning tasks.")
    parser.add_argument("--src-release", required=True)
    parser.add_argument("--out-release", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src_release)
    out = Path(args.out_release)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)

    public_rows = list(iter_jsonl(src / "tasks.jsonl"))
    hidden_rows = list(iter_jsonl(src / "tasks_hidden_eval.jsonl"))
    trace_rows = list(iter_jsonl(src / "tasks_build_trace.jsonl"))
    internal_rows = list(iter_jsonl(src / "tasks_internal_full.jsonl"))

    public_by_id = {row["task_id"]: row for row in public_rows}
    trace_by_id = {row["task_id"]: row for row in trace_rows}
    internal_by_internal_id = {row["task_id"]: row for row in internal_rows}

    next_id = next_public_task_id(public_rows)
    added_public: List[Dict[str, Any]] = []
    added_hidden: List[Dict[str, Any]] = []
    added_trace: List[Dict[str, Any]] = []
    added_internal: List[Dict[str, Any]] = []

    for source_hidden in select_direction_sources(hidden_rows):
        public_task_id = f"RTLv3-{next_id:04d}"
        next_id += 1
        source_public = public_by_id[source_hidden["task_id"]]
        source_trace = trace_by_id[source_hidden["task_id"]]
        source_internal = internal_by_internal_id[source_hidden["internal_task_id"]]
        internal_task_id = f"venue_direction::{source_hidden['internal_task_id']}"
        public, hidden, trace, internal = build_direction_variant(
            source_public=source_public,
            source_hidden=source_hidden,
            source_trace=source_trace,
            source_internal=source_internal,
            internal_task_id=internal_task_id,
            public_task_id=public_task_id,
        )
        added_public.append(public)
        added_hidden.append(hidden)
        added_trace.append(trace)
        added_internal.append(internal)

    for source_hidden in select_planning_sources(hidden_rows):
        public_task_id = f"RTLv3-{next_id:04d}"
        next_id += 1
        source_public = public_by_id[source_hidden["task_id"]]
        source_trace = trace_by_id[source_hidden["task_id"]]
        source_internal = internal_by_internal_id[source_hidden["internal_task_id"]]
        internal_task_id = f"venue_planning::{source_hidden['internal_task_id']}"
        public, hidden, trace, internal = build_planning_variant(
            source_public=source_public,
            source_hidden=source_hidden,
            source_trace=source_trace,
            source_internal=source_internal,
            internal_task_id=internal_task_id,
            public_task_id=public_task_id,
        )
        added_public.append(public)
        added_hidden.append(hidden)
        added_trace.append(trace)
        added_internal.append(internal)

    all_public = public_rows + added_public
    all_hidden = hidden_rows + added_hidden
    all_trace = trace_rows + added_trace
    all_internal = internal_rows + added_internal

    dump_jsonl(out / "tasks.jsonl", all_public)
    dump_jsonl(out / "tasks_hidden_eval.jsonl", all_hidden)
    dump_jsonl(out / "tasks_build_trace.jsonl", all_trace)
    dump_jsonl(out / "tasks_internal_full.jsonl", all_internal)

    hidden_v3_rows = [build_hidden_eval_v3_row(hidden, {row["task_id"]: row for row in all_trace}[hidden["task_id"]]) for hidden in all_hidden]
    dump_jsonl(out / "tasks_hidden_eval_v3.jsonl", hidden_v3_rows)
    v3_manifest = {
        "release_dir": str(out),
        "output": str(out / "tasks_hidden_eval_v3.jsonl"),
        "task_count": len(hidden_v3_rows),
        "notes": [
            "Augmented with venue-aware direction-forecasting and venue-targeted planning variants.",
        ],
    }
    dump_json(out / "tasks_hidden_eval_v3_manifest.json", v3_manifest)

    family_counts = Counter(row["family"] for row in all_public)
    domain_counts = Counter()
    for row in all_public:
        domain_counts[str(row.get("domain") or "")] += 1
    manifest = {
        "release_name": out.name,
        "task_count": len(all_public),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
            "tasks_hidden_eval_v3": "tasks_hidden_eval_v3.jsonl",
        },
        "augmentation": {
            "source_release": str(src),
            "added_direction_tasks": len([row for row in added_public if row["family"] == "direction_forecasting"]),
            "added_planning_tasks": len([row for row in added_public if row["family"] == "strategic_research_planning"]),
        },
    }
    dump_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
