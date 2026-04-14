from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "releases" / "benchmark_full"
DEFAULT_OUTPUT = ROOT / "data" / "releases" / "benchmark_full_curated"

STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs.",
    "via",
    "with",
}

SPECIAL_TOKENS = {
    "llm": "LLM",
    "llms": "LLMs",
    "rag": "RAG",
    "rl": "RL",
    "sql": "SQL",
    "nlp": "NLP",
    "cot": "CoT",
    "qa": "QA",
    "3d": "3D",
    "2d": "2D",
    "text-to-sql": "Text-to-SQL",
    "zero-shot": "Zero-Shot",
    "few-shot": "Few-Shot",
    "multi-turn": "Multi-Turn",
    "tool-augmented": "Tool-Augmented",
    "domain-specific": "Domain-Specific",
    "long-term": "Long-Term",
}

CANONICAL_VENUE_BY_BUCKET = {
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "ACL": "Annual Meeting of the Association for Computational Linguistics",
    "CVPR": "Computer Vision and Pattern Recognition",
    "ECCV": "European Conference on Computer Vision",
    "EMNLP": "Conference on Empirical Methods in Natural Language Processing",
    "ICCV": "International Conference on Computer Vision",
    "ICLR": "International Conference on Learning Representations",
    "ICML": "International Conference on Machine Learning",
    "NAACL": "North American Chapter of the Association for Computational Linguistics",
    "NeurIPS": "Neural Information Processing Systems",
    "SIGIR": "International ACM SIGIR Conference on Research and Development in Information Retrieval",
    "WSDM": "Web Search and Data Mining",
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


def normalize_space(text: str) -> str:
    return " ".join(str(text or "").split())


def smart_title_case(text: str) -> str:
    words = normalize_space(text).replace("_", " ").split()
    out: List[str] = []
    for idx, word in enumerate(words):
        lower = word.lower()
        if lower in SPECIAL_TOKENS:
            out.append(SPECIAL_TOKENS[lower])
            continue
        if "-" in word:
            parts = []
            for part in word.split("-"):
                lower_part = part.lower()
                if lower_part in SPECIAL_TOKENS:
                    parts.append(SPECIAL_TOKENS[lower_part])
                elif lower_part in STOPWORDS and idx > 0:
                    parts.append(lower_part)
                else:
                    parts.append(lower_part[:1].upper() + lower_part[1:])
            out.append("-".join(parts))
            continue
        if lower in STOPWORDS and idx > 0:
            out.append(lower)
        else:
            out.append(lower[:1].upper() + lower[1:])
    return " ".join(out)


def trace_topic(trace_row: Dict[str, Any]) -> str:
    meta = trace_row.get("public_metadata") or {}
    topic = str(meta.get("topic_title") or meta.get("topic") or "").strip()
    return smart_title_case(topic) if topic else ""


def normalize_public_title(public_row: Dict[str, Any], trace_row: Dict[str, Any]) -> str:
    family = str(public_row.get("family") or "")
    subtype = str(public_row.get("subtype") or "")
    topic = trace_topic(trace_row)
    if family == "strategic_research_planning" and subtype == "agenda_priority_selection" and topic:
        return f"Planning Near-Term Research Directions in {topic}"
    if family == "venue_aware_research_positioning" and subtype == "venue_aware_direction_forecast" and topic:
        return f"Forecasting the Next Research Direction in {topic} and Its Likely Venue Fit"
    if family == "venue_aware_research_positioning" and subtype == "venue_targeted_planning":
        return public_row.get("title") or ""
    if family == "strategic_research_planning" and subtype == "comparative_opportunity_prioritization":
        text = str(public_row.get("title") or "")
        if text.startswith("Comparative Prioritization: "):
            lhs_rhs = text[len("Comparative Prioritization: ") :]
            parts = lhs_rhs.split(" vs. ")
            if len(parts) == 2:
                return f"Comparative Prioritization: {smart_title_case(parts[0])} vs. {smart_title_case(parts[1])}"
    return smart_title_case(str(public_row.get("title") or ""))


def canonical_venue_name(bucket: str, fallback: str) -> str:
    bucket = str(bucket or "").strip()
    if bucket in CANONICAL_VENUE_BY_BUCKET:
        return CANONICAL_VENUE_BY_BUCKET[bucket]
    return str(fallback or "")


def should_drop(public_row: Dict[str, Any], trace_row: Dict[str, Any]) -> Tuple[bool, str]:
    family = str(public_row.get("family") or "")
    subtype = str(public_row.get("subtype") or "")
    support = trace_row.get("support_context") or {}
    if family == "strategic_research_planning" and subtype == "q1_agenda_priority_selection":
        return True, "q1_agenda_priority_selection_removed_for_structural_inconsistency"
    if family == "strategic_research_planning" and subtype == "agenda_priority_selection":
        if not (support.get("candidate_directions") or []):
            return True, "agenda_priority_selection_without_public_candidates"
    return False, ""


def build_curated_release(source_dir: Path, output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    public_rows = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks.jsonl")}
    hidden_rows = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_hidden_eval.jsonl")}
    trace_rows = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_build_trace.jsonl")}
    internal_rows = {row["task_id"]: row for row in iter_jsonl(source_dir / "tasks_internal_full.jsonl")}

    dropped: List[Dict[str, str]] = []
    kept_ids: List[str] = []
    for task_id, public_row in public_rows.items():
        trace_row = trace_rows.get(task_id) or {}
        drop, reason = should_drop(public_row, trace_row)
        if drop:
            dropped.append({"task_id": task_id, "reason": reason})
        else:
            kept_ids.append(task_id)

    kept_ids = sorted(kept_ids)
    curated_public: List[Dict[str, Any]] = []
    curated_hidden: List[Dict[str, Any]] = []
    curated_trace: List[Dict[str, Any]] = []
    curated_internal: List[Dict[str, Any]] = []
    venue_repairs: List[Dict[str, str]] = []

    for task_id in kept_ids:
        public_row = dict(public_rows[task_id])
        hidden_row = dict(hidden_rows[task_id])
        trace_row = dict(trace_rows[task_id])
        internal_row = dict(internal_rows.get(task_id) or {})

        title = normalize_public_title(public_row, trace_row)
        if title:
            public_row["title"] = title
            hidden_row["title"] = title
            if internal_row:
                internal_row["title"] = title

        if public_row.get("family") == "venue_aware_research_positioning":
            gt = dict(hidden_row.get("ground_truth") or {})
            trace_gt = dict(trace_row.get("ground_truth") or {})
            support = dict(trace_row.get("support_context") or {})

            venue_forecast = dict(gt.get("venue_forecast") or trace_gt.get("venue_forecast") or support.get("venue_forecast") or {})
            bucket = str(gt.get("target_venue_bucket") or trace_gt.get("target_venue_bucket") or venue_forecast.get("likely_bucket") or "")
            if bucket:
                fixed_name = canonical_venue_name(bucket, str(gt.get("target_venue_name") or trace_gt.get("target_venue_name") or venue_forecast.get("likely_venue") or ""))
                if venue_forecast:
                    old_name = str(venue_forecast.get("likely_venue") or "")
                    venue_forecast["likely_venue"] = fixed_name
                    if old_name != fixed_name:
                        venue_repairs.append({"task_id": task_id, "bucket": bucket, "old": old_name, "new": fixed_name})
                if public_row.get("subtype") == "venue_targeted_planning":
                    gt["target_venue_bucket"] = bucket
                    gt["target_venue_name"] = fixed_name
                    trace_gt["target_venue_bucket"] = bucket
                    trace_gt["target_venue_name"] = fixed_name
                    support["target_venue_bucket"] = bucket
                    support["target_venue_name"] = fixed_name
                if venue_forecast:
                    gt["venue_forecast"] = venue_forecast
                    trace_gt["venue_forecast"] = venue_forecast
                    support["venue_forecast"] = venue_forecast

            hidden_row["ground_truth"] = gt
            trace_row["ground_truth"] = trace_gt
            trace_row["support_context"] = support
            if internal_row:
                internal_gt = dict(internal_row.get("ground_truth") or {})
                if gt:
                    internal_gt.update(gt)
                internal_row["ground_truth"] = internal_gt
                internal_support = dict(internal_row.get("support_context") or {})
                if support:
                    internal_support.update(support)
                internal_row["support_context"] = internal_support

        if not internal_row:
            internal_row = {
                "task_id": task_id,
                "family": public_row.get("family"),
                "subtype": public_row.get("subtype"),
                "domain": public_row.get("domain"),
                "horizon": public_row.get("horizon"),
                "title": public_row.get("title"),
                "question": public_row.get("question"),
                "gold_answer": hidden_row.get("gold_answer"),
                "expected_answer_points": hidden_row.get("expected_answer_points") or [],
                "ground_truth": hidden_row.get("ground_truth") or trace_row.get("ground_truth"),
                "support_context": trace_row.get("support_context"),
                "time_context": trace_row.get("time_context"),
                "seed": trace_row.get("seed"),
                "public_metadata": hidden_row.get("public_metadata") or trace_row.get("public_metadata") or {},
                "quality_signals": trace_row.get("quality_signals") or {},
                "source_task_id": hidden_row.get("internal_task_id"),
            }

        curated_public.append(public_row)
        curated_hidden.append(hidden_row)
        curated_trace.append(trace_row)
        curated_internal.append(internal_row)

    dump_jsonl(output_dir / "tasks.jsonl", curated_public)
    dump_jsonl(output_dir / "tasks_hidden_eval.jsonl", curated_hidden)
    dump_jsonl(output_dir / "tasks_build_trace.jsonl", curated_trace)
    dump_jsonl(output_dir / "tasks_internal_full.jsonl", curated_internal)
    dump_json(output_dir / "dropped_tasks.json", dropped)
    dump_json(output_dir / "venue_repairs.json", venue_repairs)

    family_counts = Counter(row["family"] for row in curated_public)
    domain_counts = Counter(row["domain"] for row in curated_public)
    subtype_counts = Counter((row["family"], row.get("subtype") or "") for row in curated_public)
    manifest = {
        "release_name": output_dir.name,
        "source_release": str(source_dir),
        "task_count": len(curated_public),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "subtype_counts": {f"{family}::{subtype}": count for (family, subtype), count in sorted(subtype_counts.items())},
        "dropped_task_count": len(dropped),
        "venue_repair_count": len(venue_repairs),
        "files": {
            "tasks_public": "tasks.jsonl",
            "tasks_hidden_eval": "tasks_hidden_eval.jsonl",
            "tasks_build_trace": "tasks_build_trace.jsonl",
            "tasks_internal_full": "tasks_internal_full.jsonl",
            "dropped_tasks": "dropped_tasks.json",
            "venue_repairs": "venue_repairs.json",
        },
        "curation_policy": {
            "dropped_tasks": ["strategic_research_planning agenda tasks without explicit candidate directions in trace support"],
            "dropped_q1_agenda_tasks": ["remove q1 agenda-priority planning tasks because they do not form a stable candidate-ranking protocol"],
            "venue_repair": "canonicalize venue names to be consistent with the selected venue bucket",
            "title_normalization": "normalize planning and venue titles using trace-backed topic names",
        },
    }
    dump_json(output_dir / "manifest.json", manifest)

    readme = f"""# {output_dir.name}

## Summary
- tasks: {len(curated_public)}
- source release: {source_dir.name}
- dropped tasks: {len(dropped)}
- venue repairs: {len(venue_repairs)}

## Family counts
- bottleneck_opportunity_discovery: {family_counts.get('bottleneck_opportunity_discovery', 0)}
- direction_forecasting: {family_counts.get('direction_forecasting', 0)}
- strategic_research_planning: {family_counts.get('strategic_research_planning', 0)}
- venue_aware_research_positioning: {family_counts.get('venue_aware_research_positioning', 0)}

## Curation policy
- Drop agenda-priority planning tasks that do not expose public candidate directions and therefore remain structurally ambiguous.
- Canonicalize venue names in hidden and trace files so the selected venue bucket and venue name do not conflict.
- Normalize planning and venue titles using trace-backed topic names to improve surface quality.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated benchmark release from benchmark_full.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_curated_release(Path(args.source_dir), Path(args.output_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
