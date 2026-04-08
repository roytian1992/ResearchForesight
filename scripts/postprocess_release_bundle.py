from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify(row.get(key)) for key in fieldnames})


def join_rubric_dimensions(rubric: dict[str, Any] | None) -> str:
    dims = (rubric or {}).get("core_dimensions") or []
    items = []
    for dim in dims:
        name = str(dim.get("name") or "").strip()
        weight = dim.get("weight")
        if name:
            items.append(f"{name}[{weight}]")
    return " || ".join(items)


def build_csvs(release_dir: Path) -> None:
    public_rows = list(iter_jsonl(release_dir / "tasks.jsonl"))
    hidden_rows = list(iter_jsonl(release_dir / "tasks_hidden_eval.jsonl"))
    trace_rows = list(iter_jsonl(release_dir / "tasks_build_trace.jsonl"))
    hidden_by_id = {row["task_id"]: row for row in hidden_rows}
    trace_by_id = {row["task_id"]: row for row in trace_rows}

    public_csv_rows: list[dict[str, Any]] = []
    hidden_csv_rows: list[dict[str, Any]] = []
    merged_csv_rows: list[dict[str, Any]] = []
    trace_csv_rows: list[dict[str, Any]] = []

    for row in public_rows:
        task_id = row["task_id"]
        hidden = hidden_by_id.get(task_id, {})
        trace = trace_by_id.get(task_id, {})
        deliverable = row.get("deliverable_spec") or {}
        public_meta = hidden.get("public_metadata") or trace.get("public_metadata") or {}
        seed = trace.get("seed") or {}
        time_context = trace.get("time_context") or {}
        support = trace.get("support_context") or {}
        gt = trace.get("ground_truth") or {}
        judge = hidden.get("judge") or trace.get("judge") or {}

        public_csv_rows.append(
            {
                "task_id": task_id,
                "family": row.get("family"),
                "subtype": row.get("subtype"),
                "domain": row.get("domain"),
                "horizon": row.get("horizon"),
                "title": row.get("title"),
                "question": row.get("question"),
                "time_cutoff": row.get("time_cutoff"),
                "deliverable_format": deliverable.get("format"),
                "deliverable_requirements": " || ".join(deliverable.get("requirements") or []),
            }
        )

        hidden_csv_rows.append(
            {
                "task_id": task_id,
                "internal_task_id": hidden.get("internal_task_id"),
                "family": hidden.get("family"),
                "domain": hidden.get("domain"),
                "title": hidden.get("title"),
                "gold_answer": hidden.get("gold_answer"),
                "expected_answer_points": " || ".join(hidden.get("expected_answer_points") or []),
                "rubric_dimensions": join_rubric_dimensions(hidden.get("evaluation_rubric")),
                "judge_overall": judge.get("overall_score"),
                "public_topic": public_meta.get("topic"),
                "public_future_themes": " || ".join(public_meta.get("future_themes") or []),
            }
        )

        merged_csv_rows.append(
            {
                "task_id": task_id,
                "family": row.get("family"),
                "subtype": row.get("subtype"),
                "domain": row.get("domain"),
                "horizon": row.get("horizon"),
                "title": row.get("title"),
                "time_cutoff": row.get("time_cutoff"),
                "question": row.get("question"),
                "gold_answer": hidden.get("gold_answer"),
                "expected_answer_points": " || ".join(hidden.get("expected_answer_points") or []),
                "public_topic": public_meta.get("topic"),
                "public_future_themes": " || ".join(public_meta.get("future_themes") or []),
                "rubric_dimensions": join_rubric_dimensions(hidden.get("evaluation_rubric")),
                "seed_dimension_id": seed.get("dimension_id"),
                "historical_paper_count": (support.get("historical_stats") or {}).get("paper_count"),
                "future_half_paper_count": (gt.get("future_half_stats") or {}).get("paper_count"),
                "target_window_paper_count": (gt.get("target_window_stats") or {}).get("paper_count"),
                "judge_overall": judge.get("overall_score"),
            }
        )

        trace_csv_rows.append(
            {
                "task_id": task_id,
                "internal_task_id": trace.get("internal_task_id"),
                "family": trace.get("family"),
                "domain": trace.get("domain"),
                "seed_packet_id": seed.get("packet_id"),
                "seed_node_id": seed.get("node_id"),
                "seed_dimension_id": seed.get("dimension_id"),
                "history_end": time_context.get("history_end"),
                "history_structure_slice": time_context.get("history_structure_slice"),
                "historical_paper_count": (support.get("historical_stats") or {}).get("paper_count"),
                "historical_top_conf_count": (support.get("historical_stats") or {}).get("top_conf_count"),
                "historical_top_conf_share": (support.get("historical_stats") or {}).get("top_conf_share"),
                "historical_citation_median": (support.get("historical_stats") or {}).get("citation_median"),
                "top_limitations": " || ".join(
                    [str(item.get("name") or "") for item in (support.get("top_limitations") or []) if item.get("name")]
                ),
                "top_future_work": " || ".join(
                    [str(item.get("direction") or "") for item in (support.get("top_future_work") or []) if item.get("direction")]
                ),
                "history_representative_titles": " || ".join(
                    [str(item.get("title") or "") for item in (support.get("history_representative_papers") or []) if item.get("title")]
                ),
                "future_half_paper_count": (gt.get("future_half_stats") or {}).get("paper_count"),
                "future_half_top_conf_count": (gt.get("future_half_stats") or {}).get("top_conf_count"),
                "future_half_top_conf_share": (gt.get("future_half_stats") or {}).get("top_conf_share"),
                "trajectory_label": (gt.get("trajectory") or {}).get("trajectory_label"),
                "target_window_paper_count": (gt.get("target_window_stats") or {}).get("paper_count"),
                "target_window_top_conf_count": (gt.get("target_window_stats") or {}).get("top_conf_count"),
                "target_window_top_conf_share": (gt.get("target_window_stats") or {}).get("top_conf_share"),
                "judge_overall": judge.get("overall_score"),
                "heuristic_score": (trace.get("quality_signals") or {}).get("heuristic_score"),
            }
        )

    csv_dir = release_dir / "csv"
    write_csv(
        csv_dir / "tasks_public.csv",
        public_csv_rows,
        [
            "task_id",
            "family",
            "subtype",
            "domain",
            "horizon",
            "title",
            "question",
            "time_cutoff",
            "deliverable_format",
            "deliverable_requirements",
        ],
    )
    write_csv(
        csv_dir / "tasks_hidden_eval.csv",
        hidden_csv_rows,
        [
            "task_id",
            "internal_task_id",
            "family",
            "domain",
            "title",
            "gold_answer",
            "expected_answer_points",
            "rubric_dimensions",
            "judge_overall",
            "public_topic",
            "public_future_themes",
        ],
    )
    write_csv(
        csv_dir / "tasks_overview_merged.csv",
        merged_csv_rows,
        [
            "task_id",
            "family",
            "subtype",
            "domain",
            "horizon",
            "title",
            "time_cutoff",
            "question",
            "gold_answer",
            "expected_answer_points",
            "public_topic",
            "public_future_themes",
            "rubric_dimensions",
            "seed_dimension_id",
            "historical_paper_count",
            "future_half_paper_count",
            "target_window_paper_count",
            "judge_overall",
        ],
    )
    write_csv(
        csv_dir / "tasks_build_trace_summary.csv",
        trace_csv_rows,
        [
            "task_id",
            "internal_task_id",
            "family",
            "domain",
            "seed_packet_id",
            "seed_node_id",
            "seed_dimension_id",
            "history_end",
            "history_structure_slice",
            "historical_paper_count",
            "historical_top_conf_count",
            "historical_top_conf_share",
            "historical_citation_median",
            "top_limitations",
            "top_future_work",
            "history_representative_titles",
            "future_half_paper_count",
            "future_half_top_conf_count",
            "future_half_top_conf_share",
            "trajectory_label",
            "target_window_paper_count",
            "target_window_top_conf_count",
            "target_window_top_conf_share",
            "judge_overall",
            "heuristic_score",
        ],
    )


def build_readme(release_dir: Path) -> None:
    manifest = json.loads((release_dir / "manifest.json").read_text(encoding="utf-8"))
    tasks = list(iter_jsonl(release_dir / "tasks.jsonl"))
    cutoffs = sorted({str(row.get("time_cutoff") or "").strip() for row in tasks if str(row.get("time_cutoff") or "").strip()})
    domains = sorted({str(row.get("domain") or "").strip() for row in tasks if str(row.get("domain") or "").strip()})
    families = sorted({str(row.get("family") or "").strip() for row in tasks if str(row.get("family") or "").strip()})
    readme = f"""# {release_dir.name}

## Summary
- tasks: {manifest.get('task_count')}
- families: {len(families)}
- domains: {len(domains)}
- per family × domain cap: {((manifest.get('selection_policy') or {}).get('max_per_family_domain'))}
- history cutoff: {', '.join(cutoffs)}
- future windows:
  - quarterly: 2025-09-01 ~ 2025-11-30, 2025-12-01 ~ 2026-02-28
  - half-year: 2025-09-01 ~ 2026-02-28

## Task families
1. direction_forecasting
2. bottleneck_opportunity_discovery
3. strategic_research_planning

## Domains
{chr(10).join(f'- {domain}' for domain in domains)}

## Construction notes
This release is built from:
- quarterly TaxoAdapt taxonomy snapshots
- node-level venue/citation aggregates
- CoI-style support packets
- selective paper-structure extraction
- LLM rewrite and LLM-as-judge filtering

## Environment caveat
During construction, direct arXiv source/html fetching was unavailable from the current runtime environment.
Therefore the selective paper evidence layer was instantiated with abstract-derived normalized content and abstract-conditioned structure extraction rather than source-tex/html full text.
The benchmark release remains temporally valid, but this evidence layer should be upgraded to full-text extraction in a later refresh when arXiv connectivity is available.
"""
    (release_dir / "README.md").write_text(readme, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess a benchmark release into CSV summaries and README.")
    parser.add_argument("--release-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    build_csvs(release_dir)
    build_readme(release_dir)
    print(f"postprocessed {release_dir}")


if __name__ == "__main__":
    main()
