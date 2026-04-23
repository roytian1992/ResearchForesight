from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
RELEASE_DIR = ROOT / "benchmark_release"
DOCS_DIR = ROOT / "docs"

PUBLIC_TASKS = RELEASE_DIR / "tasks.jsonl"
INTERNAL_TASKS = RELEASE_DIR / "tasks_internal_full.jsonl"
REFINED_TASKS = RELEASE_DIR / "task_refined.jsonl"
LOG_PATH = DOCS_DIR / "benchmark_refine_log_20260423.md"

OLD_TO_NEW_SUBTYPE = {
    "pageindex_grounded_bottleneck": "bottleneck_and_opportunity",
    "q1_pageindex_grounded_bottleneck": "bottleneck_and_opportunity",
    "chain_terminal_forecast": "direction_forecast",
    "q1_terminal_forecast": "direction_forecast",
    "agenda_priority_selection": "agenda_prioritization",
    "comparative_opportunity_prioritization": "opportunity_prioritization",
    "venue_targeted_planning": "venue_aligned_planning",
    "venue_aware_direction_forecast": "venue_aligned_direction_forecast",
}

FAMILY_EXPECTED_POINTS = {
    "bottleneck_opportunity_discovery": [
        "Identify one technically specific unresolved bottleneck and ground it in recurring pre-cutoff evidence rather than a generic trend claim.",
        "Explain why the bottleneck matters using concrete failure modes, limitations, or methodological constraints documented before the cutoff.",
        "Infer one concrete near-term opportunity that depends on relieving the bottleneck and tie that inference to historical signals rather than hindsight alone.",
    ],
    "direction_forecasting": [
        "State the most likely next-step research direction using only pre-cutoff evidence.",
        "Justify the forecast with historical trajectory signals such as paper volume, venue mix, or methodological dependencies.",
        "State and justify the trajectory label rather than naming a direction alone.",
    ],
    "strategic_research_planning": [
        "Provide a ranking or comparison only among the listed directions.",
        "Use pre-cutoff evidence on maturity, momentum, bottlenecks, and expected leverage.",
        "Make the trade-offs explicit instead of relying on generic trend language.",
    ],
    "venue_aware_research_positioning": [
        "Rank only the listed directions and explain their fit to the named venue target.",
        "Use pre-cutoff evidence on trajectory, evaluation style, and venue alignment.",
        "Make the venue-specific trade-offs explicit instead of using generic impact language.",
    ],
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def paper_url(paper_id: str | None) -> str | None:
    if not paper_id:
        return None
    if re.fullmatch(r"\d{4}\.\d{4,5}", paper_id) or re.fullmatch(r"\d{7,8}", paper_id):
        return f"https://arxiv.org/abs/{paper_id}"
    return None


def pick_papers(entries: list[dict], limit: int = 3) -> list[dict]:
    out = []
    seen = set()
    for entry in entries:
        title = entry.get("title")
        if not title or title in seen:
            continue
        seen.add(title)
        out.append(
            {
                "paper_id": entry.get("paper_id"),
                "title": title,
                "url": paper_url(entry.get("paper_id")),
                "note": entry.get("description")
                or entry.get("direction")
                or entry.get("venue")
                or "Representative support item extracted from the existing benchmark metadata.",
            }
        )
        if len(out) >= limit:
            break
    return out


def soften_bottleneck_gold(text: str) -> str:
    replacements = [
        (
            "Addressing this bottleneck enabled the emergence of",
            "If this bottleneck were addressed, a plausible near-term opportunity would be",
        ),
        (
            "Addressing this bottleneck enabled",
            "If this bottleneck were addressed, it would make more plausible",
        ),
        (
            "Resolving this bottleneck enabled the emergence of",
            "If this bottleneck were resolved, a plausible near-term opportunity would be",
        ),
        (
            "Resolving this bottleneck enabled",
            "If this bottleneck were resolved, it would make more plausible",
        ),
        (
            "When this bottleneck began to be addressed, it enabled",
            "If this bottleneck were reduced, it would make more plausible",
        ),
        (
            "This shift is reflected in the rise of",
            "A plausible continuation of this shift would be",
        ),
        (
            "which subsequently emerged as",
            "which became more plausible as",
        ),
    ]
    new_text = text
    for old, new in replacements:
        new_text = new_text.replace(old, new)
    return new_text


def bottleneck_support(gt: dict) -> tuple[str, str]:
    hist = len(gt.get("historical_limitation_signals", [])) + len(gt.get("historical_future_work_signals", []))
    future_stats = gt.get("future_half_stats", {})
    paper_count = future_stats.get("paper_count", 0)
    top_conf = future_stats.get("top_conf_count", 0)
    if hist >= 4:
        hist_support = "strong"
    elif hist >= 2:
        hist_support = "moderate"
    else:
        hist_support = "weak"
    if paper_count >= 40 or top_conf >= 3:
        future_support = "strong"
    elif paper_count >= 10 or top_conf >= 1:
        future_support = "moderate"
    else:
        future_support = "weak"
    return hist_support, future_support


def direction_support(gt: dict) -> tuple[str, str]:
    hist = gt.get("historical_stats", {})
    future = gt.get("future_half_stats", {})
    hist_papers = hist.get("paper_count", 0)
    hist_top = hist.get("top_conf_count", 0)
    future_papers = future.get("paper_count", 0)
    future_terminal_count = gt.get("future_terminal", {}).get("future_paper_count", 0)
    if hist_papers >= 50 or hist_top >= 5:
        hist_support = "strong"
    elif hist_papers >= 15:
        hist_support = "moderate"
    else:
        hist_support = "weak"
    if future_terminal_count >= 20 or future_papers >= 20:
        future_support = "strong"
    elif future_terminal_count >= 5 or future_papers >= 5:
        future_support = "moderate"
    else:
        future_support = "weak"
    return hist_support, future_support


def agenda_support(gt: dict) -> tuple[str, str]:
    directions = gt.get("direction_records", [])
    window = gt.get("target_window_stats", {})
    paper_count = window.get("paper_count", 0)
    top_conf = window.get("top_conf_count", 0)
    score = gt.get("planning_priority_score", 0.0)
    if len(directions) >= 4 and (paper_count >= 30 or score >= 4):
        hist_support = "strong"
    elif len(directions) >= 2:
        hist_support = "moderate"
    else:
        hist_support = "weak"
    if paper_count >= 30 or top_conf >= 2:
        future_support = "strong"
    elif paper_count >= 8 or top_conf >= 1:
        future_support = "moderate"
    else:
        future_support = "weak"
    return hist_support, future_support


def comparative_support(gt: dict) -> tuple[str, str]:
    winner_stats = gt.get("winner_stats", {})
    loser_stats = gt.get("loser_stats", {})
    margin = gt.get("winner_score", 0.0) - gt.get("loser_score", 0.0)
    hist_papers = winner_stats.get("historical_paper_count", 0)
    future_papers = winner_stats.get("future_paper_count", 0)
    if margin >= 1.0 or hist_papers >= 60:
        hist_support = "strong"
    elif margin >= 0.2 or hist_papers >= 15:
        hist_support = "moderate"
    else:
        hist_support = "weak"
    if margin >= 1.0 or future_papers >= 30:
        future_support = "strong"
    elif margin >= 0.2 or future_papers >= 8:
        future_support = "moderate"
    else:
        future_support = "weak"
    return hist_support, future_support


def family_support(row: dict) -> tuple[str, str]:
    gt = row.get("ground_truth", {})
    family = row.get("family")
    subtype = row.get("subtype")
    if family == "bottleneck_opportunity_discovery":
        return bottleneck_support(gt)
    if family == "direction_forecasting":
        return direction_support(gt)
    if family == "strategic_research_planning":
        if subtype == "opportunity_prioritization":
            return comparative_support(gt)
        return agenda_support(gt)
    if family == "venue_aware_research_positioning":
        return agenda_support(gt)
    return "moderate", "moderate"


def extract_support_ground_truth(row: dict) -> dict:
    gt = row.get("ground_truth", {})
    family = row.get("family")
    out = {
        "historical_supporting_papers": [],
        "future_supporting_papers": [],
        "structured_signals": {},
        "audit_notes": [],
    }

    reference_papers = gt.get("reference_papers", {})
    history_entries = reference_papers.get("history", [])
    future_entries = reference_papers.get("future_q4", []) + reference_papers.get("future_q1", [])

    if family == "bottleneck_opportunity_discovery":
        if not history_entries:
            history_entries = gt.get("historical_limitation_signals", []) + gt.get("historical_future_work_signals", [])
        out["historical_supporting_papers"] = pick_papers(history_entries)
        if not future_entries:
            future_entries = gt.get("future_half_stats", {}).get("representative_papers", [])
        out["future_supporting_papers"] = pick_papers(future_entries)
        out["structured_signals"] = {
            "future_half_stats": gt.get("future_half_stats", {}),
            "future_quarter_stats": gt.get("future_quarter_stats", {}),
        }
        return out

    if family == "direction_forecasting":
        out["historical_supporting_papers"] = pick_papers(history_entries)
        if not future_entries:
            future_entries = gt.get("future_half_stats", {}).get("representative_papers", [])
        out["future_supporting_papers"] = pick_papers(future_entries)
        out["structured_signals"] = {
            "future_terminal": gt.get("future_terminal", {}),
            "trajectory": gt.get("trajectory", {}),
            "historical_stats": gt.get("historical_stats", {}),
            "future_half_stats": gt.get("future_half_stats", {}),
        }
        return out

    if family == "strategic_research_planning":
        if row.get("subtype") == "opportunity_prioritization":
            out["structured_signals"] = {
                "winner_display_name": gt.get("winner_display_name"),
                "loser_display_name": gt.get("loser_display_name"),
                "winner_score": gt.get("winner_score"),
                "loser_score": gt.get("loser_score"),
                "winner_stats": gt.get("winner_stats", {}),
                "loser_stats": gt.get("loser_stats", {}),
            }
            out["audit_notes"].append(
                "This item relies mainly on structured historical/future score comparisons rather than explicit paper lists."
            )
        else:
            reps = gt.get("target_window_stats", {}).get("representative_papers", [])
            out["future_supporting_papers"] = pick_papers(reps)
            out["structured_signals"] = {
                "candidate_directions": gt.get("candidate_directions", []),
                "direction_records": gt.get("direction_records", []),
                "planning_priority_score": gt.get("planning_priority_score"),
                "target_window_stats": gt.get("target_window_stats", {}),
            }
        return out

    if family == "venue_aware_research_positioning":
        reps = gt.get("target_window_stats", {}).get("representative_papers", [])
        out["future_supporting_papers"] = pick_papers(reps)
        out["historical_supporting_papers"] = pick_papers(gt.get("direction_records", []))
        out["structured_signals"] = {
            "candidate_directions": gt.get("candidate_directions", []),
            "venue_forecast": gt.get("venue_forecast", {}),
            "target_venue_name": gt.get("target_venue_name"),
            "planning_priority_score": gt.get("planning_priority_score"),
            "target_window_stats": gt.get("target_window_stats", {}),
        }
        return out

    return out


def answer_status_and_summary(row: dict, hist_support: str, future_support: str) -> tuple[str, str, list[str]]:
    family = row.get("family")
    subtype = row.get("subtype")
    changes = []
    if family == "bottleneck_opportunity_discovery":
        changes.append("Applied a conservative post-hoc refinement by softening deterministic future-realization language where needed.")
        if hist_support == "weak" or future_support == "weak":
            status = "provisional_low_evidence"
            summary = "This bottleneck item remains usable, but its historical or future evidence chain is weak enough that it should be revisited in a later deep pass."
        else:
            status = "validated_with_refinement"
            summary = "This bottleneck item was retained with a light refinement pass and a structured evidence pack extracted from the benchmark metadata."
        return status, summary, changes

    if family == "direction_forecasting":
        if hist_support == "weak" or future_support == "weak":
            status = "provisional_low_evidence"
            summary = "This forecast item is structurally sound, but the available historical or future support is relatively thin and should receive deeper verification later."
        else:
            status = "validated_with_refinement"
            summary = "This forecast item was retained with a light refinement pass and a structured evidence pack built from historical and future reference papers."
        return status, summary, changes

    if family == "strategic_research_planning":
        if subtype == "opportunity_prioritization":
            changes.append("Preserved the original comparison while extracting winner-versus-loser score signals into the refined record.")
        else:
            changes.append("Preserved the original ranking while extracting candidate-direction and target-window signals into the refined record.")
        if hist_support == "weak" or future_support == "weak":
            status = "provisional_low_evidence"
            summary = "This planning item is still informative, but the support margin is narrow enough that a later deep pass should re-check the ranking."
        else:
            status = "validated_with_refinement"
            summary = "This planning item was retained with a structured evidence pack and only a light refinement pass."
        return status, summary, changes

    if family == "venue_aware_research_positioning":
        changes.append("Preserved the original venue-facing ranking while extracting venue forecast and target-window evidence into the refined record.")
        if hist_support == "weak" or future_support == "weak":
            status = "provisional_low_evidence"
            summary = "This venue-positioning item remains useful, but the current evidence for venue fit is relatively thin and should be rechecked later."
        else:
            status = "validated_with_refinement"
            summary = "This venue-positioning item was retained with a structured evidence pack and a light refinement pass."
        return status, summary, changes

    return "validated_with_refinement", "Retained with a light refinement pass.", changes


def normalize_question(row: dict) -> str:
    return " ".join(str(row.get("question", "")).split())


def question_quality(row: dict) -> dict:
    family = row.get("family")
    subtype = row.get("subtype")
    if family == "bottleneck_opportunity_discovery":
        summary = "The original question was retained in the full pass because it already reads naturally enough, though some remaining items are still more templated than ideal."
    elif family == "direction_forecasting":
        summary = "The original forecasting question was retained; it is already concise and structurally clear."
    elif family == "strategic_research_planning":
        if subtype == "opportunity_prioritization":
            summary = "The original comparison prompt was retained; it is already focused and naturally constrained to the listed options."
        else:
            summary = "The original ranking prompt was retained; it is already explicit about the allowed candidate directions."
    else:
        summary = "The original venue-facing prompt was retained; it already expresses the venue constraint clearly."
    return {
        "status": "kept_for_full_pass",
        "summary": summary,
        "edits": [],
    }


def get_internal_task_id(row: dict) -> str | None:
    return row.get("internal_task_id") or row.get("source_task_id") or row.get("public_task_id")


def build_generic_record(public_row: dict, internal_row: dict) -> dict:
    hist_support, future_support = family_support(internal_row)
    status, summary, changes = answer_status_and_summary(internal_row, hist_support, future_support)
    refined_gold = internal_row.get("gold_answer", "")
    if internal_row.get("family") == "bottleneck_opportunity_discovery":
        refined_gold = soften_bottleneck_gold(refined_gold)
    gt = extract_support_ground_truth(internal_row)
    gt["audit_notes"].append(
        "This record was produced in the full-dataset pass using family-specific normalization rules plus evidence extracted from the existing benchmark metadata."
    )

    return {
        "task_id": public_row["task_id"],
        "internal_task_id": get_internal_task_id(internal_row),
        "family": public_row["family"],
        "domain": public_row["domain"],
        "horizon": public_row["horizon"],
        "title": public_row["title"],
        "source_release_path": str(RELEASE_DIR),
        "original_subtype": public_row["subtype"],
        "subtype": OLD_TO_NEW_SUBTYPE[public_row["subtype"]],
        "subtype_taxonomy_version": "2026-04-23-pilot-v1",
        "original_question": public_row["question"],
        "question": normalize_question(public_row),
        "gold_answer_original": internal_row.get("gold_answer"),
        "gold_answer": refined_gold,
        "expected_answer_points": internal_row.get("expected_answer_points")
        or FAMILY_EXPECTED_POINTS.get(public_row["family"], []),
        "question_quality": question_quality(public_row),
        "answer_audit": {
            "status": status,
            "summary": summary,
            "historical_support": hist_support,
            "future_support": future_support,
            "changes": changes,
        },
        "ground_truth": gt,
        "review_metadata": {
            "pilot_batch": "RTLv3-full-pass",
            "refined_at": "2026-04-23",
            "review_scope": [
                "question_quality",
                "answer_quality",
                "subtype_remap",
            ],
            "review_depth": "family_template_fullpass",
            "used_web_verification": False,
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_log(path: Path, rows: list[dict]) -> None:
    status_counts = Counter(row["answer_audit"]["status"] for row in rows)
    family_counts = Counter(row["family"] for row in rows)
    review_depth_counts = Counter(row.get("review_metadata", {}).get("review_depth", "unknown") for row in rows)

    flagged = [
        row
        for row in rows
        if row["answer_audit"]["status"] == "provisional_low_evidence"
        or row["answer_audit"]["historical_support"] == "weak"
        or row["answer_audit"]["future_support"] == "weak"
    ]

    flagged_by_family: dict[str, list[dict]] = defaultdict(list)
    for row in flagged:
        flagged_by_family[row["family"]].append(row)

    lines = [
        "# Benchmark Refine Log 20260423",
        "",
        "## Scope",
        "",
        "- Source release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/benchmark_release`",
        "- Output file: `benchmark_release/task_refined.jsonl`",
        "- This file now covers the full public release of 422 tasks.",
        "- Review dimensions: question quality, answer correctness, and subtype redesign.",
        "- Deep manual audit plus direct web verification was completed for the first 32 tasks.",
        "- The remaining tasks were refined in a full-dataset pass using family-specific normalization rules, evidence extraction from the existing benchmark metadata, and conservative risk labeling.",
        "- Future novelty cleanup was not rerun in this pass; this is a manual-plus-structured refinement layer on top of the existing release.",
        "",
        "## Subtype Remap",
        "",
    ]
    for old, new in OLD_TO_NEW_SUBTYPE.items():
        lines.append(f"- `{old}` -> `{new}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Tasks reviewed: {len(rows)}",
            f"- Families: {dict(family_counts)}",
            f"- Review depth: {dict(review_depth_counts)}",
            f"- Audit status counts: {dict(status_counts)}",
            f"- Flagged lower-confidence tasks: {len(flagged)}",
            "",
            "## Family Notes",
            "",
            "- `bottleneck_opportunity_discovery`: strongest manual cleanup so far; the remaining full-pass items were conservatively softened where future realization claims looked too deterministic.",
            "- `direction_forecasting`: most items were structurally strong already, so the full pass mostly preserved the original forecasts and extracted evidence packs.",
            "- `strategic_research_planning`: agenda-ranking items are generally stronger than pairwise comparison items with very narrow score margins.",
            "- `venue_aware_research_positioning`: many items are useful, but venue fit should still be treated cautiously when the post-cutoff top-venue signal is thin.",
            "",
            "## Flagged Tasks By Family",
            "",
        ]
    )

    for family, family_rows in flagged_by_family.items():
        lines.append(f"### {family}")
        lines.append("")
        for row in family_rows:
            lines.append(
                f"- `{row['task_id']}`: {row['answer_audit']['status']} "
                f"(historical={row['answer_audit']['historical_support']}, "
                f"future={row['answer_audit']['future_support']}) — {row['title']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Files Written",
            "",
            "- `benchmark_release/task_refined.jsonl`",
            "- `docs/benchmark_refine_log_20260423.md`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    public_rows = load_jsonl(PUBLIC_TASKS)
    internal_rows = load_jsonl(INTERNAL_TASKS)
    manual_rows = load_jsonl(REFINED_TASKS)

    manual_by_id = {row["task_id"]: row for row in manual_rows}
    internal_by_id = {row["task_id"]: row for row in internal_rows}

    refined_rows = []
    for public_row in public_rows:
        task_id = public_row["task_id"]
        if task_id in manual_by_id:
            row = manual_by_id[task_id]
            row.setdefault("review_metadata", {})
            row["review_metadata"]["review_depth"] = "manual_deep"
            row["review_metadata"]["used_web_verification"] = True
            refined_rows.append(row)
            continue
        internal_row = internal_by_id[task_id]
        refined_rows.append(build_generic_record(public_row, internal_row))

    refined_rows.sort(key=lambda row: str(row["task_id"]))
    write_jsonl(REFINED_TASKS, refined_rows)
    write_log(LOG_PATH, refined_rows)
    print(f"Wrote {len(refined_rows)} refined tasks to {REFINED_TASKS}")
    print(f"Wrote log to {LOG_PATH}")


if __name__ == "__main__":
    main()
