from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
REFINED_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
ROUND3_LOG = ROOT / "docs" / "benchmark_refine_log_20260423_round3.md"


def load_rows() -> list[dict]:
    with REFINED_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_rows(rows: list[dict]) -> None:
    with REFINED_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_directions(row: dict) -> list[str]:
    gt = row.get("ground_truth", {})
    signals = gt.get("structured_signals", {})
    dirs = signals.get("candidate_directions") or gt.get("candidate_directions") or []
    return [d for d in dirs if d]


def target_venue_name(row: dict) -> str:
    gt = row.get("ground_truth", {})
    signals = gt.get("structured_signals", {})
    return (
        signals.get("target_venue_name")
        or gt.get("target_venue_name")
        or row.get("title", "").split(" for ")[-1]
    )


def soften_forecast_gold(text: str) -> str:
    replacements = [
        ("The most likely next step is", "A plausible next-step direction is"),
        ("The most promising next-step research direction is", "A plausible next-step research direction is"),
        ("The next prominent research direction is", "A plausible next-step research direction is"),
        ("The most promising research direction is", "A plausible research direction is"),
        ("with primary traction expected in", "with"),
        ("with the highest probability of traction in", "with"),
        ("This is supported by", "This is plausibly supported by"),
    ]
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def fix_single_candidate(row: dict) -> None:
    direction = candidate_directions(row)
    direction_text = direction[0] if direction else "the listed research direction"
    venue = target_venue_name(row)
    row["question"] = (
        f"Using literature published on or before August 31, 2025, assess whether {direction_text} is the most credible venue-aligned direction for {venue} over the following six months. "
        "Justify the assessment with pre-cutoff evidence on technical need, momentum, and venue fit."
    )
    row["question_quality"] = {
        "status": "substantially_rewritten_round3",
        "summary": "The original item posed a one-candidate ranking problem. Round 3 converts it into a venue-fit assessment task so the question is structurally valid.",
        "edits": [
            "Reframed a degenerate one-option ranking prompt as an evidence-based assessment prompt.",
        ],
    }
    row["answer_audit"] = {
        "status": "substantially_corrected",
        "summary": "This venue-positioning item had a one-candidate ranking structure that was hard to justify as written. Round 3 keeps the underlying direction but rewrites the task as a venue-fit assessment instead of a fake ranking.",
        "historical_support": row["answer_audit"].get("historical_support", "weak"),
        "future_support": row["answer_audit"].get("future_support", "moderate"),
        "changes": [
            "Converted a one-option venue-ranking task into an assessment task.",
            "Preserved the core direction while reducing structural awkwardness.",
        ],
    }
    row.setdefault("ground_truth", {})
    row["ground_truth"].setdefault("audit_notes", [])
    row["ground_truth"]["audit_notes"].append(
        "Round 3 converted this one-candidate venue-ranking item into an assessment-style task."
    )
    row.setdefault("review_metadata", {})
    row["review_metadata"]["review_depth"] = "manual_structural_round3"
    row["review_metadata"]["round3_venue_fix"] = True


def fix_direction_forecast(row: dict) -> None:
    row["question"] = (
        "Using literature published on or before August 31, 2025, identify the most plausible next-step research direction in this area for the following six months and name the likeliest top-tier venue bucket fit. "
        "Justify both the direction forecast and the venue fit using only pre-cutoff evidence, and treat both as probabilistic forecasts rather than certainties."
    )
    row["gold_answer"] = soften_forecast_gold(row.get("gold_answer", ""))
    row["question_quality"] = {
        "status": "light_rewrite_round3",
        "summary": "Round 3 softens the forecasting prompt so it asks for a probabilistic venue-aligned forecast rather than implying stronger certainty than the evidence supports.",
        "edits": [
            "Added an explicit probabilistic framing to the venue-aligned forecast question.",
        ],
    }
    row["answer_audit"]["summary"] = (
        "Round 3 softened the venue-fit and next-direction language for this weak-evidence forecast, but the item remains lower confidence until a later deep pass adds stronger historical support."
    )
    row["answer_audit"]["changes"] = row["answer_audit"].get("changes", []) + [
        "Softened venue-fit certainty in the prompt and answer without changing the underlying forecast."
    ]
    row.setdefault("ground_truth", {})
    row["ground_truth"].setdefault("audit_notes", [])
    row["ground_truth"]["audit_notes"].append(
        "Round 3 softened the direction-and-venue forecast language because the supporting evidence remains thin."
    )
    row.setdefault("review_metadata", {})
    row["review_metadata"]["review_depth"] = "manual_structural_round3"
    row["review_metadata"]["round3_venue_fix"] = True


def write_log(rows: list[dict], single_count: int, forecast_count: int) -> None:
    targeted = [r for r in rows if r.get("review_metadata", {}).get("round3_venue_fix")]
    lines = [
        "# Benchmark Refine Log 20260423 Round 3",
        "",
        "## Scope",
        "",
        "- This supplemental log records the third optimization round focused on remaining low-confidence venue-aware tasks.",
        f"- Total venue tasks touched in round 3: {len(targeted)}",
        f"- Single-candidate venue-planning tasks rewritten as assessments: {single_count}",
        f"- Weak-evidence venue-direction forecasts softened and normalized: {forecast_count}",
        "",
        "## Main Changes",
        "",
        "- Converted one-candidate venue ranking items into evidence-based assessment prompts.",
        "- Softened overconfident venue-fit language in weak-evidence venue-direction forecasts.",
        "- Preserved the underlying benchmark intent while making the questions more honest and structurally cleaner.",
        "",
        "## Targeted Tasks",
        "",
    ]
    for row in sorted(targeted, key=lambda r: r["task_id"]):
        lines.append(
            f"- `{row['task_id']}`: {row['answer_audit']['status']} "
            f"(historical={row['answer_audit']['historical_support']}, future={row['answer_audit']['future_support']}) — {row['title']}"
        )
    lines.append("")
    lines.append("## Files Updated")
    lines.append("")
    lines.append("- `benchmark_release/task_refined.jsonl`")
    lines.append("- `docs/benchmark_refine_log_20260423_round3.md`")
    ROUND3_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    single_count = 0
    forecast_count = 0

    for row in rows:
        if row.get("family") != "venue_aware_research_positioning":
            continue
        if row.get("answer_audit", {}).get("status") != "provisional_low_evidence":
            continue

        dirs = candidate_directions(row)
        if row.get("original_subtype") == "venue_targeted_planning" and len(dirs) == 1:
            fix_single_candidate(row)
            single_count += 1
        elif row.get("original_subtype") == "venue_aware_direction_forecast":
            fix_direction_forecast(row)
            forecast_count += 1

    write_rows(rows)
    write_log(rows, single_count, forecast_count)
    print(f"Round 3 updated {single_count} single-candidate venue tasks and {forecast_count} venue-direction forecast tasks.")


if __name__ == "__main__":
    main()
