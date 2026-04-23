from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
REFINED_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
ROUND4_LOG = ROOT / "docs" / "benchmark_refine_log_20260423_round4.md"


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
    return signals.get("candidate_directions") or gt.get("candidate_directions") or []


def fix_single_candidate_planning(row: dict) -> None:
    direction = candidate_directions(row)
    direction_text = direction[0] if direction else "the listed research direction"
    row["question"] = (
        f"Using literature published on or before August 31, 2025, assess whether {direction_text} should be treated as the leading near-term research direction for the following six months. "
        "Justify the assessment with pre-cutoff evidence on technical need, momentum, dependencies, and likely downstream impact."
    )
    row["question_quality"] = {
        "status": "substantially_rewritten_round4",
        "summary": "The original item posed a one-candidate ranking problem. Round 4 converts it into an evidence-based assessment task.",
        "edits": [
            "Reframed a degenerate one-option planning prompt into an assessment prompt.",
        ],
    }
    row["answer_audit"] = {
        "status": "substantially_corrected",
        "summary": "This planning item had a structural flaw because it asked for a ranking over a single candidate direction. Round 4 preserves the underlying research direction but rewrites the task as an assessment rather than a fake ranking.",
        "historical_support": row["answer_audit"].get("historical_support", "weak"),
        "future_support": row["answer_audit"].get("future_support", "moderate"),
        "changes": [
            "Converted a one-option ranking task into an evidence-based assessment task.",
            "Preserved the intended research direction while making the task form more defensible.",
        ],
    }
    row.setdefault("ground_truth", {})
    row["ground_truth"].setdefault("audit_notes", [])
    row["ground_truth"]["audit_notes"].append(
        "Round 4 converted this one-candidate planning item into an assessment-style task."
    )
    row.setdefault("review_metadata", {})
    row["review_metadata"]["review_depth"] = "manual_structural_round4"
    row["review_metadata"]["round4_planning_fix"] = True


def write_log(rows: list[dict]) -> None:
    targeted = [r for r in rows if r.get("review_metadata", {}).get("round4_planning_fix")]
    lines = [
        "# Benchmark Refine Log 20260423 Round 4",
        "",
        "## Scope",
        "",
        "- This supplemental log records the fourth optimization round focused on structurally weak strategic-planning tasks.",
        f"- Planning tasks rewritten in round 4: {len(targeted)}",
        "",
        "## Main Changes",
        "",
        "- Converted one-candidate planning-ranking items into evidence-based assessment prompts.",
        "- Preserved the intended near-term direction while removing the fake-ranking structure.",
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
    lines.append("- `docs/benchmark_refine_log_20260423_round4.md`")
    ROUND4_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    count = 0
    for row in rows:
        if row.get("family") != "strategic_research_planning":
            continue
        if row.get("answer_audit", {}).get("status") != "provisional_low_evidence":
            continue
        if len(candidate_directions(row)) != 1:
            continue
        fix_single_candidate_planning(row)
        count += 1
    write_rows(rows)
    write_log(rows)
    print(f"Round 4 updated {count} strategic-planning tasks.")


if __name__ == "__main__":
    main()
