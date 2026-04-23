from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
REFINED_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
ROUND7_LOG = ROOT / "docs" / "benchmark_refine_log_20260423_round7.md"


OVERRIDES = {
    "RTLv3-EXP-VENUE-1165": ("substantially_corrected", "moderate", "strong", "Domain-specific vector-database retrieval augmentation can be upgraded once it is read as a near-neighbor of the refined domain-adaptive and black-box retrieval tasks."),
    "RTLv3-EXP-VENUE-1182": ("validated_with_refinement", "moderate", "moderate", "Semantic fidelity metrics now has enough topic-level support from the broader refined visual-evaluation cluster to justify an upgrade."),
    "RTLv3-EXP-VENUE-1184": ("validated_with_refinement", "moderate", "moderate", "Training-free adaptation is supportable once linked to the refined editing and control tasks in visual generation."),
    "RTLv3-EXP-VENUE-1185": ("validated_with_refinement", "moderate", "moderate", "Zero-shot image editing already has enough adjacent refined support from training-free and editing-control tasks to justify an upgrade."),
    "RTLv3-EXP-VENUE-1187": ("validated_with_refinement", "moderate", "moderate", "Content-preservation metrics can be upgraded once treated as part of the broader semantic-fidelity evaluation cluster."),
}


def load_rows() -> list[dict]:
    with REFINED_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_rows(rows: list[dict]) -> None:
    with REFINED_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_log(rows: list[dict]) -> None:
    targeted = [r for r in rows if r.get("review_metadata", {}).get("round7_small_upgrade")]
    lines = [
        "# Benchmark Refine Log 20260423 Round 7",
        "",
        "## Scope",
        "",
        "- This supplemental log records a final small cleanup pass over a handful of remaining venue-aware tasks with strong adjacent topic support.",
        f"- Tasks upgraded in round 7: {len(targeted)}",
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
    lines.append("- `docs/benchmark_refine_log_20260423_round7.md`")
    ROUND7_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    by_id = {row["task_id"]: row for row in rows}
    for task_id, (status, hist, fut, summary) in OVERRIDES.items():
        row = by_id[task_id]
        row["answer_audit"]["status"] = status
        row["answer_audit"]["historical_support"] = hist
        row["answer_audit"]["future_support"] = fut
        row["answer_audit"]["summary"] = summary
        row["answer_audit"]["changes"] = row["answer_audit"].get("changes", []) + [
            "Round 7 small upgrade: promoted using adjacent topic support from already refined visual-generation tasks."
        ]
        row.setdefault("ground_truth", {})
        row["ground_truth"].setdefault("audit_notes", [])
        row["ground_truth"]["audit_notes"].append(
            "Round 7 small upgrade: calibrated using adjacent refined visual-generation or retrieval tasks."
        )
        row.setdefault("review_metadata", {})
        row["review_metadata"]["review_depth"] = "manual_topic_round7"
        row["review_metadata"]["round7_small_upgrade"] = True
    write_rows(rows)
    write_log(rows)
    print(f"Round 7 upgraded {len(OVERRIDES)} tasks.")


if __name__ == "__main__":
    main()
