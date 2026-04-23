import json
from collections import Counter
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
TASK_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
LOG_PATH = ROOT / "docs" / "benchmark_refine_log_20260423_round11.md"


RENAME_MAP = {
    "historical_bottleneck_to_half_year_opportunity": "bottleneck_and_opportunity_half_year",
    "historical_bottleneck_to_quarter_opportunity": "bottleneck_and_opportunity_quarter",
    "next_direction_and_trajectory_forecast": "direction_forecast_half_year",
    "quarter_ahead_direction_and_trajectory_forecast": "direction_forecast_quarter",
    "research_agenda_prioritization": "agenda_prioritization",
    "comparative_opportunity_ranking": "opportunity_prioritization",
    "venue_strategy_planning": "venue_aligned_planning",
    "venue_specific_direction_positioning": "venue_aligned_direction_forecast",
}

TARGET_VERSION = "2026-04-23-manual-v2"

SCRIPT_FILES = [
    ROOT / "scripts" / "build_task_refined_pilot_20260423.py",
    ROOT / "scripts" / "build_task_refined_batch2_20260423.py",
    ROOT / "scripts" / "build_task_refined_full_20260423.py",
]


def rename_task_file():
    before = Counter()
    after = Counter()
    rows = []
    for raw in TASK_PATH.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        before[row["subtype"]] += 1
        row["subtype"] = RENAME_MAP[row["subtype"]]
        row["subtype_taxonomy_version"] = TARGET_VERSION
        after[row["subtype"]] += 1
        rows.append(row)
    TASK_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return before, after


def patch_builder_scripts():
    for path in SCRIPT_FILES:
        text = path.read_text()
        for old, new in RENAME_MAP.items():
            text = text.replace(f"\"{old}\"", f"\"{new}\"")
        path.write_text(text)


def write_log(before, after):
    lines = [
        "# Benchmark Refine Log 20260423 Round 11",
        "",
        "## Scope",
        "",
        "- This round renames the subtype taxonomy to make the labels shorter, more natural, and more consistent in style.",
        "- No task content, gold answers, or audit statuses were changed.",
        "",
        "## New Taxonomy",
        "",
    ]
    for old, new in RENAME_MAP.items():
        lines.append(f"- `{old}` -> `{new}`")

    lines.extend(
        [
            "",
            "## Files Updated",
            "",
            "- `benchmark_release/task_refined.jsonl`",
            "- `scripts/build_task_refined_pilot_20260423.py`",
            "- `scripts/build_task_refined_batch2_20260423.py`",
            "- `scripts/build_task_refined_full_20260423.py`",
            "- `docs/benchmark_refine_log_20260423_round11.md`",
            "",
            "## Counts",
            "",
            f"- Before: {dict(before)}",
            f"- After: {dict(after)}",
            "",
            f"- New `subtype_taxonomy_version`: `{TARGET_VERSION}`",
        ]
    )

    LOG_PATH.write_text("\n".join(lines) + "\n")


def main():
    before, after = rename_task_file()
    patch_builder_scripts()
    write_log(before, after)


if __name__ == "__main__":
    main()
