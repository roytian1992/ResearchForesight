import json
from collections import Counter
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
TASK_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
LOG_PATH = ROOT / "docs" / "benchmark_refine_log_20260423_round12.md"


TASK_SUBTYPE_MAP = {
    "bottleneck_and_opportunity": "bottleneck_and_opportunity",
    "direction_forecast": "direction_forecast",
    "agenda_prioritization": "agenda_prioritization",
    "opportunity_prioritization": "opportunity_prioritization",
    "venue_aligned_planning": "venue_aligned_planning",
    "venue_aligned_direction_forecast": "venue_aligned_direction_forecast",
    "historical_bottleneck_to_half_year_opportunity": "bottleneck_and_opportunity",
    "historical_bottleneck_to_quarter_opportunity": "bottleneck_and_opportunity",
    "next_direction_and_trajectory_forecast": "direction_forecast",
    "quarter_ahead_direction_and_trajectory_forecast": "direction_forecast",
    "research_agenda_prioritization": "agenda_prioritization",
    "comparative_opportunity_ranking": "opportunity_prioritization",
    "venue_strategy_planning": "venue_aligned_planning",
    "venue_specific_direction_positioning": "venue_aligned_direction_forecast",
    "bottleneck_and_opportunity_half_year": "bottleneck_and_opportunity",
    "bottleneck_and_opportunity_quarter": "bottleneck_and_opportunity",
    "direction_forecast_half_year": "direction_forecast",
    "direction_forecast_quarter": "direction_forecast",
    "agenda_prioritization": "agenda_prioritization",
    "opportunity_prioritization": "opportunity_prioritization",
    "venue_aligned_planning": "venue_aligned_planning",
    "venue_aligned_direction_forecast": "venue_aligned_direction_forecast",
}

ORIGINAL_SUBTYPE_MAP = {
    "pageindex_grounded_bottleneck": "bottleneck_and_opportunity",
    "q1_pageindex_grounded_bottleneck": "bottleneck_and_opportunity",
    "chain_terminal_forecast": "direction_forecast",
    "q1_terminal_forecast": "direction_forecast",
    "agenda_priority_selection": "agenda_prioritization",
    "comparative_opportunity_prioritization": "opportunity_prioritization",
    "venue_targeted_planning": "venue_aligned_planning",
    "venue_aware_direction_forecast": "venue_aligned_direction_forecast",
}

TARGET_VERSION = "2026-04-23-taxonomy-v3"

SCRIPT_FILES = [
    ROOT / "scripts" / "build_task_refined_pilot_20260423.py",
    ROOT / "scripts" / "build_task_refined_batch2_20260423.py",
    ROOT / "scripts" / "build_task_refined_full_20260423.py",
]


def relabel_task_file():
    before = Counter()
    after = Counter()
    rows = []
    for raw in TASK_PATH.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        before[row["subtype"]] += 1
        row["subtype"] = TASK_SUBTYPE_MAP[row["subtype"]]
        row["subtype_taxonomy_version"] = TARGET_VERSION
        after[row["subtype"]] += 1
        rows.append(row)
    TASK_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return before, after


def patch_builder_scripts():
    for path in SCRIPT_FILES:
        text = path.read_text()
        for old, new in ORIGINAL_SUBTYPE_MAP.items():
            text = text.replace(f"\"{old}\": \"{old}\"", f"\"{old}\": \"{new}\"")
            text = text.replace(f"'{old}': '{old}'", f"'{old}': '{new}'")

        for old, new in [
            ("historical_bottleneck_to_half_year_opportunity", "bottleneck_and_opportunity"),
            ("historical_bottleneck_to_quarter_opportunity", "bottleneck_and_opportunity"),
            ("next_direction_and_trajectory_forecast", "direction_forecast"),
            ("quarter_ahead_direction_and_trajectory_forecast", "direction_forecast"),
            ("research_agenda_prioritization", "agenda_prioritization"),
            ("comparative_opportunity_ranking", "opportunity_prioritization"),
            ("venue_strategy_planning", "venue_aligned_planning"),
            ("venue_specific_direction_positioning", "venue_aligned_direction_forecast"),
        ]:
            text = text.replace(f"\"{old}\"", f"\"{new}\"")

        path.write_text(text)


def write_log(before, after):
    lines = [
        "# Benchmark Refine Log 20260423 Round 12",
        "",
        "## Scope",
        "",
        "- This round redesigns the subtype taxonomy instead of only renaming labels.",
        "- Principle: `family` and `horizon` already exist as separate fields, so `subtype` should only express the task form.",
        "- Result: subtype labels are shorter, more natural, and orthogonal to horizon.",
        "",
        "## New Subtype Definitions",
        "",
        "- `bottleneck_and_opportunity`: identify a historically grounded bottleneck and infer the downstream opportunity.",
        "- `direction_forecast`: predict the next concrete research direction and/or near-term trajectory.",
        "- `agenda_prioritization`: prioritize a research agenda or ordered strategic plan.",
        "- `opportunity_prioritization`: compare and rank candidate opportunities or directions.",
        "- `venue_aligned_planning`: build a venue-oriented research plan or priority ordering.",
        "- `venue_aligned_direction_forecast`: forecast a likely next direction together with venue fit.",
        "",
        "## Mapping From Previous Labels",
        "",
    ]

    for old, new in [
        ("historical_bottleneck_to_half_year_opportunity", "bottleneck_and_opportunity"),
        ("historical_bottleneck_to_quarter_opportunity", "bottleneck_and_opportunity"),
        ("next_direction_and_trajectory_forecast", "direction_forecast"),
        ("quarter_ahead_direction_and_trajectory_forecast", "direction_forecast"),
        ("research_agenda_prioritization", "agenda_prioritization"),
        ("comparative_opportunity_ranking", "opportunity_prioritization"),
        ("venue_strategy_planning", "venue_aligned_planning"),
        ("venue_specific_direction_positioning", "venue_aligned_direction_forecast"),
    ]:
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
            "- `docs/benchmark_refine_log_20260423_round12.md`",
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
    before, after = relabel_task_file()
    patch_builder_scripts()
    write_log(before, after)


if __name__ == "__main__":
    main()
