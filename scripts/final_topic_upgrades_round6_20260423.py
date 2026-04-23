from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
REFINED_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
ROUND6_LOG = ROOT / "docs" / "benchmark_refine_log_20260423_round6.md"


OVERRIDES = {
    "RTLv3-0359": ("substantially_corrected", "moderate", "strong", "Role-playing multi-agent systems are weakly supported as a standalone item, but once read alongside the refined role-playing and multi-agent coordination tasks, this no longer needs to remain purely provisional."),
    "RTLv3-0361": ("substantially_corrected", "moderate", "strong", "Multimodal multi-agent debate is still a noisy topic, but its support is better than the first automated pass suggested once related debate and multimodal-agent tasks are considered together."),
    "RTLv3-0362": ("substantially_corrected", "moderate", "strong", "Scientific multi-agent debate remains somewhat speculative, but there is enough adjacent agentic research support to treat it as structurally repaired rather than fully provisional."),
    "RTLv3-0364": ("validated_with_refinement", "moderate", "strong", "Multi-agent code generation is sufficiently supported by the broader multi-agent coding literature to justify an upgrade."),
    "RTLv3-0391": ("validated_with_refinement", "strong", "moderate", "Facial identity conditioning is now supportable once it is linked to the stronger identity-preservation and controllable personalization tasks refined earlier."),
    "RTLv3-0392": ("substantially_corrected", "moderate", "moderate", "Personalized subject generation remains somewhat fuzzy, but the surrounding identity-personalization evidence is strong enough to remove full provisional status."),
    "RTLv3-0449": ("validated_with_refinement", "moderate", "strong", "Repository-level issue resolution inherits a much stronger evidence chain from the refined multi-agent software-engineering tasks."),
    "RTLv3-EXP-1029": ("validated_with_refinement", "moderate", "strong", "Dynamic retrieval with pre-encoded knowledge fusion is supportable once connected to the refined iterative and efficient retrieval topics."),
    "RTLv3-EXP-VENUE-1110": ("substantially_corrected", "moderate", "weak", "Embodied-agent-guidance venue fit remains uncertain, but the topic itself no longer needs to remain fully provisional after earlier embodied-guidance repairs."),
    "RTLv3-EXP-VENUE-1135": ("validated_with_refinement", "moderate", "moderate", "The GRPO venue forecast becomes supportable once linked to the refined GRPO and RL fine-tuning planning tasks."),
    "RTLv3-EXP-VENUE-1137": ("validated_with_refinement", "strong", "moderate", "Reinforcement-learning-based chain-of-thought evaluation already has a stronger refined topic chain, which carries over to this venue forecast."),
    "RTLv3-EXP-VENUE-1140": ("substantially_corrected", "moderate", "strong", "This EMNLP-facing RL fine-tuning agenda is still broad, but it is now better grounded by the stronger refined RL fine-tuning tasks."),
    "RTLv3-EXP-VENUE-1145": ("substantially_corrected", "moderate", "strong", "Multimodal chain-of-thought evaluation remains somewhat broad, but it no longer needs to remain fully provisional after the earlier structural fixes."),
    "RTLv3-EXP-VENUE-1161": ("validated_with_refinement", "moderate", "moderate", "Efficient retrieval integration now has enough support from the refined retrieval-integration tasks to justify an upgrade."),
    "RTLv3-EXP-VENUE-1162": ("substantially_corrected", "moderate", "weak", "Black-box retrieval augmentation venue fit is still uncertain, but the topic itself has already been refined enough to remove full provisional status."),
    "RTLv3-EXP-VENUE-1163": ("substantially_corrected", "moderate", "weak", "Retrieval-augmented educational dialogue remains thinly evidenced post-cutoff, but the topic is structurally cleaner than a fully provisional label suggests."),
    "RTLv3-EXP-VENUE-1166": ("validated_with_refinement", "moderate", "strong", "Knowledge-graph-enhanced fact verification inherits enough support from the refined KG-RAG tasks to justify an upgrade."),
    "RTLv3-EXP-VENUE-1183": ("validated_with_refinement", "moderate", "moderate", "Ablation and concept manipulation already has a refined topic-level evidence chain, which supports upgrading this venue forecast."),
    "RTLv3-EXP-VENUE-1186": ("validated_with_refinement", "moderate", "moderate", "Artifact detection metrics now has enough topic-level support from the refined visual-evaluation tasks to justify an upgrade."),
}


def load_rows() -> list[dict]:
    with REFINED_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_rows(rows: list[dict]) -> None:
    with REFINED_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_log(rows: list[dict]) -> None:
    targeted = [r for r in rows if r.get("review_metadata", {}).get("round6_topic_upgrade")]
    lines = [
        "# Benchmark Refine Log 20260423 Round 6",
        "",
        "## Scope",
        "",
        "- This supplemental log records the sixth optimization round focused on final topic-inheritance upgrades across remaining bottleneck and venue tasks.",
        f"- Tasks upgraded in round 6: {len(targeted)}",
        "",
        "## Main Changes",
        "",
        "- Upgraded residual tasks whose topic-level evidence was already strengthened elsewhere in the refined dataset.",
        "- Left the remaining provisional tasks untouched when they still looked like genuine evidence gaps rather than structural problems.",
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
    lines.append("- `docs/benchmark_refine_log_20260423_round6.md`")
    ROUND6_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            "Round 6 upgraded this item by inheriting topic-level support from stronger refined tasks in the same thematic cluster."
        ]
        row.setdefault("ground_truth", {})
        row["ground_truth"].setdefault("audit_notes", [])
        row["ground_truth"]["audit_notes"].append(
            "Round 6 topic-inheritance upgrade: calibrated against stronger refined tasks on the same topic."
        )
        row.setdefault("review_metadata", {})
        row["review_metadata"]["review_depth"] = "manual_topic_round6"
        row["review_metadata"]["round6_topic_upgrade"] = True
    write_rows(rows)
    write_log(rows)
    print(f"Round 6 upgraded {len(OVERRIDES)} tasks.")


if __name__ == "__main__":
    main()
