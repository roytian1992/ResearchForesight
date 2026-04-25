from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from researchworld.corpus import iter_jsonl


REFINED_TASK_FILENAME = "task_refined.jsonl"
LEGACY_PUBLIC_TASK_FILENAME = "tasks.jsonl"
LEGACY_HIDDEN_FILENAMES = {
    "base": "tasks_hidden_eval.jsonl",
    "v3": "tasks_hidden_eval_v3.jsonl",
    "v3_1": "tasks_hidden_eval_v3_1.jsonl",
}

PUBLIC_TASK_KEYS = [
    "schema_version",
    "task_id",
    "family",
    "subtype",
    "domain",
    "horizon",
    "title",
    "question",
    "time_cutoff",
    "deliverable_spec",
    "answer_contract",
]

EVAL_TARGET_KEYS = [
    "slot_targets",
    "claim_bank",
    "component_targets",
    "future_alignment_targets",
    "temporal_policy",
]


def uses_refined_release(release_dir: Path) -> bool:
    return (release_dir / REFINED_TASK_FILENAME).exists()


def _split_future_window_label(label: Any) -> Tuple[str, str]:
    value = str(label or "").strip()
    if "_to_" not in value:
        return "", ""
    start, end = value.split("_to_", 1)
    return start.strip(), end.strip()


def _default_judge_profile(family: str) -> Dict[str, Any]:
    from researchworld.eval_v3 import FAMILY_SCORE_WEIGHTS, JUDGE_DIMENSIONS

    return {
        "mode": "benchmark_aware_structured_judge",
        "dimensions": list(JUDGE_DIMENSIONS.get(family) or []),
        "score_weights": dict(FAMILY_SCORE_WEIGHTS.get(family) or {}),
    }


def _derive_temporal_policy(row: Dict[str, Any]) -> Dict[str, Any]:
    eval_targets = row.get("eval_targets") or {}
    temporal_policy = dict((eval_targets.get("temporal_policy") or {}))
    time_context = (row.get("trace") or {}).get("time_context") or {}
    future_label = (
        str(temporal_policy.get("future_window_label") or "").strip()
        or str(time_context.get("future_window") or "").strip()
        or str(((time_context.get("future_windows") or {}).get("halfyear_2025q4_2026q1")) or "").strip()
    )
    future_start, future_end = _split_future_window_label(future_label)
    if not temporal_policy.get("history_cutoff"):
        temporal_policy["history_cutoff"] = time_context.get("history_end") or row.get("time_cutoff")
    if not temporal_policy.get("history_slice"):
        temporal_policy["history_slice"] = time_context.get("history_structure_slice")
    if future_label and not temporal_policy.get("future_window_label"):
        temporal_policy["future_window_label"] = future_label
    if future_start and not temporal_policy.get("future_start"):
        temporal_policy["future_start"] = future_start
    if future_end and not temporal_policy.get("future_end"):
        temporal_policy["future_end"] = future_end
    temporal_policy.setdefault("history_evidence_only_for_reasoning", True)
    temporal_policy.setdefault("future_window_hidden_for_fact_verification", True)
    return temporal_policy


def normalize_refined_eval_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    eval_targets = row.get("eval_targets") or {}
    for key in EVAL_TARGET_KEYS:
        if key not in normalized and key in eval_targets:
            normalized[key] = eval_targets.get(key)
    if not normalized.get("temporal_policy"):
        normalized["temporal_policy"] = _derive_temporal_policy(row)
    if not normalized.get("judge_profile"):
        normalized["judge_profile"] = _default_judge_profile(str(normalized.get("family") or ""))
    return normalized


def build_public_task_view(row: Dict[str, Any]) -> Dict[str, Any]:
    public_row = {key: row.get(key) for key in PUBLIC_TASK_KEYS if key in row}
    if "deliverable_spec" in public_row and "expected_output" not in public_row:
        public_row["expected_output"] = public_row.get("deliverable_spec")
    return public_row


def load_refined_rows(release_dir: Path) -> List[Dict[str, Any]]:
    return [normalize_refined_eval_row(row) for row in iter_jsonl(release_dir / REFINED_TASK_FILENAME)]


def load_release_public_tasks(release_dir: Path) -> List[Dict[str, Any]]:
    if uses_refined_release(release_dir):
        return [build_public_task_view(row) for row in load_refined_rows(release_dir)]
    return list(iter_jsonl(release_dir / LEGACY_PUBLIC_TASK_FILENAME))


def load_release_public_by_id(release_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("task_id") or ""): row for row in load_release_public_tasks(release_dir)}


def load_release_eval_rows(release_dir: Path, *, variant: str = "v3_1") -> List[Dict[str, Any]]:
    if uses_refined_release(release_dir):
        return load_refined_rows(release_dir)
    filename = LEGACY_HIDDEN_FILENAMES.get(variant, LEGACY_HIDDEN_FILENAMES["v3_1"])
    return list(iter_jsonl(release_dir / filename))


def load_release_eval_by_id(release_dir: Path, *, variant: str = "v3_1") -> Dict[str, Dict[str, Any]]:
    return {str(row.get("task_id") or ""): row for row in load_release_eval_rows(release_dir, variant=variant)}


def load_release_task_views(release_dir: Path, *, eval_variant: str = "v3_1") -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    public_by_id = load_release_public_by_id(release_dir)
    eval_by_id = load_release_eval_by_id(release_dir, variant=eval_variant)
    return public_by_id, eval_by_id
