from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.eval_v3 import build_hidden_eval_v3_row
from researchworld.eval_v3_1 import build_hidden_eval_v3_1_row


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "over",
    "the",
    "to",
    "via",
    "with",
}
GENERIC_MODIFIERS = {
    "analysis",
    "analyses",
    "approach",
    "approaches",
    "architecture",
    "architectures",
    "benchmark",
    "benchmarks",
    "driven",
    "evaluation",
    "evaluations",
    "framework",
    "frameworks",
    "guided",
    "improved",
    "improving",
    "method",
    "methods",
    "model",
    "models",
    "novel",
    "protocol",
    "protocols",
    "robust",
    "scalable",
    "study",
    "studies",
    "system",
    "systems",
}
STRING_KEYS = ("display_name", "name", "direction", "topic", "topic_title", "title")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_hidden_v3_manifest(release_dir: Path, rows: Sequence[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    family_counts = Counter()
    domain_counts = Counter()
    claim_type_counts = Counter()
    slot_field_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row.get("family") or "")
        domain = str(row.get("domain") or "")
        family_counts[family] += 1
        domain_counts[domain] += 1
        for claim in row.get("claim_bank") or []:
            claim_type_counts[str(claim.get("claim_type") or "")] += 1
        for key in (row.get("slot_targets") or {}).keys():
            slot_field_counts[family][str(key)] += 1
    return {
        "release_dir": str(release_dir),
        "output": str(output_path),
        "task_count": len(rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "claim_type_counts": dict(claim_type_counts),
        "slot_field_coverage": {family: dict(counter) for family, counter in slot_field_counts.items()},
        "notes": [
            "v3 hidden eval adds slot_targets, claim_bank, and judge_profile on top of hidden eval.",
            "This manifest was regenerated after future novelty post-processing.",
        ],
    }


def build_hidden_v31_manifest(release_dir: Path, rows: Sequence[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
    family_counts = Counter()
    domain_counts = Counter()
    component_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    future_unit_counts = Counter()
    for row in rows:
        family = str(row.get("family") or "")
        domain = str(row.get("domain") or "")
        family_counts[family] += 1
        domain_counts[domain] += 1
        for component in ((row.get("component_targets") or {}).get("components") or []):
            component_counts[family][str(component.get("name") or "")] += 1
        future_unit_counts[family] += len(((row.get("future_alignment_targets") or {}).get("alignment_units") or []))
    return {
        "release_dir": str(release_dir),
        "output": str(output_path),
        "task_count": len(rows),
        "family_counts": dict(family_counts),
        "domain_counts": dict(domain_counts),
        "component_coverage": {family: dict(counter) for family, counter in component_counts.items()},
        "future_unit_counts": dict(future_unit_counts),
        "notes": [
            "v3.1 adds component_targets and future_alignment_targets on top of v3 hidden eval.",
            "This manifest was regenerated after future novelty post-processing.",
        ],
    }


def normalize_ws(text: Any) -> str:
    return " ".join(str(text or "").replace("_", " ").replace("-", " ").split()).strip()


def stem_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def tokenize(text: Any) -> List[str]:
    return [stem_token(token.lower()) for token in TOKEN_RE.findall(normalize_ws(text))]


def label_from_item(item: Any) -> str:
    if isinstance(item, str):
        return normalize_ws(item)
    if isinstance(item, dict):
        for key in STRING_KEYS:
            value = normalize_ws(item.get(key))
            if value:
                return value
    return ""


def ensure_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def full_tokens(text: Any) -> List[str]:
    return [token for token in tokenize(text) if token not in STOPWORDS]


def reduced_tokens(text: Any) -> List[str]:
    return [token for token in full_tokens(text) if token not in GENERIC_MODIFIERS]


def token_set(text: Any, *, reduced: bool) -> set[str]:
    values = reduced_tokens(text) if reduced else full_tokens(text)
    return set(values)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    na = normalize_ws(a).lower()
    nb = normalize_ws(b).lower()
    if na == nb:
        return 1.0
    full_a = token_set(a, reduced=False)
    full_b = token_set(b, reduced=False)
    reduced_a = token_set(a, reduced=True)
    reduced_b = token_set(b, reduced=True)
    return max(
        jaccard(full_a, full_b),
        jaccard(reduced_a, reduced_b),
        containment(reduced_a, reduced_b),
        containment(full_a, full_b) * 0.9,
    )


def is_duplicate_like(candidate: str, anchor: str) -> Tuple[bool, float, str]:
    cand = normalize_ws(candidate)
    base = normalize_ws(anchor)
    if not cand or not base:
        return False, 0.0, ""
    if cand.lower() == base.lower():
        return True, 1.0, "exact_match"
    reduced_c = token_set(cand, reduced=True)
    reduced_b = token_set(base, reduced=True)
    full_c = token_set(cand, reduced=False)
    full_b = token_set(base, reduced=False)
    if reduced_c and reduced_b and reduced_c == reduced_b:
        return True, 0.96, "same_reduced_tokens"
    jac_reduced = jaccard(reduced_c, reduced_b)
    cont_reduced = containment(reduced_c, reduced_b)
    jac_full = jaccard(full_c, full_b)
    if jac_reduced >= 0.82:
        return True, jac_reduced, "high_reduced_jaccard"
    if cont_reduced >= 0.88 and max(len(reduced_c), len(reduced_b)) - min(len(reduced_c), len(reduced_b)) <= 1:
        return True, cont_reduced, "near_subset_reduced"
    if jac_full >= 0.9:
        return True, jac_full, "high_full_jaccard"
    return False, max(jac_reduced, cont_reduced, jac_full), ""


def dedupe_keep_order(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        norm = normalize_ws(value)
        key = norm.lower()
        if not norm or key in seen:
            continue
        seen.add(key)
        out.append(norm)
    return out


def extract_history_labels(hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    public = ensure_dict(hidden_row.get("public_metadata") or trace_row.get("public_metadata"))
    gt = ensure_dict(trace_row.get("ground_truth"))
    support = ensure_dict(trace_row.get("support_context"))

    for key in ("topic", "topic_title"):
        labels.append(public.get(key) or "")
    for item in support.get("top_limitations") or []:
        labels.append(label_from_item(item))
    for item in support.get("top_future_work") or []:
        labels.append(label_from_item(item))
    for item in support.get("candidate_directions") or []:
        labels.append(label_from_item(item))
    for item in support.get("history_chain") or []:
        labels.append(label_from_item(item))
    for item in gt.get("historical_limitation_signals") or []:
        labels.append(label_from_item(item))
    for item in gt.get("historical_future_work_signals") or []:
        labels.append(label_from_item(item))
    return dedupe_keep_order(labels)


def extract_future_labels(hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    public = ensure_dict(hidden_row.get("public_metadata") or trace_row.get("public_metadata"))
    gt = ensure_dict(trace_row.get("ground_truth"))
    for key in (
        "future_descendants",
        "emergent_descendants",
        "realized_opportunity_directions",
        "direction_records",
    ):
        for item in gt.get(key) or []:
            labels.append(label_from_item(item))
    for item in public.get("future_themes") or []:
        labels.append(label_from_item(item))
    return dedupe_keep_order(labels)


def best_match(candidate: str, bank: Sequence[str]) -> Tuple[str, float, str]:
    best_label = ""
    best_score = 0.0
    best_reason = ""
    for anchor in bank:
        dup, score, reason = is_duplicate_like(candidate, anchor)
        if score > best_score:
            best_label = anchor
            best_score = score
            best_reason = reason if dup else ""
    return best_label, best_score, best_reason


def prune_future_labels(history_labels: Sequence[str], future_labels: Sequence[str]) -> Dict[str, Any]:
    kept: List[str] = []
    removed: List[Dict[str, Any]] = []
    fallback_kept = False

    for label in future_labels:
        hist_match, hist_score, hist_reason = best_match(label, history_labels)
        if hist_reason:
            removed.append(
                {
                    "label": label,
                    "reason": "duplicate_like_history",
                    "matched_history_label": hist_match,
                    "match_score": round(hist_score, 4),
                    "match_rule": hist_reason,
                }
            )
            continue
        future_match, future_score, future_reason = best_match(label, kept)
        if future_reason and future_score >= 0.86:
            removed.append(
                {
                    "label": label,
                    "reason": "duplicate_like_future_peer",
                    "matched_future_label": future_match,
                    "match_score": round(future_score, 4),
                    "match_rule": future_reason,
                }
            )
            continue
        kept.append(label)

    if not kept and future_labels:
        scored = []
        for label in future_labels:
            _, score, _ = best_match(label, history_labels)
            scored.append((score, label))
        scored.sort(key=lambda item: (item[0], len(item[1])))
        kept = [scored[0][1]]
        fallback_kept = True

    return {
        "kept_labels": kept,
        "removed_labels": removed,
        "fallback_kept": fallback_kept,
    }


def filter_labeled_list(items: Any, kept_labels: set[str]) -> Any:
    if not isinstance(items, list):
        return items
    out = []
    for item in items:
        label = label_from_item(item)
        if label and label in kept_labels:
            out.append(item)
    return out


def keep_future_paper(title: str, kept_labels: Sequence[str], history_labels: Sequence[str]) -> Tuple[bool, Dict[str, Any]]:
    title_norm = normalize_ws(title)
    if not title_norm:
        return False, {"future_score": 0.0, "history_score": 0.0}
    future_score = max((similarity(title_norm, label) for label in kept_labels), default=0.0)
    history_score = max((similarity(title_norm, label) for label in history_labels), default=0.0)
    keep = bool(kept_labels) and (future_score >= 0.18) and (future_score >= history_score + 0.05)
    return keep, {
        "future_score": round(future_score, 4),
        "history_score": round(history_score, 4),
    }


def filter_future_papers(items: Any, kept_labels: Sequence[str], history_labels: Sequence[str]) -> Tuple[Any, Dict[str, Any]]:
    if not isinstance(items, list):
        return items, {"removed_count": 0, "fallback_kept": False}
    kept = []
    removed = []
    scored_rows = []
    for item in items:
        title = label_from_item(item)
        keep, scores = keep_future_paper(title, kept_labels, history_labels)
        scored_rows.append((scores["future_score"] - scores["history_score"], item, scores))
        if keep:
            kept.append(item)
        else:
            removed.append({"title": title, **scores})
    fallback_kept = False
    if items and not kept and kept_labels:
        scored_rows.sort(key=lambda row: row[0], reverse=True)
        best_delta, best_item, scores = scored_rows[0]
        if best_delta > -0.05:
            kept = [best_item]
            fallback_kept = True
            removed = [row for row in removed if row.get("title") != label_from_item(best_item)]
    return kept, {
        "removed_count": len(removed),
        "removed_examples": removed[:5],
        "fallback_kept": fallback_kept,
    }


def patch_rows(
    hidden_row: Dict[str, Any],
    trace_row: Dict[str, Any],
    internal_row: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]], Dict[str, Any]]:
    hidden = json.loads(json.dumps(hidden_row, ensure_ascii=False))
    trace = json.loads(json.dumps(trace_row, ensure_ascii=False))
    internal = json.loads(json.dumps(internal_row, ensure_ascii=False)) if internal_row is not None else None

    history_labels = extract_history_labels(hidden, trace)
    future_labels = extract_future_labels(hidden, trace)
    decision = prune_future_labels(history_labels, future_labels)
    kept_labels = set(decision["kept_labels"])

    hidden_public = ensure_dict(hidden.get("public_metadata"))
    hidden["public_metadata"] = hidden_public
    trace_public = ensure_dict(trace.get("public_metadata"))
    trace["public_metadata"] = trace_public
    hidden_public["future_themes"] = list(decision["kept_labels"])
    trace_public["future_themes"] = list(decision["kept_labels"])
    if internal is not None:
        internal_public = ensure_dict(internal.get("public_metadata"))
        internal["public_metadata"] = internal_public
        internal_public["future_themes"] = list(decision["kept_labels"])

    for row in [hidden, trace, internal]:
        if row is None:
            continue
        gt = ensure_dict(row.get("ground_truth"))
        for key in ("future_descendants", "emergent_descendants", "realized_opportunity_directions", "direction_records"):
            if key in gt:
                gt[key] = filter_labeled_list(gt.get(key), kept_labels)
        ref = gt.get("reference_papers") or {}
        if isinstance(ref, dict):
            for key in ("future_q4", "future_q1"):
                if key in ref:
                    filtered, _ = filter_future_papers(ref.get(key), decision["kept_labels"], history_labels)
                    ref[key] = filtered
        row["ground_truth"] = gt

    support = ensure_dict(trace.get("support_context"))
    support_report: Dict[str, Any] = {}
    if "future_validation_set" in support:
        filtered, support_report = filter_future_papers(
            support.get("future_validation_set"),
            decision["kept_labels"],
            history_labels,
        )
        support["future_validation_set"] = filtered
    trace["support_context"] = support
    if internal is not None and isinstance(internal.get("support_context"), dict) and "future_validation_set" in internal["support_context"]:
        filtered, _ = filter_future_papers(
            internal["support_context"].get("future_validation_set"),
            decision["kept_labels"],
            history_labels,
        )
        internal["support_context"]["future_validation_set"] = filtered

    report = {
        "task_id": hidden.get("task_id"),
        "family": hidden.get("family"),
        "topic": hidden_public.get("topic") or trace_public.get("topic") or "",
        "history_label_count": len(history_labels),
        "original_future_labels": future_labels,
        "kept_future_labels": decision["kept_labels"],
        "removed_future_labels": decision["removed_labels"],
        "fallback_kept_future_label": decision["fallback_kept"],
        "support_future_validation_update": support_report,
    }
    return hidden, trace, internal, report


def copy_release_file(src_dir: Path, dst_dir: Path, name: str) -> None:
    src = src_dir / name
    if src.exists():
        shutil.copy2(src, dst_dir / name)


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process benchmark GT to remove post-cutoff future labels that merely restate pre-cutoff ideas.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_by_id = {row["task_id"]: row for row in iter_jsonl(release_dir / "tasks_hidden_eval.jsonl")}
    trace_by_id = {row["task_id"]: row for row in iter_jsonl(release_dir / "tasks_build_trace.jsonl")}
    internal_by_id = {}
    internal_path = release_dir / "tasks_internal_full.jsonl"
    if internal_path.exists():
        internal_by_id = {row["task_id"]: row for row in iter_jsonl(internal_path)}

    patched_hidden: List[Dict[str, Any]] = []
    patched_trace: List[Dict[str, Any]] = []
    patched_internal: List[Dict[str, Any]] = []
    hidden_v3_rows: List[Dict[str, Any]] = []
    hidden_v31_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    summary = Counter()

    for task_id in sorted(hidden_by_id):
        hidden = hidden_by_id[task_id]
        trace = trace_by_id[task_id]
        internal = internal_by_id.get(task_id)
        patched_h, patched_t, patched_i, report = patch_rows(hidden, trace, internal)
        patched_hidden.append(patched_h)
        patched_trace.append(patched_t)
        if patched_i is not None:
            patched_internal.append(patched_i)
        hidden_v3 = build_hidden_eval_v3_row(patched_h, patched_t)
        hidden_v31 = build_hidden_eval_v3_1_row(hidden_v3, patched_t)
        hidden_v3_rows.append(hidden_v3)
        hidden_v31_rows.append(hidden_v31)
        report_rows.append(report)

        summary["task_count"] += 1
        summary["kept_future_labels"] += len(report["kept_future_labels"])
        summary["removed_future_labels"] += len(report["removed_future_labels"])
        if report["removed_future_labels"]:
            summary["tasks_touched"] += 1
        if report["fallback_kept_future_label"]:
            summary["fallback_tasks"] += 1
        if (report.get("support_future_validation_update") or {}).get("removed_count"):
            summary["tasks_with_future_paper_prune"] += 1

    copy_release_file(release_dir, output_dir, "tasks.jsonl")
    copy_release_file(release_dir, output_dir, "task_ids.txt")
    copy_release_file(release_dir, output_dir, "manifest.json")
    dump_jsonl(output_dir / "tasks_hidden_eval.jsonl", patched_hidden)
    dump_jsonl(output_dir / "tasks_build_trace.jsonl", patched_trace)
    if patched_internal:
        dump_jsonl(output_dir / "tasks_internal_full.jsonl", patched_internal)
    dump_jsonl(output_dir / "tasks_hidden_eval_v3.jsonl", hidden_v3_rows)
    dump_jsonl(output_dir / "tasks_hidden_eval_v3_1.jsonl", hidden_v31_rows)
    dump_json(
        output_dir / "tasks_hidden_eval_v3_manifest.json",
        build_hidden_v3_manifest(release_dir, hidden_v3_rows, output_dir / "tasks_hidden_eval_v3.jsonl"),
    )
    dump_json(
        output_dir / "tasks_hidden_eval_v3_1_manifest.json",
        build_hidden_v31_manifest(release_dir, hidden_v31_rows, output_dir / "tasks_hidden_eval_v3_1.jsonl"),
    )

    for name in ("kb", "future_kb"):
        src = release_dir / name
        if src.exists():
            ensure_symlink(src, output_dir / name)

    manifest = {}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    manifest["release_name"] = output_dir.name
    manifest["source_release"] = str(release_dir)
    manifest["future_novelty_postprocess"] = {
        "script": str(Path(__file__).name),
        "task_count": summary["task_count"],
        "tasks_touched": summary["tasks_touched"],
        "removed_future_labels": summary["removed_future_labels"],
        "fallback_tasks": summary["fallback_tasks"],
        "tasks_with_future_paper_prune": summary["tasks_with_future_paper_prune"],
        "notes": [
            "This pass removes post-cutoff future labels that heuristically restate pre-cutoff ideas.",
            "It keeps statistical GT untouched and rewrites semantic future GT only.",
            "If all future labels would be removed for a task, the least-history-overlapping label is kept as a fallback.",
        ],
    }
    dump_json(manifest_path, manifest)

    report = {
        "release_dir": str(release_dir),
        "output_dir": str(output_dir),
        **{key: int(value) for key, value in summary.items()},
        "examples": report_rows[:12],
    }
    dump_json(output_dir / "future_novelty_dedup_report.json", report)
    dump_json(output_dir / "future_novelty_dedup_rows.json", {"rows": report_rows})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
