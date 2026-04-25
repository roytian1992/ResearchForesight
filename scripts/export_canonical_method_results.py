from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results"
DEFAULT_RELEASE_DIR = (
    ROOT
    / "data"
    / "releases"
    / "researchforesight_refined_422"
)

FAMILY_TO_AUX_NAME = {
    "bottleneck_opportunity_discovery": "Opportunity Grounding",
    "direction_forecasting": "Forecast Grounding",
    "strategic_research_planning": "Strategic Execution Grounding",
    "venue_aware_research_positioning": "Venue Positioning Grounding",
}

FAMILY_TO_AUX_SCORE_KEY = {
    "bottleneck_opportunity_discovery": "opportunity_grounding_score",
    "direction_forecasting": "forecast_grounding_score",
    "strategic_research_planning": "strategic_execution_grounding_score",
    "venue_aware_research_positioning": "venue_positioning_grounding_score",
}


def _jsonl(path: str) -> str:
    return str(Path(path))


CANONICAL_METHOD_SPECS: List[Dict[str, Any]] = [
    {
        "method_key": "native_llm",
        "method_name": "Native LLM",
        "code_files": [
            "scripts/run_nonagent_baseline.py",
            "src/researchworld/baseline_runner.py",
            "src/researchworld/offline_kb.py",
            "src/researchworld/llm.py",
        ],
        "bundles": {
            "v31": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "native_llm_experiment100_finalmetrics_20260418"
                        / "eval_v31"
                        / "results_eval_v3_1.jsonl"
                    ),
                    "mode": "replace_row",
                }
            ],
            "v4": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "native_llm_experiment100_finalmetrics_20260418"
                        / "eval_v4"
                        / "results_eval_v4.jsonl"
                    ),
                    "mode": "replace_row",
                }
            ],
            "aux": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "native_llm_experiment100_finalmetrics_20260418"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "native_llm_experiment100_aux_venueprior_20260420"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "families": ["venue_aware_research_positioning"],
                    "mode": "replace_row",
                },
            ],
        },
    },
    {
        "method_key": "hybrid_rag",
        "method_name": "Hybrid RAG",
        "code_files": [
            "scripts/run_nonagent_baseline.py",
            "src/researchworld/baseline_runner.py",
            "src/researchworld/offline_kb.py",
            "src/researchworld/llm.py",
        ],
        "bundles": {
            "v31": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "hybrid_rag_experiment100_evidenceexp_v1_parallel4"
                        / "eval_v31"
                        / "results_eval_v3_1.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT / "hybrid_rag_bd50_v31_lenient_20260418" / "results_eval_v3_1.jsonl"
                    ),
                    "families": [
                        "bottleneck_opportunity_discovery",
                        "direction_forecasting",
                    ],
                    "mode": "patch_fact_only",
                },
            ],
            "v4": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "hybrid_rag_experiment100_evidenceexp_v1_parallel4"
                        / "eval_v4"
                        / "results_eval_v4.jsonl"
                    ),
                    "mode": "replace_row",
                }
            ],
            "aux": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "hybrid_rag_experiment100_evidenceexp_v1_parallel4"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT / "hybrid_rag_bd50_aux_lenient_20260418" / "results_eval_aux.jsonl"
                    ),
                    "families": [
                        "bottleneck_opportunity_discovery",
                        "direction_forecasting",
                    ],
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "hybrid_rag_experiment100_strategic25_aux_reval_20260418"
                        / "results_eval_aux.jsonl"
                    ),
                    "families": ["strategic_research_planning"],
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "hybrid_rag_experiment100_aux_venueprior_20260420"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "families": ["venue_aware_research_positioning"],
                    "mode": "replace_row",
                },
            ],
        },
    },
    {
        "method_key": "coi",
        "method_name": "CoI",
        "code_files": [
            "scripts/run_coi_agent_offline.py",
            "scripts/run_coi_agent_offline_sharded.py",
            "src/researchworld/coi_agent_offline.py",
            "src/researchworld/coi_offline_retrieval.py",
            "src/researchworld/fulltext_cache.py",
            "src/researchworld/offline_kb.py",
            "src/researchworld/llm.py",
            "src/researchworld/research_arc_kb.py",
            "src/researchworld/research_arc_v2.py",
            "src/researchworld/retrieval_fusion.py",
        ],
        "bundles": {
            "v31": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "coi_experiment100_focusfixv3_20260420_parallel4"
                        / "final_metrics"
                        / "eval_v31"
                        / "results_eval_v3_1.jsonl"
                    ),
                    "mode": "replace_row",
                }
            ],
            "v4": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "coi_experiment100_focusfixv3_20260420_parallel4"
                        / "final_metrics"
                        / "eval_v4"
                        / "results_eval_v4.jsonl"
                    ),
                    "mode": "replace_row",
                }
            ],
            "aux": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "coi_experiment100_focusfixv3_20260420_parallel4"
                        / "final_metrics"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "mode": "replace_row",
                }
            ],
        },
    },
    {
        "method_key": "researchagent",
        "method_name": "ResearchAgent",
        "code_files": [
            "scripts/run_researchagent_offline.py",
            "src/researchworld/researchagent_offline.py",
            "src/researchworld/researchagent_prompts.py",
            "src/researchworld/offline_kb.py",
            "src/researchworld/llm.py",
            "src/researchworld/research_arc_v2.py",
            "src/researchworld/retrieval_fusion.py",
        ],
        "bundles": {
            "v31": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "researchagent_experiment100_qwen8002_20260420_parallel4"
                        / "final_metrics"
                        / "eval_v31"
                        / "results_eval_v3_1.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "researchagent_experiment100_forecast_venue50_renderfix_20260421_parallel4"
                        / "final_metrics"
                        / "eval_v31"
                        / "results_eval_v3_1.jsonl"
                    ),
                    "families": [
                        "direction_forecasting",
                        "venue_aware_research_positioning",
                    ],
                    "mode": "replace_row",
                }
            ],
            "v4": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "researchagent_experiment100_qwen8002_20260420_parallel4"
                        / "final_metrics"
                        / "eval_v4"
                        / "results_eval_v4.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "researchagent_experiment100_forecast_venue50_renderfix_20260421_parallel4"
                        / "final_metrics"
                        / "eval_v4"
                        / "results_eval_v4.jsonl"
                    ),
                    "families": [
                        "direction_forecasting",
                        "venue_aware_research_positioning",
                    ],
                    "mode": "replace_row",
                }
            ],
            "aux": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "researchagent_experiment100_qwen8002_20260420_parallel4"
                        / "final_metrics"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "researchagent_experiment100_forecast_venue50_renderfix_20260421_parallel4"
                        / "final_metrics"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "families": [
                        "direction_forecasting",
                        "venue_aware_research_positioning",
                    ],
                    "mode": "replace_row",
                },
            ],
        },
    },
    {
        "method_key": "aris",
        "method_name": "ARIS",
        "code_files": [
            "scripts/run_aris_offline.py",
            "src/researchworld/aris_offline.py",
            "src/researchworld/offline_kb.py",
            "src/researchworld/llm.py",
            "src/researchworld/research_arc_v2.py",
            "src/researchworld/research_arc_kb.py",
            "src/researchworld/retrieval_fusion.py",
        ],
        "bundles": {
            "v31": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "aris_experiment100_strategic_forecasting50_20260418_parallel4"
                        / "eval_v31"
                        / "results_eval_v3_1.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "aris_experiment100_bottleneck_venue50_20260418_parallel4"
                        / "eval_v31"
                        / "results_eval_v3_1.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT / "aris_bd50_v31_lenient_20260418" / "results_eval_v3_1.jsonl"
                    ),
                    "families": [
                        "bottleneck_opportunity_discovery",
                        "direction_forecasting",
                    ],
                    "mode": "patch_fact_only",
                },
            ],
            "v4": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "aris_experiment100_strategic_forecasting50_20260418_parallel4"
                        / "eval_v4"
                        / "results_eval_v4.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "aris_experiment100_bottleneck_venue50_20260418_parallel4"
                        / "eval_v4"
                        / "results_eval_v4.jsonl"
                    ),
                    "mode": "replace_row",
                },
            ],
            "aux": [
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "aris_experiment100_strategic_forecasting50_20260418_parallel4"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "aris_experiment100_bottleneck_venue50_20260418_parallel4"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT / "aris_bd50_aux_lenient_20260418" / "results_eval_aux.jsonl"
                    ),
                    "families": [
                        "bottleneck_opportunity_discovery",
                        "direction_forecasting",
                    ],
                    "mode": "replace_row",
                },
                {
                    "path": _jsonl(
                        RESULTS_ROOT
                        / "aris_experiment100_bottleneck_venue50_aux_venueprior_20260420"
                        / "eval_aux"
                        / "results_eval_aux.jsonl"
                    ),
                    "families": ["venue_aware_research_positioning"],
                    "mode": "replace_row",
                },
            ],
        },
    },
]

METHOD_EXPORT_ORDER = {
    spec["method_key"]: idx for idx, spec in enumerate(CANONICAL_METHOD_SPECS)
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _export_code_snapshot(*, method_dir: Path, spec: Mapping[str, Any]) -> Dict[str, Any]:
    code_files = [str(x) for x in spec.get("code_files") or [] if str(x).strip()]
    exported_files: List[Dict[str, Any]] = []
    code_root = method_dir / "code"
    if code_root.exists():
        shutil.rmtree(code_root)
    code_root.mkdir(parents=True, exist_ok=True)
    for rel in code_files:
        src = ROOT / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst = code_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        exported_files.append(
            {
                "relative_path": rel,
                "source_path": str(src),
                "snapshot_path": str(dst),
            }
        )
    return {
        "method_key": str(spec["method_key"]),
        "method_name": str(spec["method_name"]),
        "snapshot_root": str(code_root),
        "file_count": len(exported_files),
        "files": exported_files,
    }


def _mean(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None]
    return round(mean(clean), 4) if clean else 0.0


def _matches_family(row: Mapping[str, Any], families: Sequence[str] | None) -> bool:
    if not families:
        return True
    return str(row.get("family") or "") in set(families)


def _patch_v31_fact_only(base_row: Dict[str, Any], patch_row: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base_row)
    out.setdefault("scores", {})
    out["scores"]["fact_precision_score"] = (
        patch_row.get("scores", {}) or {}
    ).get("fact_precision_score", out["scores"].get("fact_precision_score"))
    if "fact_eval" in patch_row:
        out["fact_eval"] = copy.deepcopy(patch_row["fact_eval"])
    return out


def _merge_bundle_rows(steps: Sequence[Mapping[str, Any]]) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    rows_by_task: Dict[str, Dict[str, Any]] = {}
    provenance_by_task: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        path = Path(str(step["path"]))
        if not path.exists():
            raise FileNotFoundError(path)
        rows = _load_jsonl(path)
        families = step.get("families")
        mode = str(step.get("mode") or "replace_row")
        for row in rows:
            task_id = str(row.get("task_id") or "").strip()
            if not task_id or not _matches_family(row, families):
                continue
            if mode == "replace_row" or task_id not in rows_by_task:
                rows_by_task[task_id] = copy.deepcopy(row)
            elif mode == "patch_fact_only":
                rows_by_task[task_id] = _patch_v31_fact_only(rows_by_task[task_id], row)
            else:
                raise ValueError(f"unsupported merge mode: {mode}")
            provenance_by_task[task_id] = {
                "path": str(path),
                "mode": mode,
                "families": list(families or []),
            }
    return rows_by_task, provenance_by_task


def _task_field(task_row: Mapping[str, Any], key: str, fallback: Any = None) -> Any:
    value = task_row.get(key)
    return fallback if value is None else value


def _primary_dimensions(v31_row: Mapping[str, Any], v4_row: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    fact_eval = v31_row.get("fact_eval", {}) or {}
    future_eval = v31_row.get("future_alignment_eval", {}) or {}
    trace_eval = v4_row.get("evidence_traceability_eval", {}) or {}
    trace_dims = trace_eval.get("rubric_scores", {}) or {}
    return {
        "evidence_grounded_factuality": {
            "precision_score": float(fact_eval.get("precision_score") or 0.0),
            "coverage_score": float(fact_eval.get("coverage_score") or 0.0),
        },
        "future_alignment": {
            "weighted_unit_alignment": float(future_eval.get("weighted_unit_alignment") or 0.0),
            "alignment_coverage": float(future_eval.get("alignment_coverage") or 0.0),
            "mean_specificity": float(future_eval.get("mean_specificity") or 0.0),
        },
        "evidence_traceability": {
            key: float(value or 0.0) for key, value in trace_dims.items()
        },
    }


def _family_aux_payload(aux_row: Mapping[str, Any]) -> Dict[str, Any]:
    family = str(aux_row.get("family") or "")
    aux_eval = aux_row.get("family_aux_eval", {}) or {}
    scores = aux_row.get("scores", {}) or {}
    score_key = next((key for key in scores.keys() if key.endswith("_score")), FAMILY_TO_AUX_SCORE_KEY.get(family, "family_aux_score"))
    metric_name = str(aux_eval.get("family_aux_metric_name") or FAMILY_TO_AUX_NAME.get(family) or "Family Auxiliary")
    return {
        "metric_name": metric_name,
        "score_key": score_key,
        "score": float(scores.get(score_key) or aux_eval.get(score_key) or 0.0),
        "dimensions": {key: float(value or 0.0) for key, value in (aux_eval.get("rubric_scores", {}) or {}).items()},
        "details": copy.deepcopy(aux_eval),
    }


def _diagnostics(v31_row: Mapping[str, Any], v4_row: Mapping[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in ("retrieval_diagnostics", "agent_diagnostics"):
        if key in v31_row:
            payload[key] = copy.deepcopy(v31_row[key])
    if "support_profile" in v4_row:
        payload["support_profile"] = copy.deepcopy(v4_row["support_profile"])
    return payload


def _build_task_result(
    *,
    task_row: Mapping[str, Any],
    method_key: str,
    method_name: str,
    v31_row: Mapping[str, Any],
    v4_row: Mapping[str, Any],
    aux_row: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    family = str(task_row.get("family") or v31_row.get("family") or "")
    primary_dimensions = _primary_dimensions(v31_row, v4_row)
    family_aux = _family_aux_payload(aux_row)
    answer = (
        v31_row.get("answer")
        or v4_row.get("answer")
        or aux_row.get("answer")
        or ""
    )
    return {
        "task_id": str(task_row.get("task_id") or ""),
        "title": _task_field(task_row, "title", (v31_row.get("metadata", {}) or {}).get("task_title")),
        "question": _task_field(task_row, "question"),
        "family": family,
        "domain": _task_field(task_row, "domain", v31_row.get("domain")),
        "time_cutoff": _task_field(task_row, "time_cutoff", (v31_row.get("metadata", {}) or {}).get("time_cutoff")),
        "method_key": method_key,
        "method_name": method_name,
        "answer": answer,
        "metadata": copy.deepcopy(v31_row.get("metadata", {}) or v4_row.get("metadata", {}) or aux_row.get("metadata", {})),
        "diagnostics": _diagnostics(v31_row, v4_row),
        "source_provenance": copy.deepcopy(provenance),
        "primary_metrics": {
            "evidence_grounded_factuality": {
                "score": float((v31_row.get("scores", {}) or {}).get("fact_precision_score") or 0.0),
                "dimensions": primary_dimensions["evidence_grounded_factuality"],
                "details": copy.deepcopy(v31_row.get("fact_eval", {}) or {}),
            },
            "future_alignment": {
                "score": float((v31_row.get("scores", {}) or {}).get("future_alignment_score") or 0.0),
                "dimensions": primary_dimensions["future_alignment"],
                "details": copy.deepcopy(v31_row.get("future_alignment_eval", {}) or {}),
            },
            "evidence_traceability": {
                "score": float((v4_row.get("scores", {}) or {}).get("evidence_traceability_score") or 0.0),
                "dimensions": primary_dimensions["evidence_traceability"],
                "details": copy.deepcopy(v4_row.get("evidence_traceability_eval", {}) or {}),
            },
        },
        "family_auxiliary": family_aux,
    }


def _nested_dimension_accumulator() -> Dict[str, List[float]]:
    return defaultdict(list)


def _add_dimensions(target: MutableMapping[str, List[float]], dims: Mapping[str, Any]) -> None:
    for key, value in dims.items():
        target[key].append(float(value or 0.0))


def _summarize_task_rows(method_key: str, method_name: str, rows: Sequence[Mapping[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_domain: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row["family"])].append(row)
        by_domain[str(row["domain"])].append(row)

    def score_block(group: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        return {
            "evidence_grounded_factuality": _mean([r["primary_metrics"]["evidence_grounded_factuality"]["score"] for r in group]),
            "future_alignment": _mean([r["primary_metrics"]["future_alignment"]["score"] for r in group]),
            "evidence_traceability": _mean([r["primary_metrics"]["evidence_traceability"]["score"] for r in group]),
        }

    def aux_block(group: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        buckets = {name: [] for name in FAMILY_TO_AUX_NAME.values()}
        for row in group:
            payload = row["family_auxiliary"]
            buckets[payload["metric_name"]].append(payload["score"])
        return {name: _mean(vals) for name, vals in buckets.items() if vals}

    def dimension_block(group: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
        fact_dims = _nested_dimension_accumulator()
        future_dims = _nested_dimension_accumulator()
        trace_dims = _nested_dimension_accumulator()
        aux_dims_by_family: Dict[str, Dict[str, List[float]]] = defaultdict(_nested_dimension_accumulator)
        for row in group:
            _add_dimensions(fact_dims, row["primary_metrics"]["evidence_grounded_factuality"]["dimensions"])
            _add_dimensions(future_dims, row["primary_metrics"]["future_alignment"]["dimensions"])
            _add_dimensions(trace_dims, row["primary_metrics"]["evidence_traceability"]["dimensions"])
            _add_dimensions(aux_dims_by_family[row["family"]], row["family_auxiliary"]["dimensions"])
        return {
            "evidence_grounded_factuality": {key: _mean(vals) for key, vals in fact_dims.items()},
            "future_alignment": {key: _mean(vals) for key, vals in future_dims.items()},
            "evidence_traceability": {key: _mean(vals) for key, vals in trace_dims.items()},
            "family_auxiliary": {
                family: {key: _mean(vals) for key, vals in dims.items()}
                for family, dims in aux_dims_by_family.items()
            },
        }

    score_summary = {
        "method_key": method_key,
        "method_name": method_name,
        "task_count": len(rows),
        "overall": {
            "primary_metrics": score_block(rows),
            "family_auxiliary": aux_block(rows),
        },
        "by_family": {
            family: {
                "task_count": len(group),
                "primary_metrics": score_block(group),
                "family_auxiliary": {
                    "metric_name": FAMILY_TO_AUX_NAME[family],
                    "score": _mean([r["family_auxiliary"]["score"] for r in group]),
                },
            }
            for family, group in sorted(by_family.items())
        },
        "by_domain": {
            domain: {
                "task_count": len(group),
                "primary_metrics": score_block(group),
                "family_auxiliary": aux_block(group),
            }
            for domain, group in sorted(by_domain.items())
        },
    }

    dimension_summary = {
        "method_key": method_key,
        "method_name": method_name,
        "task_count": len(rows),
        "overall": dimension_block(rows),
        "by_family": {
            family: dimension_block(group)
            for family, group in sorted(by_family.items())
        },
        "by_domain": {
            domain: dimension_block(group)
            for domain, group in sorted(by_domain.items())
        },
    }
    return score_summary, dimension_summary


def _validate_counts(rows: Sequence[Mapping[str, Any]], expected_task_ids: Sequence[str]) -> None:
    got = {str(row["task_id"]) for row in rows}
    expected = set(expected_task_ids)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise SystemExit(
            json.dumps(
                {
                    "error": "task-id mismatch in canonical export",
                    "missing": missing[:20],
                    "missing_count": len(missing),
                    "extra": extra[:20],
                    "extra_count": len(extra),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def _build_method_package(
    *,
    spec: Mapping[str, Any],
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    release_dir: Path,
    output_root: Path,
) -> Dict[str, Any]:
    method_key = str(spec["method_key"])
    method_name = str(spec["method_name"])
    merged_rows: Dict[str, Dict[str, Dict[str, Any]]] = {}
    provenance: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for bundle in ("v31", "v4", "aux"):
        rows_by_task, bundle_provenance = _merge_bundle_rows(spec["bundles"][bundle])
        for task_id, row in rows_by_task.items():
            merged_rows.setdefault(task_id, {})[bundle] = row
            provenance[task_id][bundle] = bundle_provenance[task_id]

    missing_by_bundle: Dict[str, List[str]] = defaultdict(list)
    task_results: List[Dict[str, Any]] = []
    for task_id, task_row in tasks_by_id.items():
        packs = merged_rows.get(task_id, {})
        for bundle in ("v31", "v4", "aux"):
            if bundle not in packs:
                missing_by_bundle[bundle].append(task_id)
        if set(packs) != {"v31", "v4", "aux"}:
            continue
        task_results.append(
            _build_task_result(
                task_row=task_row,
                method_key=method_key,
                method_name=method_name,
                v31_row=packs["v31"],
                v4_row=packs["v4"],
                aux_row=packs["aux"],
                provenance=provenance[task_id],
            )
        )

    if any(missing_by_bundle.values()):
        raise SystemExit(
            json.dumps(
                {
                    "error": f"incomplete bundles for {method_key}",
                    "missing_by_bundle": {k: v[:20] for k, v in missing_by_bundle.items() if v},
                    "missing_counts": {k: len(v) for k, v in missing_by_bundle.items() if v},
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    task_results.sort(key=lambda row: str(row["task_id"]))
    _validate_counts(task_results, list(tasks_by_id))
    score_summary, dimension_summary = _summarize_task_rows(method_key, method_name, task_results)

    method_dir = output_root / method_key
    code_manifest = _export_code_snapshot(method_dir=method_dir, spec=spec)
    provenance_payload = {
        "method_key": method_key,
        "method_name": method_name,
        "release_dir": str(release_dir),
        "bundle_sources": spec["bundles"],
        "code_files": list(spec.get("code_files") or []),
        "task_count": len(task_results),
    }
    _write_jsonl(method_dir / "task_level_results.jsonl", task_results)
    _write_json(method_dir / "score_summary.json", score_summary)
    _write_json(method_dir / "dimension_summary.json", dimension_summary)
    _write_json(method_dir / "provenance.json", provenance_payload)
    _write_json(method_dir / "code_manifest.json", code_manifest)
    return {
        "method_key": method_key,
        "method_name": method_name,
        "task_count": len(task_results),
        "output_dir": str(method_dir),
        "score_summary": score_summary,
        "dimension_summary": dimension_summary,
        "provenance": provenance_payload,
        "code_manifest": code_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical per-method result packages for the current experiment100 comparison.")
    parser.add_argument(
        "--release-dir",
        default=str(DEFAULT_RELEASE_DIR),
        help="Release directory used to recover canonical task metadata.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_ROOT / "experiment100_canonical_method_exports_20260421"),
        help="Directory where per-method canonical export folders will be written.",
    )
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    tasks_path = release_dir / "task_refined.jsonl"
    if not tasks_path.exists():
        raise FileNotFoundError(tasks_path)
    tasks_by_id = {str(row["task_id"]): row for row in _load_jsonl(tasks_path)}
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    exported: List[Dict[str, Any]] = []
    for spec in CANONICAL_METHOD_SPECS:
        exported.append(
            _build_method_package(
                spec=spec,
                tasks_by_id=tasks_by_id,
                release_dir=release_dir,
                output_root=output_root,
            )
        )

    top_level_scores = []
    for item in exported:
        summary = item["score_summary"]
        top_level_scores.append(
            {
                "method_key": item["method_key"],
                "method_name": item["method_name"],
                "task_count": item["task_count"],
                **summary["overall"]["primary_metrics"],
                **summary["overall"]["family_auxiliary"],
            }
        )
    all_dimension_summaries = [
        {
            "method_key": item["method_key"],
            "method_name": item["method_name"],
            "task_count": item["task_count"],
            "overall": item["dimension_summary"]["overall"],
            "by_family": item["dimension_summary"]["by_family"],
        }
        for item in exported
    ]
    top_level_scores.sort(key=lambda row: METHOD_EXPORT_ORDER.get(row["method_key"], 999))
    all_dimension_summaries.sort(key=lambda row: METHOD_EXPORT_ORDER.get(row["method_key"], 999))
    index_payload = {
        "release_dir": str(release_dir),
        "output_dir": str(output_root),
        "method_count": len(exported),
        "methods": [
            {
                "method_key": item["method_key"],
                "method_name": item["method_name"],
                "task_count": item["task_count"],
                "output_dir": item["output_dir"],
            }
            for item in exported
        ],
    }
    _write_json(output_root / "method_index.json", index_payload)
    _write_json(output_root / "all_methods_score_summary.json", top_level_scores)
    _write_json(output_root / "all_methods_dimension_summary.json", all_dimension_summaries)
    print(json.dumps(index_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
