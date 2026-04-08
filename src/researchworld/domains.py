from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from researchworld.config import load_yaml


def load_domain_registry(path: str | Path) -> Dict[str, Dict[str, Any]]:
    obj = load_yaml(path)
    rows = obj.get("domains") or []
    registry: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        domain_id = row.get("domain_id")
        if isinstance(domain_id, str) and domain_id:
            registry[domain_id] = row
    return registry


def load_domain_seed_queries(path: str | Path) -> Dict[str, Dict[str, Any]]:
    obj = load_yaml(path)
    rows = obj.get("domains") or {}
    return rows if isinstance(rows, dict) else {}


def load_domain_config(path: str | Path) -> Dict[str, Any]:
    return load_yaml(path)


def resolve_domain_bundle(
    project_root: Path,
    benchmark_cfg: Dict[str, Any],
    domain_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    project = benchmark_cfg.get("project") or {}
    registry_path = project_root / project.get("domain_registry_path", "configs/domain_registry.yaml")
    seed_queries_path = project_root / project.get("domain_seed_queries_path", "configs/domain_seed_queries.yaml")

    domain_id = str(domain_cfg.get("domain_id") or "")
    if not domain_id:
        raise ValueError("domain config missing domain_id")

    registry = load_domain_registry(registry_path)
    seed_queries = load_domain_seed_queries(seed_queries_path)
    registry_row = registry.get(domain_id) or {}
    seed_row = seed_queries.get(domain_id) or {}
    annotation_cfg = domain_cfg.get("annotation") or {}
    raw_inputs = domain_cfg.get("raw_inputs") or project.get("raw_inputs") or []
    if not isinstance(raw_inputs, list):
        raise ValueError(f"raw_inputs must be a list for domain {domain_id}")

    taxonomy_path = project_root / str(
        domain_cfg.get("taxonomy_path")
        or registry_row.get("taxonomy_path")
        or project.get("taxonomy_path")
    )
    prompt_id = str(annotation_cfg.get("prompt_id") or "")
    if not prompt_id:
        raise ValueError(f"domain config missing annotation.prompt_id for {domain_id}")

    return {
        "domain_id": domain_id,
        "display_name": str(registry_row.get("display_name") or domain_id),
        "taxonomy_path": taxonomy_path,
        "prompt_id": prompt_id,
        "raw_inputs": [project_root / str(path) for path in raw_inputs],
        "seed_queries": {
            "anchor_terms": list(seed_row.get("anchor_terms") or []),
            "positive_terms": list(seed_row.get("positive_terms") or []),
            "caution_terms": list(seed_row.get("caution_terms") or []),
            "negative_terms": list(seed_row.get("negative_terms") or []),
        },
        "corpus": domain_cfg.get("corpus") or {},
        "annotation": annotation_cfg,
        "clean": domain_cfg.get("clean") or {},
        "registry_row": registry_row,
    }
