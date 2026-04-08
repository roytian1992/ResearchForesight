from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.config import load_yaml
from researchworld.llm import OpenAICompatChatClient, load_openai_compat_config
from researchworld.taxoadapt import TemporalTaxoAdaptRunner, load_domain_papers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal TaxoAdapt-style taxonomy induction for a benchmark domain.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "taxoadapt_temporal.yaml"), help="TaxoAdapt YAML config.")
    parser.add_argument(
        "--benchmark-config",
        default=str(ROOT / "configs" / "benchmark.yaml"),
        help="Benchmark config YAML used to resolve the LLM profile.",
    )
    parser.add_argument("--domain-id", required=True, help="Domain id to run, e.g. rag_and_retrieval_structuring.")
    parser.add_argument("--run-name", default="", help="Optional output run name. Default uses 'default'.")
    parser.add_argument("--years", default="", help="Optional comma-separated override, e.g. 2023,2024,2025,2026.")
    parser.add_argument("--max-papers-per-year", type=int, default=0, help="Optional cap for pilot runs.")
    parser.add_argument("--workers", type=int, default=0, help="Optional LLM worker override.")
    parser.add_argument("--max-density", type=int, default=0, help="Optional density threshold override.")
    parser.add_argument("--max-depth", type=int, default=0, help="Optional depth override.")
    parser.add_argument("--slice-ids", default="", help="Optional comma-separated override, e.g. 2023Q1,2023Q2.")
    return parser.parse_args()


def resolve_settings(config: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
    defaults = dict(config.get("defaults") or {})
    domains = config.get("domains") or {}
    domain_cfg = dict(domains.get(domain_id) or {})
    if not domain_cfg:
        raise SystemExit(f"Unknown domain_id: {domain_id}")
    merged = dict(defaults)
    merged.update(domain_cfg)
    return merged


def parse_years(text: str, fallback: List[int]) -> List[int]:
    if not text.strip():
        return fallback
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_slice_ids(text: str, fallback: List[str]) -> List[str]:
    if not text.strip():
        return fallback
    return [part.strip() for part in text.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    settings = resolve_settings(config, args.domain_id)
    benchmark_cfg = load_yaml(args.benchmark_config)
    project = benchmark_cfg.get("project") or {}

    llm_cfg = load_openai_compat_config((ROOT / project["llm_config_path"]).resolve())
    llm = OpenAICompatChatClient(llm_cfg)

    time_slices = list(settings.get("time_slices") or [])
    slice_id_sequence = [str(item["slice_id"]) for item in time_slices]
    if time_slices:
        selected_slice_ids = parse_slice_ids(args.slice_ids, slice_id_sequence)
        time_slices = [item for item in time_slices if str(item["slice_id"]) in set(selected_slice_ids)]
        year_sequence = []
        time_sequence = [str(item["slice_id"]) for item in time_slices]
    else:
        year_sequence = parse_years(args.years, [int(year) for year in settings.get("year_sequence") or [2023, 2024, 2025, 2026]])
        time_sequence = [str(year) for year in year_sequence]
    run_name = args.run_name.strip() or "default"
    output_dir = ROOT / str(settings["output_dir"]) / run_name

    papers = load_domain_papers(
        ROOT / str(settings["input_path"]),
        scope_labels_path=(ROOT / str(settings["scope_labels_path"])) if settings.get("scope_labels_path") else None,
        allowed_scope=str(settings.get("allowed_scope") or "core_domain"),
        year_sequence=year_sequence,
        time_slices=time_slices,
        max_papers_per_year=args.max_papers_per_year or int(settings.get("max_papers_per_year") or 0),
        min_abstract_chars=int(settings.get("min_abstract_chars") or 40),
    )

    runner = TemporalTaxoAdaptRunner(
        project_root=ROOT,
        llm=llm,
        domain_id=args.domain_id,
        topic=str(settings["topic"]),
        papers=papers,
        output_dir=output_dir,
        dimensions=settings.get("dimensions"),
        year_sequence=year_sequence,
        time_sequence=time_sequence,
        init_levels=int(settings.get("init_levels") or 1),
        max_depth=args.max_depth or int(settings.get("max_depth") or 3),
        max_density=args.max_density or int(settings.get("max_density") or 40),
        max_children_per_node=int(settings.get("max_children_per_node") or 5),
        bootstrap_paper_sample_size=int(settings.get("bootstrap_paper_sample_size") or 20),
        width_paper_sample_size=int(settings.get("width_paper_sample_size") or 40),
        depth_paper_sample_size=int(settings.get("depth_paper_sample_size") or 40),
        candidate_label_top_k=int(settings.get("candidate_label_top_k") or 20),
        min_candidate_votes=int(settings.get("min_candidate_votes") or 2),
        workers=args.workers or int(settings.get("workers") or 6),
        request_timeout=int(settings.get("request_timeout") or 240),
        max_retries=int(settings.get("max_retries") or 2),
        temperature_bootstrap=float(settings.get("temperature_bootstrap") or 0.1),
        temperature_routing=float(settings.get("temperature_routing") or 0.0),
        temperature_classification=float(settings.get("temperature_classification") or 0.0),
        temperature_expansion=float(settings.get("temperature_expansion") or 0.4),
        temperature_clustering=float(settings.get("temperature_clustering") or 0.2),
    )
    summary = runner.run()

    print(f"domain_id: {args.domain_id}")
    print(f"run_name: {run_name}")
    print(f"paper_count: {summary['paper_count']}")
    print(f"node_count: {summary['node_count']}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
