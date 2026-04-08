from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.config import load_yaml
from researchworld.llm import OpenAICompatChatClient, load_openai_compat_config
from researchworld.taxoadapt import TaxonomyNode, TemporalTaxoAdaptRunner, load_domain_papers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume temporal TaxoAdapt from an existing yearly snapshot.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "taxoadapt_temporal.yaml"))
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--run-name", required=True, help="Existing run directory name, e.g. full_v1")
    parser.add_argument("--snapshot-year", type=int, default=0, help="Latest completed year to restore from, e.g. 2024")
    parser.add_argument("--snapshot-slice", default="", help="Latest completed slice to restore from, e.g. 2024Q3")
    parser.add_argument("--target-year", type=int, default=0, help="Optional last year to execute. Default runs remaining configured years.")
    parser.add_argument("--target-slice", default="", help="Optional last slice to execute. Default runs remaining configured slices.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-density", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=0)
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


def restore_runner_state(
    runner: TemporalTaxoAdaptRunner,
    *,
    run_dir: Path,
    snapshot_label: str,
) -> None:
    taxonomy_nodes_path = run_dir / snapshot_label / "taxonomy_nodes.json"
    assignments_path = run_dir / snapshot_label / "paper_assignments.jsonl"
    if not taxonomy_nodes_path.exists():
        raise SystemExit(f"Missing snapshot taxonomy: {taxonomy_nodes_path}")
    if not assignments_path.exists():
        raise SystemExit(f"Missing snapshot assignments: {assignments_path}")

    runner.nodes = {}
    runner.roots = {}
    runner.dimension_label_inventory.clear()
    runner.paper_dimension_membership.clear()

    node_rows = json.load(open(taxonomy_nodes_path, "r", encoding="utf-8"))
    for row in node_rows:
        node = TaxonomyNode(
            node_id=row["node_id"],
            label=row["label"],
            display_name=row["display_name"],
            description=row["description"],
            dimension_id=row["dimension_id"],
            level=int(row["level"]),
            created_year=row.get("created_year"),
            source=row["source"],
            parent_id=row["parent_id"],
            child_ids=list(row.get("child_ids") or []),
            created_time_slice=row.get("created_time_slice"),
        )
        runner.nodes[node.node_id] = node
        runner.dimension_label_inventory[node.dimension_id].add(node.label)
        if node.parent_id is None:
            runner.roots[node.dimension_id] = node.node_id

    for line in open(assignments_path, "r", encoding="utf-8"):
        obj = json.loads(line)
        paper_id = obj["paper_id"]
        runner.paper_dimension_membership[paper_id].update(obj.get("dimension_membership") or [])
        for _dim_id, assignments in (obj.get("dimension_assignments") or {}).items():
            for assignment in assignments:
                node_id = assignment["node_id"]
                current = node_id
                while current:
                    runner.nodes[current].paper_ids.add(paper_id)
                    current = runner.nodes[current].parent_id

    runner.year_summaries = []
    for label in runner.time_sequence:
        if runner.time_sequence.index(label) > runner.time_sequence.index(snapshot_label):
            break
        summary_path = run_dir / label / "summary.json"
        if summary_path.exists():
            runner.year_summaries.append(json.load(open(summary_path, "r", encoding="utf-8")))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    settings = resolve_settings(config, args.domain_id)
    benchmark_cfg = load_yaml(ROOT / "configs" / "benchmark.yaml")
    project = benchmark_cfg.get("project") or {}

    run_dir = ROOT / str(settings["output_dir"]) / args.run_name
    time_slices = list(settings.get("time_slices") or [])
    if time_slices:
        full_sequence = [str(item["slice_id"]) for item in time_slices]
        snapshot_label = args.snapshot_slice.strip()
        if not snapshot_label:
            raise SystemExit("quarterly resume requires --snapshot-slice")
        if snapshot_label not in full_sequence:
            raise SystemExit(f"snapshot slice {snapshot_label} not in configured sequence {full_sequence}")
        target_label = args.target_slice.strip() or full_sequence[-1]
        if target_label not in full_sequence:
            raise SystemExit(f"target slice {target_label} not in configured sequence {full_sequence}")
        selected_sequence = full_sequence[: full_sequence.index(target_label) + 1]
    else:
        full_years = [int(year) for year in settings.get("year_sequence") or [2023, 2024, 2025, 2026]]
        if args.snapshot_year not in full_years:
            raise SystemExit(f"snapshot year {args.snapshot_year} not in configured sequence {full_years}")
        snapshot_label = str(args.snapshot_year)
        target_year = args.target_year or full_years[-1]
        selected_sequence = [str(year) for year in full_years if year <= target_year]

    llm_cfg = load_openai_compat_config((ROOT / project["llm_config_path"]).resolve())
    llm = OpenAICompatChatClient(llm_cfg)
    papers = load_domain_papers(
        ROOT / str(settings["input_path"]),
        scope_labels_path=(ROOT / str(settings["scope_labels_path"])) if settings.get("scope_labels_path") else None,
        allowed_scope=str(settings.get("allowed_scope") or "core_domain"),
        year_sequence=[int(x) for x in selected_sequence if x.isdigit()],
        time_slices=[item for item in time_slices if str(item["slice_id"]) in set(selected_sequence)] if time_slices else None,
        max_papers_per_year=int(settings.get("max_papers_per_year") or 0),
        min_abstract_chars=int(settings.get("min_abstract_chars") or 40),
    )

    runner = TemporalTaxoAdaptRunner(
        project_root=ROOT,
        llm=llm,
        domain_id=args.domain_id,
        topic=str(settings["topic"]),
        papers=papers,
        output_dir=run_dir,
        dimensions=settings.get("dimensions"),
        year_sequence=[int(x) for x in selected_sequence if x.isdigit()] or [2023],
        time_sequence=selected_sequence,
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

    restore_runner_state(runner, run_dir=run_dir, snapshot_label=snapshot_label)

    pending_labels = selected_sequence[selected_sequence.index(snapshot_label) + 1 :]
    if not pending_labels:
        print("No pending years to run.")
        return

    try:
        for label in pending_labels:
            new_paper_ids = list(runner.papers_by_year.get(label) or [])
            print(f"[resume] slice={label} new_papers={len(new_paper_ids)}")
            route_map = runner._route_papers_to_dimensions(new_paper_ids)
            print(f"[resume] slice={label} route_done papers={len(route_map)}")
            runner._bootstrap_missing_roots(label, route_map)
            print(f"[resume] slice={label} bootstrap_done")
            runner._ingest_year(label, route_map)
            print(f"[resume] slice={label} ingest_done")
            runner._write_year_snapshot(label, route_map=route_map)
            print(f"[resume] slice={label} snapshot_written")

        final_summary = {
            "domain_id": runner.domain_id,
            "topic": runner.topic,
            "paper_count": len(runner.papers),
            "years": runner.year_summaries,
            "node_count": len(runner.nodes),
            "dimension_node_counts": {
                dim.id: len([node for node in runner.nodes.values() if node.dimension_id == dim.id])
                for dim in runner.dimensions
            },
        }
        (run_dir / "final_summary.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[resume] final_summary_written path={run_dir / 'final_summary.json'}")
    except Exception:
        error_suffix = snapshot_label.replace('/', '_')
        error_path = run_dir / f"resume_error_from_{error_suffix}.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[resume] failed; traceback written to {error_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
