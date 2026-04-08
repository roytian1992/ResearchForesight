from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.config import load_yaml
from researchworld.corpus import (
    heuristic_profile_from_seed_terms,
    iter_jsonl,
    normalize_paper,
    summarize_candidates,
    write_jsonl,
)
from researchworld.domains import resolve_domain_bundle, load_domain_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize and pre-filter papers for a target benchmark domain.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"), help="Benchmark config YAML.")
    parser.add_argument("--domain-config", required=True, help="Domain pipeline config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_cfg = load_yaml(args.config)
    domain_cfg = load_domain_config(args.domain_config)
    domain = resolve_domain_bundle(ROOT, benchmark_cfg, domain_cfg)

    corpus_cfg = domain["corpus"]
    raw_inputs = domain["raw_inputs"]
    merged_output = ROOT / corpus_cfg["merged_output"]
    candidate_output = ROOT / corpus_cfg["candidate_output"]
    summary_output = ROOT / corpus_cfg["summary_output"]

    merged_output.parent.mkdir(parents=True, exist_ok=True)
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    dedup = {}
    for raw_input in raw_inputs:
        if not raw_input.exists():
            print(f"[WARN] missing input: {raw_input}")
            continue
        for record in iter_jsonl(raw_input):
            paper = normalize_paper(record, source_path=str(raw_input))
            current = dedup.get(paper["paper_id"])
            if current is None or (paper.get("updated") or "") > (current.get("updated") or ""):
                dedup[paper["paper_id"]] = paper

    merged_rows = []
    candidate_rows = []
    seed_queries = domain["seed_queries"]
    for paper_id in sorted(dedup):
        paper = dedup[paper_id]
        profile = heuristic_profile_from_seed_terms(
            paper,
            anchor_terms=seed_queries.get("anchor_terms") or None,
            positive_terms=seed_queries["positive_terms"],
            caution_terms=seed_queries["caution_terms"],
            negative_terms=seed_queries["negative_terms"],
            positive_key=f"{domain['domain_id']}_keyword_hits",
        )
        paper.update(profile)
        merged_rows.append(paper)
        if paper["candidate_tier"] != "out":
            candidate_rows.append(paper)

    candidate_rows.sort(
        key=lambda row: (
            row.get("candidate_tier") != "core_candidate",
            -float(row.get("heuristic_score", 0.0)),
            row.get("published") or "",
        )
    )

    write_jsonl(merged_output, merged_rows)
    write_jsonl(candidate_output, candidate_rows)

    summary = {
        "domain_id": domain["domain_id"],
        "display_name": domain["display_name"],
        "inputs": [str(path) for path in raw_inputs],
        "merged": summarize_candidates(merged_rows),
        "candidates": summarize_candidates(candidate_rows),
        "seed_queries": seed_queries,
    }
    with open(summary_output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Domain: {domain['domain_id']}")
    print(f"Merged papers: {len(merged_rows)} -> {merged_output}")
    print(f"Candidate papers: {len(candidate_rows)} -> {candidate_output}")
    print(f"Summary: {summary_output}")


if __name__ == "__main__":
    main()
