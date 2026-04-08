from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.analytics import TAXONOMY_KEYS, count_labels
from researchworld.benchmark import load_rows_by_paper_id, merge_papers_with_labels
from researchworld.config import load_yaml
from researchworld.domains import load_domain_config, resolve_domain_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the clean core-domain corpus for a target benchmark domain.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"), help="Benchmark config YAML.")
    parser.add_argument("--domain-config", required=True, help="Domain pipeline config YAML.")
    parser.add_argument("--papers", default="", help="Override normalized-paper JSONL.")
    parser.add_argument("--labels", default="", help="Override label JSONL.")
    parser.add_argument("--output", default="", help="Override clean core-corpus JSONL output.")
    parser.add_argument("--stats-output", default="", help="Override JSON stats output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_cfg = load_yaml(args.config)
    domain_cfg = load_domain_config(args.domain_config)
    domain = resolve_domain_bundle(ROOT, benchmark_cfg, domain_cfg)
    clean_cfg = domain["clean"]
    annotation_cfg = domain["annotation"]
    corpus_cfg = domain["corpus"]

    papers_path = Path(args.papers) if args.papers else ROOT / corpus_cfg["merged_output"]
    labels_path = Path(args.labels) if args.labels else ROOT / annotation_cfg["output_path"]
    output_path = Path(args.output) if args.output else ROOT / clean_cfg["output_path"]
    stats_path = Path(args.stats_output) if args.stats_output else ROOT / clean_cfg["stats_output"]

    papers = load_rows_by_paper_id(papers_path)
    labels = load_rows_by_paper_id(labels_path)
    rows = merge_papers_with_labels(papers, labels)
    labeled_rows = [row for row in rows if row.get("scope_decision")]
    core_rows = [row for row in rows if row.get("scope_decision") == "core_domain"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in core_rows:
            serializable = dict(row)
            serializable.pop("published_date", None)
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    by_month = Counter()
    for row in core_rows:
        month = str(row.get("published") or "")[:7]
        if month:
            by_month[month] += 1

    stats = {
        "domain_id": domain["domain_id"],
        "display_name": domain["display_name"],
        "paper_count": len(core_rows),
        "labeled_paper_count": len(labeled_rows),
        "unlabeled_paper_count": len(rows) - len(labeled_rows),
        "scope_counts": Counter(str(row.get("scope_decision") or "unlabeled") for row in labeled_rows),
        "by_month": dict(sorted(by_month.items())),
        "top_labels": {
            key: count_labels(core_rows, key).most_common(20)
            for key in TAXONOMY_KEYS
        },
        "input_paths": {
            "papers": str(papers_path),
            "labels": str(labels_path),
        },
    }
    stats["scope_counts"] = dict(stats["scope_counts"])

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)

    print(f"Domain: {domain['domain_id']}")
    print(f"Core papers: {len(core_rows)}")
    print(f"Output: {output_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
