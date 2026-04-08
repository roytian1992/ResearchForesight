from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.domains import load_domain_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a domain label file to the current candidate pool and tier policy.")
    parser.add_argument("--domain-config", required=True, help="Domain pipeline config YAML.")
    parser.add_argument("--labels", default="", help="Override input label JSONL. Defaults to domain annotation.output_path.")
    parser.add_argument("--output", default="", help="Override output label JSONL. Defaults to in-place rewrite.")
    parser.add_argument(
        "--keep-tier",
        default="",
        help="Override candidate tier policy. Defaults to domain annotation.candidate_tier; use 'any' to keep all current candidates.",
    )
    return parser.parse_args()


def load_allowed_ids(candidate_path: Path, keep_tier: str) -> Dict[str, str]:
    allowed: Dict[str, str] = {}
    for row in iter_jsonl(candidate_path):
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            continue
        candidate_tier = str(row.get("candidate_tier") or "")
        if keep_tier != "any" and candidate_tier != keep_tier:
            continue
        allowed[paper_id] = candidate_tier
    return allowed


def atomic_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def main() -> None:
    args = parse_args()
    domain_cfg = load_domain_config(args.domain_config)
    annotation_cfg = domain_cfg.get("annotation") or {}
    corpus_cfg = domain_cfg.get("corpus") or {}

    labels_path = Path(args.labels) if args.labels else ROOT / str(annotation_cfg["output_path"])
    output_path = Path(args.output) if args.output else labels_path
    candidate_path = ROOT / str(corpus_cfg["candidate_output"])
    keep_tier = str(args.keep_tier or annotation_cfg.get("candidate_tier") or "any")

    allowed_ids = load_allowed_ids(candidate_path, keep_tier)
    latest_rows: Dict[str, dict] = {}
    total_rows = 0
    kept_rows = 0
    dropped_rows = 0

    for row in iter_jsonl(labels_path):
        total_rows += 1
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id:
            dropped_rows += 1
            continue
        if paper_id not in allowed_ids:
            dropped_rows += 1
            continue
        latest_rows[paper_id] = row
        kept_rows += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        for paper_id in sorted(latest_rows):
            handle.write(json.dumps(latest_rows[paper_id], ensure_ascii=False) + "\n")

    if output_path == labels_path:
        atomic_replace(temp_path, output_path)
    else:
        atomic_replace(temp_path, output_path)

    print(
        json.dumps(
            {
                "domain_id": domain_cfg.get("domain_id"),
                "labels_path": str(labels_path),
                "output_path": str(output_path),
                "candidate_path": str(candidate_path),
                "keep_tier": keep_tier,
                "total_rows": total_rows,
                "kept_rows": kept_rows,
                "unique_kept_paper_ids": len(latest_rows),
                "dropped_rows": dropped_rows,
                "allowed_candidate_ids": len(allowed_ids),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
