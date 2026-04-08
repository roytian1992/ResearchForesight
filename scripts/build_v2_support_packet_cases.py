from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.benchmark_v2 import (
    DEFAULT_DOMAINS,
    choose_case_papers,
    dump_json,
    dump_jsonl,
    load_all_packets,
    load_selected_seed_packets,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-domain support-packet cases for selective full-text processing.")
    parser.add_argument("--selected-seeds", default=str(ROOT / "data" / "support_packets" / "selected_seed_nodes.json"))
    parser.add_argument("--all-packets", default=str(ROOT / "data" / "support_packets" / "all_node_support_packets.json"))
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "support_packets" / "fulltext_cases"))
    parser.add_argument("--history-k", type=int, default=3)
    parser.add_argument("--future-k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    selected = load_selected_seed_packets(args.selected_seeds)
    packets = load_all_packets(args.all_packets)

    manifest = {"domains": {}, "total_cases": 0, "total_unique_papers": 0}
    global_unique = set()
    for domain in domains:
        rows = []
        unique_ids = set()
        for seed in selected:
            if seed.get("domain") != domain:
                continue
            packet = packets.get(str(seed["packet_id"]))
            if not packet:
                continue
            paper_ids = choose_case_papers(packet, history_k=args.history_k, future_k=args.future_k)
            unique_ids.update(paper_ids["history_paper_ids"])
            unique_ids.update(paper_ids["future_paper_ids"])
            row = {
                "case_id": f"{domain}::{packet['node_id'].replace('/', '__')}",
                "domain": domain,
                "packet_id": packet["packet_id"],
                "family_targets": [
                    "direction_forecasting",
                    "bottleneck_opportunity_discovery",
                    "strategic_research_planning",
                ],
                "node_id": packet["node_id"],
                "display_name": packet["display_name"],
                "dimension_id": packet["dimension_id"],
                "history_paper_ids": paper_ids["history_paper_ids"],
                "future_paper_ids": paper_ids["future_paper_ids"],
                "history_end_date": packet["history_end_date"],
                "history_structure_slice": packet["history_structure_slice"],
            }
            rows.append(row)
        rows.sort(key=lambda row: row["case_id"])
        dump_jsonl(out_dir / domain / "support_packet_cases.jsonl", rows)
        dump_json(
            out_dir / domain / "manifest.json",
            {
                "domain": domain,
                "case_count": len(rows),
                "unique_paper_count": len(unique_ids),
            },
        )
        manifest["domains"][domain] = {
            "case_count": len(rows),
            "unique_paper_count": len(unique_ids),
        }
        manifest["total_cases"] += len(rows)
        global_unique.update(unique_ids)
        print(domain, "cases", len(rows), "unique_papers", len(unique_ids))
    manifest["total_unique_papers"] = len(global_unique)
    dump_json(out_dir / "manifest.json", manifest)
    print("total_cases", manifest["total_cases"])
    print("total_unique_papers", manifest["total_unique_papers"])


if __name__ == "__main__":
    main()
