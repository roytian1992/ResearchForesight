from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.benchmark_v2 import DEFAULT_DOMAINS
from researchworld.family_pipelines import build_all_family_packets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build family-specific benchmark packets on top of the shared backbone.")
    parser.add_argument("--out-root", default=str(ROOT / "data" / "family_packets" / "v1"))
    parser.add_argument("--selected-seeds", default=str(ROOT / "data" / "support_packets" / "selected_seed_nodes.json"))
    parser.add_argument("--all-packets", default=str(ROOT / "data" / "support_packets" / "all_node_support_packets.json"))
    parser.add_argument("--domains", default=",".join(DEFAULT_DOMAINS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    manifest = build_all_family_packets(
        out_root=Path(args.out_root),
        selected_seeds_path=args.selected_seeds,
        all_packets_path=args.all_packets,
        domains=domains,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
