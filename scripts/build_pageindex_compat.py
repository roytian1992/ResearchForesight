from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.pageindex_compat import build_pageindex_tree
from researchworld.technical_vision import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a PageIndex-compatible tree from normalized paper content.")
    parser.add_argument("--content", required=True, help="Paper content JSONL.")
    parser.add_argument("--output", required=True, help="PageIndex-compatible JSONL output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.content)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            index_row = build_pageindex_tree(row)
            handle.write(json.dumps(index_row, ensure_ascii=False) + "\n")
            count += 1
    print(f"Output: {out_path}")
    print(f"Rows: {count}")


if __name__ == "__main__":
    main()
