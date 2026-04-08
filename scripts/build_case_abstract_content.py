from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.benchmark import load_rows_by_paper_id
from researchworld.content import build_fallback_content_row
from researchworld.technical_vision import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build abstract-only normalized content rows for support-packet cases.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--papers", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def collect_case_paper_ids(path: Path) -> list[str]:
    rows = load_jsonl(path)
    out = []
    seen: Set[str] = set()
    for row in rows:
        for key in ("history_paper_ids", "future_paper_ids"):
            for paper_id in row.get(key) or []:
                paper_id = str(paper_id or "").strip()
                if not paper_id or paper_id in seen:
                    continue
                seen.add(paper_id)
                out.append(paper_id)
    return out


def main() -> None:
    args = parse_args()
    papers = load_rows_by_paper_id(args.papers)
    paper_ids = collect_case_paper_ids(Path(args.cases))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for paper_id in paper_ids:
            row = papers.get(paper_id)
            if not row:
                continue
            content_row = build_fallback_content_row(
                paper_id=paper_id,
                source_paper_id=str(row.get("source_paper_id") or paper_id),
                title=str(row.get("title") or ""),
                abstract=str(row.get("abstract") or ""),
            )
            handle.write(json.dumps(content_row, ensure_ascii=False) + "\n")
            count += 1
    print("output", out_path)
    print("rows", count)


if __name__ == "__main__":
    main()
