from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from requests import RequestException

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.benchmark import load_rows_by_paper_id
from researchworld.content import (
    arxiv_html_url,
    arxiv_source_url,
    build_content_row_from_html,
    build_content_row_from_source_blob,
    build_fallback_content_row,
    fetch_url_bytes,
)
from researchworld.technical_vision import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch source-first full-text content for case-linked benchmark papers.")
    parser.add_argument("--cases", required=True, help="Research case JSONL.")
    parser.add_argument("--papers", required=True, help="Merged papers JSONL.")
    parser.add_argument("--output", required=True, help="Content JSONL output.")
    parser.add_argument("--error-output", required=True, help="Error JSONL output.")
    parser.add_argument("--raw-dir", required=True, help="Directory for cached raw source archives and html.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent fetch workers.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout.")
    parser.add_argument("--resume", action="store_true", help="Skip already fetched paper_ids.")
    parser.add_argument("--limit", type=int, default=0, help="Optional paper limit.")
    return parser.parse_args()


def load_seen_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen: Set[str] = set()
    for row in load_jsonl(path):
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            seen.add(paper_id)
    return seen


def collect_case_paper_ids(cases_path: Path) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for row in load_jsonl(cases_path):
        for key in ("history_paper_ids", "future_paper_ids"):
            for paper_id in row.get(key) or []:
                if isinstance(paper_id, str) and paper_id and paper_id not in seen:
                    seen.add(paper_id)
                    ordered.append(paper_id)
    return ordered


def write_bytes(path: Path, blob: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(blob)


def fetch_one(paper: Dict, raw_dir: Path, timeout: int) -> Tuple[bool, Dict]:
    paper_id = str(paper.get("paper_id") or "")
    source_paper_id = str(paper.get("source_paper_id") or paper_id)
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")

    try:
        source_blob, _ = fetch_url_bytes(arxiv_source_url(source_paper_id), timeout=timeout)
        write_bytes(raw_dir / f"{paper_id}.source.tar.gz", source_blob)
        return True, build_content_row_from_source_blob(
            paper_id=paper_id,
            source_paper_id=source_paper_id,
            title=title,
            abstract=abstract,
            blob=source_blob,
        )
    except Exception as source_exc:
        html_error = ""
        try:
            html_blob, content_type = fetch_url_bytes(arxiv_html_url(source_paper_id), timeout=timeout)
            if "text/html" in content_type:
                write_bytes(raw_dir / f"{paper_id}.html", html_blob)
                return True, build_content_row_from_html(
                    paper_id=paper_id,
                    source_paper_id=source_paper_id,
                    title=title,
                    abstract=abstract,
                    html=html_blob.decode("utf-8", errors="ignore"),
                )
        except Exception as html_exc:
            html_error = str(html_exc)

        fallback = build_fallback_content_row(
            paper_id=paper_id,
            source_paper_id=source_paper_id,
            title=title,
            abstract=abstract,
        )
        fallback["fetch_warnings"] = {
            "source_error": str(source_exc),
            "html_error": html_error,
        }
        if fallback.get("section_count"):
            return True, fallback
        return False, {
            "paper_id": paper_id,
            "source_paper_id": source_paper_id,
            "source_error": str(source_exc),
            "html_error": html_error,
        }


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    papers = load_rows_by_paper_id(args.papers)
    output_path = Path(args.output)
    error_path = Path(args.error_output)
    raw_dir = Path(args.raw_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    seen_ids = load_seen_ids(output_path) if args.resume else set()
    paper_ids = []
    for paper_id in collect_case_paper_ids(cases_path):
        if paper_id in seen_ids:
            continue
        if paper_id in papers:
            paper_ids.append(paper_id)
        if args.limit and len(paper_ids) >= args.limit:
            break

    mode = "a" if args.resume else "w"
    with open(output_path, mode, encoding="utf-8") as out_handle, open(error_path, mode, encoding="utf-8") as err_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(fetch_one, papers[paper_id], raw_dir, args.timeout): paper_id
                for paper_id in paper_ids
            }
            processed = 0
            for future in concurrent.futures.as_completed(futures):
                ok, row = future.result()
                handle = out_handle if ok else err_handle
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                processed += 1
                print(f"Fetched {processed}/{len(paper_ids)}: {row.get('paper_id')}")

    print(f"Output: {output_path}")
    print(f"Errors: {error_path}")


if __name__ == "__main__":
    main()
