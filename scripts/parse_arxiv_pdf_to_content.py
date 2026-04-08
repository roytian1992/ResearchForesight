from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.content import build_content_row_from_pdf_text, build_fallback_content_row, extract_pdf_text_with_pdftotext
from researchworld.technical_vision import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse cached arXiv PDFs into content.jsonl rows.")
    parser.add_argument("--papers-jsonl", required=True, help="Paper metadata JSONL.")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing cached PDFs named by source_paper_id.pdf.")
    parser.add_argument("--output", required=True, help="Output content JSONL.")
    parser.add_argument("--error-output", required=True, help="Error JSONL.")
    parser.add_argument("--paper-ids-file", default="", help="Optional paper_id file.")
    parser.add_argument("--source-paper-ids-file", default="", help="Optional source_paper_id file.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fallback-abstract", action="store_true", help="Write abstract_only fallback rows when PDF parsing fails.")
    return parser.parse_args()


def load_papers_by_id(path: str) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}
    for row in load_jsonl(path):
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id:
            rows[paper_id] = row
    return rows


def normalize_line_ids(path: str) -> List[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_paper_id(source_paper_id: str) -> str:
    text = str(source_paper_id or "").strip()
    if "v" in text and text.rsplit("v", 1)[-1].isdigit():
        return text.rsplit("v", 1)[0]
    return text


def load_seen_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen: Set[str] = set()
    for row in load_jsonl(path):
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id:
            seen.add(paper_id)
    return seen


def build_targets(args: argparse.Namespace, papers_by_id: Dict[str, Dict]) -> List[Dict]:
    selected: List[Dict] = []
    seen = set()

    if args.paper_ids_file:
        for paper_id in normalize_line_ids(args.paper_ids_file):
            if paper_id not in papers_by_id:
                raise SystemExit(f"paper_id not found in papers-jsonl: {paper_id}")
            key = str(paper_id)
            if key in seen:
                continue
            seen.add(key)
            selected.append(papers_by_id[paper_id])

    if args.source_paper_ids_file:
        for source_paper_id in normalize_line_ids(args.source_paper_ids_file):
            paper_id = canonical_paper_id(source_paper_id)
            if paper_id not in papers_by_id:
                raise SystemExit(f"source_paper_id maps to missing paper_id in papers-jsonl: {source_paper_id}")
            key = str(paper_id)
            if key in seen:
                continue
            seen.add(key)
            selected.append(papers_by_id[paper_id])

    if not selected:
        selected = list(papers_by_id.values())
    return selected


def resolve_pdf_path(pdf_dir: Path, paper: Dict) -> Path | None:
    source_paper_id = str(paper.get("source_paper_id") or "").strip()
    paper_id = str(paper.get("paper_id") or "").strip()
    candidates = []
    if source_paper_id:
        candidates.append(pdf_dir / f"{source_paper_id}.pdf")
    if paper_id:
        candidates.append(pdf_dir / f"{paper_id}.pdf")
    for path in candidates:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def parse_one(paper: Dict, pdf_dir: Path, *, timeout: int, fallback_abstract: bool) -> Tuple[bool, Dict]:
    paper_id = str(paper.get("paper_id") or "").strip()
    source_paper_id = str(paper.get("source_paper_id") or paper_id).strip()
    title = str(paper.get("title") or "")
    abstract = str(paper.get("abstract") or "")
    pdf_path = resolve_pdf_path(pdf_dir, paper)
    if pdf_path is None:
        if fallback_abstract:
            row = build_fallback_content_row(
                paper_id=paper_id,
                source_paper_id=source_paper_id,
                title=title,
                abstract=abstract,
            )
            row["fetch_warnings"] = {"pdf_error": "missing pdf file"}
            return True, row
        return False, {"paper_id": paper_id, "source_paper_id": source_paper_id, "error": "missing pdf file"}
    try:
        pdf_text = extract_pdf_text_with_pdftotext(pdf_path, timeout=timeout)
        row = build_content_row_from_pdf_text(
            paper_id=paper_id,
            source_paper_id=source_paper_id,
            title=title,
            abstract=abstract,
            pdf_text=pdf_text,
            source_url=str(paper.get("pdf_url") or ""),
        )
        row["pdf_path"] = str(pdf_path)
        row["pdf_char_count"] = len(pdf_text)
        return True, row
    except Exception as exc:
        if fallback_abstract:
            row = build_fallback_content_row(
                paper_id=paper_id,
                source_paper_id=source_paper_id,
                title=title,
                abstract=abstract,
            )
            row["fetch_warnings"] = {"pdf_error": str(exc), "pdf_path": str(pdf_path)}
            return True, row
        return False, {"paper_id": paper_id, "source_paper_id": source_paper_id, "error": str(exc), "pdf_path": str(pdf_path)}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    error_path = Path(args.error_output)
    pdf_dir = Path(args.pdf_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    papers_by_id = load_papers_by_id(args.papers_jsonl)
    targets = build_targets(args, papers_by_id)
    seen_ids = load_seen_ids(output_path) if args.resume else set()
    targets = [paper for paper in targets if str(paper.get("paper_id") or "") not in seen_ids]

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as out_handle, error_path.open(mode, encoding="utf-8") as err_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    parse_one,
                    paper,
                    pdf_dir,
                    timeout=args.timeout,
                    fallback_abstract=args.fallback_abstract,
                ): paper
                for paper in targets
            }
            processed = 0
            for future in concurrent.futures.as_completed(futures):
                ok, row = future.result()
                processed += 1
                print(f"[parse_arxiv_pdf_to_content] {processed}/{len(targets)} {row.get('source_paper_id')} ok={ok}", flush=True)
                handle = out_handle if ok else err_handle
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()

    print(json.dumps({"output": str(output_path), "error_output": str(error_path), "target_count": len(targets)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
