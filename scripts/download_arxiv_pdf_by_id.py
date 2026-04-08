from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.content import arxiv_pdf_url
from researchworld.technical_vision import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download versioned arXiv PDFs by source_paper_id.")
    parser.add_argument("--out-dir", required=True, help="PDF output directory.")
    parser.add_argument("--manifest-out", default="", help="Optional JSONL manifest output.")
    parser.add_argument("--papers-jsonl", default="", help="Optional paper metadata JSONL used to map paper_id -> source_paper_id.")
    parser.add_argument("--paper-ids-file", default="", help="Optional paper_id file, one per line.")
    parser.add_argument("--source-paper-ids-file", default="", help="Optional source_paper_id file, one per line.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-unversioned", action="store_true", help="Allow raw arXiv ids without explicit version suffix.")
    return parser.parse_args()


def normalize_line_ids(path: str) -> List[str]:
    if not path:
        return []
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_papers_by_id(path: str) -> Dict[str, Dict]:
    if not path:
        return {}
    rows: Dict[str, Dict] = {}
    for row in load_jsonl(path):
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id:
            rows[paper_id] = row
    return rows


def has_version_suffix(source_paper_id: str) -> bool:
    lowered = str(source_paper_id or "").strip().lower()
    return bool(lowered) and "v" in lowered and lowered.rsplit("v", 1)[-1].isdigit()


def canonical_paper_id(source_paper_id: str) -> str:
    text = str(source_paper_id or "").strip()
    if "v" in text and text.rsplit("v", 1)[-1].isdigit():
        return text.rsplit("v", 1)[0]
    return text


def iter_targets(args: argparse.Namespace) -> List[Dict]:
    papers_by_id = load_papers_by_id(args.papers_jsonl)
    targets: List[Dict] = []
    seen = set()

    for paper_id in normalize_line_ids(args.paper_ids_file):
        paper = papers_by_id.get(paper_id)
        if not paper:
            raise SystemExit(f"paper_id not found in papers-jsonl: {paper_id}")
        source_paper_id = str(paper.get("source_paper_id") or "").strip()
        if not source_paper_id:
            raise SystemExit(f"paper_id has empty source_paper_id: {paper_id}")
        key = (paper_id, source_paper_id)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            {
                "paper_id": paper_id,
                "source_paper_id": source_paper_id,
                "title": str(paper.get("title") or ""),
                "pdf_url": str(paper.get("pdf_url") or arxiv_pdf_url(source_paper_id)),
            }
        )

    for source_paper_id in normalize_line_ids(args.source_paper_ids_file):
        paper_id = canonical_paper_id(source_paper_id)
        paper = papers_by_id.get(paper_id, {})
        key = (paper_id, source_paper_id)
        if key in seen:
            continue
        seen.add(key)
        targets.append(
            {
                "paper_id": paper_id,
                "source_paper_id": source_paper_id,
                "title": str(paper.get("title") or ""),
                "pdf_url": str(paper.get("pdf_url") or arxiv_pdf_url(source_paper_id)),
            }
        )
    return targets


def download_one(target: Dict, out_dir: Path, *, timeout: int, allow_unversioned: bool, resume: bool) -> Tuple[bool, Dict]:
    paper_id = str(target.get("paper_id") or "").strip()
    source_paper_id = str(target.get("source_paper_id") or "").strip()
    if not source_paper_id:
        return False, {"paper_id": paper_id, "source_paper_id": source_paper_id, "error": "empty source_paper_id"}
    if not allow_unversioned and not has_version_suffix(source_paper_id):
        return False, {
            "paper_id": paper_id,
            "source_paper_id": source_paper_id,
            "error": "source_paper_id must include explicit arXiv version suffix",
        }

    pdf_url = str(target.get("pdf_url") or arxiv_pdf_url(source_paper_id))
    out_path = out_dir / f"{source_paper_id}.pdf"
    tmp_path = out_dir / f"{source_paper_id}.pdf.part"
    if resume and out_path.exists() and out_path.stat().st_size > 0:
        return True, {
            "paper_id": paper_id,
            "source_paper_id": source_paper_id,
            "pdf_url": pdf_url,
            "pdf_path": str(out_path),
            "status": "cached",
        }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        errors: List[str] = []
        proc = None
        downloader = ""

        aria2 = shutil.which("aria2c")
        if aria2:
            proc = subprocess.run(
                [
                    aria2,
                    "--dir",
                    str(out_path.parent),
                    "--out",
                    tmp_path.name,
                    "--max-connection-per-server=8",
                    "--split=8",
                    "--min-split-size=1M",
                    "--continue=true",
                    "--timeout=120",
                    "--connect-timeout=20",
                    "--summary-interval=0",
                    "--console-log-level=warn",
                    "--file-allocation=none",
                    "--user-agent=ResearchTrajectoryLab/0.1",
                    pdf_url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout + 10,
                check=False,
            )
            if proc.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
                downloader = "aria2c"
            else:
                errors.append(f"aria2c rc={proc.returncode} stderr={proc.stderr.strip()[:300]} stdout={proc.stdout.strip()[:300]}")

        if not downloader:
            curl = shutil.which("curl")
            if not curl:
                raise RuntimeError("; ".join(errors) if errors else "neither aria2c nor curl is available")
            proc = subprocess.run(
                [
                    curl,
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    str(timeout),
                    "-A",
                    "ResearchTrajectoryLab/0.1",
                    "-C",
                    "-",
                    "-o",
                    str(tmp_path),
                    pdf_url,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout + 10,
                check=False,
            )
            if proc.returncode != 0:
                errors.append(f"curl rc={proc.returncode} stderr={proc.stderr.strip()[:400]}")
                raise RuntimeError("; ".join(errors))
            downloader = "curl"
        elapsed = time.monotonic() - started
        total_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
        if total_bytes <= 0:
            raise RuntimeError("empty file after download")
        with tmp_path.open("rb") as check_handle:
            prefix = check_handle.read(8)
        if not prefix.startswith(b"%PDF-"):
            raise RuntimeError("downloaded file is not a valid PDF prefix")
        os.replace(tmp_path, out_path)
        return True, {
            "paper_id": paper_id,
            "source_paper_id": source_paper_id,
            "pdf_url": pdf_url,
            "pdf_path": str(out_path),
            "bytes": total_bytes,
            "elapsed_sec": round(elapsed, 3),
            "downloader": downloader,
            "status": "downloaded",
        }
    except Exception as exc:
        try:
            if tmp_path.exists():
                aria2_state = tmp_path.with_name(tmp_path.name + ".aria2")
                if not aria2_state.exists():
                    tmp_path.unlink()
        except Exception:
            pass
        return False, {
            "paper_id": paper_id,
            "source_paper_id": source_paper_id,
            "pdf_url": pdf_url,
            "error": str(exc),
        }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = iter_targets(args)
    if not targets:
        raise SystemExit("No download targets found.")

    manifest_path = Path(args.manifest_out) if args.manifest_out else None
    if manifest_path:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

    success_rows: List[Dict] = []
    error_rows: List[Dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                download_one,
                target,
                out_dir,
                timeout=args.timeout,
                allow_unversioned=args.allow_unversioned,
                resume=args.resume,
            ): target
            for target in targets
        }
        processed = 0
        for future in concurrent.futures.as_completed(futures):
            ok, row = future.result()
            processed += 1
            print(f"[download_arxiv_pdf_by_id] {processed}/{len(targets)} {row.get('source_paper_id')} ok={ok}", flush=True)
            if ok:
                success_rows.append(row)
            else:
                error_rows.append(row)

    if manifest_path:
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in success_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "target_count": len(targets),
        "downloaded_count": len(success_rows),
        "error_count": len(error_rows),
        "out_dir": str(out_dir),
        "manifest_out": str(manifest_path) if manifest_path else "",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if error_rows:
        print(json.dumps({"errors": error_rows[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
