from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, Tuple

import arxiv


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest arXiv papers month-by-month for a target benchmark domain.")
    parser.add_argument("--domain-id", required=True, help="Domain id defined in configs/domain_harvest_queries.yaml.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "domain_harvest_queries.yaml"),
        help="Harvest query config YAML.",
    )
    parser.add_argument("--start", default="", help="Optional override start datetime, e.g. 2023-01-01T00:00:00")
    parser.add_argument("--end", default="", help="Optional override end datetime, e.g. 2026-02-01T00:00:00")
    parser.add_argument("--download-pdf", action="store_true", help="Download PDFs in addition to metadata.")
    parser.add_argument("--page-size", type=int, default=0, help="Optional override page size.")
    parser.add_argument("--delay-seconds", type=float, default=-1.0, help="Optional override request delay.")
    parser.add_argument("--num-retries", type=int, default=-1, help="Optional override retry count.")
    parser.add_argument("--max-results", type=int, default=0, help="Optional global cap for smoke tests.")
    parser.add_argument(
        "--min-window-days",
        type=int,
        default=1,
        help="Smallest time window size to keep splitting down to when arXiv returns server errors.",
    )
    return parser.parse_args()


def parse_dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


def yyyymmddhhmm(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def month_ranges(start: datetime, end: datetime) -> Iterator[Tuple[datetime, datetime]]:
    cur = datetime(start.year, start.month, 1, 0, 0)
    while cur < end:
        if cur.month == 12:
            nxt = datetime(cur.year + 1, 1, 1, 0, 0)
        else:
            nxt = datetime(cur.year, cur.month + 1, 1, 0, 0)
        yield max(cur, start), min(nxt, end)
        cur = nxt


def load_downloaded_ids(meta_file: Path) -> set[str]:
    ids: set[str] = set()
    if not meta_file.exists():
        return ids
    with open(meta_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            paper_id = obj.get("id")
            if isinstance(paper_id, str) and paper_id.strip():
                ids.add(paper_id.strip())
    return ids


def safe_download_pdf(result: arxiv.Result, pdf_dir: Path, paper_id: str) -> bool:
    out_path = pdf_dir / f"{paper_id}.pdf"
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    try:
        result.download_pdf(dirpath=str(pdf_dir), filename=f"{paper_id}.pdf")
        return True
    except Exception:
        return False


def load_domain_cfg(config_path: Path, domain_id: str) -> Dict:
    obj = load_yaml(config_path)
    domains = obj.get("domains") or {}
    if domain_id not in domains:
        raise SystemExit(f"Unknown domain_id: {domain_id}")
    cfg = domains[domain_id]
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid config for domain_id: {domain_id}")
    return cfg


def window_label(start: datetime, end: datetime) -> str:
    return f"{start:%Y-%m-%d} -> {end:%Y-%m-%d}"


def iter_results_with_fallback(
    client: arxiv.Client,
    base_query: str,
    seg_start: datetime,
    seg_end: datetime,
    *,
    domain_id: str,
    min_window: timedelta,
    retry_same_window: int = 1,
) -> Iterator[arxiv.Result]:
    date_filter = f"submittedDate:[{yyyymmddhhmm(seg_start)} TO {yyyymmddhhmm(seg_end)}]"
    query = f"{base_query} AND {date_filter}"
    search = arxiv.Search(
        query=query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Ascending,
    )

    try:
        yield from client.results(search)
        return
    except Exception as exc:
        exc_text = str(exc)
        if retry_same_window > 0 and ("HTTP 429" in exc_text or "HTTP 503" in exc_text):
            wait_seconds = 60 if "HTTP 429" in exc_text else 20
            print(
                f"[WARN] [{domain_id}] retrying window {window_label(seg_start, seg_end)} "
                f"after {wait_seconds}s due to {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(wait_seconds)
            yield from iter_results_with_fallback(
                client,
                base_query,
                seg_start,
                seg_end,
                domain_id=domain_id,
                min_window=min_window,
                retry_same_window=retry_same_window - 1,
            )
            return

        span = seg_end - seg_start
        if span <= min_window:
            print(
                f"[WARN] [{domain_id}] failed on narrow window {window_label(seg_start, seg_end)}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(5)
            return

        mid = seg_start + (span / 2)
        if mid <= seg_start or mid >= seg_end:
            print(
                f"[WARN] [{domain_id}] cannot split window {window_label(seg_start, seg_end)} further: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(5)
            return

        print(
            f"[WARN] [{domain_id}] splitting failed window {window_label(seg_start, seg_end)} "
            f"into {window_label(seg_start, mid)} and {window_label(mid, seg_end)} "
            f"after {type(exc).__name__}: {exc}",
            flush=True,
        )
        time.sleep(5)
        yield from iter_results_with_fallback(
            client,
            base_query,
            seg_start,
            mid,
            domain_id=domain_id,
            min_window=min_window,
            retry_same_window=retry_same_window,
        )
        yield from iter_results_with_fallback(
            client,
            base_query,
            mid,
            seg_end,
            domain_id=domain_id,
            min_window=min_window,
            retry_same_window=retry_same_window,
        )


def main() -> None:
    args = parse_args()
    domain_cfg = load_domain_cfg(Path(args.config), args.domain_id)

    save_dir = ROOT / str(domain_cfg["save_dir"])
    meta_file = ROOT / str(domain_cfg["meta_file"])
    pdf_dir = save_dir / "pdfs"
    download_pdf = bool(domain_cfg.get("download_pdf", False) or args.download_pdf)
    page_size = int(args.page_size or domain_cfg.get("page_size", 200))
    delay_seconds = float(args.delay_seconds if args.delay_seconds >= 0 else domain_cfg.get("delay_seconds", 3.0))
    num_retries = int(args.num_retries if args.num_retries >= 0 else domain_cfg.get("num_retries", 8))
    start = parse_dt(args.start or str(domain_cfg["start"]))
    end = parse_dt(args.end or str(domain_cfg["end"]))
    base_query = str(domain_cfg["base_query"]).strip()
    min_window = timedelta(days=max(args.min_window_days, 1))

    save_dir.mkdir(parents=True, exist_ok=True)
    meta_file.parent.mkdir(parents=True, exist_ok=True)
    if download_pdf:
        pdf_dir.mkdir(parents=True, exist_ok=True)

    downloaded_ids = load_downloaded_ids(meta_file)
    print(f"Domain: {args.domain_id}")
    print(f"Already have {len(downloaded_ids)} papers")
    print(f"Meta file: {meta_file}")

    client = arxiv.Client(
        page_size=page_size,
        delay_seconds=delay_seconds,
        num_retries=num_retries,
    )

    count = len(downloaded_ids)
    with open(meta_file, "a", encoding="utf-8") as handle:
        for seg_start, seg_end in month_ranges(start, end):
            month_tag = seg_start.strftime("%Y-%m")
            print(f"[{args.domain_id}] harvesting {month_tag} ...")
            time.sleep(0.5)
            for result in iter_results_with_fallback(
                client,
                base_query,
                seg_start,
                seg_end,
                domain_id=args.domain_id,
                min_window=min_window,
            ):
                paper_id = result.get_short_id()
                if paper_id in downloaded_ids:
                    continue

                meta = {
                    "id": paper_id,
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "abstract": result.summary,
                    "published": result.published.isoformat() if result.published else None,
                    "updated": result.updated.isoformat() if result.updated else None,
                    "categories": list(result.categories),
                    "pdf_url": result.pdf_url,
                    "entry_id": result.entry_id,
                }
                handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
                handle.flush()

                if download_pdf:
                    safe_download_pdf(result, pdf_dir, paper_id)

                downloaded_ids.add(paper_id)
                count += 1
                if count % 50 == 0:
                    print(f"[{args.domain_id}] total downloaded: {count}")
                if args.max_results and count >= args.max_results:
                    print(f"[{args.domain_id}] reached max_results={args.max_results}")
                    return

    print(f"Finished. Total downloaded: {count}")


if __name__ == "__main__":
    main()
