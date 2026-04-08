from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.config import load_yaml
from researchworld.corpus import canonical_arxiv_id


S2_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "abstract",
        "authors",
        "year",
        "publicationDate",
        "externalIds",
        "citationCount",
        "venue",
        "publicationVenue",
        "publicationTypes",
        "openAccessPdf",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest arXiv-identified papers for a domain via Semantic Scholar bulk search."
    )
    parser.add_argument("--domain-id", required=True, help="Domain id defined in configs/domain_harvest_queries.yaml.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "domain_harvest_queries.yaml"),
        help="Harvest query config YAML.",
    )
    parser.add_argument(
        "--seed-config",
        default=str(ROOT / "configs" / "domain_seed_queries.yaml"),
        help="Seed query YAML used for local filtering.",
    )
    parser.add_argument("--start", default="", help="Optional override start date, e.g. 2023-01-01")
    parser.add_argument("--end", default="", help="Optional override end date, e.g. 2026-03-01")
    parser.add_argument("--sleep", type=float, default=0.3, help="Sleep seconds between API calls.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds.")
    parser.add_argument("--max-pages-per-query", type=int, default=0, help="Optional page cap per query.")
    parser.add_argument(
        "--stop-after-empty-pages",
        type=int,
        default=2,
        help="Stop paginating a query after this many consecutive pages with zero kept papers.",
    )
    parser.add_argument(
        "--min-positive-hits",
        type=int,
        default=2,
        help="Minimum positive keyword hits required after local filtering.",
    )
    return parser.parse_args()


def parse_dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def compile_term_pattern(terms: list[str]) -> re.Pattern[str] | None:
    cleaned = sorted({normalize_text(term).lower() for term in terms if normalize_text(term)}, key=len, reverse=True)
    if not cleaned:
        return None
    return re.compile("|".join(re.escape(term) for term in cleaned), re.I)


def term_hits(pattern: re.Pattern[str] | None, text: str) -> list[str]:
    if pattern is None:
        return []
    seen: set[str] = set()
    hits: list[str] = []
    for match in pattern.finditer(text.lower()):
        value = match.group(0).lower()
        if value not in seen:
            seen.add(value)
            hits.append(value)
    return hits


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except Exception:
            pass
    return min(60.0, 2.0**attempt)


def fetch_page(query: str, *, token: str | None, timeout: int) -> dict[str, Any]:
    params = {"query": query, "fields": S2_FIELDS}
    if token:
        params["token"] = token
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urlencode(params)
    headers = {"User-Agent": "ResearchTrajectoryLab semantic-scholar bulk harvest"}

    last_error: Exception | None = None
    for attempt in range(5):
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code in {429, 500, 502, 503, 504} and attempt < 4:
                time.sleep(_retry_delay(attempt, exc.headers.get("Retry-After")))
                continue
            raise
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(_retry_delay(attempt, None))
                continue
            raise
    if last_error is not None:
        raise last_error
    return {}


def paper_date(item: dict[str, Any]) -> str:
    date = normalize_text(item.get("publicationDate") or "")
    if date:
        return date
    year = item.get("year")
    if isinstance(year, int):
        return f"{year:04d}-01-01"
    return ""


def in_range(date_text: str, *, start: str, end: str) -> bool:
    if not date_text:
        return False
    return start <= date_text < end


def to_record(
    item: dict[str, Any],
    *,
    query: str,
    positive_hits: list[str],
    caution_hits: list[str],
    negative_hits: list[str],
) -> dict[str, Any] | None:
    external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    arxiv_id = normalize_text(external_ids.get("ArXiv") or "")
    if not arxiv_id:
        return None

    canonical_id, versioned_id = canonical_arxiv_id(arxiv_id)
    authors = item.get("authors") or []
    author_names = [normalize_text(author.get("name")) for author in authors if normalize_text(author.get("name"))]
    published = paper_date(item)

    return {
        "id": versioned_id or canonical_id,
        "title": normalize_text(item.get("title") or ""),
        "authors": author_names,
        "abstract": normalize_text(item.get("abstract") or ""),
        "published": published or None,
        "updated": published or None,
        "categories": [],
        "pdf_url": f"https://arxiv.org/pdf/{canonical_id}.pdf",
        "entry_id": f"https://arxiv.org/abs/{canonical_id}",
        "semantic_scholar_paper_id": item.get("paperId"),
        "citation_count": item.get("citationCount"),
        "venue": item.get("venue"),
        "publication_venue": item.get("publicationVenue"),
        "publication_types": item.get("publicationTypes"),
        "external_ids": external_ids,
        "harvest_source": "semantic_scholar_bulk_search",
        "harvest_query": query,
        "harvest_positive_hits": positive_hits,
        "harvest_caution_hits": caution_hits,
        "harvest_negative_hits": negative_hits,
    }


def choose_better(current: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    current_score = (
        len(current.get("harvest_positive_hits") or []),
        len(normalize_text(current.get("abstract") or "")),
        1 if current.get("published") else 0,
    )
    candidate_score = (
        len(candidate.get("harvest_positive_hits") or []),
        len(normalize_text(candidate.get("abstract") or "")),
        1 if candidate.get("published") else 0,
    )
    return candidate if candidate_score > current_score else current


def load_domain_cfg(config_path: Path, domain_id: str) -> dict[str, Any]:
    obj = load_yaml(config_path)
    domains = obj.get("domains") or {}
    if domain_id not in domains:
        raise SystemExit(f"Unknown domain_id: {domain_id}")
    cfg = domains[domain_id]
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid config for domain_id: {domain_id}")
    return cfg


def load_seed_cfg(seed_path: Path, domain_id: str) -> dict[str, Any]:
    obj = load_yaml(seed_path)
    domains = obj.get("domains") or {}
    cfg = domains.get(domain_id) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid seed config for domain_id: {domain_id}")
    return cfg


def main() -> None:
    args = parse_args()
    domain_cfg = load_domain_cfg(Path(args.config), args.domain_id)
    seed_cfg = load_seed_cfg(Path(args.seed_config), args.domain_id)

    queries = [normalize_text(query) for query in (domain_cfg.get("semantic_scholar_queries") or []) if normalize_text(query)]
    if not queries:
        raise SystemExit(f"No semantic_scholar_queries configured for {args.domain_id}")

    positive_terms = list(seed_cfg.get("positive_terms") or [])
    caution_terms = list(seed_cfg.get("caution_terms") or [])
    negative_terms = list(seed_cfg.get("negative_terms") or [])

    positive_pattern = compile_term_pattern(positive_terms)
    caution_pattern = compile_term_pattern(caution_terms)
    negative_pattern = compile_term_pattern(negative_terms)

    meta_file = ROOT / str(domain_cfg["meta_file"])
    meta_file.parent.mkdir(parents=True, exist_ok=True)

    start = (args.start or str(domain_cfg["start"])).split("T", 1)[0]
    end = (args.end or str(domain_cfg["end"])).split("T", 1)[0]
    parse_dt(start)
    parse_dt(end)

    print(f"Domain: {args.domain_id}")
    print(f"Output: {meta_file}")
    print(f"Date range: {start} -> {end}")
    print(f"Queries: {len(queries)}")

    kept: dict[str, dict[str, Any]] = {}
    summary_queries: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()

    for query in queries:
        token: str | None = None
        page_idx = 0
        empty_keep_streak = 0
        query_seen = 0
        query_kept = 0
        query_dedup_new = 0

        print(f"[{args.domain_id}] query={query!r}")
        while True:
            if args.max_pages_per_query and page_idx >= args.max_pages_per_query:
                break

            data = fetch_page(query, token=token, timeout=args.timeout)
            rows = data.get("data") or []
            token = data.get("token")
            page_idx += 1
            page_kept = 0

            for item in rows:
                query_seen += 1
                if not isinstance(item, dict):
                    reason_counter["non_dict"] += 1
                    continue

                external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
                if not normalize_text(external_ids.get("ArXiv") or ""):
                    reason_counter["missing_arxiv_id"] += 1
                    continue

                published = paper_date(item)
                if not in_range(published, start=start, end=end):
                    reason_counter["out_of_range"] += 1
                    continue

                text = f"{normalize_text(item.get('title') or '')}\n{normalize_text(item.get('abstract') or '')}"
                positive_hits = term_hits(positive_pattern, text)
                caution_hits = term_hits(caution_pattern, text)
                negative_hits = term_hits(negative_pattern, text)

                if len(positive_hits) < args.min_positive_hits:
                    reason_counter["too_few_positive_hits"] += 1
                    continue
                if negative_hits:
                    reason_counter["negative_hit"] += 1
                    continue

                record = to_record(
                    item,
                    query=query,
                    positive_hits=positive_hits,
                    caution_hits=caution_hits,
                    negative_hits=negative_hits,
                )
                if record is None:
                    reason_counter["record_build_failed"] += 1
                    continue

                page_kept += 1
                query_kept += 1
                canonical_id, _ = canonical_arxiv_id(str(record["id"]))
                if canonical_id not in kept:
                    kept[canonical_id] = record
                    query_dedup_new += 1
                else:
                    kept[canonical_id] = choose_better(kept[canonical_id], record)

            if page_kept == 0:
                empty_keep_streak += 1
            else:
                empty_keep_streak = 0

            print(
                f"  page={page_idx} fetched={len(rows)} kept={page_kept} unique_total={len(kept)} "
                f"next_token={bool(token)}",
                flush=True,
            )

            if not token:
                break
            if args.stop_after_empty_pages and empty_keep_streak >= args.stop_after_empty_pages:
                break
            time.sleep(args.sleep)

        summary_queries.append(
            {
                "query": query,
                "pages": page_idx,
                "seen": query_seen,
                "kept_before_dedup": query_kept,
                "new_unique": query_dedup_new,
            }
        )

    rows = sorted(kept.values(), key=lambda row: (row.get("published") or "", row.get("id") or ""))
    with meta_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "domain_id": args.domain_id,
        "output_path": str(meta_file),
        "date_range": {"start": start, "end_exclusive": end},
        "query_count": len(queries),
        "paper_count": len(rows),
        "queries": summary_queries,
        "filter_reasons": dict(reason_counter),
    }
    summary_path = meta_file.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Saved {len(rows)} papers -> {meta_file}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
