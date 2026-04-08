from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener

TOP_VENUE_RULES = [
    ("NeurIPS", ["neural information processing systems", "neurips"]),
    ("ICML", ["international conference on machine learning", "icml"]),
    ("ICLR", ["international conference on learning representations", "iclr"]),
    ("AAAI", ["aaai conference on artificial intelligence", "aaai"]),
    ("IJCAI", ["international joint conference on artificial intelligence", "ijcai"]),
    ("ACL", ["annual meeting of the association for computational linguistics", " acl ", "proceedings of the 61st annual meeting of the association for computational linguistics", "findings of the association for computational linguistics", "association for computational linguistics"]),
    ("EMNLP", ["conference on empirical methods in natural language processing", "emnlp"]),
    ("NAACL", ["north american chapter of the association for computational linguistics", "naacl"]),
    ("COLING", ["international conference on computational linguistics", "coling"]),
    ("SIGIR", ["international acm sigir conference", "sigir"]),
    ("WWW", ["web conference", "world wide web conference", "acm web conference"]),
    ("WSDM", ["web search and data mining", "wsdm"]),
    ("KDD", ["knowledge discovery and data mining", "kdd"]),
    ("CVPR", ["computer vision and pattern recognition", "cvpr"]),
    ("ICCV", ["international conference on computer vision", "iccv"]),
    ("ECCV", ["european conference on computer vision", "eccv"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich papers with publication venue and citation metadata from OpenAlex.")
    parser.add_argument("--input", required=True, help="Input JSONL with paper_id/title/published/authors.")
    parser.add_argument("--output", required=True, help="Output JSONL sidecar file.")
    parser.add_argument("--summary-output", required=True, help="Output summary JSON.")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between requests.")
    parser.add_argument("--resume", action="store_true", help="Skip paper_ids already present in the output.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to process.")
    parser.add_argument("--sort-by-published", action="store_true", help="Sort ascending by published time before processing.")
    parser.add_argument("--mailto", default="local@example.com", help="Mailto parameter for OpenAlex politeness pool.")
    parser.add_argument("--per-page", type=int, default=8, help="How many OpenAlex candidates to request.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers for OpenAlex lookups.")
    return parser.parse_args()


def normalize_title(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def last_name(name: str) -> str:
    parts = normalize_name(name).split()
    return parts[-1] if parts else ""


def venue_bucket(*venue_texts: str) -> str:
    hay = " ".join(str(x or "") for x in venue_texts).lower()
    hay = f" {hay} "
    for bucket, needles in TOP_VENUE_RULES:
        if any(needle in hay for needle in needles):
            return bucket
    return ""


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            paper_id = row.get("paper_id")
            if isinstance(paper_id, str) and paper_id:
                seen.add(paper_id)
    return seen


def parse_year(text: str) -> int | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).year
    except Exception:
        return None


def work_title(result: dict[str, Any]) -> str:
    return str(result.get("title") or result.get("display_name") or "")


def is_preprint_like(result: dict[str, Any]) -> bool:
    if str(result.get("type") or "") == "preprint":
        return True
    doi = str(result.get("doi") or "").lower()
    if "10.48550/arxiv" in doi:
        return True
    primary = result.get("primary_location") or {}
    source = primary.get("source") or {}
    names = [
        str(source.get("display_name") or "").lower(),
        str(primary.get("raw_source_name") or "").lower(),
    ]
    return any("arxiv" in name for name in names)


def candidate_context(result: dict[str, Any]) -> dict[str, Any]:
    primary = result.get("primary_location") or {}
    source = primary.get("source") or {}
    raw_source_name = str(primary.get("raw_source_name") or "").strip()
    source_display_name = str(source.get("display_name") or "").strip()
    raw_type = str(primary.get("raw_type") or "").strip().lower()
    source_type = str(source.get("type") or "").strip().lower()
    doi = str(result.get("doi") or "")
    return {
        "primary": primary,
        "source": source,
        "raw_source_name": raw_source_name,
        "source_display_name": source_display_name,
        "raw_type": raw_type,
        "source_type": source_type,
        "doi": doi,
        "has_non_arxiv_doi": bool(doi and "10.48550/arxiv" not in doi.lower()),
        "venue_bucket": venue_bucket(raw_source_name, source_display_name),
    }


def score_candidate(row: dict[str, Any], result: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    src_norm = normalize_title(str(row.get("title") or ""))
    dst_norm = normalize_title(work_title(result))
    title_sim = SequenceMatcher(None, src_norm, dst_norm).ratio()
    exact = src_norm == dst_norm

    src_year = parse_year(str(row.get("published") or ""))
    dst_year = result.get("publication_year")
    year_score = 0.0
    if isinstance(dst_year, int) and src_year is not None:
        diff = dst_year - src_year
        if 0 <= diff <= 3:
            year_score = 1.0
        elif -1 <= diff <= 4:
            year_score = 0.5
        elif abs(diff) <= 5:
            year_score = 0.2

    src_authors = {last_name(name) for name in (row.get("authors") or []) if last_name(name)}
    dst_authors = {
        last_name((auth.get("author") or {}).get("display_name") or "")
        for auth in (result.get("authorships") or [])
        if last_name((auth.get("author") or {}).get("display_name") or "")
    }
    overlap = len(src_authors & dst_authors)
    overlap_ratio = overlap / max(1, min(max(len(src_authors), 1), 3))

    ctx = candidate_context(result)
    version = str((ctx["primary"] or {}).get("version") or "").lower()
    raw_type = ctx["raw_type"]
    source_type = ctx["source_type"]
    is_preprint = is_preprint_like(result)

    score = 0.0
    score += 5.0 if exact else 3.0 * title_sim
    score += year_score
    score += 1.5 * overlap_ratio
    if not is_preprint:
        score += 2.0
    if version == "publishedversion":
        score += 1.0
    elif version == "acceptedversion":
        score += 0.4
    if "proceedings" in raw_type:
        score += 1.0
    if "journal" in raw_type:
        score += 0.8
    if source_type in {"conference", "journal"}:
        score += 0.8
    if ctx["has_non_arxiv_doi"]:
        score += 0.8
    if ctx["venue_bucket"]:
        score += 1.0

    meta = {
        "title_similarity": round(title_sim, 4),
        "exact_title": exact,
        "author_overlap": overlap,
        "author_overlap_ratio": round(overlap_ratio, 4),
        "year_score": year_score,
        "venue_bucket": ctx["venue_bucket"],
        "raw_source_name": ctx["raw_source_name"],
        "source_display_name": ctx["source_display_name"],
        "raw_type": raw_type,
        "source_type": source_type,
        "has_non_arxiv_doi": ctx["has_non_arxiv_doi"],
        "is_preprint_like": is_preprint,
        "version": version,
        "cited_by_count": result.get("cited_by_count"),
    }
    return score, meta


def publication_confidence(score: float, meta: dict[str, Any]) -> str:
    if meta["is_preprint_like"]:
        return "low"
    trusted_venue = bool(
        meta["has_non_arxiv_doi"]
        or meta["raw_source_name"]
        or meta["source_display_name"]
        or meta["raw_type"] in {"journal-article", "proceedings-article", "book-chapter"}
        or meta["source_type"] in {"journal", "conference"}
        or meta["venue_bucket"]
    )
    if meta["exact_title"] and meta["author_overlap"] >= 1 and trusted_venue:
        return "high"
    if score >= 6.5 and trusted_venue and meta["title_similarity"] >= 0.95:
        return "medium"
    return "low"


def build_session():
    return build_opener()


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except Exception:
            pass
    return min(60.0, 2.0**attempt)


def fetch_openalex(session, *, title: str, per_page: int, mailto: str) -> list[dict[str, Any]]:
    params = {"search": title, "per-page": per_page}
    if mailto:
        params["mailto"] = mailto
    url = "https://api.openalex.org/works?" + urlencode(params)
    headers = {"User-Agent": "ResearchWorld publication enrichment"}

    last_error: Exception | None = None
    for attempt in range(5):
        request = Request(url, headers=headers)
        try:
            with session.open(request, timeout=30) as response:
                payload = json.load(response)
            return payload.get("results", [])
        except HTTPError as exc:
            last_error = exc
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                try:
                    retry_seconds = float(retry_after) if retry_after is not None else None
                except Exception:
                    retry_seconds = None
                if retry_seconds is not None and retry_seconds > 600:
                    raise RuntimeError(
                        f"OpenAlex rate limit exceeded; retry-after={int(retry_seconds)}s looks like daily budget exhaustion."
                    ) from exc
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
    return []


_THREAD_LOCAL = threading.local()


def get_thread_session():
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = build_session()
        _THREAD_LOCAL.session = session
    return session


def enrich_one_row(row: dict[str, Any], *, sleep: float, per_page: int, mailto: str) -> dict[str, Any]:
    paper_id = str(row.get("paper_id") or "")
    title = str(row.get("title") or "")
    record: dict[str, Any] = {
        "paper_id": paper_id,
        "source_paper_id": row.get("source_paper_id"),
        "title": title,
        "published": row.get("published"),
        "authors": row.get("authors") or [],
        "status": "unmatched",
        "match_source": "openalex_title_search",
    }
    try:
        session = get_thread_session()
        results = fetch_openalex(session, title=title, per_page=per_page, mailto=mailto)
        scored = []
        for result in results:
            score, meta = score_candidate(row, result)
            scored.append((score, meta, result))
        scored.sort(key=lambda item: item[0], reverse=True)

        best_preprint = None
        best_publication = None
        for score, meta, result in scored:
            if meta["is_preprint_like"] and best_preprint is None and meta["title_similarity"] >= 0.95:
                best_preprint = (score, meta, result)
            conf = publication_confidence(score, meta)
            if best_publication is None and conf != "low":
                best_publication = (score, meta, result, conf)

        if best_preprint is not None:
            _, pre_meta, pre_result = best_preprint
            record.update(
                {
                    "preprint_work_id": pre_result.get("id"),
                    "preprint_doi": pre_result.get("doi"),
                    "preprint_openalex_type": pre_result.get("type"),
                    "preprint_cited_by_count": pre_result.get("cited_by_count"),
                    "preprint_title_similarity": pre_meta["title_similarity"],
                }
            )

        if best_publication is not None:
            score, meta, best, conf = best_publication
            ctx = candidate_context(best)
            primary = ctx["primary"]
            source = ctx["source"]
            raw_name = ctx["raw_source_name"] or ctx["source_display_name"]
            raw_type = ctx["raw_type"] or "unknown"
            source_type = ctx["source_type"] or "unknown"
            bucket = meta["venue_bucket"] or None
            is_conf = bool(bucket) or ("proceedings" in raw_type) or source_type == "conference"
            pub_citations = best.get("cited_by_count")
            record.update(
                {
                    "status": "matched",
                    "match_confidence": conf,
                    "match_score": round(score, 4),
                    "matched_work_id": best.get("id"),
                    "matched_title": work_title(best),
                    "matched_publication_year": best.get("publication_year"),
                    "matched_doi": best.get("doi"),
                    "matched_type": best.get("type"),
                    "matched_cited_by_count": pub_citations,
                    "matched_ids": best.get("ids"),
                    "published_venue_name": raw_name,
                    "published_venue_type": "conference" if is_conf else ("journal" if ("journal" in raw_type or source_type == "journal") else raw_type),
                    "published_raw_type": raw_type,
                    "published_source_type": source_type,
                    "published_source_display_name": source.get("display_name"),
                    "published_version": primary.get("version"),
                    "is_top_ai_venue": bool(bucket),
                    "top_venue_bucket": bucket,
                    "preferred_cited_by_count": pub_citations,
                    "preferred_citation_source": "publication",
                    "evidence": meta,
                }
            )
        elif best_preprint is not None:
            record.update(
                {
                    "status": "citation_only",
                    "preferred_cited_by_count": record.get("preprint_cited_by_count"),
                    "preferred_citation_source": "preprint",
                }
            )
    except Exception as exc:
        record["status"] = "error"
        record["error"] = str(exc)
    finally:
        if sleep:
            time.sleep(sleep)
    return record


def summarize_sidecar(path: Path) -> dict[str, Any]:
    counts = Counter()
    venue_counts = Counter()
    raw_type_counts = Counter()
    source_type_counts = Counter()
    confidence_counts = Counter()
    preferred_citation_available = 0
    total_citations = 0
    max_citations = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts[row.get("status") or "unknown"] += 1
            if row.get("is_top_ai_venue"):
                venue_counts[row.get("top_venue_bucket") or "unknown"] += 1
            if row.get("published_raw_type"):
                raw_type_counts[row["published_raw_type"]] += 1
            if row.get("published_source_type"):
                source_type_counts[row["published_source_type"]] += 1
            if row.get("match_confidence"):
                confidence_counts[row["match_confidence"]] += 1
            preferred = row.get("preferred_cited_by_count")
            if isinstance(preferred, int):
                preferred_citation_available += 1
                total_citations += preferred
                max_citations = max(max_citations, preferred)

    processed = sum(counts.values())
    return {
        "processed_count": processed,
        "status_counts": dict(counts),
        "top_venue_counts": dict(venue_counts.most_common()),
        "published_raw_type_counts": dict(raw_type_counts.most_common()),
        "published_source_type_counts": dict(source_type_counts.most_common()),
        "confidence_counts": dict(confidence_counts),
        "preferred_citation_available_count": preferred_citation_available,
        "preferred_citation_available_ratio": round(preferred_citation_available / processed, 4) if processed else 0.0,
        "average_preferred_citations": round(total_citations / preferred_citation_available, 4) if preferred_citation_available else 0.0,
        "max_preferred_citations": max_citations,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_path)
    if args.sort_by_published:
        rows.sort(key=lambda row: str(row.get("published") or ""))
    if args.limit:
        rows = rows[: args.limit]

    seen_ids = load_seen_ids(output_path) if args.resume else set()
    mode = "a" if args.resume else "w"
    pending_rows = [
        row for row in rows if str(row.get("paper_id") or "") and str(row.get("paper_id") or "") not in seen_ids
    ]

    with output_path.open(mode, encoding="utf-8") as out:
        if args.workers <= 1:
            for idx, row in enumerate(pending_rows, start=1):
                record = enrich_one_row(row, sleep=args.sleep, per_page=args.per_page, mailto=args.mailto)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                if idx % 25 == 0:
                    print(f"processed {idx}/{len(pending_rows)}")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(
                        enrich_one_row,
                        row,
                        sleep=args.sleep,
                        per_page=args.per_page,
                        mailto=args.mailto,
                    ): str(row.get("paper_id") or "")
                    for row in pending_rows
                }
                processed = 0
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    processed += 1
                    if processed % 25 == 0:
                        print(f"processed {processed}/{len(pending_rows)}")

    summary = summarize_sidecar(output_path)
    summary.update(
        {
            "input": str(input_path),
            "output": str(output_path),
            "pending_processed_this_run": len(pending_rows),
            "resume": args.resume,
        }
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
