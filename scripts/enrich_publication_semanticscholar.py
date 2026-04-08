from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TOP_VENUE_RULES = [
    ("NeurIPS", ["neural information processing systems", "neurips"]),
    ("ICML", ["international conference on machine learning", "icml"]),
    ("ICLR", ["international conference on learning representations", "iclr"]),
    ("AAAI", ["aaai conference on artificial intelligence", "aaai"]),
    ("IJCAI", ["international joint conference on artificial intelligence", "ijcai"]),
    ("ACL", ["annual meeting of the association for computational linguistics", "association for computational linguistics", "findings of the association for computational linguistics", " acl "]),
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

S2_FIELDS = ",".join(
    [
        "title",
        "citationCount",
        "venue",
        "publicationVenue",
        "publicationTypes",
        "year",
        "externalIds",
        "journal",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich arXiv papers with publication venue and citation metadata from Semantic Scholar."
    )
    parser.add_argument("--input", required=True, help="Input JSONL with paper_id/title/published/authors.")
    parser.add_argument("--output", required=True, help="Output JSONL sidecar file.")
    parser.add_argument("--summary-output", required=True, help="Output summary JSON.")
    parser.add_argument("--batch-size", type=int, default=100, help="How many arXiv ids to query per batch.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between batches.")
    parser.add_argument("--resume", action="store_true", help="Skip paper_ids already present in the output.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to process.")
    parser.add_argument("--sort-by-published", action="store_true", help="Sort ascending by published time before processing.")
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    return parser.parse_args()


def normalize_title(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def venue_bucket(*venue_texts: str) -> str:
    hay = " ".join(str(x or "") for x in venue_texts).lower()
    hay = f" {hay} "
    for bucket, needles in TOP_VENUE_RULES:
        if any(needle in hay for needle in needles):
            return bucket
    return ""


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_seen_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for row in iter_jsonl(path):
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            seen.add(paper_id)
    return seen


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except Exception:
            pass
    return min(60.0, 2.0**attempt)


def fetch_batch(arxiv_ids: list[str], *, timeout: int) -> list[dict[str, Any] | None]:
    url = "https://api.semanticscholar.org/graph/v1/paper/batch?" + urlencode({"fields": S2_FIELDS})
    payload = {"ids": [f"ARXIV:{arxiv_id}" for arxiv_id in arxiv_ids]}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "User-Agent": "ResearchWorld publication enrichment",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(5):
        request = Request(url, data=body, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.load(response)
            if isinstance(data, list):
                return data
            raise RuntimeError(f"Unexpected Semantic Scholar response type: {type(data).__name__}")
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
    return []


def is_arxiv_like_name(name: str) -> bool:
    lowered = str(name or "").strip().lower()
    return lowered in {"", "arxiv", "arxiv.org", "corr"} or "arxiv" in lowered


def infer_venue_type(
    venue_name: str,
    publication_venue: dict[str, Any] | None,
    journal_name: str,
    dblp_id: str,
    publication_types: list[str],
) -> str:
    if publication_venue:
        venue_type = str(publication_venue.get("type") or "").strip().lower()
        if venue_type in {"conference", "journal"}:
            return venue_type
    lowered_types = {str(item).strip().lower() for item in publication_types if str(item).strip()}
    if "conference" in lowered_types:
        return "conference"
    if "journalarticle" in lowered_types:
        return "journal"
    dblp_id = str(dblp_id or "")
    if dblp_id.startswith("conf/"):
        return "conference"
    if dblp_id.startswith("journals/") and not dblp_id.startswith("journals/corr/"):
        return "journal"

    hay = " ".join([venue_name, journal_name]).lower()
    if any(token in hay for token in ["conference", "symposium", "workshop", "proceedings", "meeting"]):
        return "conference"
    if any(token in hay for token in ["journal", "transactions", "letters", "review"]):
        return "journal"
    return "unknown"


def extract_publication_metadata(item: dict[str, Any]) -> dict[str, Any]:
    publication_venue = item.get("publicationVenue") if isinstance(item.get("publicationVenue"), dict) else None
    venue_name = str(item.get("venue") or "").strip()
    journal = item.get("journal") if isinstance(item.get("journal"), dict) else {}
    journal_name = str(journal.get("name") or "").strip()
    external_ids = item.get("externalIds") if isinstance(item.get("externalIds"), dict) else {}
    dblp_id = str(external_ids.get("DBLP") or "").strip()
    doi = str(external_ids.get("DOI") or "").strip()
    publication_types = [str(item) for item in (item.get("publicationTypes") or []) if str(item).strip()]

    aliases: list[str] = []
    if publication_venue:
        if publication_venue.get("name"):
            aliases.append(str(publication_venue.get("name")))
        for alt in publication_venue.get("alternate_names") or []:
            if alt:
                aliases.append(str(alt))
    if venue_name:
        aliases.append(venue_name)
    if journal_name:
        aliases.append(journal_name)

    non_arxiv_aliases = [alias for alias in aliases if alias and not is_arxiv_like_name(alias)]
    has_non_arxiv_dblp = bool(dblp_id and not dblp_id.startswith("journals/corr/"))
    has_non_arxiv_publication = bool(non_arxiv_aliases or has_non_arxiv_dblp)

    preferred_venue_name = ""
    if publication_venue and publication_venue.get("name") and not is_arxiv_like_name(str(publication_venue.get("name"))):
        preferred_venue_name = str(publication_venue.get("name"))
    elif venue_name and not is_arxiv_like_name(venue_name):
        preferred_venue_name = venue_name
    elif journal_name and not is_arxiv_like_name(journal_name):
        preferred_venue_name = journal_name

    bucket = venue_bucket(*aliases, dblp_id)
    venue_type = infer_venue_type(
        preferred_venue_name or venue_name,
        publication_venue,
        journal_name,
        dblp_id,
        publication_types,
    )
    doi_is_arxiv = doi.lower().startswith("10.48550/arxiv.")

    return {
        "publication_venue": publication_venue,
        "venue_name": venue_name,
        "journal_name": journal_name,
        "external_ids": external_ids,
        "dblp_id": dblp_id,
        "doi": doi,
        "doi_is_arxiv": doi_is_arxiv,
        "aliases": aliases,
        "has_non_arxiv_publication": has_non_arxiv_publication,
        "preferred_venue_name": preferred_venue_name,
        "venue_type": venue_type,
        "top_venue_bucket": bucket or None,
    }


def build_record(row: dict[str, Any], item: dict[str, Any] | None) -> dict[str, Any]:
    paper_id = str(row.get("paper_id") or "")
    title = str(row.get("title") or "")
    record: dict[str, Any] = {
        "paper_id": paper_id,
        "source_paper_id": row.get("source_paper_id"),
        "title": title,
        "published": row.get("published"),
        "authors": row.get("authors") or [],
        "match_source": "semantic_scholar_arxiv_batch",
    }

    if not isinstance(item, dict) or not item:
        record["status"] = "unmatched"
        return record

    matched_title = str(item.get("title") or "")
    title_similarity = SequenceMatcher(None, normalize_title(title), normalize_title(matched_title)).ratio()
    citation_count = item.get("citationCount")
    publication_year = item.get("year")
    meta = extract_publication_metadata(item)
    is_top_venue = bool(meta["top_venue_bucket"])

    status = "matched" if meta["has_non_arxiv_publication"] else "citation_only"
    publication_venue = meta["publication_venue"] or {}
    journal = item.get("journal") if isinstance(item.get("journal"), dict) else {}

    record.update(
        {
            "status": status,
            "match_confidence": "high" if title_similarity >= 0.95 else "medium",
            "matched_semantic_scholar_paper_id": item.get("paperId"),
            "matched_title": matched_title,
            "matched_publication_year": publication_year,
            "matched_doi": meta["doi"] or None,
            "matched_doi_is_arxiv": meta["doi_is_arxiv"],
            "matched_dblp_id": meta["dblp_id"] or None,
            "matched_cited_by_count": citation_count,
            "matched_external_ids": meta["external_ids"],
            "semantic_scholar_venue": meta["venue_name"] or None,
            "semantic_scholar_journal_name": meta["journal_name"] or None,
            "semantic_scholar_publication_types": item.get("publicationTypes") or [],
            "published_venue_name": meta["preferred_venue_name"] or None,
            "published_venue_type": meta["venue_type"],
            "published_source_display_name": publication_venue.get("name"),
            "is_top_ai_venue": is_top_venue,
            "top_venue_bucket": meta["top_venue_bucket"],
            "preferred_cited_by_count": citation_count,
            "preferred_citation_source": "semantic_scholar",
            "evidence": {
                "title_similarity": round(title_similarity, 4),
                "has_non_arxiv_publication": meta["has_non_arxiv_publication"],
                "venue_aliases": meta["aliases"],
                "publication_venue": publication_venue,
                "semantic_scholar_venue": meta["venue_name"] or None,
                "semantic_scholar_journal": journal,
            },
        }
    )
    return record


def summarize_sidecar(path: Path) -> dict[str, Any]:
    counts = Counter()
    venue_counts = Counter()
    venue_type_counts = Counter()
    confidence_counts = Counter()
    citation_available = 0
    total_citations = 0
    max_citations = 0

    for row in iter_jsonl(path):
        counts[row.get("status") or "unknown"] += 1
        if row.get("is_top_ai_venue"):
            venue_counts[row.get("top_venue_bucket") or "unknown"] += 1
        if row.get("published_venue_type"):
            venue_type_counts[row.get("published_venue_type")] += 1
        if row.get("match_confidence"):
            confidence_counts[row.get("match_confidence")] += 1
        preferred = row.get("preferred_cited_by_count")
        if isinstance(preferred, int):
            citation_available += 1
            total_citations += preferred
            max_citations = max(max_citations, preferred)

    processed = sum(counts.values())
    return {
        "processed_count": processed,
        "status_counts": dict(counts),
        "top_venue_counts": dict(venue_counts.most_common()),
        "published_venue_type_counts": dict(venue_type_counts.most_common()),
        "confidence_counts": dict(confidence_counts),
        "preferred_citation_available_count": citation_available,
        "preferred_citation_available_ratio": round(citation_available / processed, 4) if processed else 0.0,
        "average_preferred_citations": round(total_citations / citation_available, 4) if citation_available else 0.0,
        "max_preferred_citations": max_citations,
    }


def batched(rows: list[dict[str, Any]], batch_size: int):
    for idx in range(0, len(rows), batch_size):
        yield rows[idx : idx + batch_size]


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
    pending_rows = [
        row for row in rows if str(row.get("paper_id") or "") and str(row.get("paper_id") or "") not in seen_ids
    ]
    mode = "a" if args.resume else "w"

    with output_path.open(mode, encoding="utf-8") as out:
        total_batches = (len(pending_rows) + args.batch_size - 1) // args.batch_size if pending_rows else 0
        for batch_idx, batch_rows in enumerate(batched(pending_rows, args.batch_size), start=1):
            batch_ids = [str(row["paper_id"]) for row in batch_rows]
            try:
                payload = fetch_batch(batch_ids, timeout=args.timeout)
                if len(payload) != len(batch_rows):
                    raise RuntimeError(
                        f"Semantic Scholar batch size mismatch: requested {len(batch_rows)}, got {len(payload)}"
                    )
                for row, item in zip(batch_rows, payload):
                    record = build_record(row, item if isinstance(item, dict) else None)
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                for row in batch_rows:
                    record = {
                        "paper_id": row.get("paper_id"),
                        "source_paper_id": row.get("source_paper_id"),
                        "title": row.get("title"),
                        "published": row.get("published"),
                        "authors": row.get("authors") or [],
                        "status": "error",
                        "match_source": "semantic_scholar_arxiv_batch",
                        "error": str(exc),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            processed = min(batch_idx * args.batch_size, len(pending_rows))
            print(f"processed {processed}/{len(pending_rows)} (batch {batch_idx}/{total_batches})")
            if args.sleep and batch_idx < total_batches:
                time.sleep(args.sleep)

    summary = summarize_sidecar(output_path)
    summary.update(
        {
            "input": str(input_path),
            "output": str(output_path),
            "pending_processed_this_run": len(pending_rows),
            "resume": args.resume,
            "backend": "semantic_scholar",
            "batch_size": args.batch_size,
        }
    )
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
