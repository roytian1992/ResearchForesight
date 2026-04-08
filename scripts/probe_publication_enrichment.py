from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_TOP_VENUES = {
    "NeurIPS",
    "ICML",
    "ICLR",
    "AAAI",
    "IJCAI",
    "ACL",
    "EMNLP",
    "NAACL",
    "COLING",
    "SIGIR",
    "WWW",
    "WSDM",
    "KDD",
    "CVPR",
    "ICCV",
    "ECCV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe publication-venue enrichment coverage for arXiv papers.")
    parser.add_argument("--input", required=True, help="Input JSONL with at least paper_id/title/published.")
    parser.add_argument("--output", required=True, help="Output JSON report.")
    parser.add_argument("--sample-size", type=int, default=20, help="How many papers to probe.")
    parser.add_argument("--published-before", default="2025-01-01", help="Only probe papers published before this ISO date.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Sleep between API requests.")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout.")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_arxiv_only(venue_name: str, publication_venue: dict[str, Any] | None) -> bool:
    lowered = (venue_name or "").strip().lower()
    if lowered in {"arxiv", "arxiv.org", "corr", ""}:
        return True
    if publication_venue:
        pub_name = str(publication_venue.get("name") or "").strip().lower()
        if pub_name in {"arxiv", "arxiv.org", "corr"}:
            return True
    return False


def venue_aliases(publication_venue: dict[str, Any] | None, venue_name: str) -> set[str]:
    aliases: set[str] = set()
    if venue_name:
        aliases.add(venue_name)
    if publication_venue:
        if publication_venue.get("name"):
            aliases.add(str(publication_venue["name"]))
        for alt in publication_venue.get("alternate_names") or []:
            aliases.add(str(alt))
    return {alias.strip() for alias in aliases if str(alias).strip()}


def is_top_venue(publication_venue: dict[str, Any] | None, venue_name: str) -> bool:
    aliases = venue_aliases(publication_venue, venue_name)
    return any(alias in DEFAULT_TOP_VENUES for alias in aliases)


def fetch_semantic_scholar(arxiv_id: str, *, timeout: int) -> tuple[int, dict[str, Any] | None, str | None]:
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/ARXIV:{arxiv_id}"
        "?fields=title,year,venue,publicationVenue,publicationTypes,externalIds"
    )
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "ResearchWorld publication probe"})
    except Exception as exc:
        return 0, None, str(exc)
    try:
        payload = response.json()
    except Exception:
        payload = None
    return response.status_code, payload, None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = [
        row
        for row in iter_jsonl(input_path)
        if str(row.get("published") or "") < args.published_before and row.get("paper_id")
    ]
    random.seed(args.seed)
    sample = random.sample(rows, min(args.sample_size, len(rows)))
    sample.sort(key=lambda row: str(row.get("published") or ""))

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(sample, start=1):
        arxiv_id = str(row["paper_id"])
        status_code, payload, error = fetch_semantic_scholar(arxiv_id, timeout=args.timeout)
        result: dict[str, Any] = {
            "paper_id": arxiv_id,
            "title": row.get("title"),
            "published": row.get("published"),
            "api_status": status_code,
        }
        if error is not None:
            result["status"] = "request_error"
            result["error"] = error
        elif status_code == 200 and isinstance(payload, dict):
            venue_name = str(payload.get("venue") or "")
            publication_venue = payload.get("publicationVenue") if isinstance(payload.get("publicationVenue"), dict) else None
            result.update(
                {
                    "status": "ok",
                    "semantic_scholar_title": payload.get("title"),
                    "semantic_scholar_year": payload.get("year"),
                    "venue": venue_name,
                    "publication_venue": publication_venue,
                    "publication_types": payload.get("publicationTypes") or [],
                    "external_ids": payload.get("externalIds") or {},
                    "has_non_arxiv_publication": (not is_arxiv_only(venue_name, publication_venue)),
                    "is_top_venue": is_top_venue(publication_venue, venue_name),
                }
            )
        else:
            result["status"] = "api_error"
            result["error"] = payload
        results.append(result)
        print("[{}/{}] {} -> {}".format(idx, len(sample), arxiv_id, result.get("status")))
        if idx < len(sample):
            time.sleep(args.sleep_seconds)

    ok_rows = [row for row in results if row.get("status") == "ok"]
    non_arxiv = [row for row in ok_rows if row.get("has_non_arxiv_publication")]
    top_rows = [row for row in ok_rows if row.get("is_top_venue")]
    report = {
        "input": str(input_path),
        "sample_size": len(sample),
        "api_ok": len(ok_rows),
        "api_error": len(results) - len(ok_rows),
        "non_arxiv_publication_matches": len(non_arxiv),
        "top_venue_matches": len(top_rows),
        "top_venue_rate_over_ok": (len(top_rows) / len(ok_rows)) if ok_rows else 0.0,
        "non_arxiv_rate_over_ok": (len(non_arxiv) / len(ok_rows)) if ok_rows else 0.0,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("Report: {}".format(output_path))


if __name__ == "__main__":
    main()
