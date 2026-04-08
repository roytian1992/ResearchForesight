from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DOMAINS = [
    "llm_agent",
    "llm_finetuning_post_training",
    "rag_and_retrieval_structuring",
    "visual_generative_modeling_and_diffusion",
]

DOMAIN_PUBLIC = {
    "llm_agent": "LLM agents",
    "llm_finetuning_post_training": "LLM fine-tuning and post-training",
    "rag_and_retrieval_structuring": "RAG and retrieval structuring",
    "visual_generative_modeling_and_diffusion": "Visual generative modeling and diffusion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export hidden future-only KB for benchmark-aware fact evaluation.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--domains", nargs="*", default=DEFAULT_DOMAINS)
    parser.add_argument("--future-start", default="")
    parser.add_argument("--future-end", default="")
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_date(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:10]


def infer_range(release_dir: Path) -> tuple[str, str]:
    v3_path = release_dir / "tasks_hidden_eval_v3.jsonl"
    starts = set()
    ends = set()
    for row in iter_jsonl(v3_path):
        policy = row.get("temporal_policy") or {}
        if policy.get("future_start"):
            starts.add(str(policy["future_start"]))
        if policy.get("future_end"):
            ends.add(str(policy["future_end"]))
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError(f"Unable to infer unique future range from {v3_path}")
    return next(iter(starts)), next(iter(ends))


def paper_in_range(row: dict, future_start: str, future_end: str) -> bool:
    published = normalize_date(row.get("published"))
    if not published:
        return False
    return future_start <= published <= future_end


def summarize_publication(row: dict) -> dict:
    publication = row.get("publication_enrichment") or {}
    citation_count = publication.get("preferred_cited_by_count")
    if citation_count is None:
        citation_count = publication.get("matched_cited_by_count")
    return {
        "status": publication.get("status"),
        "match_source": publication.get("match_source"),
        "match_confidence": publication.get("match_confidence"),
        "matched_semantic_scholar_paper_id": publication.get("matched_semantic_scholar_paper_id"),
        "publication_year": publication.get("matched_publication_year"),
        "doi": publication.get("matched_doi"),
        "venue_name": publication.get("published_venue_name") or publication.get("semantic_scholar_venue"),
        "venue_type": publication.get("published_venue_type"),
        "source_display_name": publication.get("published_source_display_name"),
        "semantic_scholar_venue": publication.get("semantic_scholar_venue"),
        "journal_name": publication.get("semantic_scholar_journal_name"),
        "publication_types": publication.get("semantic_scholar_publication_types") or [],
        "is_top_ai_venue": bool(publication.get("is_top_ai_venue")),
        "top_venue_bucket": publication.get("top_venue_bucket"),
        "citation_count": citation_count,
        "citation_source": publication.get("preferred_citation_source"),
    }


def simplify_paper(row: dict, *, domain_slug: str) -> dict:
    return {
        "paper_id": row.get("paper_id"),
        "source_paper_id": row.get("source_paper_id"),
        "domain_slug": domain_slug,
        "domain": DOMAIN_PUBLIC.get(domain_slug, domain_slug),
        "title": row.get("title"),
        "abstract": row.get("abstract"),
        "authors": row.get("authors") or [],
        "categories": row.get("categories") or [],
        "published": row.get("published"),
        "published_date": normalize_date(row.get("published")),
        "updated": row.get("updated"),
        "pdf_url": row.get("pdf_url"),
        "candidate_tier": row.get("candidate_tier"),
        "heuristic_score": row.get("heuristic_score"),
        "publication": summarize_publication(row),
    }


def attach_common_fields(rows: List[dict], *, domain_slug: str, visible_papers: Dict[str, dict]) -> List[dict]:
    enriched: List[dict] = []
    for row in rows:
        paper_id = str(row.get("paper_id"))
        paper = visible_papers.get(paper_id)
        if not paper:
            continue
        merged = dict(row)
        merged["domain_slug"] = domain_slug
        merged["domain"] = DOMAIN_PUBLIC.get(domain_slug, domain_slug)
        merged["published_date"] = paper.get("published_date")
        enriched.append(merged)
    return enriched


def flatten_sections(content_rows: List[dict], *, domain_slug: str, visible_papers: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for content in content_rows:
        paper_id = str(content.get("paper_id"))
        paper = visible_papers.get(paper_id)
        if not paper:
            continue
        for section_index, section in enumerate(content.get("sections") or []):
            text = str(section.get("text") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "paper_id": paper_id,
                    "source_paper_id": content.get("source_paper_id"),
                    "domain_slug": domain_slug,
                    "domain": DOMAIN_PUBLIC.get(domain_slug, domain_slug),
                    "paper_title": content.get("title") or paper.get("title"),
                    "published_date": paper.get("published_date"),
                    "section_id": section.get("section_id"),
                    "section_index": section_index,
                    "section_title": section.get("title"),
                    "level": section.get("level"),
                    "text": text,
                    "source_type": content.get("source_type"),
                    "source_url": content.get("source_url"),
                }
            )
    return rows


def export_domain(domain_slug: str, *, out_dir: Path, future_start: str, future_end: str) -> dict:
    papers_src = ROOT / "data" / "domains" / domain_slug / "interim" / "papers_merged.publication_enriched.semanticscholar.jsonl"
    pageindex_src = ROOT / "data" / "support_packets" / "pageindex" / domain_slug / "pageindex.jsonl"
    structures_src = ROOT / "data" / "support_packets" / "paper_structures" / domain_slug / "paper_structures.jsonl"
    content_src = ROOT / "data" / "support_packets" / "fulltext_content" / domain_slug / "content.jsonl"

    raw_papers = list(iter_jsonl(papers_src))
    visible_papers_list = [
        simplify_paper(row, domain_slug=domain_slug)
        for row in raw_papers
        if paper_in_range(row, future_start, future_end)
    ]
    visible_papers = {str(row["paper_id"]): row for row in visible_papers_list}

    pageindex_raw = list(iter_jsonl(pageindex_src)) if pageindex_src.exists() else []
    pageindex_rows = attach_common_fields(
        [row for row in pageindex_raw if str(row.get("paper_id")) in visible_papers],
        domain_slug=domain_slug,
        visible_papers=visible_papers,
    )
    structures_raw = list(iter_jsonl(structures_src)) if structures_src.exists() else []
    structure_rows = attach_common_fields(
        [row for row in structures_raw if str(row.get("paper_id")) in visible_papers],
        domain_slug=domain_slug,
        visible_papers=visible_papers,
    )
    content_raw = list(iter_jsonl(content_src)) if content_src.exists() else []
    content_rows = attach_common_fields(
        [row for row in content_raw if str(row.get("paper_id")) in visible_papers],
        domain_slug=domain_slug,
        visible_papers=visible_papers,
    )
    section_rows = flatten_sections(content_rows, domain_slug=domain_slug, visible_papers=visible_papers)

    domain_out = out_dir / "domains" / domain_slug
    files = {
        "papers": domain_out / "papers.jsonl",
        "pageindex": domain_out / "pageindex.jsonl",
        "structures": domain_out / "structures.jsonl",
        "content": domain_out / "content.jsonl",
        "sections": domain_out / "sections.jsonl",
        "manifest": domain_out / "manifest.json",
    }
    write_jsonl(visible_papers_list, files["papers"])
    write_jsonl(pageindex_rows, files["pageindex"])
    write_jsonl(structure_rows, files["structures"])
    write_jsonl(content_rows, files["content"])
    write_jsonl(section_rows, files["sections"])

    domain_manifest = {
        "domain_slug": domain_slug,
        "domain": DOMAIN_PUBLIC.get(domain_slug, domain_slug),
        "future_start": future_start,
        "future_end": future_end,
        "coverage": {
            "source_papers_total": len(raw_papers),
            "papers_exported": len(visible_papers_list),
            "pageindex_rows_exported": len(pageindex_rows),
            "structures_rows_exported": len(structure_rows),
            "content_rows_exported": len(content_rows),
            "section_rows_exported": len(section_rows),
        },
        "files": {name: str(path.relative_to(out_dir)) for name, path in files.items()},
        "source_files": {
            "papers": str(papers_src.relative_to(ROOT)),
            "pageindex": str(pageindex_src.relative_to(ROOT)) if pageindex_src.exists() else None,
            "structures": str(structures_src.relative_to(ROOT)) if structures_src.exists() else None,
            "content": str(content_src.relative_to(ROOT)) if content_src.exists() else None,
        },
        "notes": [
            "This KB is hidden and intended only for future-window fact verification.",
            "It must not be exposed to evaluated systems.",
        ],
    }
    write_json(domain_manifest, files["manifest"])
    return domain_manifest


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    future_start, future_end = (args.future_start.strip(), args.future_end.strip())
    if not future_start or not future_end:
        future_start, future_end = infer_range(release_dir)
    out_dir = Path(args.out_dir) if args.out_dir else release_dir / "kb_future_hidden"
    if out_dir.exists() and not args.keep_existing:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain_manifests = [export_domain(domain, out_dir=out_dir, future_start=future_start, future_end=future_end) for domain in args.domains]
    root_manifest = {
        "release_name": release_dir.name,
        "kb_name": f"{release_dir.name}_future_hidden_kb",
        "future_start": future_start,
        "future_end": future_end,
        "domains": domain_manifests,
        "visibility": "hidden_eval_only",
        "notes": [
            "This KB contains only future-window papers and is for evaluator use only.",
            "Do not expose this KB to the evaluated model or agent.",
        ],
    }
    write_json(root_manifest, out_dir / "manifest.json")
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
