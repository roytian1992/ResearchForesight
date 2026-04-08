from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

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
    parser = argparse.ArgumentParser(
        description="Export a frozen offline knowledge base for benchmark_v2 evaluation."
    )
    parser.add_argument(
        "--release-dir",
        default=str(ROOT / "data" / "releases" / "benchmark_v2_20260329"),
        help="Benchmark release directory.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output KB directory. Defaults to <release-dir>/kb",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=DEFAULT_DOMAINS,
        help="Domain slugs to export.",
    )
    parser.add_argument(
        "--history-cutoff",
        default="",
        help="History cutoff date (YYYY-MM-DD). If omitted, infer from release tasks.jsonl.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep an existing output directory instead of deleting it first.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def normalize_date(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10:
        return text[:10]
    return text


def infer_history_cutoff(release_dir: Path) -> str:
    tasks_path = release_dir / "tasks.jsonl"
    cutoffs = sorted(
        {
            str((row.get("time_cutoff") or "")).strip()
            for row in iter_jsonl(tasks_path)
            if str((row.get("time_cutoff") or "")).strip()
        }
    )
    if not cutoffs:
        raise ValueError(f"Unable to infer history cutoff from {tasks_path}")
    if len(cutoffs) != 1:
        raise ValueError(f"Expected exactly one history cutoff in {tasks_path}, got: {cutoffs}")
    return cutoffs[0]


def paper_visible(row: dict, history_cutoff: str) -> bool:
    published = normalize_date(row.get("published"))
    return bool(published and published <= history_cutoff)


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


def flatten_sections(content_rows: list[dict], *, domain_slug: str, visible_papers: dict[str, dict]) -> list[dict]:
    rows: list[dict] = []
    for content in content_rows:
        paper_id = content.get("paper_id")
        paper = visible_papers.get(str(paper_id))
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


def attach_common_fields(rows: list[dict], *, domain_slug: str, visible_papers: dict[str, dict]) -> list[dict]:
    enriched: list[dict] = []
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


def export_domain(
    *,
    domain_slug: str,
    release_dir: Path,
    out_dir: Path,
    history_cutoff: str,
) -> dict:
    papers_src = ROOT / "data" / "domains" / domain_slug / "interim" / "papers_merged.publication_enriched.semanticscholar.jsonl"
    pageindex_src = ROOT / "data" / "support_packets" / "pageindex" / domain_slug / "pageindex.jsonl"
    structures_src = ROOT / "data" / "support_packets" / "paper_structures" / domain_slug / "paper_structures.jsonl"
    content_src = ROOT / "data" / "support_packets" / "fulltext_content" / domain_slug / "content.jsonl"

    if not papers_src.exists():
        raise FileNotFoundError(f"Missing source papers file: {papers_src}")

    raw_papers = list(iter_jsonl(papers_src))
    visible_papers_list = [
        simplify_paper(row, domain_slug=domain_slug)
        for row in raw_papers
        if paper_visible(row, history_cutoff)
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
    domain_out.mkdir(parents=True, exist_ok=True)
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

    coverage = {
        "source_papers_total": len(raw_papers),
        "papers_exported": len(visible_papers_list),
        "pageindex_rows_exported": len(pageindex_rows),
        "structures_rows_exported": len(structure_rows),
        "content_rows_exported": len(content_rows),
        "section_rows_exported": len(section_rows),
        "papers_missing_published_date": sum(1 for row in raw_papers if not normalize_date(row.get("published"))),
        "papers_with_pageindex_share": round(len(pageindex_rows) / len(visible_papers_list), 4) if visible_papers_list else 0.0,
        "papers_with_structures_share": round(len(structure_rows) / len(visible_papers_list), 4) if visible_papers_list else 0.0,
        "papers_with_content_share": round(len(content_rows) / len(visible_papers_list), 4) if visible_papers_list else 0.0,
    }
    domain_manifest = {
        "domain_slug": domain_slug,
        "domain": DOMAIN_PUBLIC.get(domain_slug, domain_slug),
        "history_cutoff": history_cutoff,
        "source_files": {
            "papers": str(papers_src.relative_to(ROOT)),
            "pageindex": str(pageindex_src.relative_to(ROOT)) if pageindex_src.exists() else None,
            "structures": str(structures_src.relative_to(ROOT)) if structures_src.exists() else None,
            "content": str(content_src.relative_to(ROOT)) if content_src.exists() else None,
        },
        "files": {name: str(path.relative_to(out_dir)) for name, path in files.items()},
        "coverage": coverage,
        "notes": [
            "This KB contains only papers published on or before the history cutoff.",
            "pageindex/content/structures are partial layers and do not cover all exported papers.",
            "The current content layer is abstract-derived normalized content rather than uniformly recovered raw full text.",
        ],
    }
    write_json(domain_manifest, files["manifest"])
    return domain_manifest


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    if not release_dir.exists():
        raise FileNotFoundError(f"Missing release directory: {release_dir}")

    history_cutoff = args.history_cutoff.strip() or infer_history_cutoff(release_dir)
    out_dir = Path(args.out_dir) if args.out_dir else release_dir / "kb"
    if out_dir.exists() and not args.keep_existing:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domain_manifests = []
    for domain_slug in args.domains:
        domain_manifests.append(
            export_domain(
                domain_slug=domain_slug,
                release_dir=release_dir,
                out_dir=out_dir,
                history_cutoff=history_cutoff,
            )
        )

    root_manifest = {
        "release_name": release_dir.name,
        "history_cutoff": history_cutoff,
        "kb_name": f"{release_dir.name}_offline_kb",
        "domains": domain_manifests,
        "files": {
            "manifest": "manifest.json",
            "domains_dir": "domains",
        },
        "temporal_policy": {
            "mode": "history_frozen",
            "allowed_evidence": f"Only papers with published_date <= {history_cutoff} are included.",
            "disallowed_sources": [
                "future papers after cutoff",
                "tasks_hidden_eval.jsonl",
                "tasks_internal_full.jsonl",
                "tasks_build_trace.jsonl",
                "support packets used during benchmark construction",
            ],
        },
        "usage_modes": {
            "native_llm": "Use only tasks.jsonl question text. Do not load this KB.",
            "llm_plus_rag": "Use this frozen KB as the only retrieval source.",
            "agent": "Use this frozen KB plus non-network tools over the same KB.",
        },
        "notes": [
            "The KB is release-aligned and safe for offline evaluation under the benchmark_v2 temporal protocol.",
            "Section/pageindex/structure layers are partial and should be treated as auxiliary evidence layers.",
            "For reproducibility and corpus extension, use a separate fetch-and-parse toolkit rather than modifying the official evaluation KB in place.",
        ],
    }
    write_json(root_manifest, out_dir / "manifest.json")
    print(json.dumps(root_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
