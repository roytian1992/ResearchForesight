from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer


PUBLIC_DOMAIN_TO_ID = {
    "LLM agents": "llm_agent",
    "LLM finetuning and post-training": "llm_finetuning_post_training",
    "LLM fine-tuning and post-training": "llm_finetuning_post_training",
    "RAG and retrieval structuring": "rag_and_retrieval_structuring",
    "Visual generative modeling and diffusion": "visual_generative_modeling_and_diffusion",
}


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\./+]{0,63}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_ws(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def clip_text(text: Any, limit: int) -> str:
    value = normalize_ws(text)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def parse_day(value: Any) -> Optional[datetime]:
    raw = normalize_ws(value)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw[:10])
    except Exception:
        return None


def on_or_before(date_value: Any, cutoff_value: Any) -> bool:
    if not cutoff_value:
        return True
    dt = parse_day(date_value)
    cutoff = parse_day(cutoff_value)
    if cutoff is None:
        return True
    if dt is None:
        return True
    return dt.date() <= cutoff.date()


def tokenize(text: Any) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def dedupe(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        item = normalize_ws(value)
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


@dataclass
class RetrievalDoc:
    doc_id: str
    paper_id: str
    title: str
    text: str
    meta: Dict[str, Any]


class HybridRetriever:
    def __init__(self, docs: List[RetrievalDoc]):
        self.docs = docs
        if not docs:
            self.doc_texts = []
            self.doc_tokens = []
            self.bm25 = None
            self.vectorizer = None
            self.doc_matrix = None
            return
        self.doc_texts = [doc.text for doc in docs]
        self.doc_tokens = [tokenize(doc.text) for doc in docs]
        self.bm25 = BM25Okapi(self.doc_tokens)
        max_df = 1.0 if len(docs) < 5 else 0.98
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=max_df,
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def retrieve(self, query: str, *, top_k: int = 10) -> List[Tuple[RetrievalDoc, Dict[str, float]]]:
        if not self.docs:
            return []
        query_tokens = tokenize(query)
        bm25_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
        tfidf_query = self.vectorizer.transform([query])
        dense_scores = (self.doc_matrix @ tfidf_query.T).toarray().reshape(-1)

        def normalize(scores: np.ndarray) -> np.ndarray:
            if scores.size == 0:
                return scores
            max_v = float(scores.max())
            min_v = float(scores.min())
            if math.isclose(max_v, min_v):
                return np.zeros_like(scores)
            return (scores - min_v) / (max_v - min_v)

        bm25_norm = normalize(bm25_scores)
        dense_norm = normalize(dense_scores)
        fused = 0.55 * bm25_norm + 0.45 * dense_norm
        order = np.argsort(-fused)[:top_k]
        rows: List[Tuple[RetrievalDoc, Dict[str, float]]] = []
        for idx in order:
            rows.append(
                (
                    self.docs[int(idx)],
                    {
                        "hybrid_score": float(fused[int(idx)]),
                        "bm25_score": float(bm25_scores[int(idx)]),
                        "tfidf_score": float(dense_scores[int(idx)]),
                    },
                )
            )
        return rows


class OfflineDomainKB:
    def __init__(self, domain_dir: Path):
        self.domain_dir = domain_dir
        self.manifest = json.loads((domain_dir / "manifest.json").read_text(encoding="utf-8"))
        self.domain_slug = str(self.manifest["domain_slug"])
        self.domain = str(self.manifest["domain"])
        self._papers = list(iter_jsonl(domain_dir / "papers.jsonl"))
        self._pageindex = list(iter_jsonl(domain_dir / "pageindex.jsonl")) if (domain_dir / "pageindex.jsonl").exists() else []
        self._structures = list(iter_jsonl(domain_dir / "structures.jsonl")) if (domain_dir / "structures.jsonl").exists() else []
        self._sections = list(iter_jsonl(domain_dir / "sections.jsonl")) if (domain_dir / "sections.jsonl").exists() else []

        self.papers_by_id = {str(row.get("paper_id") or ""): row for row in self._papers}
        self.pageindex_by_id = {str(row.get("paper_id") or ""): row for row in self._pageindex}
        self.structures_by_id = {str(row.get("paper_id") or ""): row for row in self._structures}
        self.sections_by_paper: Dict[str, List[Dict[str, Any]]] = {}
        for row in self._sections:
            self.sections_by_paper.setdefault(str(row.get("paper_id") or ""), []).append(row)

        self._paper_docs: Optional[List[RetrievalDoc]] = None
        self._paper_retriever_cache: Dict[str, HybridRetriever] = {}
        self._paper_docs_cache: Dict[str, List[RetrievalDoc]] = {}
        self._structure_docs_cache: Dict[Tuple[str, Tuple[str, ...]], List[RetrievalDoc]] = {}
        self._structure_retriever_cache: Dict[Tuple[str, Tuple[str, ...]], HybridRetriever] = {}
        self._section_retrievers: Dict[Tuple[str, Tuple[str, ...]], HybridRetriever] = {}
        self._pageindex_retrievers: Dict[Tuple[str, Tuple[str, ...]], HybridRetriever] = {}

    def _cache_cutoff_key(self, cutoff_date: Optional[str]) -> str:
        return str(cutoff_date or "")

    def _allowed_paper(self, paper_id: str, cutoff_date: Optional[str]) -> bool:
        paper = self.get_paper(paper_id) or {}
        return on_or_before(paper.get("published_date"), cutoff_date)

    def paper_docs(self, *, cutoff_date: Optional[str] = None) -> List[RetrievalDoc]:
        cache_key = self._cache_cutoff_key(cutoff_date)
        if cache_key not in self._paper_docs_cache:
            docs: List[RetrievalDoc] = []
            for row in self._papers:
                if not on_or_before(row.get("published_date"), cutoff_date):
                    continue
                pub = row.get("publication") or {}
                title = str(row.get("title") or "")
                abstract = str(row.get("abstract") or "")
                text = "\n".join(
                    part
                    for part in [
                        f"Title: {title}",
                        f"Published: {row.get('published_date') or row.get('published') or ''}",
                        f"Venue: {pub.get('venue_name') or ''}",
                        f"Top venue: {pub.get('top_venue_bucket') or ''}",
                        f"Citations: {pub.get('citation_count') if pub.get('citation_count') is not None else ''}",
                        f"Abstract: {abstract}",
                    ]
                    if normalize_ws(part)
                )
                docs.append(
                    RetrievalDoc(
                        doc_id=f"paper::{row.get('paper_id')}",
                        paper_id=str(row.get("paper_id") or ""),
                        title=title,
                        text=text,
                        meta={
                            "published_date": row.get("published_date"),
                            "publication": pub,
                        },
                    )
                )
            self._paper_docs_cache[cache_key] = docs
        return self._paper_docs_cache[cache_key]

    def paper_retriever(self, *, cutoff_date: Optional[str] = None) -> HybridRetriever:
        cache_key = self._cache_cutoff_key(cutoff_date)
        if cache_key not in self._paper_retriever_cache:
            self._paper_retriever_cache[cache_key] = HybridRetriever(self.paper_docs(cutoff_date=cutoff_date))
        return self._paper_retriever_cache[cache_key]

    def structure_docs(self, *, cutoff_date: Optional[str] = None, paper_ids: Optional[Iterable[str]] = None) -> List[RetrievalDoc]:
        key = (
            self._cache_cutoff_key(cutoff_date),
            tuple(sorted({str(x) for x in paper_ids})) if paper_ids is not None else tuple(),
        )
        if key not in self._structure_docs_cache:
            docs: List[RetrievalDoc] = []
            allowed = None if paper_ids is None else {str(x) for x in paper_ids}
            for row in self._structures:
                paper_id = str(row.get("paper_id") or "")
                if allowed is not None and paper_id not in allowed:
                    continue
                if not self._allowed_paper(paper_id, cutoff_date):
                    continue
                title = str(row.get("title") or self.papers_by_id.get(paper_id, {}).get("title") or "")
                limitations = [
                    x.get("name") if isinstance(x, dict) else str(x)
                    for x in (row.get("explicit_limitations") or [])
                ]
                future_work = [
                    x.get("direction") if isinstance(x, dict) else str(x)
                    for x in (row.get("future_work") or [])
                ]
                core_ideas = [
                    x.get("name") if isinstance(x, dict) else str(x)
                    for x in (row.get("core_ideas") or [])
                ]
                text = "\n".join(
                    part
                    for part in [
                        f"Title: {title}",
                        f"Problem statement: {row.get('problem_statement') or ''}",
                        f"Explicit limitations: {'; '.join([x for x in limitations if x])}",
                        f"Core ideas: {'; '.join([x for x in core_ideas if x])}",
                        f"Future work: {'; '.join([x for x in future_work if x])}",
                    ]
                    if normalize_ws(part)
                )
                docs.append(
                    RetrievalDoc(
                        doc_id=f"struct::{paper_id}",
                        paper_id=paper_id,
                        title=title,
                        text=text,
                        meta={
                            "problem_statement": row.get("problem_statement"),
                            "limitations": [x for x in limitations if x],
                            "future_work": [x for x in future_work if x],
                            "core_ideas": [x for x in core_ideas if x],
                        },
                    )
                )
            self._structure_docs_cache[key] = docs
        return self._structure_docs_cache[key]

    def structure_retriever(self, *, cutoff_date: Optional[str] = None, paper_ids: Optional[Iterable[str]] = None) -> HybridRetriever:
        key = (
            self._cache_cutoff_key(cutoff_date),
            tuple(sorted({str(x) for x in paper_ids})) if paper_ids is not None else tuple(),
        )
        if key not in self._structure_retriever_cache:
            self._structure_retriever_cache[key] = HybridRetriever(self.structure_docs(cutoff_date=cutoff_date, paper_ids=paper_ids))
        return self._structure_retriever_cache[key]

    def section_docs(self, *, cutoff_date: Optional[str] = None, paper_ids: Optional[Iterable[str]] = None) -> List[RetrievalDoc]:
        allowed = None if paper_ids is None else {str(x) for x in paper_ids}
        docs: List[RetrievalDoc] = []
        rows = self._sections if allowed is None else [
            row for pid in allowed for row in self.sections_by_paper.get(pid, [])
        ]
        for row in rows:
            paper_id = str(row.get("paper_id") or "")
            if not self._allowed_paper(paper_id, cutoff_date):
                continue
            docs.append(
                RetrievalDoc(
                    doc_id=f"section::{paper_id}::{row.get('section_id')}",
                    paper_id=paper_id,
                    title=f"{row.get('paper_title') or ''} / {row.get('section_title') or ''}",
                    text="\n".join(
                        part
                        for part in [
                            f"Paper: {row.get('paper_title') or ''}",
                            f"Section: {row.get('section_title') or ''}",
                            f"Level: {row.get('level')}",
                            str(row.get("text") or ""),
                        ]
                        if normalize_ws(part)
                    ),
                    meta={
                        "paper_title": row.get("paper_title"),
                        "section_title": row.get("section_title"),
                        "section_id": row.get("section_id"),
                        "published_date": row.get("published_date"),
                        "level": row.get("level"),
                    },
                )
            )
        return docs

    def section_retriever(self, *, cutoff_date: Optional[str] = None, paper_ids: Optional[Iterable[str]] = None) -> HybridRetriever:
        key = (
            self._cache_cutoff_key(cutoff_date),
            tuple(sorted({str(x) for x in paper_ids})) if paper_ids is not None else tuple(),
        )
        if key not in self._section_retrievers:
            self._section_retrievers[key] = HybridRetriever(self.section_docs(cutoff_date=cutoff_date, paper_ids=paper_ids))
        return self._section_retrievers[key]

    def pageindex_docs(self, *, cutoff_date: Optional[str] = None, paper_ids: Optional[Iterable[str]] = None) -> List[RetrievalDoc]:
        allowed = None if paper_ids is None else {str(x) for x in paper_ids}
        docs: List[RetrievalDoc] = []
        for row in self._pageindex:
            paper_id = str(row.get("paper_id") or "")
            if allowed is not None and paper_id not in allowed:
                continue
            if not self._allowed_paper(paper_id, cutoff_date):
                continue
            paper_title = str(row.get("paper_title") or (self.papers_by_id.get(paper_id, {}) or {}).get("title") or "")
            for node in row.get("nodes") or []:
                text = "\n".join(
                    part for part in [
                        f"Paper: {paper_title}",
                        f"Section: {node.get('normalized_title') or node.get('title') or ''}",
                        f"Path: {node.get('section_path') or ''}",
                        f"Kind: {node.get('kind') or ''}",
                        str(node.get("summary") or ""),
                        str(node.get("text") or ""),
                    ] if normalize_ws(part)
                )
                docs.append(
                    RetrievalDoc(
                        doc_id=f"page::{paper_id}::{node.get('node_id')}",
                        paper_id=paper_id,
                        title=f"{paper_title} / {node.get('normalized_title') or node.get('title') or ''}",
                        text=text,
                        meta={
                            "paper_title": paper_title,
                            "section_title": node.get("normalized_title") or node.get("title"),
                            "section_path": node.get("section_path"),
                            "kind": node.get("kind"),
                            "summary": node.get("summary"),
                        },
                    )
                )
        return docs

    def pageindex_retriever(self, *, cutoff_date: Optional[str] = None, paper_ids: Optional[Iterable[str]] = None) -> HybridRetriever:
        key = (
            self._cache_cutoff_key(cutoff_date),
            tuple(sorted({str(x) for x in paper_ids})) if paper_ids is not None else tuple(),
        )
        if key not in self._pageindex_retrievers:
            self._pageindex_retrievers[key] = HybridRetriever(self.pageindex_docs(cutoff_date=cutoff_date, paper_ids=paper_ids))
        return self._pageindex_retrievers[key]

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.papers_by_id.get(str(paper_id))

    def get_structure(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.structures_by_id.get(str(paper_id))

    def get_pageindex(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.pageindex_by_id.get(str(paper_id))


class OfflineKnowledgeBase:
    def __init__(self, kb_dir: Path):
        self.kb_dir = kb_dir
        self.manifest = json.loads((kb_dir / "manifest.json").read_text(encoding="utf-8"))
        domains_dir = kb_dir / "domains"
        self.domains: Dict[str, OfflineDomainKB] = {}
        for child in sorted(domains_dir.iterdir()):
            if child.is_dir() and (child / "manifest.json").exists():
                domain = OfflineDomainKB(child)
                self.domains[domain.domain_slug] = domain

    def domain(self, domain_id: str) -> OfflineDomainKB:
        return self.domains[domain_id]


def merge_multi_query_results(
    retriever: HybridRetriever,
    queries: Iterable[str],
    *,
    top_k_per_query: int = 8,
    limit: int = 10,
) -> List[Tuple[RetrievalDoc, Dict[str, Any]]]:
    merged: Dict[str, Tuple[RetrievalDoc, Dict[str, Any]]] = {}
    for query in dedupe(queries):
        rows = retriever.retrieve(query, top_k=top_k_per_query)
        for rank, (doc, scores) in enumerate(rows, start=1):
            bonus = 1.0 / (rank + 1)
            existing = merged.get(doc.doc_id)
            if existing is None:
                merged[doc.doc_id] = (
                    doc,
                    {
                        **scores,
                        "combined_score": float(scores.get("hybrid_score") or 0.0) + bonus,
                        "matched_queries": [query],
                    },
                )
            else:
                _, prev = existing
                prev["combined_score"] = float(prev.get("combined_score") or 0.0) + float(scores.get("hybrid_score") or 0.0) + bonus
                prev["bm25_score"] = max(float(prev.get("bm25_score") or 0.0), float(scores.get("bm25_score") or 0.0))
                prev["tfidf_score"] = max(float(prev.get("tfidf_score") or 0.0), float(scores.get("tfidf_score") or 0.0))
                prev["hybrid_score"] = max(float(prev.get("hybrid_score") or 0.0), float(scores.get("hybrid_score") or 0.0))
                matched = list(prev.get("matched_queries") or [])
                if query not in matched:
                    matched.append(query)
                prev["matched_queries"] = matched[:6]
    ranked = sorted(merged.values(), key=lambda item: float(item[1].get("combined_score") or 0.0), reverse=True)
    return ranked[:limit]
