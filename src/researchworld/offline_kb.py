from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

def _load_tfidf_vectorizer():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        return TfidfVectorizer
    except Exception:  # pragma: no cover
        return None

try:
    from researchworld.llm import (
        OpenAICompatChatClient,
        OpenAICompatEmbeddingClient,
        complete_json_object,
        load_openai_compat_config,
        load_openai_compat_embedding_config,
    )
except Exception:  # pragma: no cover
    OpenAICompatChatClient = None  # type: ignore
    OpenAICompatEmbeddingClient = None  # type: ignore
    complete_json_object = None  # type: ignore
    load_openai_compat_config = None  # type: ignore
    load_openai_compat_embedding_config = None  # type: ignore


PUBLIC_DOMAIN_TO_ID = {
    "LLM agents": "llm_agent",
    "LLM finetuning and post-training": "llm_finetuning_post_training",
    "LLM fine-tuning and post-training": "llm_finetuning_post_training",
    "RAG and retrieval structuring": "rag_and_retrieval_structuring",
    "Visual generative modeling and diffusion": "visual_generative_modeling_and_diffusion",
}


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\./+]{0,63}")


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EMBEDDING_CONFIG = ROOT / "configs" / "embedding" / "bge_m3.local.yaml"
DEFAULT_LLM_CONFIG = ROOT / "configs" / "llm" / "qwen3_235b_8002.local.yaml"


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


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def sentence_aware_chunks(text: Any, *, chunk_chars: int = 1050, overlap_chars: int = 160) -> List[str]:
    value = normalize_ws(text)
    if not value:
        return []
    if len(value) <= chunk_chars:
        return [value]
    sentences = [normalize_ws(x) for x in SENTENCE_RE.split(value) if normalize_ws(x)]
    if len(sentences) <= 1:
        chunks: List[str] = []
        start = 0
        while start < len(value):
            end = min(len(value), start + chunk_chars)
            chunks.append(value[start:end].strip())
            if end >= len(value):
                break
            start = max(start + 1, end - overlap_chars)
        return [x for x in chunks if x]
    chunks = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
            continue
        if len(current) + 1 + len(sentence) <= chunk_chars:
            current = f"{current} {sentence}"
            continue
        chunks.append(current)
        overlap = ""
        tail = current
        if overlap_chars > 0 and len(tail) > overlap_chars:
            overlap = tail[-overlap_chars:].strip()
            cut = overlap.find(" ")
            if cut > 0:
                overlap = overlap[cut + 1 :].strip()
        current = f"{overlap} {sentence}".strip() if overlap else sentence
    if current:
        chunks.append(current)
    return [x for x in chunks if x]


@dataclass
class RetrievalDoc:
    doc_id: str
    paper_id: str
    title: str
    text: str
    meta: Dict[str, Any]


class EmbeddingVectorCache:
    """Small SQLite cache for OpenAI-compatible embedding vectors."""

    def __init__(
        self,
        *,
        client: Any,
        cache_path: Path,
        namespace: str,
        batch_size: int = 96,
    ):
        self.client = client
        self.cache_path = cache_path
        self.namespace = namespace
        self.batch_size = max(1, int(batch_size))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.cache_path), timeout=60.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    namespace TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dims INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (namespace, text_hash, model)
                )
                """
            )

    def _hash_text(self, text: str) -> str:
        import hashlib

        cleaned = normalize_ws(text)
        return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    def _read_many(self, keys: List[str]) -> Dict[str, np.ndarray]:
        if not keys:
            return {}
        out: Dict[str, np.ndarray] = {}
        model = str(getattr(getattr(self.client, "config", None), "model_name", ""))
        with self._connect() as conn:
            for start in range(0, len(keys), 900):
                chunk = keys[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT text_hash, dims, vector
                    FROM embeddings
                    WHERE namespace = ? AND model = ? AND text_hash IN ({placeholders})
                    """,
                    [self.namespace, model, *chunk],
                ).fetchall()
                for text_hash, dims, blob in rows:
                    vec = np.frombuffer(blob, dtype=np.float32)
                    if int(dims) == int(vec.size) and vec.size:
                        out[str(text_hash)] = vec
        return out

    def _write_many(self, rows: List[Tuple[str, np.ndarray]]) -> None:
        if not rows:
            return
        model = str(getattr(getattr(self.client, "config", None), "model_name", ""))
        payload = [
            (
                self.namespace,
                text_hash,
                model,
                int(vector.size),
                np.asarray(vector, dtype=np.float32).tobytes(),
            )
            for text_hash, vector in rows
            if vector.size
        ]
        if not payload:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings(namespace, text_hash, model, dims, vector)
                VALUES (?, ?, ?, ?, ?)
                """,
                payload,
            )

    def embed_texts(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        cleaned = [normalize_ws(text) for text in texts]
        keys = [self._hash_text(text) if text else "" for text in cleaned]
        unique_missing: List[Tuple[str, str]] = []
        cached = self._read_many([key for key in sorted(set(keys)) if key])
        seen_missing = set()
        for key, text in zip(keys, cleaned):
            if not key or key in cached or key in seen_missing:
                continue
            seen_missing.add(key)
            unique_missing.append((key, text))
        for start in range(0, len(unique_missing), self.batch_size):
            batch = unique_missing[start : start + self.batch_size]
            vectors = self.client.embed([text for _, text in batch], transport_retries=2, retry_delay=1.5)
            write_rows: List[Tuple[str, np.ndarray]] = []
            for (key, _), vector in zip(batch, vectors):
                arr = np.asarray(vector, dtype=np.float32)
                norm = float(np.linalg.norm(arr))
                if norm > 0.0:
                    arr = arr / norm
                cached[key] = arr
                write_rows.append((key, arr))
            self._write_many(write_rows)
        return [cached.get(key) if key else None for key in keys]


class HybridRetriever:
    def __init__(
        self,
        docs: List[RetrievalDoc],
        *,
        embedding_cache: Optional[EmbeddingVectorCache] = None,
        semantic_weight: float = 0.65,
        bm25_weight: float = 0.35,
        tfidf_weight: float = 0.0,
    ):
        self.docs = docs
        self.embedding_cache = embedding_cache
        self.semantic_weight = float(semantic_weight)
        self.bm25_weight = float(bm25_weight)
        self.tfidf_weight = float(tfidf_weight)
        self._doc_embedding_matrix: Optional[np.ndarray] = None
        self.disable_tfidf = str(os.environ.get("RTL_RETRIEVAL_DISABLE_TFIDF", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        self.lightweight_lexical = str(os.environ.get("RTL_RETRIEVAL_LIGHTWEIGHT_LEXICAL", "1")).strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if not docs:
            self.doc_texts = []
            self.doc_tokens = []
            self.doc_token_sets = []
            self.bm25 = None
            self.vectorizer = None
            self.doc_matrix = None
            self._fallback_vocab = {}
            self._fallback_idf = np.zeros(0, dtype=float)
            return
        self.doc_texts = [str(doc.meta.get("bm25_text") or doc.text) for doc in docs]
        self.embedding_texts = [str(doc.meta.get("embedding_text") or doc.text) for doc in docs]
        self.doc_tokens = [tokenize(text) for text in self.doc_texts]
        self.doc_token_sets = [set(tokens) for tokens in self.doc_tokens]
        self.bm25 = None if self.lightweight_lexical else BM25Okapi(self.doc_tokens)
        self._fallback_vocab: Dict[str, int] = {}
        self._fallback_idf = np.zeros(0, dtype=float)
        if self.disable_tfidf:
            self.vectorizer = None
            self.doc_matrix = np.zeros((len(docs), 0), dtype=float)
        else:
            tfidf_vectorizer = _load_tfidf_vectorizer()
            if tfidf_vectorizer is None:
                self.vectorizer = None
                self.doc_matrix = self._build_fallback_doc_matrix(self.doc_tokens)
                return
            max_df = 1.0 if len(docs) < 5 else 0.98
            self.vectorizer = tfidf_vectorizer(
                lowercase=True,
                strip_accents="unicode",
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_df=max_df,
            )
            self.doc_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def _build_fallback_doc_matrix(self, doc_tokens: List[List[str]]) -> np.ndarray:
        doc_freq: Dict[str, int] = {}
        term_counts_per_doc: List[Dict[str, int]] = []
        for tokens in doc_tokens:
            counts: Dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            term_counts_per_doc.append(counts)
            for tok in counts:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        vocab = {tok: idx for idx, tok in enumerate(sorted(doc_freq))}
        self._fallback_vocab = vocab
        if not vocab:
            self._fallback_idf = np.zeros(0, dtype=float)
            return np.zeros((len(doc_tokens), 0), dtype=float)
        n_docs = max(1, len(doc_tokens))
        idf = np.zeros(len(vocab), dtype=float)
        for tok, idx in vocab.items():
            idf[idx] = math.log((1.0 + n_docs) / (1.0 + float(doc_freq.get(tok, 0)))) + 1.0
        self._fallback_idf = idf
        matrix = np.zeros((len(doc_tokens), len(vocab)), dtype=float)
        for row_idx, counts in enumerate(term_counts_per_doc):
            total = float(sum(counts.values()) or 1.0)
            for tok, count in counts.items():
                col_idx = vocab.get(tok)
                if col_idx is None:
                    continue
                matrix[row_idx, col_idx] = (float(count) / total) * idf[col_idx]
            norm = float(np.linalg.norm(matrix[row_idx]))
            if norm > 0.0:
                matrix[row_idx] /= norm
        return matrix

    def _fallback_query_vector(self, query: str) -> np.ndarray:
        if not self._fallback_vocab:
            return np.zeros(0, dtype=float)
        counts: Dict[str, int] = {}
        for tok in tokenize(query):
            if tok in self._fallback_vocab:
                counts[tok] = counts.get(tok, 0) + 1
        vec = np.zeros(len(self._fallback_vocab), dtype=float)
        total = float(sum(counts.values()) or 1.0)
        for tok, count in counts.items():
            idx = self._fallback_vocab[tok]
            vec[idx] = (float(count) / total) * self._fallback_idf[idx]
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def retrieve(self, query: str, *, top_k: int = 10) -> List[Tuple[RetrievalDoc, Dict[str, float]]]:
        if not self.docs:
            return []
        query_tokens = tokenize(query)
        if self.bm25 is not None:
            bm25_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=float)
        else:
            query_set = set(query_tokens)
            query_norm = math.sqrt(float(len(query_set) or 1))
            bm25_scores = np.asarray(
                [
                    len(query_set & token_set) / max(query_norm * math.sqrt(float(len(token_set) or 1)), 1.0)
                    for token_set in self.doc_token_sets
                ],
                dtype=float,
            )
        if self.vectorizer is not None:
            tfidf_query = self.vectorizer.transform([query])
            dense_scores = (self.doc_matrix @ tfidf_query.T).toarray().reshape(-1)
        else:
            query_vec = self._fallback_query_vector(query)
            dense_scores = self.doc_matrix @ query_vec if query_vec.size else np.zeros(len(self.docs), dtype=float)

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
        semantic_scores = np.zeros(len(self.docs), dtype=float)
        semantic_norm = semantic_scores
        used_semantic = False
        if self.embedding_cache is not None:
            try:
                query_vecs = self.embedding_cache.embed_texts([query])
                query_vec = query_vecs[0] if query_vecs else None
                candidate_limit = int(os.environ.get("RTL_RETRIEVAL_VECTOR_CANDIDATE_LIMIT", "1500") or 0)
                if query_vec is not None and candidate_limit > 0 and len(self.docs) > candidate_limit:
                    lexical = 0.55 * bm25_norm + 0.45 * dense_norm
                    candidate_indices = np.argsort(-lexical)[: max(int(top_k), candidate_limit)]
                    candidate_vectors = self.embedding_cache.embed_texts(
                        [self.embedding_texts[int(idx)] for idx in candidate_indices]
                    )
                    valid_rows: List[int] = []
                    valid_vectors: List[np.ndarray] = []
                    for local_idx, vector in enumerate(candidate_vectors):
                        if vector is None or not vector.size:
                            continue
                        valid_rows.append(int(candidate_indices[local_idx]))
                        valid_vectors.append(vector.astype(np.float32))
                    if valid_vectors:
                        matrix = np.vstack(valid_vectors)
                        local_scores = matrix @ query_vec.astype(np.float32)
                        local_norm = normalize(local_scores)
                        for row_idx, score, norm_score in zip(valid_rows, local_scores, local_norm):
                            semantic_scores[row_idx] = float(score)
                            semantic_norm[row_idx] = float(norm_score)
                        used_semantic = True
                else:
                    doc_matrix = self._load_doc_embedding_matrix()
                    if query_vec is not None and doc_matrix.size:
                        semantic_scores = doc_matrix @ query_vec.astype(np.float32)
                        semantic_norm = normalize(semantic_scores)
                        used_semantic = True
            except Exception:
                used_semantic = False

        if used_semantic:
            fused = (
                self.semantic_weight * semantic_norm
                + self.bm25_weight * bm25_norm
            )
        else:
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
                        "embedding_score": float(semantic_scores[int(idx)]) if used_semantic else 0.0,
                        "retrieval_mode": "bm25_vector" if used_semantic else "bm25_tfidf_fallback",
                    },
                )
            )
        return rows

    def _load_doc_embedding_matrix(self) -> np.ndarray:
        if self._doc_embedding_matrix is not None:
            return self._doc_embedding_matrix
        if self.embedding_cache is None or not self.doc_texts:
            self._doc_embedding_matrix = np.zeros((len(self.docs), 0), dtype=np.float32)
            return self._doc_embedding_matrix
        vectors = self.embedding_cache.embed_texts(self.embedding_texts)
        dim = 0
        for vector in vectors:
            if vector is not None and vector.size:
                dim = int(vector.size)
                break
        if dim <= 0:
            self._doc_embedding_matrix = np.zeros((len(self.docs), 0), dtype=np.float32)
            return self._doc_embedding_matrix
        matrix = np.zeros((len(self.docs), dim), dtype=np.float32)
        for idx, vector in enumerate(vectors):
            if vector is None or int(vector.size) != dim:
                continue
            matrix[idx] = vector.astype(np.float32)
        self._doc_embedding_matrix = matrix
        return self._doc_embedding_matrix


class PageIndexLLMRetriever:
    """LLM selector over paper page/section index nodes.

    The first stage only narrows candidates so the LLM sees a bounded list.
    The returned ranking is produced by the LLM selector, not by the candidate
    retriever score alone.
    """

    def __init__(
        self,
        docs: List[RetrievalDoc],
        *,
        llm_client: Optional[Any],
        candidate_retriever: HybridRetriever,
    ):
        self.docs = docs
        self.llm_client = llm_client
        self.candidate_retriever = candidate_retriever
        self._cache: Dict[Tuple[str, int], List[Tuple[RetrievalDoc, Dict[str, float]]]] = {}

    def retrieve(self, query: str, *, top_k: int = 10) -> List[Tuple[RetrievalDoc, Dict[str, float]]]:
        key = (normalize_ws(query), int(top_k))
        if key in self._cache:
            return self._cache[key]
        candidate_limit = max(int(top_k) * 4, 16)
        candidates = self.candidate_retriever.retrieve(query, top_k=candidate_limit)
        if not candidates or self.llm_client is None or complete_json_object is None:
            rows = [
                (
                    doc,
                    {
                        **scores,
                        "retrieval_mode": "pageindex_candidate_fallback",
                        "llm_relevance": float(scores.get("hybrid_score") or 0.0),
                    },
                )
                for doc, scores in candidates[:top_k]
            ]
            self._cache[key] = rows
            return rows

        candidate_lines: List[str] = []
        for idx, (doc, scores) in enumerate(candidates, start=1):
            candidate_lines.append(
                "\n".join(
                    [
                        f"[{idx}] paper_id={doc.paper_id}",
                        f"title={clip_text(doc.title, 180)}",
                        f"section={clip_text(doc.meta.get('section_title') or '', 120)}",
                        f"path={clip_text(doc.meta.get('section_path') or '', 160)}",
                        f"kind={doc.meta.get('kind') or ''}",
                        f"summary/text={clip_text(doc.text, 900)}",
                        f"candidate_score={float(scores.get('hybrid_score') or 0.0):.4f}",
                    ]
                )
            )
        prompt = (
            "You are selecting relevant paper section/page-index nodes for an offline research benchmark.\n"
            "Given a research query and candidate section nodes, choose the nodes that best answer the query.\n"
            "Prefer nodes with direct methodological, limitation, experiment, future-work, or conclusion evidence.\n"
            "Do not choose a node only because it shares keywords; judge whether the section content is substantively useful.\n"
            "Return JSON only with this schema:\n"
            '{"selected":[{"index":1,"relevance":0.0,"reason":"short reason"}]}\n'
            f"Select at most {int(top_k)} nodes. relevance must be between 0 and 1.\n\n"
            f"Query:\n{query}\n\n"
            "Candidate nodes:\n"
            + "\n\n".join(candidate_lines)
        )
        try:
            obj = complete_json_object(
                self.llm_client,
                [
                    {"role": "system", "content": "Return exactly one valid JSON object. No markdown."},
                    {"role": "user", "content": prompt},
                ],
                max_parse_attempts=2,
                temperature=0.0,
                max_tokens=1400,
                timeout=90,
                transport_retries=1,
            )
            selected = obj.get("selected") if isinstance(obj, dict) else []
            rows: List[Tuple[RetrievalDoc, Dict[str, float]]] = []
            seen = set()
            for item in selected if isinstance(selected, list) else []:
                if not isinstance(item, dict):
                    continue
                index = int(item.get("index") or 0)
                if index < 1 or index > len(candidates) or index in seen:
                    continue
                seen.add(index)
                doc, scores = candidates[index - 1]
                relevance = max(0.0, min(1.0, float(item.get("relevance") or 0.0)))
                rows.append(
                    (
                        doc,
                        {
                            **scores,
                            "hybrid_score": relevance,
                            "combined_score": relevance,
                            "llm_relevance": relevance,
                            "retrieval_mode": "pageindex_llm_selector",
                            "llm_reason": clip_text(item.get("reason") or "", 240),
                        },
                    )
                )
            if not rows:
                raise ValueError("pageindex LLM selector returned no valid rows")
            rows = sorted(rows, key=lambda item: float(item[1].get("llm_relevance") or 0.0), reverse=True)[:top_k]
            self._cache[key] = rows
            return rows
        except Exception:
            rows = [
                (
                    doc,
                    {
                        **scores,
                        "retrieval_mode": "pageindex_candidate_fallback_after_llm_error",
                        "llm_relevance": float(scores.get("hybrid_score") or 0.0),
                        "llm_error": "pageindex selector failed; used first-stage candidate ranking",
                    },
                )
                for doc, scores in candidates[:top_k]
            ]
            self._cache[key] = rows
            return rows


class OfflineDomainKB:
    def __init__(
        self,
        domain_dir: Path,
        *,
        embedding_client: Optional[Any] = None,
        pageindex_llm_client: Optional[Any] = None,
        embedding_cache_dir: Optional[Path] = None,
    ):
        self.domain_dir = domain_dir
        self.manifest = json.loads((domain_dir / "manifest.json").read_text(encoding="utf-8"))
        self.domain_slug = str(self.manifest["domain_slug"])
        self.domain = str(self.manifest["domain"])
        self.embedding_client = embedding_client
        self.pageindex_llm_client = pageindex_llm_client
        self.embedding_cache_dir = embedding_cache_dir
        self._papers = list(iter_jsonl(domain_dir / "papers.jsonl"))
        self._abstract_chunks = list(iter_jsonl(domain_dir / "abstract_chunks.jsonl")) if (domain_dir / "abstract_chunks.jsonl").exists() else []
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
        self._abstract_chunk_docs_cache: Dict[str, List[RetrievalDoc]] = {}
        self._abstract_chunk_retriever_cache: Dict[str, HybridRetriever] = {}
        self._structure_docs_cache: Dict[Tuple[str, Tuple[str, ...]], List[RetrievalDoc]] = {}
        self._structure_retriever_cache: Dict[Tuple[str, Tuple[str, ...]], HybridRetriever] = {}
        self._section_retrievers: Dict[Tuple[str, Tuple[str, ...]], HybridRetriever] = {}
        self._pageindex_retrievers: Dict[Tuple[str, Tuple[str, ...]], Any] = {}

    def _embedding_cache(self, doc_type: str) -> Optional[EmbeddingVectorCache]:
        if self.embedding_client is None or self.embedding_cache_dir is None:
            return None
        namespace = f"{self.domain_slug}:{doc_type}"
        return EmbeddingVectorCache(
            client=self.embedding_client,
            cache_path=self.embedding_cache_dir / "embeddings.sqlite3",
            namespace=namespace,
        )

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
            self._paper_retriever_cache[cache_key] = HybridRetriever(
                self.paper_docs(cutoff_date=cutoff_date),
                embedding_cache=self._embedding_cache("paper"),
            )
        return self._paper_retriever_cache[cache_key]

    def abstract_chunk_docs(self, *, cutoff_date: Optional[str] = None) -> List[RetrievalDoc]:
        cache_key = self._cache_cutoff_key(cutoff_date)
        if cache_key not in self._abstract_chunk_docs_cache:
            docs: List[RetrievalDoc] = []
            if self._abstract_chunks:
                rows = self._abstract_chunks
                for row in rows:
                    paper_id = str(row.get("paper_id") or "")
                    published_date = row.get("published_date")
                    if not on_or_before(published_date, cutoff_date):
                        continue
                    title = str(row.get("title") or "")
                    chunk = str(row.get("abstract_chunk") or "")
                    bm25_text = str(row.get("bm25_text") or row.get("retrieval_text") or row.get("metadata_text") or "")
                    embedding_text = str(row.get("embedding_text") or row.get("semantic_text") or "")
                    if not bm25_text:
                        bm25_text = self._abstract_chunk_bm25_text(row=row, title=title, chunk=chunk)
                    if not embedding_text:
                        embedding_text = self._abstract_chunk_embedding_text(row=row, title=title, chunk=chunk)
                    docs.append(
                        RetrievalDoc(
                            doc_id=str(row.get("chunk_id") or f"abstract::{paper_id}::{row.get('chunk_index') or 0}"),
                            paper_id=paper_id,
                            title=title,
                            text=bm25_text,
                            meta={**row, "bm25_text": bm25_text, "embedding_text": embedding_text},
                        )
                    )
            else:
                for row in self._papers:
                    if not on_or_before(row.get("published_date"), cutoff_date):
                        continue
                    paper_id = str(row.get("paper_id") or "")
                    title = str(row.get("title") or "")
                    abstract = str(row.get("abstract") or "")
                    chunks = sentence_aware_chunks(abstract)
                    for idx, chunk in enumerate(chunks):
                        chunk_row = self._make_abstract_chunk_row(row=row, chunk=chunk, chunk_index=idx, chunk_count=len(chunks))
                        bm25_text = str(chunk_row["bm25_text"])
                        embedding_text = str(chunk_row["embedding_text"])
                        docs.append(
                            RetrievalDoc(
                                doc_id=str(chunk_row["chunk_id"]),
                                paper_id=paper_id,
                                title=title,
                                text=bm25_text,
                                meta={**chunk_row, "bm25_text": bm25_text, "embedding_text": embedding_text},
                            )
                        )
            self._abstract_chunk_docs_cache[cache_key] = docs
        return self._abstract_chunk_docs_cache[cache_key]

    def abstract_chunk_retriever(self, *, cutoff_date: Optional[str] = None) -> HybridRetriever:
        cache_key = self._cache_cutoff_key(cutoff_date)
        if cache_key not in self._abstract_chunk_retriever_cache:
            self._abstract_chunk_retriever_cache[cache_key] = HybridRetriever(
                self.abstract_chunk_docs(cutoff_date=cutoff_date),
                embedding_cache=self._embedding_cache("abstract_chunk"),
            )
        return self._abstract_chunk_retriever_cache[cache_key]

    def _make_abstract_chunk_row(self, *, row: Dict[str, Any], chunk: str, chunk_index: int, chunk_count: int) -> Dict[str, Any]:
        paper_id = str(row.get("paper_id") or "")
        pub = row.get("publication") or {}
        title = str(row.get("title") or "")
        out = {
            "chunk_id": f"{self.domain_slug}::{paper_id}::abstract::{chunk_index:03d}",
            "doc_type": "abstract_chunk",
            "domain_slug": self.domain_slug,
            "domain": self.domain,
            "paper_id": paper_id,
            "title": title,
            "abstract_chunk": chunk,
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "published_date": row.get("published_date"),
            "venue_name": pub.get("venue_name"),
            "top_venue_bucket": pub.get("top_venue_bucket"),
            "citation_count": pub.get("citation_count"),
            "source_type": "abstract_only",
        }
        out["bm25_text"] = self._abstract_chunk_bm25_text(row=out, title=title, chunk=chunk)
        out["embedding_text"] = self._abstract_chunk_embedding_text(row=out, title=title, chunk=chunk)
        return out

    def _abstract_chunk_bm25_text(self, *, row: Dict[str, Any], title: str, chunk: str) -> str:
        return "\n".join(
            part
            for part in [
                f"Title: {title}",
                f"Published: {row.get('published_date') or ''}",
                f"Venue: {row.get('venue_name') or ''}",
                f"Top venue: {row.get('top_venue_bucket') or ''}",
                f"Citations: {row.get('citation_count') if row.get('citation_count') is not None else ''}",
                f"Abstract chunk {int(row.get('chunk_index') or 0) + 1}/{int(row.get('chunk_count') or 1)}: {chunk}",
            ]
            if normalize_ws(part)
        )

    def _abstract_chunk_embedding_text(self, *, row: Dict[str, Any], title: str, chunk: str) -> str:
        return "\n".join(
            part
            for part in [
                f"Title: {title}",
                f"Domain: {row.get('domain') or self.domain}",
                f"Abstract: {chunk}",
            ]
            if normalize_ws(part)
        )

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
            self._structure_retriever_cache[key] = HybridRetriever(
                self.structure_docs(cutoff_date=cutoff_date, paper_ids=paper_ids),
                embedding_cache=self._embedding_cache("structure"),
            )
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
            self._section_retrievers[key] = HybridRetriever(
                self.section_docs(cutoff_date=cutoff_date, paper_ids=paper_ids),
                embedding_cache=self._embedding_cache("section"),
            )
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
            docs = self.pageindex_docs(cutoff_date=cutoff_date, paper_ids=paper_ids)
            candidate_retriever = HybridRetriever(
                docs,
                embedding_cache=self._embedding_cache("pageindex_candidate"),
                semantic_weight=0.45,
                bm25_weight=0.55,
            )
            self._pageindex_retrievers[key] = PageIndexLLMRetriever(
                docs,
                llm_client=self.pageindex_llm_client,
                candidate_retriever=candidate_retriever,
            )
        return self._pageindex_retrievers[key]

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.papers_by_id.get(str(paper_id))

    def get_structure(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.structures_by_id.get(str(paper_id))

    def get_pageindex(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.pageindex_by_id.get(str(paper_id))


class OfflineKnowledgeBase:
    def __init__(
        self,
        kb_dir: Path,
        *,
        embedding_client: Optional[Any] = None,
        pageindex_llm_client: Optional[Any] = None,
        embedding_config: Optional[Path | str] = None,
        llm_config: Optional[Path | str] = None,
        embedding_cache_dir: Optional[Path | str] = None,
        auto_embedding: bool = True,
        auto_pageindex_llm: bool = True,
    ):
        self.kb_dir = kb_dir
        self.manifest = json.loads((kb_dir / "manifest.json").read_text(encoding="utf-8"))
        self.embedding_client = embedding_client or self._load_default_embedding_client(
            embedding_config=embedding_config,
            auto_embedding=auto_embedding,
        )
        self.pageindex_llm_client = pageindex_llm_client or self._load_default_llm_client(
            llm_config=llm_config,
            auto_pageindex_llm=auto_pageindex_llm,
        )
        self.embedding_cache_dir = (
            Path(embedding_cache_dir)
            if embedding_cache_dir is not None
            else kb_dir / ".embedding_cache"
        )
        domains_dir = kb_dir / "domains"
        self.domains: Dict[str, OfflineDomainKB] = {}
        for child in sorted(domains_dir.iterdir()):
            if child.is_dir() and (child / "manifest.json").exists():
                domain = OfflineDomainKB(
                    child,
                    embedding_client=self.embedding_client,
                    pageindex_llm_client=self.pageindex_llm_client,
                    embedding_cache_dir=self.embedding_cache_dir if self.embedding_client is not None else None,
                )
                self.domains[domain.domain_slug] = domain

    def domain(self, domain_id: str) -> OfflineDomainKB:
        return self.domains[domain_id]

    def _load_default_embedding_client(
        self,
        *,
        embedding_config: Optional[Path | str],
        auto_embedding: bool,
    ) -> Optional[Any]:
        if not auto_embedding:
            return None
        if OpenAICompatEmbeddingClient is None or load_openai_compat_embedding_config is None:
            return None
        configured = str(embedding_config or os.environ.get("RESEARCHFORESIGHT_EMBEDDING_CONFIG") or "").strip()
        config_path = Path(configured) if configured else DEFAULT_EMBEDDING_CONFIG
        if not config_path.exists():
            return None
        try:
            return OpenAICompatEmbeddingClient(load_openai_compat_embedding_config(config_path))
        except Exception:
            return None

    def _load_default_llm_client(
        self,
        *,
        llm_config: Optional[Path | str],
        auto_pageindex_llm: bool,
    ) -> Optional[Any]:
        if not auto_pageindex_llm:
            return None
        if OpenAICompatChatClient is None or load_openai_compat_config is None:
            return None
        configured = str(llm_config or os.environ.get("RESEARCHFORESIGHT_LLM_CONFIG") or "").strip()
        config_path = Path(configured) if configured else DEFAULT_LLM_CONFIG
        if not config_path.exists():
            return None
        try:
            return OpenAICompatChatClient(load_openai_compat_config(config_path))
        except Exception:
            return None


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
