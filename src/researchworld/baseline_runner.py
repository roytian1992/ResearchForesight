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

from researchworld.corpus import iter_jsonl
from researchworld.llm import (
    OpenAICompatChatClient,
    complete_json_object,
    extract_json_object,
    load_openai_compat_config,
)


ROOT = Path(__file__).resolve().parents[2]

PUBLIC_DOMAIN_TO_ID = {
    "LLM agents": "llm_agent",
    "LLM finetuning and post-training": "llm_finetuning_post_training",
    "LLM fine-tuning and post-training": "llm_finetuning_post_training",
    "RAG and retrieval structuring": "rag_and_retrieval_structuring",
    "Visual generative modeling and diffusion": "visual_generative_modeling_and_diffusion",
}


def parse_iso_date(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw[:19], fmt)
        except ValueError:
            continue
    return None


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-\./+]{0,63}")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(text or ""))]


def clip_text(text: str, limit: int) -> str:
    value = normalize_ws(text)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def load_release_tasks(release_dir: Path) -> List[Dict[str, Any]]:
    with (release_dir / "tasks.jsonl").open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_hidden_eval(release_dir: Path) -> Dict[str, Dict[str, Any]]:
    with (release_dir / "tasks_hidden_eval.jsonl").open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {str(row["task_id"]): row for row in rows}


def load_domain_labels(domain_id: str) -> Dict[str, Dict[str, Any]]:
    path = ROOT / "data" / "domains" / domain_id / "annotations" / "paper_labels.jsonl"
    return {str(row["paper_id"]): row for row in iter_jsonl(path)}


def load_domain_metadata(domain_id: str) -> Dict[str, Dict[str, Any]]:
    path = ROOT / "data" / "domains" / domain_id / "interim" / "papers_merged.publication_enriched.semanticscholar.jsonl"
    labels = load_domain_labels(domain_id)
    rows: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        paper_id = str(row.get("paper_id") or "")
        label = labels.get(paper_id) or {}
        if str(label.get("scope_decision") or "") != "core_domain":
            continue
        rows[paper_id] = row
    return rows


def load_domain_content(domain_id: str) -> Dict[str, Dict[str, Any]]:
    path = ROOT / "data" / "support_packets" / "fulltext_content" / domain_id / "content.jsonl"
    return {str(row["paper_id"]): row for row in iter_jsonl(path)}


def load_domain_pageindex(domain_id: str) -> Dict[str, Dict[str, Any]]:
    path = ROOT / "data" / "support_packets" / "pageindex" / domain_id / "pageindex.jsonl"
    if not path.exists():
        return {}
    return {str(row["paper_id"]): row for row in iter_jsonl(path)}


def paper_publication_date(row: Dict[str, Any]) -> Optional[datetime]:
    enrichment = row.get("publication_enrichment") or {}
    for candidate in [
        enrichment.get("published_date"),
        enrichment.get("openalex_publication_date"),
        enrichment.get("crossref_published_date"),
        row.get("published"),
        row.get("updated"),
    ]:
        dt = parse_iso_date(candidate)
        if dt is not None:
            return dt
    return None


def paper_venue(row: Dict[str, Any]) -> str:
    enrichment = row.get("publication_enrichment") or {}
    return (
        enrichment.get("published_venue_name")
        or enrichment.get("semantic_scholar_venue")
        or "unknown"
    )


def paper_citations(row: Dict[str, Any]) -> int:
    enrichment = row.get("publication_enrichment") or {}
    try:
        return int(enrichment.get("preferred_cited_by_count") or 0)
    except Exception:
        return 0


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


class DomainCorpus:
    def __init__(self, domain_id: str):
        self.domain_id = domain_id
        self.metadata_by_paper = load_domain_metadata(domain_id)
        self.content_by_paper = load_domain_content(domain_id)
        self.pageindex_by_paper = load_domain_pageindex(domain_id)
        self._paper_docs_cache: Dict[str, List[RetrievalDoc]] = {}
        self._paper_retriever_cache: Dict[str, HybridRetriever] = {}

    def paper_docs(self, cutoff: str) -> List[RetrievalDoc]:
        if cutoff in self._paper_docs_cache:
            return self._paper_docs_cache[cutoff]
        cutoff_dt = parse_iso_date(cutoff)
        docs: List[RetrievalDoc] = []
        for paper_id, row in self.metadata_by_paper.items():
            pub_dt = paper_publication_date(row)
            if cutoff_dt is not None and pub_dt is not None and pub_dt.date() > cutoff_dt.date():
                continue
            content = self.content_by_paper.get(paper_id) or {}
            abstract = content.get("abstract") or row.get("abstract") or ""
            title = str(row.get("title") or content.get("title") or "")
            text = "\n".join(
                part
                for part in [
                    f"Title: {title}",
                    f"Venue: {paper_venue(row)}",
                    f"Published: {(row.get('published') or '')}",
                    f"Citations: {paper_citations(row)}",
                    f"Abstract: {abstract}",
                ]
                if normalize_ws(part)
            )
            docs.append(
                RetrievalDoc(
                    doc_id=f"paper::{paper_id}",
                    paper_id=paper_id,
                    title=title,
                    text=text,
                    meta={
                        "venue": paper_venue(row),
                        "citations": paper_citations(row),
                        "published": row.get("published"),
                    },
                )
            )
        self._paper_docs_cache[cutoff] = docs
        return docs

    def paper_retriever(self, cutoff: str) -> HybridRetriever:
        if cutoff not in self._paper_retriever_cache:
            self._paper_retriever_cache[cutoff] = HybridRetriever(self.paper_docs(cutoff))
        return self._paper_retriever_cache[cutoff]

    def node_docs(self, cutoff: str, *, paper_ids: Optional[set[str]] = None) -> List[RetrievalDoc]:
        cutoff_dt = parse_iso_date(cutoff)
        docs: List[RetrievalDoc] = []
        for paper_id, row in self.metadata_by_paper.items():
            if paper_ids is not None and paper_id not in paper_ids:
                continue
            pub_dt = paper_publication_date(row)
            if cutoff_dt is not None and pub_dt is not None and pub_dt.date() > cutoff_dt.date():
                continue
            pageindex = self.pageindex_by_paper.get(paper_id)
            if not pageindex:
                continue
            paper_title = str(pageindex.get("paper_title") or row.get("title") or "")
            for node in pageindex.get("nodes") or []:
                node_id = str(node.get("node_id") or "")
                section_title = str(node.get("normalized_title") or node.get("title") or "")
                section_path = str(node.get("section_path") or section_title)
                node_text = str(node.get("text") or "")
                full_text = "\n".join(
                    part
                    for part in [
                        f"Paper: {paper_title}",
                        f"Section: {section_title}",
                        f"Path: {section_path}",
                        f"Kind: {node.get('kind')}",
                        node_text,
                    ]
                    if normalize_ws(part)
                )
                docs.append(
                    RetrievalDoc(
                        doc_id=f"node::{paper_id}::{node_id}",
                        paper_id=paper_id,
                        title=f"{paper_title} / {section_title}",
                        text=full_text,
                        meta={
                            "paper_title": paper_title,
                            "section_title": section_title,
                            "section_path": section_path,
                            "kind": node.get("kind"),
                        },
                    )
                )
        return docs


def build_hybrid_evidence(task: Dict[str, Any], corpus: DomainCorpus, *, top_k: int = 12) -> Dict[str, Any]:
    retriever = corpus.paper_retriever(task["time_cutoff"])
    rows = retriever.retrieve(task["question"], top_k=top_k)
    evidence = []
    for rank, (doc, scores) in enumerate(rows, start=1):
        evidence.append(
            {
                "evidence_id": f"P{rank}",
                "paper_id": doc.paper_id,
                "paper_title": doc.title,
                "venue": doc.meta.get("venue"),
                "citations": doc.meta.get("citations"),
                "published": doc.meta.get("published"),
                "snippet": clip_text(doc.text, 1400),
                "scores": scores,
            }
        )
    return {
        "retrieval_mode": "hybrid_paper",
        "retrieved": evidence,
    }


def build_pageindex_evidence(
    task: Dict[str, Any],
    corpus: DomainCorpus,
    *,
    coarse_k: int = 16,
    fine_k: int = 18,
) -> Dict[str, Any]:
    paper_retriever = corpus.paper_retriever(task["time_cutoff"])
    coarse = paper_retriever.retrieve(task["question"], top_k=coarse_k)
    paper_ids = {doc.paper_id for doc, _ in coarse}

    node_docs = corpus.node_docs(task["time_cutoff"], paper_ids=paper_ids)
    if not node_docs:
        return build_hybrid_evidence(task, corpus, top_k=min(12, coarse_k))
    node_retriever = HybridRetriever(node_docs)
    fine = node_retriever.retrieve(task["question"], top_k=fine_k)

    evidence = []
    for rank, (doc, scores) in enumerate(fine, start=1):
        evidence.append(
            {
                "evidence_id": f"S{rank}",
                "paper_id": doc.paper_id,
                "paper_title": doc.meta.get("paper_title") or doc.title,
                "section_title": doc.meta.get("section_title"),
                "section_path": doc.meta.get("section_path"),
                "section_kind": doc.meta.get("kind"),
                "snippet": clip_text(doc.text, 1200),
                "scores": scores,
            }
        )
    return {
        "retrieval_mode": "pageindex_section",
        "coarse_paper_ids": sorted(paper_ids),
        "retrieved": evidence,
    }


def render_evidence_block(evidence_rows: List[Dict[str, Any]]) -> str:
    parts = []
    for row in evidence_rows:
        head = [f"[{row['evidence_id']}] {row.get('paper_title', '')}"]
        if row.get("section_path"):
            head.append(f"section={row['section_path']}")
        if row.get("venue"):
            head.append(f"venue={row['venue']}")
        if row.get("citations") is not None:
            head.append(f"citations={row['citations']}")
        parts.append(" | ".join(head))
        parts.append(row.get("snippet") or "")
        parts.append("")
    return "\n".join(parts).strip()


def answer_task(
    client: OpenAICompatChatClient,
    *,
    task: Dict[str, Any],
    baseline_name: str,
    evidence_packet: Dict[str, Any],
) -> str:
    evidence_block = render_evidence_block(evidence_packet["retrieved"])
    prompt = f"""You are answering an offline research benchmark.

Task ID: {task['task_id']}
Family: {task['family']}
Domain: {task['domain']}
Horizon: {task['horizon']}
Time cutoff: {task['time_cutoff']}
Title: {task['title']}
Question:
{task['question']}

Constraints:
- Use only the provided evidence.
- Do not claim access to future papers beyond the cutoff.
- Make a concrete conclusion instead of generic trend language.
- Cite evidence inline using the provided evidence ids like [P3] or [S5].
- If the evidence is insufficient, say what remains uncertain.

Retrieved evidence ({baseline_name}):
{evidence_block}

Write a concise but substantive research answer."""
    return client.complete_text(
        [
            {"role": "system", "content": "You are a precise research assistant."},
            {"role": "user", "content": prompt},
        ]
    )


def judge_answer(
    client: OpenAICompatChatClient,
    *,
    public_task: Dict[str, Any],
    hidden_task: Dict[str, Any],
    candidate_answer: str,
) -> Dict[str, Any]:
    rubric = hidden_task.get("evaluation_rubric") or {}
    prompt = f"""Evaluate a benchmark answer.

Public task:
{json.dumps(public_task, ensure_ascii=False, indent=2)}

Hidden reference:
gold_answer: {hidden_task.get('gold_answer')}
expected_answer_points: {json.dumps(hidden_task.get('expected_answer_points') or [], ensure_ascii=False)}
evaluation_rubric: {json.dumps(rubric, ensure_ascii=False)}

Candidate answer:
{candidate_answer}

Return a JSON object with this schema:
{{
  "overall_score": float,
  "dimension_scores": {{"dimension_name": float}},
  "verdict": "strong" | "acceptable" | "weak",
  "reasoning": "brief explanation"
}}

Scores must be in [0, 1]."""
    return complete_json_object(
        client,
        [
            {"role": "system", "content": "You are a strict but fair benchmark judge. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        transport_retries=2,
        max_parse_attempts=3,
    )


def aggregate_scores(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {"count": 0, "mean_overall_score": 0.0}
    overall = [float((row.get("judge") or {}).get("overall_score") or 0.0) for row in rows]
    by_family: Dict[str, List[float]] = {}
    by_domain: Dict[str, List[float]] = {}
    for row, score in zip(rows, overall):
        by_family.setdefault(str(row.get("family") or "unknown"), []).append(score)
        by_domain.setdefault(str(row.get("domain_id") or "unknown"), []).append(score)
    return {
        "count": len(rows),
        "mean_overall_score": round(sum(overall) / len(overall), 4),
        "family_scores": {
            key: round(sum(values) / len(values), 4) for key, values in sorted(by_family.items())
        },
        "domain_scores": {
            key: round(sum(values) / len(values), 4) for key, values in sorted(by_domain.items())
        },
    }


def run_baseline(
    *,
    release_dir: Path,
    baseline_name: str,
    output_dir: Path,
    answer_llm_config: Path,
    judge_llm_config: Optional[Path] = None,
    task_limit: Optional[int] = None,
    domain_filter: Optional[set[str]] = None,
    family_filter: Optional[set[str]] = None,
) -> Dict[str, Any]:
    tasks = load_release_tasks(release_dir)
    hidden_by_id = load_hidden_eval(release_dir) if judge_llm_config else {}
    answer_client = OpenAICompatChatClient(load_openai_compat_config(answer_llm_config))
    judge_client = (
        OpenAICompatChatClient(load_openai_compat_config(judge_llm_config))
        if judge_llm_config is not None
        else None
    )

    corpora: Dict[str, DomainCorpus] = {}
    rows_out: List[Dict[str, Any]] = []
    selected_tasks = []
    for task in tasks:
        domain_id = PUBLIC_DOMAIN_TO_ID.get(str(task.get("domain") or "").strip())
        if not domain_id:
            continue
        if domain_filter and domain_id not in domain_filter:
            continue
        if family_filter and str(task.get("family") or "") not in family_filter:
            continue
        selected_tasks.append((task, domain_id))
    if task_limit is not None:
        selected_tasks = selected_tasks[:task_limit]

    for idx, (task, domain_id) in enumerate(selected_tasks, start=1):
        print(
            f"[baseline:{baseline_name}] task {idx}/{len(selected_tasks)} "
            f"{task['task_id']} domain={domain_id} family={task['family']}",
            flush=True,
        )
        if domain_id not in corpora:
            corpora[domain_id] = DomainCorpus(domain_id)
        corpus = corpora[domain_id]
        if baseline_name == "hybrid":
            evidence_packet = build_hybrid_evidence(task, corpus)
        elif baseline_name == "pageindex":
            evidence_packet = build_pageindex_evidence(task, corpus)
        else:
            raise ValueError(f"Unsupported baseline: {baseline_name}")
        answer = answer_task(
            answer_client,
            task=task,
            baseline_name=baseline_name,
            evidence_packet=evidence_packet,
        )
        row = {
            "task_id": task["task_id"],
            "family": task["family"],
            "domain": task["domain"],
            "domain_id": domain_id,
            "baseline": baseline_name,
            "title": task["title"],
            "question": task["question"],
            "time_cutoff": task["time_cutoff"],
            "answer": answer,
            "evidence": evidence_packet,
        }
        if judge_client is not None:
            hidden_task = hidden_by_id.get(str(task["task_id"]))
            if hidden_task is not None:
                row["judge"] = judge_answer(
                    judge_client,
                    public_task=task,
                    hidden_task=hidden_task,
                    candidate_answer=answer,
                )
        rows_out.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "baseline": baseline_name,
        "release_dir": str(release_dir),
        "task_count": len(rows_out),
        "results_path": str(results_path),
        "score_summary": aggregate_scores(rows_out),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary
