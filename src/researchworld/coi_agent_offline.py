from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from researchworld.llm import OpenAICompatChatClient, OpenAICompatEmbeddingClient, complete_json_object
from researchworld.fulltext_cache import LocalFulltextCache
from researchworld.coi_offline_retrieval import CoIOfflineRetrievalAdaptor, TaskCandidatePool
from researchworld.offline_kb import OfflineKnowledgeBase, clip_text, dedupe, merge_multi_query_results, normalize_ws
from researchworld.research_arc_kb import extract_focus_text
from researchworld.research_arc_v2 import extract_task_contract
from researchworld.retrieval_fusion import build_hybrid_task_queries, merge_retrieval_runs
from researchworld.answer_adapter import apply_shared_final_adapter_to_trace_result


ROOT = Path(__file__).resolve().parents[2]


def _resolve_coi_path() -> Optional[Path]:
    env_path = os.environ.get("RESEARCHFORESIGHT_COI_PATH", "").strip()
    candidates = [
        Path(env_path) if env_path else None,
        ROOT / "external" / "CoI-Agent",
        ROOT.parent / "ResearchTrajectoryLab" / "external" / "CoI-Agent",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


COI_PATH = _resolve_coi_path()
if COI_PATH and str(COI_PATH) not in sys.path:
    sys.path.insert(0, str(COI_PATH))

from prompts.deep_research_agent_prompts import (  # type: ignore
    get_deep_reference_prompt,
    get_deep_generate_future_direciton_prompt,
    get_deep_judge_relevant_prompt,
    get_deep_search_query_prompt,
    get_deep_trend_idea_chains_prompt,
)


TAG_RE_TEMPLATE = r"<{tag}>(.*?)</{tag}>"
GENERIC_TAG_RE = re.compile(r"</?[A-Za-z_][A-Za-z0-9_:-]*>")

COI_PROFILE_CACHE_ROOT = ROOT / "tmp" / "coi_profile_cache"
COI_EMBED_MODEL_NAME = os.environ.get("RTL_COI_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip() or "sentence-transformers/all-MiniLM-L6-v2"
_COI_EMBED_MODEL = None
_COI_EMBED_MODEL_LOAD_FAILED = False
_COI_EMBED_MODEL_ERROR = ""
COI_TRANSITION_BATCH_SIZE = 8
COI_EMBED_TEXT_LIMIT = 1200
COI_DEFAULT_ANCHOR_WORKERS = 3
COI_KEYWORD_STOPWORDS = {
    "about", "after", "agent", "agents", "analysis", "approach", "approaches", "based", "benchmark", "benchmarks",
    "between", "chain", "current", "dataset", "datasets", "direction", "directions", "during", "evidence",
    "focus", "framework", "frameworks", "from", "future", "idea", "ideas", "into", "method", "methods", "model",
    "models", "paper", "papers", "problem", "problems", "published", "question", "questions", "recent", "research",
    "result", "results", "same", "step", "steps", "study", "studies", "system", "systems", "task", "tasks",
    "technical", "their", "them", "these", "this", "topic", "trajectory", "using", "work", "works",
}


def extract_tag(text: str, tag: str) -> str:
    if not text:
        return ""
    pattern = re.compile(TAG_RE_TEMPLATE.format(tag=re.escape(tag)), re.DOTALL | re.IGNORECASE)
    matches = pattern.findall(text)
    if matches:
        return " ".join(str(x).strip() for x in matches if str(x).strip()).strip()
    return ""


def parse_query_list(text: str) -> List[str]:
    raw = extract_tag(text, "queries") or text
    raw = raw.strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return dedupe(str(x) for x in value if str(x).strip())
    except Exception:
        pass
    quoted = re.findall(r'"([^"]+)"', raw)
    if quoted:
        return dedupe(quoted)
    return dedupe(re.split(r"[\n;,]+", raw))


def strip_xmlish_tags(text: str) -> str:
    return normalize_ws(GENERIC_TAG_RE.sub(" ", str(text or "")))


def _join_lines(parts: Iterable[str]) -> str:
    return "\n".join(str(x).strip() for x in parts if str(x).strip())


def _normalize_label(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ")).strip().lower()


def _humanize_label(text: Any) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = raw.split("/")[-1].replace("__", " / ").replace("_", " ")
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _clean_topic_label(text: Any) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip(" .,:;-\n\t")
    value = re.sub(r"^(direction|topic|opportunity|label|positioning)\s*:\s*", "", value, flags=re.IGNORECASE)
    return value


def _extract_question_candidate_directions(question: str) -> List[str]:
    text = re.sub(r"\s+", " ", str(question or "")).strip()
    lowered = text.lower()
    if not text or "(" not in text or ")" not in text:
        return []
    gating_markers = [
        "rank these",
        "rank the following",
        "candidate research directions",
        "candidate directions",
        "listed directions",
        "do not add new directions",
        "do not introduce new candidate directions",
    ]
    if not any(marker in lowered for marker in gating_markers):
        return []
    matches = list(re.finditer(r"\((\d+|one|two|three|four|five|six)\)\s*", text, flags=re.IGNORECASE))
    if not matches:
        return []
    stop_pattern = re.compile(
        r"(?:;\s*provide\b|\.?\s*provide\b|;\s*do not\b|\.?\s*do not\b|;\s*limit\b|\.?\s*limit\b|;\s*rank only\b|\.?\s*rank only\b)",
        flags=re.IGNORECASE,
    )
    out: List[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end]
        stop = stop_pattern.search(chunk)
        if stop:
            chunk = chunk[: stop.start()]
        chunk = _clean_topic_label(chunk.strip(" ;,."))
        if chunk:
            out.append(chunk)
    return dedupe(out)


def _extract_comparative_candidate_directions(text: Any) -> List[str]:
    raw = re.sub(r"\s+", " ", str(text or "")).strip(" ?.")
    if not raw:
        return []
    patterns = [
        r":\s*([^:]+?)\s+vs\.?\s+([^?]+)$",
        r":\s*([^:]+?)\s+versus\s+([^?]+)$",
        r":\s*([^:]+?)\s+or\s+([^?]+)$",
        r"prioritized[^:]*:\s*([^:]+?)\s+or\s+([^?]+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if not match:
            continue
        left = _clean_topic_label(match.group(1))
        right = _clean_topic_label(match.group(2))
        candidates = [
            value.rstrip(".,;: ")
            for value in [left, right]
            if value and len(_keyword_terms(value, limit=8)) >= 2
        ]
        if len(candidates) >= 2:
            return dedupe(candidates)
    return []


def _task_candidate_directions(task: Dict[str, Any]) -> List[str]:
    contract_candidates = [
        str(x).strip()
        for x in (extract_task_contract(task).get("candidate_directions") or [])
        if str(x).strip()
    ]
    if contract_candidates:
        return dedupe(contract_candidates)
    from_question = _extract_question_candidate_directions(str(task.get("question") or ""))
    if from_question:
        return dedupe(from_question)
    comparative = _extract_comparative_candidate_directions(task.get("question"))
    if comparative:
        return dedupe(comparative)
    return _extract_comparative_candidate_directions(task.get("title"))


def _extract_target_venue(question: str) -> str:
    text = str(question or "")
    patterns = [
        r"venues?\s+(?:such as|similar to)\s+([A-Z][A-Za-z0-9/-]+)",
        r"for\s+([A-Z][A-Za-z0-9/-]+)-style venues",
        r"for\s+top-tier\s+([A-Z][A-Za-z0-9/-]+)\s+venues",
        r"for\s+([A-Z][A-Za-z0-9/-]+)\s+submissions",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().rstrip(".,")
    for venue in ["ICLR", "NeurIPS", "ICML", "ACL", "EMNLP", "NAACL", "AAAI", "IJCAI", "SIGIR", "KDD", "CVPR", "ECCV", "ICCV"]:
        if venue in text:
            return venue
    return ""


def _extract_venue_topic_text(task: Dict[str, Any]) -> str:
    question = normalize_ws(task.get("question") or "")
    title = normalize_ws(task.get("title") or "")
    patterns = [
        r"next-step directions in (.+?) should be prioritized",
        r"which one or two next-step directions in (.+?) should be prioritized",
        r"rank the following candidate directions in (.+?) by",
        r"rank the following directions in (.+?) by",
        r"rank these candidate directions in (.+?) by",
        r"rank these (.+?) research directions based",
        r"identify one concrete next-step direction within (.+?) that is most likely",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            value = _clean_topic_label(match.group(1))
            value = re.sub(r"\s+for\s+[A-Z][A-Za-z0-9/-]+(?:-like)?\s+venues?$", "", value, flags=re.IGNORECASE)
            value = re.sub(r"\s+for\s+[A-Z][A-Za-z0-9/-]+\s+submissions?$", "", value, flags=re.IGNORECASE)
            if value:
                return value
    value = title
    leading_patterns = [
        r"^Venue-Targeted Prioritization of Research Directions in\s+",
        r"^Forecasting Top-Venue Traction in\s+",
        r"^Strategic Prioritization of\s+",
        r"^Prioritizing\s+",
    ]
    for pattern in leading_patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+for\s+[A-Z][A-Za-z0-9/-]+(?:-like)?\s+venues?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+for\s+[A-Z][A-Za-z0-9/-]+\s+Conference\s+Submissions?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+for\s+[A-Z][A-Za-z0-9/-]+\s+Submissions?$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+Directions?$", "", value, flags=re.IGNORECASE)
    return _clean_topic_label(value)


def _extract_bottleneck_focus_text(task: Dict[str, Any]) -> str:
    title = normalize_ws(task.get("title") or "")
    if title:
        patterns = [
            r"^Identifying a Key Bottleneck in\s+",
            r"^Identifying a Critical Bottleneck in\s+",
            r"^Identifying Bottlenecks in\s+",
            r"^Identifying Bottlenecks and Future Opportunities in\s+",
            r"^Bottleneck and Opportunity Discovery in\s+",
            r"^Bottleneck and Opportunity Discovery for\s+",
        ]
        value = title
        for pattern in patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()
        if value and value != title:
            return _clean_topic_label(value)
    return _clean_topic_label(extract_focus_text(task) or title or task.get("question") or "")


_VENUE_COMPATIBLE_BUCKETS: Dict[str, List[str]] = {
    "acl": ["acl", "emnlp", "naacl"],
    "emnlp": ["emnlp", "acl", "naacl"],
    "naacl": ["naacl", "acl", "emnlp"],
    "iclr": ["iclr", "neurips", "icml", "aaai", "ijcai"],
    "neurips": ["neurips", "iclr", "icml", "aaai", "ijcai"],
    "icml": ["icml", "iclr", "neurips", "aaai", "ijcai"],
    "aaai": ["aaai", "ijcai", "iclr", "neurips", "icml"],
    "ijcai": ["ijcai", "aaai", "iclr", "neurips", "icml"],
    "sigir": ["sigir", "kdd"],
    "kdd": ["kdd", "sigir"],
    "cvpr": ["cvpr", "eccv", "iccv"],
    "eccv": ["eccv", "cvpr", "iccv"],
    "iccv": ["iccv", "cvpr", "eccv"],
}


def _target_venue_bucket_from_name(name: Any) -> str:
    norm = _clean_topic_label(name).lower()
    alias_map = {
        "neurips": ["neurips", "neural information processing systems"],
        "iclr": ["iclr"],
        "icml": ["icml"],
        "aaai": ["aaai"],
        "ijcai": ["ijcai"],
        "acl": ["acl"],
        "emnlp": ["emnlp"],
        "naacl": ["naacl"],
        "sigir": ["sigir"],
        "kdd": ["kdd"],
        "cvpr": ["cvpr"],
        "eccv": ["eccv"],
        "iccv": ["iccv"],
    }
    for bucket in _VENUE_COMPATIBLE_BUCKETS:
        if norm == bucket:
            return bucket
        if any(alias == norm or alias in norm for alias in alias_map.get(bucket, [bucket])):
            return bucket
    return ""


def _compatible_venue_buckets(bucket: str) -> List[str]:
    norm = _clean_topic_label(bucket).lower()
    return dedupe(_VENUE_COMPATIBLE_BUCKETS.get(norm, [norm])) if norm else []


def _counter_rows(counter: Counter[str], *, limit: int = 6) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, count in counter.most_common(limit):
        clean_label = normalize_ws(label)
        if not clean_label or count <= 0:
            continue
        rows.append({"label": clean_label, "count": int(count)})
    return rows


def _render_counter_rows(rows: List[Dict[str, Any]], *, limit: int = 4) -> str:
    parts = []
    for row in (rows or [])[:limit]:
        label = normalize_ws((row or {}).get("label") or "")
        count = int((row or {}).get("count") or 0)
        if not label or count <= 0:
            continue
        parts.append(f"{label} x{count}")
    return ", ".join(parts)


def _detect_venue_package_patterns(text: str) -> List[str]:
    lowered = str(text or "").lower()
    patterns = {
        "new_method": ["method", "framework", "architecture", "approach", "model", "algorithm"],
        "dataset_or_benchmark": ["dataset", "benchmark", "corpus", "resource", "leaderboard"],
        "empirical_comparison": ["empirical", "comparison", "baseline", "state-of-the-art", "sota"],
        "analysis_or_diagnosis": ["analysis", "diagnostic", "probing", "error analysis", "failure analysis"],
        "system_or_efficiency": ["system", "pipeline", "efficiency", "latency", "throughput", "runtime"],
        "human_centered": ["human evaluation", "user study", "annotator", "preference", "human judgment"],
    }
    hits = []
    for label, keywords in patterns.items():
        if any(keyword in lowered for keyword in keywords):
            hits.append(label)
    return hits


def _detect_venue_evaluation_patterns(text: str) -> List[str]:
    lowered = str(text or "").lower()
    patterns = {
        "benchmark_eval": ["benchmark", "leaderboard", "baseline", "state-of-the-art", "sota"],
        "ablation_analysis": ["ablation", "analysis", "error analysis", "case study", "probing"],
        "human_eval": ["human evaluation", "user study", "annotator", "manual inspection", "human judgment"],
        "robustness_generalization": ["robust", "generalization", "out-of-domain", "transfer", "cross-domain"],
        "efficiency_measurement": ["efficiency", "latency", "runtime", "throughput", "cost"],
        "resource_release": ["dataset", "corpus", "resource", "release", "benchmark suite"],
    }
    hits = []
    for label, keywords in patterns.items():
        if any(keyword in lowered for keyword in keywords):
            hits.append(label)
    return hits


def _signal_rows(counter: Counter[str], *, limit: int = 6) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, count in counter.most_common(limit):
        clean = _clean_topic_label(label)
        if not clean or count <= 0:
            continue
        rows.append({"label": clean, "count": int(count)})
    return rows


def short_focus_terms(text: str, *, limit: int = 10) -> str:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", str(text or "").lower())
    stop = {
        "based", "before", "after", "using", "identify", "concrete", "research", "literature", "published",
        "long", "large", "model", "models", "agent", "agents", "framework", "frameworks", "technical",
        "unresolved", "subsequent", "period", "response", "grounded", "exclusively", "evidence", "signals",
    }
    picked: List[str] = []
    for token in tokens:
        if token in stop:
            continue
        picked.append(token)
        if len(picked) >= limit:
            break
    return " ".join(picked).strip()


def _cache_key(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _keyword_terms(text: Any, *, limit: int = 48) -> List[str]:
    out: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", str(text or "").lower()):
        if token in COI_KEYWORD_STOPWORDS:
            continue
        if token not in seen:
            out.append(token)
            seen.add(token)
        if len(out) >= limit:
            break
    return out


def _keyword_overlap_score(left: Any, right: Any) -> float:
    left_terms = set(_keyword_terms(left))
    right_terms = set(_keyword_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    overlap = left_terms & right_terms
    if not overlap:
        return 0.0
    precision = len(overlap) / len(right_terms)
    recall = len(overlap) / len(left_terms)
    return 2 * precision * recall / max(precision + recall, 1e-8)


def _normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-9:
        base = 0.5 if hi > 0 else 0.0
        return [base for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _cosine_similarity(left: Optional[List[float]], right: Optional[List[float]]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _get_coi_embed_model():
    global _COI_EMBED_MODEL, _COI_EMBED_MODEL_LOAD_FAILED, _COI_EMBED_MODEL_ERROR
    if _COI_EMBED_MODEL is not None:
        return _COI_EMBED_MODEL
    if _COI_EMBED_MODEL_LOAD_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        _COI_EMBED_MODEL = SentenceTransformer(COI_EMBED_MODEL_NAME)
    except Exception as exc:
        _COI_EMBED_MODEL_LOAD_FAILED = True
        _COI_EMBED_MODEL_ERROR = str(exc)
        _COI_EMBED_MODEL = None
    return _COI_EMBED_MODEL


class CoIAgentOffline:
    def __init__(
        self,
        *,
        kb: OfflineKnowledgeBase,
        main_client: OpenAICompatChatClient,
        cheap_client: Optional[OpenAICompatChatClient] = None,
        embedding_client: Optional[OpenAICompatEmbeddingClient] = None,
        fulltext_cache: Optional[LocalFulltextCache] = None,
        allow_fulltext_fetch: bool = False,
    ):
        self.kb = kb
        self.main_client = main_client
        self.cheap_client = cheap_client or main_client
        self.embedding_client = embedding_client
        self.fulltext_cache = fulltext_cache
        self.allow_fulltext_fetch = allow_fulltext_fetch
        self.max_chain_length = 5
        self.min_chain_length = 3
        self.max_anchor_papers = 4
        self.verbose = os.environ.get("RTL_VERBOSE_COI", "").strip().lower() in {"1", "true", "yes", "y"}
        self.retrieval_adaptor = CoIOfflineRetrievalAdaptor(kb=kb, cheap_client=self.cheap_client, main_client=self.main_client)
        self._paper_content_cache: Dict[Tuple[str, str], str] = {}
        self._paper_profile_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._embedding_text_cache: Dict[str, List[float]] = {}
        self._task_query_cache: Dict[str, List[str]] = {}
        self._entity_summary_cache: Dict[str, str] = {}
        self._cache_lock = threading.RLock()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[CoI-Agent-Offline][debug] {message}", flush=True)

    def _cache_path(self, namespace: str, key: str) -> Path:
        return COI_PROFILE_CACHE_ROOT / namespace / f"{key}.json"

    def _read_cached_json(self, namespace: str, key: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(namespace, key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    def _write_cached_json(self, namespace: str, key: str, payload: Dict[str, Any]) -> None:
        path = self._cache_path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def _read_cached_text(self, namespace: str, key: str) -> str:
        payload = self._read_cached_json(namespace, key)
        if isinstance(payload, dict):
            return str(payload.get("text") or "").strip()
        return ""

    def _write_cached_text(self, namespace: str, key: str, text: str, **meta: Any) -> None:
        payload: Dict[str, Any] = {"text": str(text or "")}
        payload.update(meta)
        self._write_cached_json(namespace, key, payload)

    def _embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        cleaned = [clip_text(normalize_ws(text), COI_EMBED_TEXT_LIMIT) for text in texts]
        if self.embedding_client is not None:
            with self._cache_lock:
                missing_texts = [text for text in cleaned if text and text not in self._embedding_text_cache]
            if missing_texts:
                try:
                    embeddings = self.embedding_client.embed(missing_texts, transport_retries=2, retry_delay=1.5)
                    with self._cache_lock:
                        for text, vector in zip(missing_texts, embeddings):
                            self._embedding_text_cache[text] = [float(x) for x in vector]
                except Exception as exc:
                    self._log(f"embed_service_failed model={self.embedding_client.config.model_name} error={exc}")
            with self._cache_lock:
                cached_vectors = [self._embedding_text_cache.get(text) for text in cleaned]
            if any(vector is not None for vector in cached_vectors):
                return cached_vectors
        model = _get_coi_embed_model()
        if model is None:
            if _COI_EMBED_MODEL_ERROR:
                self._log(f"embed_model_unavailable model={COI_EMBED_MODEL_NAME} error={_COI_EMBED_MODEL_ERROR}")
            return [None for _ in cleaned]
        missing_texts: List[str] = []
        missing_indices: List[int] = []
        with self._cache_lock:
            for idx, text in enumerate(cleaned):
                if text and text not in self._embedding_text_cache:
                    missing_texts.append(text)
                    missing_indices.append(idx)
        if missing_texts:
            try:
                embeddings = model.encode(missing_texts, normalize_embeddings=True)
                with self._cache_lock:
                    for text, vector in zip(missing_texts, embeddings):
                        self._embedding_text_cache[text] = [float(x) for x in vector]
            except Exception as exc:
                self._log(f"embed_encode_failed model={COI_EMBED_MODEL_NAME} error={exc}")
                with self._cache_lock:
                    return [self._embedding_text_cache.get(text) for text in cleaned]
        with self._cache_lock:
            return [self._embedding_text_cache.get(text) for text in cleaned]

    def _complete(self, client: OpenAICompatChatClient, prompt: str, *, max_tokens: int = 1400, temperature: float = 0.2) -> str:
        return client.complete_text(
            [
                {"role": "system", "content": "You are a precise research agent. Follow the requested format exactly."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=180,
            transport_retries=2,
        )

    def _contract(self, task: Dict[str, Any]) -> Dict[str, Any]:
        contract = extract_task_contract(task)
        contract["topic_text"] = str(contract.get("topic_text") or extract_focus_text(task) or "").strip()
        return contract

    def _chain_candidate_labels(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Dict[str, Any],
    ) -> Dict[str, List[str]]:
        def _clean_label(raw: Any) -> str:
            value = normalize_ws(raw or "")
            if not value:
                return ""
            if len(value) > 160:
                return ""
            if value.count("\n") > 0:
                return ""
            lower = value.lower()
            noisy_markers = (
                "{background:",
                "background:",
                "novelty:",
                "contribution:",
                "methods:",
                "detail reason:",
                "limitation:",
            )
            if any(marker in lower for marker in noisy_markers):
                return ""
            if value.count(":") >= 4:
                return ""
            return value

        packet_labels: List[str] = []
        descendant_labels: List[str] = []
        contract_candidates = _task_candidate_directions(task)
        if candidate_pool is not None:
            for packet in candidate_pool.packets:
                packet_labels.append(_clean_label(_humanize_label(packet.get("display_name") or packet.get("node_id") or "")))
                for row in (packet.get("emergent_descendants") or [])[:10]:
                    descendant_labels.append(_clean_label(_humanize_label(row.get("display_name") or row.get("node_id") or "")))
        profile_titles = [_clean_label(_humanize_label((row or {}).get("title") or "")) for row in (chain.get("profiles") or [])]
        profile_ideas = [_clean_label((row or {}).get("idea") or "") for row in (chain.get("profiles") or [])]
        family = str(task.get("family") or "")
        planning_candidates = dedupe(
            [
                *[_clean_label(x) for x in contract_candidates],
                *descendant_labels,
                *packet_labels,
                *profile_titles,
                *profile_ideas,
            ]
        )[:18]
        return {
            "packet_labels": [x for x in dedupe(packet_labels) if x][:8],
            "descendant_labels": [x for x in dedupe(descendant_labels) if x][:12],
            "planning_candidates": [x for x in planning_candidates if x][:18],
            "family": [family],
        }

    def _canonicalize_text(self, text: str, pool: Iterable[str]) -> str:
        value = str(text or "").strip()
        if not value:
            return ""
        norm = _normalize_label(value)
        best = value
        value_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", norm))
        scored_best: Tuple[float, int, str] = (0.0, 10**9, value)
        for candidate in pool:
            cand = str(candidate or "").strip()
            if not cand:
                continue
            if len(cand) > 160 or cand.count(":") >= 4 or "{background:" in cand.lower():
                continue
            cand_norm = _normalize_label(cand)
            if norm == cand_norm or norm in cand_norm or cand_norm in norm:
                best = cand
                break
            cand_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", cand_norm))
            if not value_tokens or not cand_tokens:
                continue
            overlap = value_tokens & cand_tokens
            if len(overlap) < 2:
                continue
            precision = len(overlap) / max(1, len(value_tokens))
            recall = len(overlap) / max(1, len(cand_tokens))
            score = max(precision, recall)
            if score >= 0.6:
                candidate_rank = (score, len(cand), cand)
                if candidate_rank[0] > scored_best[0] or (candidate_rank[0] == scored_best[0] and candidate_rank[1] < scored_best[1]):
                    scored_best = candidate_rank
        else:
            if scored_best[0] >= 0.6:
                best = scored_best[2]
        return best

    def _bottleneck_candidate_labels(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        labels: List[str] = []
        base = self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain or {})
        labels.extend(base.get("descendant_labels") or [])
        if candidate_pool is not None:
            for packet in candidate_pool.packets[:4]:
                for row in (packet.get("top_limitations") or [])[:8]:
                    labels.append(str((row or {}).get("name") or ""))
                for row in (packet.get("top_future_work") or [])[:8]:
                    labels.append(str((row or {}).get("direction") or (row or {}).get("name") or ""))
                labels.append(_humanize_label(packet.get("display_name") or packet.get("node_id") or ""))
        labels.extend(base.get("planning_candidates") or [])
        return [x for x in dedupe(labels) if normalize_ws(x)][:24]

    def _bottleneck_label_state(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chain = chain or {}
        bottleneck_labels: List[str] = []
        opportunity_labels: List[str] = []
        pair_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        title_like_blocks: List[str] = []

        def _add_title_like(text: Any) -> None:
            value = normalize_ws(_humanize_label(text or ""))
            if not value:
                return
            if len(_keyword_terms(value, limit=24)) < 3:
                return
            title_like_blocks.append(value)

        def _add_label(target: List[str], text: Any, *, reject_title_like: bool = False) -> None:
            value = _clean_topic_label(text)
            if not normalize_ws(value):
                return
            if len(value) > 140 or value.count(":") >= 3:
                return
            if len(_keyword_terms(value, limit=16)) < 2:
                return
            if reject_title_like and self._is_title_like_bottleneck_opportunity_label(value, title_like_blocks):
                return
            if target is bottleneck_labels and self._is_focus_level_bottleneck_label(task, value):
                return
            if target is opportunity_labels and self._is_focus_level_opportunity_label(task, value):
                return
            target.append(value)

        def _add_pair(bottleneck: Any, opportunity: Any, paper_id: Any = None) -> None:
            left = _clean_topic_label(bottleneck)
            right = _clean_topic_label(opportunity)
            if not normalize_ws(left) or not normalize_ws(right):
                return
            if self._is_title_like_bottleneck_opportunity_label(left, title_like_blocks):
                return
            if self._is_title_like_bottleneck_opportunity_label(right, title_like_blocks):
                return
            if self._is_focus_level_bottleneck_label(task, left):
                return
            if self._is_focus_level_opportunity_label(task, right):
                return
            if self._labels_too_similar(left, right):
                return
            key = (left, right)
            meta = pair_meta.setdefault(
                key,
                {
                    "bottleneck": left,
                    "opportunity": right,
                    "support_count": 0,
                    "paper_ids": [],
                },
            )
            meta["support_count"] += 1
            if paper_id and str(paper_id).strip():
                meta["paper_ids"] = dedupe([*meta["paper_ids"], str(paper_id).strip()])

        base = self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain)
        if candidate_pool is not None:
            for paper in candidate_pool.papers[:32]:
                _add_title_like((paper or {}).get("title") or "")
        for profile in (chain.get("profiles") or [])[:12]:
            _add_title_like((profile or {}).get("title") or "")
        for label in base.get("descendant_labels") or []:
            _add_label(opportunity_labels, label, reject_title_like=True)

        if candidate_pool is not None:
            for packet in candidate_pool.packets[:4]:
                for row in (packet.get("top_limitations") or [])[:10]:
                    _add_label(bottleneck_labels, (row or {}).get("name") or row or "", reject_title_like=True)
                for row in (packet.get("top_future_work") or [])[:10]:
                    _add_label(
                        opportunity_labels,
                        (row or {}).get("direction") or (row or {}).get("name") or row or "",
                        reject_title_like=True,
                    )
                for row in (packet.get("emergent_descendants") or [])[:10]:
                    _add_label(
                        opportunity_labels,
                        _humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or ""),
                        reject_title_like=True,
                    )

        signals = (chain.get("bottleneck_signals") or {}) if isinstance(chain, dict) else {}
        for row in (signals.get("recurring_limitations") or [])[:10]:
            _add_label(bottleneck_labels, (row or {}).get("label") or (row or {}).get("name") or "", reject_title_like=True)
        for row in (signals.get("recurring_problem_statements") or [])[:6]:
            _add_label(bottleneck_labels, (row or {}).get("label") or (row or {}).get("name") or "", reject_title_like=True)
        for row in (signals.get("future_work_signals") or [])[:10]:
            _add_label(opportunity_labels, (row or {}).get("label") or (row or {}).get("name") or "", reject_title_like=True)

        for profile in (chain.get("profiles") or [])[:8]:
            paper_id = profile.get("paper_id")
            limitations = [
                _clean_topic_label(x)
                for x in (profile.get("structure_limitations") or [])
                if _clean_topic_label(x)
            ]
            future_work = [
                _clean_topic_label(x)
                for x in (profile.get("structure_future_work") or [])
                if _clean_topic_label(x)
            ]
            problem_statement = _clean_topic_label(clip_text(profile.get("problem_statement") or "", 220))
            for label in limitations[:4]:
                _add_label(bottleneck_labels, label, reject_title_like=True)
            if problem_statement:
                _add_label(bottleneck_labels, problem_statement, reject_title_like=True)
            for label in future_work[:4]:
                _add_label(opportunity_labels, label, reject_title_like=True)
            for left in limitations[:3]:
                for right in future_work[:3]:
                    _add_pair(left, right, paper_id)
            if problem_statement:
                for right in future_work[:3]:
                    _add_pair(problem_statement, right, paper_id)

        for label in base.get("planning_candidates") or []:
            text = _clean_topic_label(label)
            if not text:
                continue
            lower = text.lower()
            if any(token in lower for token in ["limitation", "failure", "challenge", "constraint", "bottleneck", "conflict", "error", "gap"]):
                _add_label(bottleneck_labels, text, reject_title_like=True)
            elif any(token in lower for token in ["reasoning", "planning", "training", "retrieval", "evaluation", "alignment", "verification", "monitoring"]):
                _add_label(opportunity_labels, text, reject_title_like=True)

        title_like_blocks = [x for x in dedupe(title_like_blocks) if normalize_ws(x)][:48]
        bottleneck_labels = [
            x
            for x in dedupe(bottleneck_labels)
            if normalize_ws(x)
            and not self._is_title_like_bottleneck_opportunity_label(x, title_like_blocks)
            and not self._is_focus_level_bottleneck_label(task, x)
        ][:24]
        opportunity_labels = [
            x
            for x in dedupe(opportunity_labels)
            if normalize_ws(x)
            and not self._is_title_like_bottleneck_opportunity_label(x, title_like_blocks)
            and not self._is_focus_level_opportunity_label(task, x)
            and not any(self._labels_too_similar(x, y) for y in bottleneck_labels[:12])
        ][:24]
        paired_unlocks = sorted(
            pair_meta.values(),
            key=lambda row: (
                -self._score_bottleneck_pair(
                    task=task,
                    bottleneck=(row or {}).get("bottleneck") or "",
                    opportunity=(row or {}).get("opportunity") or "",
                    support_count=(row or {}).get("support_count") or 0,
                ),
                -int(row.get("support_count") or 0),
            ),
        )[:12]
        return {
            "bottleneck_candidates": bottleneck_labels,
            "opportunity_candidates": opportunity_labels,
            "paired_unlocks": paired_unlocks,
            "title_like_blocks": title_like_blocks,
        }

    def _merge_bottleneck_label_states(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        bottleneck_labels: List[str] = []
        opportunity_labels: List[str] = []
        title_like_blocks: List[str] = []
        pair_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for state in states:
            for label in state.get("bottleneck_candidates") or []:
                bottleneck_labels.append(str(label))
            for label in state.get("opportunity_candidates") or []:
                opportunity_labels.append(str(label))
            for label in state.get("title_like_blocks") or []:
                title_like_blocks.append(str(label))
            for row in state.get("paired_unlocks") or []:
                left = _clean_topic_label((row or {}).get("bottleneck") or "")
                right = _clean_topic_label((row or {}).get("opportunity") or "")
                if not left or not right:
                    continue
                key = (left, right)
                meta = pair_meta.setdefault(
                    key,
                    {
                        "bottleneck": left,
                        "opportunity": right,
                        "support_count": 0,
                        "paper_ids": [],
                    },
                )
                meta["support_count"] += int((row or {}).get("support_count") or 0)
                meta["paper_ids"] = dedupe([*meta["paper_ids"], *[str(x) for x in ((row or {}).get("paper_ids") or []) if str(x).strip()]])
        title_like_blocks = [x for x in dedupe(title_like_blocks) if normalize_ws(x)][:64]
        bottleneck_labels = [
            x
            for x in dedupe(bottleneck_labels)
            if normalize_ws(x) and not self._is_title_like_bottleneck_opportunity_label(x, title_like_blocks)
        ][:24]
        opportunity_labels = [
            x
            for x in dedupe(opportunity_labels)
            if normalize_ws(x)
            and not self._is_title_like_bottleneck_opportunity_label(x, title_like_blocks)
            and not any(self._labels_too_similar(x, y) for y in bottleneck_labels[:12])
        ][:24]
        paired_unlocks = sorted(
            (
                row
                for row in pair_meta.values()
                if not self._is_title_like_bottleneck_opportunity_label((row or {}).get("bottleneck") or "", title_like_blocks)
                and not self._is_title_like_bottleneck_opportunity_label((row or {}).get("opportunity") or "", title_like_blocks)
            ),
            key=lambda row: (
                -int(row.get("support_count") or 0),
                len(str(row.get("bottleneck") or "")) + len(str(row.get("opportunity") or "")),
            ),
        )[:12]
        return {
            "bottleneck_candidates": bottleneck_labels,
            "opportunity_candidates": opportunity_labels,
            "paired_unlocks": paired_unlocks,
            "title_like_blocks": title_like_blocks,
        }

    def _aggregate_bottleneck_label_state(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        states = [
            self._bottleneck_label_state(
                task=task,
                candidate_pool=candidate_pool,
                chain=run.get("chain") or {},
            )
            for run in candidate_runs[: max(4, self._candidate_anchor_count(task))]
        ]
        if candidate_pool is not None:
            states.append(self._bottleneck_label_state(task=task, candidate_pool=candidate_pool, chain={}))
        return self._merge_bottleneck_label_states(states)

    def _is_focus_level_bottleneck_label(self, task: Dict[str, Any], text: Any) -> bool:
        value = normalize_ws(text or "")
        focus = normalize_ws(_extract_bottleneck_focus_text(task))
        if not value or not focus:
            return False
        value_norm = _normalize_label(value)
        focus_norm = _normalize_label(focus)
        value_terms = set(_keyword_terms(value, limit=16))
        focus_terms = set(_keyword_terms(focus, limit=16))
        if len(value_terms) < 2 or len(focus_terms) < 2:
            return False
        overlap = len(value_terms & focus_terms) / max(1, min(len(value_terms), len(focus_terms)))
        exactish = value_norm == focus_norm or value_norm in focus_norm or focus_norm in value_norm
        bottleneck_markers = {
            "lack", "lacks", "lacking", "inability", "unable", "failure", "failures", "constraint", "constraints",
            "gap", "gaps", "bottleneck", "tradeoff", "tradeoffs", "drift", "instability", "inconsistency",
            "latency", "cost", "noise", "sparsity", "coverage", "observability", "verification", "credit",
        }
        if any(marker in value_norm for marker in bottleneck_markers):
            return False
        return exactish or (overlap >= 0.9 and len(value_terms) <= len(focus_terms) + 1)

    def _is_focus_level_opportunity_label(self, task: Dict[str, Any], text: Any) -> bool:
        value = normalize_ws(text or "")
        focus = normalize_ws(_extract_bottleneck_focus_text(task))
        if not value or not focus:
            return False
        value_norm = _normalize_label(value)
        focus_norm = _normalize_label(focus)
        value_terms = set(_keyword_terms(value, limit=16))
        focus_terms = set(_keyword_terms(focus, limit=16))
        if len(value_terms) < 2 or len(focus_terms) < 2:
            return False
        overlap = len(value_terms & focus_terms) / max(1, min(len(value_terms), len(focus_terms)))
        exactish = value_norm == focus_norm or value_norm in focus_norm or focus_norm in value_norm
        opportunity_markers = {
            "adaptive", "verification", "monitoring", "alignment", "retrieval", "planning", "training",
            "evaluation", "reconsolidation", "tracking", "augmentation", "benchmark", "orchestration",
            "feedback", "memory", "control", "grounding", "state", "reasoning",
        }
        has_new_directional_signal = bool(value_terms - focus_terms) and any(marker in value_norm for marker in opportunity_markers)
        if has_new_directional_signal:
            return False
        return exactish or (overlap >= 0.92 and len(value_terms) <= len(focus_terms) + 1)

    def _best_matching_contract_candidate(self, task: Dict[str, Any], text: Any) -> str:
        value = normalize_ws(text or "")
        if not value:
            return ""
        candidates = [normalize_ws(x) for x in (_task_candidate_directions(task) or []) if normalize_ws(x)]
        if not candidates:
            return ""
        best = ""
        best_score = 0.0
        for candidate in candidates:
            score = _keyword_overlap_score(value, candidate)
            if _normalize_label(value) == _normalize_label(candidate):
                return candidate
            if score > best_score:
                best_score = score
                best = candidate
        return best if best_score >= 0.45 else ""

    def _score_bottleneck_pair(
        self,
        *,
        task: Dict[str, Any],
        bottleneck: Any,
        opportunity: Any,
        support_count: Any = None,
    ) -> float:
        left = normalize_ws(bottleneck or "")
        right = normalize_ws(opportunity or "")
        if not left or not right:
            return -999.0
        score = float(support_count or 0) * 2.5
        if self._is_focus_level_bottleneck_label(task, left):
            score -= 2.5
        else:
            score += 0.8
        if self._is_focus_level_opportunity_label(task, right):
            score -= 2.0
        else:
            score += 0.8
        left_norm = _normalize_label(left)
        if any(marker in left_norm for marker in ["lack", "inability", "failure", "constraint", "gap", "bottleneck", "drift", "instability", "conflict", "observability"]):
            score += 0.7
        right_norm = _normalize_label(right)
        if any(marker in right_norm for marker in ["adaptive", "verification", "monitoring", "alignment", "retrieval", "planning", "training", "evaluation", "augmentation", "benchmark", "tracking", "orchestration"]):
            score += 0.4
        overlap = _keyword_overlap_score(left, right)
        if 0.08 <= overlap <= 0.65:
            score += 0.25
        if self._labels_too_similar(left, right):
            score -= 2.5
        score += min(len(_keyword_terms(left, limit=16)), 6) * 0.04
        score += min(len(_keyword_terms(right, limit=16)), 6) * 0.04
        return score

    def _labels_too_similar(self, left: Any, right: Any) -> bool:
        left_text = normalize_ws(left or "")
        right_text = normalize_ws(right or "")
        if not left_text or not right_text:
            return False
        if _normalize_label(left_text) == _normalize_label(right_text):
            return True
        overlap = _keyword_overlap_score(left_text, right_text)
        if overlap >= 0.82:
            return True
        left_terms = set(_keyword_terms(left_text, limit=16))
        right_terms = set(_keyword_terms(right_text, limit=16))
        if left_terms and right_terms and (left_terms <= right_terms or right_terms <= left_terms):
            return True
        return False

    def _is_title_like_bottleneck_opportunity_label(self, text: Any, title_pool: Iterable[str]) -> bool:
        value = normalize_ws(text or "")
        if not value:
            return False
        lowered = value.lower()
        if lowered.startswith("paper ") or lowered.startswith("title:"):
            return True
        if value.count(":") >= 1 and len(_keyword_terms(value, limit=24)) >= 5:
            return True
        for title in title_pool:
            title_text = normalize_ws(title or "")
            if not title_text:
                continue
            if _normalize_label(value) == _normalize_label(title_text):
                return True
            if _keyword_overlap_score(value, title_text) >= 0.86:
                return True
        return False

    def _flatten_bottleneck_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        flat = dict(payload or {})
        nested_candidates: List[Dict[str, Any]] = []
        for key in ["primary_direction", "primary_item", "selected_item", "answer", "result"]:
            value = flat.get(key)
            if isinstance(value, dict):
                nested_candidates.append(value)
        items = flat.get("items")
        if isinstance(items, list) and items:
            first = items[0]
            if isinstance(first, dict):
                nested_candidates.append(first)
        nested = next(
            (
                candidate
                for candidate in nested_candidates
                if isinstance(candidate, dict)
                and any(normalize_ws(candidate.get(k) or "") for k in ["bottleneck", "opportunity", "linkage"])
            ),
            None,
        )
        if nested:
            for key in ["bottleneck", "opportunity", "linkage", "evidence_ids"]:
                if key not in flat or not flat.get(key):
                    flat[key] = nested.get(key)
        return flat

    def _repair_bottleneck_payload(
        self,
        *,
        task: Dict[str, Any],
        payload: Dict[str, Any],
        label_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        repaired = self._flatten_bottleneck_payload(payload or {})
        bottleneck_pool = list(label_state.get("bottleneck_candidates") or [])
        opportunity_pool = list(label_state.get("opportunity_candidates") or [])
        pair_rows = list(label_state.get("paired_unlocks") or [])
        title_pool = list(label_state.get("title_like_blocks") or [])
        contract_match = self._best_matching_contract_candidate(task, repaired.get("opportunity") or "")

        bottleneck = self._canonicalize_text(str(repaired.get("bottleneck") or ""), bottleneck_pool)
        opportunity = self._canonicalize_text(str(repaired.get("opportunity") or ""), opportunity_pool)
        if contract_match:
            opportunity = contract_match

        if not bottleneck and pair_rows:
            bottleneck = str(pair_rows[0].get("bottleneck") or "")
        if not bottleneck and bottleneck_pool:
            bottleneck = bottleneck_pool[0]
        if bottleneck and self._is_focus_level_bottleneck_label(task, bottleneck):
            paired_match = next(
                (
                    row
                    for row in pair_rows
                    if not self._is_focus_level_bottleneck_label(task, (row or {}).get("bottleneck") or "")
                ),
                None,
            )
            if paired_match:
                bottleneck = str((paired_match or {}).get("bottleneck") or "")
        if bottleneck and self._is_title_like_bottleneck_opportunity_label(bottleneck, title_pool):
            paired_match = next(
                (
                    row
                    for row in pair_rows
                    if not self._is_title_like_bottleneck_opportunity_label((row or {}).get("bottleneck") or "", title_pool)
                ),
                None,
            )
            if paired_match:
                bottleneck = str((paired_match or {}).get("bottleneck") or "")
            else:
                alternative = next(
                    (
                        label
                        for label in bottleneck_pool
                        if not self._is_title_like_bottleneck_opportunity_label(label, title_pool)
                    ),
                    "",
                )
                if alternative:
                    bottleneck = alternative
        if not opportunity:
            paired_match = next(
                (
                    row
                    for row in pair_rows
                    if not bottleneck
                    or self._canonicalize_text(str((row or {}).get("bottleneck") or ""), [bottleneck]) == bottleneck
                    or _keyword_overlap_score((row or {}).get("bottleneck") or "", bottleneck) >= 0.55
                ),
                None,
            )
            if paired_match:
                opportunity = str((paired_match or {}).get("opportunity") or "")
        if not opportunity and opportunity_pool:
            opportunity = opportunity_pool[0]
        if not opportunity:
            opportunity = self._best_matching_contract_candidate(task, bottleneck) or opportunity
        if opportunity and self._is_focus_level_opportunity_label(task, opportunity):
            paired_match = next(
                (
                    row
                    for row in pair_rows
                    if not self._is_focus_level_opportunity_label(task, (row or {}).get("opportunity") or "")
                    and (
                        not bottleneck
                        or self._canonicalize_text(str((row or {}).get("bottleneck") or ""), [bottleneck]) == bottleneck
                        or _keyword_overlap_score((row or {}).get("bottleneck") or "", bottleneck) >= 0.55
                    )
                ),
                None,
            )
            if paired_match:
                opportunity = str((paired_match or {}).get("opportunity") or "")
            else:
                alternative = next(
                    (
                        label
                        for label in opportunity_pool
                        if not self._is_focus_level_opportunity_label(task, label)
                        and not self._is_title_like_bottleneck_opportunity_label(label, title_pool)
                        and not self._labels_too_similar(bottleneck, label)
                    ),
                    "",
                )
                if alternative:
                    opportunity = alternative
        contract_match = self._best_matching_contract_candidate(task, opportunity)
        if contract_match:
            opportunity = contract_match

        if opportunity and self._is_title_like_bottleneck_opportunity_label(opportunity, title_pool):
            paired_match = next(
                (
                    row
                    for row in pair_rows
                    if (
                        not bottleneck
                        or self._canonicalize_text(str((row or {}).get("bottleneck") or ""), [bottleneck]) == bottleneck
                        or _keyword_overlap_score((row or {}).get("bottleneck") or "", bottleneck) >= 0.55
                    )
                    and not self._is_title_like_bottleneck_opportunity_label((row or {}).get("opportunity") or "", title_pool)
                ),
                None,
            )
            if paired_match:
                opportunity = str((paired_match or {}).get("opportunity") or "")
            else:
                alternative = next(
                    (
                        label
                        for label in opportunity_pool
                        if not self._is_title_like_bottleneck_opportunity_label(label, title_pool)
                        and not self._labels_too_similar(bottleneck, label)
                    ),
                    "",
                )
                if alternative:
                    opportunity = alternative
        contract_match = self._best_matching_contract_candidate(task, opportunity)
        if contract_match:
            opportunity = contract_match

        if self._labels_too_similar(bottleneck, opportunity):
            paired_match = next(
                (
                    row
                    for row in pair_rows
                    if (
                        not bottleneck
                        or self._canonicalize_text(str((row or {}).get("bottleneck") or ""), [bottleneck]) == bottleneck
                        or _keyword_overlap_score((row or {}).get("bottleneck") or "", bottleneck) >= 0.55
                    )
                    and not self._labels_too_similar(bottleneck, (row or {}).get("opportunity") or "")
                ),
                None,
            )
            if paired_match:
                opportunity = str((paired_match or {}).get("opportunity") or "")
            else:
                alternative = next(
                    (label for label in opportunity_pool if not self._labels_too_similar(bottleneck, label)),
                    "",
                )
                if alternative:
                    opportunity = alternative

        linkage = normalize_ws(repaired.get("linkage") or "")
        if linkage and (":" in linkage and len(_keyword_terms(linkage, limit=24)) >= 8):
            linkage = re.sub(r"\s+", " ", linkage)
        if bottleneck and opportunity and not linkage:
            linkage = f"Resolving {bottleneck} would make {opportunity} a viable near-term technical direction."

        repaired["bottleneck"] = bottleneck
        repaired["opportunity"] = opportunity
        repaired["linkage"] = linkage
        return repaired

    def _align_bottleneck_payload_to_head_analysis(
        self,
        *,
        task: Dict[str, Any],
        payload: Dict[str, Any],
        label_state: Dict[str, Any],
        head_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        aligned = self._repair_bottleneck_payload(task=task, payload=payload, label_state=label_state)
        title_pool = list(label_state.get("title_like_blocks") or [])
        selected_bottleneck = _clean_topic_label(head_analysis.get("selected_bottleneck") or "")
        selected_opportunity = _clean_topic_label(head_analysis.get("selected_opportunity") or "")
        selected_contract_match = self._best_matching_contract_candidate(task, selected_opportunity)
        if selected_contract_match:
            selected_opportunity = selected_contract_match
        if (
            selected_bottleneck
            and not self._is_title_like_bottleneck_opportunity_label(selected_bottleneck, title_pool)
            and not self._is_focus_level_bottleneck_label(task, selected_bottleneck)
        ):
            aligned["bottleneck"] = selected_bottleneck
        if (
            selected_opportunity
            and not self._is_title_like_bottleneck_opportunity_label(selected_opportunity, title_pool)
            and not self._is_focus_level_opportunity_label(task, selected_opportunity)
            and not self._labels_too_similar(aligned.get("bottleneck") or "", selected_opportunity)
        ):
            aligned["opportunity"] = selected_opportunity
        if not normalize_ws(aligned.get("linkage") or ""):
            aligned["linkage"] = normalize_ws(head_analysis.get("linkage") or "")
        return aligned

    def _forecast_candidate_labels(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        labels: List[str] = []
        labels.extend(_task_candidate_directions(task))
        base = self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain or {})
        labels.extend(base.get("descendant_labels") or [])
        labels.extend(base.get("packet_labels") or [])
        if candidate_pool is not None:
            for packet in candidate_pool.packets[:4]:
                labels.append(_humanize_label(packet.get("display_name") or packet.get("node_id") or ""))
                for row in (packet.get("emergent_descendants") or [])[:10]:
                    labels.append(_humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or ""))
                for row in (packet.get("top_future_work") or [])[:10]:
                    labels.append(str((row or {}).get("direction") or (row or {}).get("name") or ""))
        for row in (chain or {}).get("profiles") or []:
            idea = normalize_ws((row or {}).get("idea") or "")
            if not idea:
                continue
            if len(idea) > 96 or "." in idea or ":" in idea:
                continue
            if len(_keyword_terms(idea, limit=24)) > 10:
                continue
            labels.append(idea)
        filtered: List[str] = []
        for label in dedupe(labels):
            text = normalize_ws(label or "")
            if not text:
                continue
            if len(text) > 120:
                continue
            if text.count(":") >= 3:
                continue
            lower = text.lower()
            if "{background:" in lower:
                continue
            if lower.startswith("paper ") or lower.startswith("title:"):
                continue
            filtered.append(text)
        return filtered[:24]

    def _aggregate_forecast_labels(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> List[str]:
        labels: List[str] = []
        labels.extend(_task_candidate_directions(task))
        for run in candidate_runs[: max(4, self._candidate_anchor_count(task))]:
            labels.extend(
                self._forecast_candidate_labels(
                    task=task,
                    candidate_pool=candidate_pool,
                    chain=run.get("chain") or {},
                )
            )
        if candidate_pool is not None:
            labels.extend(self._forecast_candidate_labels(task=task, candidate_pool=candidate_pool, chain={}))
        return [x for x in dedupe(labels) if normalize_ws(x)][:24]

    def _strategic_candidate_labels(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        base = self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain or {})
        contract_candidates = _task_candidate_directions(task)
        return [x for x in dedupe(contract_candidates or base.get("planning_candidates") or []) if normalize_ws(x)][:24]

    # Stable family path: forecast logic is now intentionally isolated so later
    # bottleneck / venue iterations do not need to touch this candidate shaping.
    def _project_direction_structured_answer(
        self,
        *,
        task: Dict[str, Any],
        contract_candidates: List[str],
        forecast_labels: List[str],
    ) -> Tuple[Dict[str, Any], str, List[str]]:
        schema_hint = {
            "trajectory_label": "accelerating",
            "primary_direction": "...",
            "secondary_directions": ["..."],
            "supporting_signals": [
                {"signal": "...", "evidence_ids": ["P1", "F1"]}
            ],
            "counter_signals": [
                {"signal": "...", "impact": "...", "evidence_ids": ["T2"]}
            ],
            "rationale": "...",
            "calibration": "...",
            "evidence_ids": ["P1"],
        }
        family_rule = (
            "Return exactly one trajectory_label from accelerating, fragmenting, steady, cooling, and exactly one primary_direction. "
            "You may include up to 2 secondary_directions, but they must be explicitly weaker or more contingent than the primary_direction. "
            "supporting_signals must name 2-3 concrete pre-cutoff signals, frictions, or momentum cues with evidence ids. "
            "calibration must explain why the chosen direction is the mainline near-term expectation rather than a certainty."
        )
        family_rule += (
            " Prefer descendant-style or future-work-style candidate labels over paper titles, benchmark names, or full method titles. "
            "If candidate labels are available, primary_direction should be a reusable technical direction label rather than a specific paper title."
        )
        candidate_block = contract_candidates or forecast_labels
        return schema_hint, family_rule, candidate_block

    # Stable family path: strategic logic is separated for the same reason as
    # forecasting. Future bottleneck / venue adaptation should not edit this block.
    def _project_strategic_structured_answer(
        self,
        *,
        task: Dict[str, Any],
        contract: Dict[str, Any],
        contract_candidates: List[str],
        strategic_labels: List[str],
    ) -> Tuple[Dict[str, Any], str, List[str]]:
        schema_hint = {
            "ranked_directions": [
                {"rank": 1, "direction": "...", "rationale": "...", "evidence_ids": ["P1", "F2"]}
            ],
            "first_milestone": "...",
            "dependency_chain": "...",
            "defer_rationale": "...",
            "risk_or_kill_criterion": "...",
        }
        family_rule = f"Return up to {int(contract.get('max_items') or 3)} ranked directions. Include one first_milestone, one dependency_chain, one defer_rationale, and one risk_or_kill_criterion."
        if contract_candidates:
            family_rule += f" Use only these candidate directions: {json.dumps(contract_candidates, ensure_ascii=False)}."
        candidate_block = contract_candidates or strategic_labels
        return schema_hint, family_rule, candidate_block

    def _project_structured_answer(
        self,
        *,
        task: Dict[str, Any],
        chain_text: str,
        trend: str,
        future: str,
        human: str,
        evidence: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Dict[str, Any],
    ) -> Dict[str, Any]:
        contract = self._contract(task)
        family = str(task.get("family") or "")
        labels = self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain)
        contract_candidates = _task_candidate_directions(task)
        forecast_labels = self._forecast_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain)
        strategic_labels = self._strategic_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain)
        bottleneck_state = self._bottleneck_label_state(task=task, candidate_pool=candidate_pool, chain=chain)
        target_venue = _extract_target_venue(str(task.get("question") or ""))
        evidence_refs = []
        for group in ["papers", "fulltext"]:
            for row in (evidence.get(group) or [])[:5]:
                evidence_refs.append(
                    {
                        "evidence_id": row.get("evidence_id"),
                        "paper_id": row.get("paper_id"),
                        "title": row.get("title"),
                    }
                )
        if family == "strategic_research_planning":
            schema_hint, family_rule, candidate_block = self._project_strategic_structured_answer(
                task=task,
                contract=contract,
                contract_candidates=contract_candidates,
                strategic_labels=strategic_labels,
            )
        elif family == "venue_aware_research_positioning":
            schema_hint = {
                "ranked_directions": [
                    {"rank": 1, "direction": "...", "venue_fit": "...", "evidence_ids": ["P1", "F2"]}
                ],
                "primary_positioning": "...",
                "target_venue": target_venue or "...",
                "contribution_package": "...",
                "venue_fit_rationale": "...",
                "evaluation_signature": "...",
                "nearby_but_wrong_positioning": "...",
                "evidence_ids": ["P1", "F2"],
            }
            family_rule = "Return one venue-facing positioning judgment with an explicit contribution package, venue-fit rationale, evaluation signature, and one nearby but weaker alternative."
            if target_venue:
                family_rule += f" Explicitly name {target_venue} in the reasoning."
            if contract_candidates:
                family_rule += f" If the task lists candidate directions, rank only these directions: {json.dumps(contract_candidates, ensure_ascii=False)}."
            candidate_block = contract_candidates or labels["planning_candidates"]
        elif family == "direction_forecasting":
            schema_hint, family_rule, candidate_block = self._project_direction_structured_answer(
                task=task,
                contract_candidates=contract_candidates,
                forecast_labels=forecast_labels or labels["descendant_labels"] or labels["planning_candidates"],
            )
        else:
            schema_hint = {
                "bottleneck": "...",
                "opportunity": "...",
                "linkage": "...",
                "evidence_ids": ["P1", "F2"],
            }
            family_rule = (
                "Return exactly one bottleneck and one opportunity. "
                "The bottleneck must read like a recurring technical limitation explicitly visible in historical evidence, not a synthesized grand diagnosis. "
                "The opportunity must directly answer that bottleneck and should be phrased as a one-step technical direction rather than a paper title, branded framework name, or project title. "
                "If the evidence mostly mentions a paper or system title, rewrite it into the underlying reusable technical direction label. "
                "Choose the bottleneck from bottleneck_candidates when possible and the opportunity from opportunity_candidates when possible. "
                "Do not use the same label for both fields."
            )
            candidate_block = bottleneck_state
        candidate_prompt_block = candidate_block if isinstance(candidate_block, dict) else candidate_block[:16]
        prompt = f"""Adapt the Chain-of-Ideas reasoning to the benchmark schema.

Task:
{task.get('question')}

Task contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Chain of ideas:
{chain_text}

Trend:
{trend}

Future direction:
{future}

Human reasoning:
{human}

Candidate labels:
{json.dumps(candidate_prompt_block, ensure_ascii=False, indent=2)}

Evidence refs:
{json.dumps(evidence_refs, ensure_ascii=False, indent=2)}

Rules:
- Preserve the CoI reasoning; do not invent a different line of argument.
- {family_rule}
- Prefer reusable terminology visible in candidate labels and supported future-work phrasing. Do not copy raw evidence titles as the bottleneck or opportunity label.
- Cite evidence ids when possible.
- Output JSON only.

Schema:
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}
"""
        draft = complete_json_object(
            self.main_client,
            [
                {"role": "system", "content": "You are a precise benchmark schema adapter. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.1,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
        critique_prompt = f"""Repair the structured answer so that it strictly matches the task family and benchmark contract.

Task:
{task.get('question')}

Contract:
{json.dumps(contract, ensure_ascii=False, indent=2)}

Candidate labels:
{json.dumps(candidate_prompt_block, ensure_ascii=False, indent=2)}

Draft JSON:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Rules:
- Keep the output faithful to the Chain-of-Ideas reasoning.
- Reject vague generic directions when more specific candidate labels are available.
- Ensure the schema is complete and concise.
- Output JSON only.
"""
        final = complete_json_object(
            self.main_client,
            [
                {"role": "system", "content": "You are a strict benchmark schema critic. Output JSON only."},
                {"role": "user", "content": critique_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.0,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
        return {"draft": draft, "final": final, "candidate_labels": candidate_block}

    def _verbalize_structured_answer(
        self,
        *,
        task: Dict[str, Any],
        projected: Dict[str, Any],
    ) -> str:
        family = str(task.get("family") or "")
        contract = self._contract(task)
        target_venue = _extract_target_venue(str(task.get("question") or ""))
        payload = projected.get("final") or projected.get("draft") or {}
        if isinstance(payload, dict) and isinstance(payload.get("items"), list) and payload.get("items"):
            first = payload["items"][0]
            if isinstance(first, dict):
                normalized = dict(first)
                for key in (
                    "trajectory_label",
                    "next_directions",
                    "rationale",
                    "ranked_directions",
                    "bottleneck",
                    "opportunity",
                    "linkage",
                    "evidence_ids",
                    "first_milestone",
                    "dependency_chain",
                    "defer_rationale",
                    "risk_or_kill_criterion",
                    "primary_positioning",
                    "target_venue",
                    "contribution_package",
                    "venue_fit_rationale",
                    "evaluation_signature",
                    "nearby_but_wrong_positioning",
                ):
                    if key not in normalized and key in payload:
                        normalized[key] = payload.get(key)
                payload = normalized
        pool = projected.get("candidate_labels") or []
        if family == "strategic_research_planning":
            rows = payload.get("ranked_directions") or []
            lines: List[str] = []
            top_direction = ""
            second_direction = ""
            for idx, row in enumerate(rows, start=1):
                direction = self._canonicalize_text(str((row or {}).get("direction") or ""), pool)
                rationale = normalize_ws((row or {}).get("rationale") or "")
                evidence_ids = [str(x).strip() for x in ((row or {}).get("evidence_ids") or []) if str(x).strip()]
                if not direction:
                    continue
                if not top_direction:
                    top_direction = direction
                elif not second_direction:
                    second_direction = direction
                suffix = f" Evidence anchors: {', '.join(evidence_ids[:3])}." if evidence_ids else ""
                lines.append(f"{idx}. {direction} — {rationale}.{suffix}".replace("..", "."))
            milestone = normalize_ws(payload.get("first_milestone") or "")
            dependency = normalize_ws(payload.get("dependency_chain") or "")
            defer_rationale = normalize_ws(payload.get("defer_rationale") or "")
            risk = normalize_ws(payload.get("risk_or_kill_criterion") or "")
            if milestone:
                prefix = f"First milestone for {top_direction}: " if top_direction else "First milestone: "
                lines.append(f"{prefix}{milestone}.")
            if dependency:
                if top_direction and second_direction:
                    lines.append(f"Dependency chain favoring {top_direction} over {second_direction}: {dependency}.")
                elif top_direction:
                    lines.append(f"Dependency chain for {top_direction}: {dependency}.")
                else:
                    lines.append(f"Dependency: {dependency}.")
            if defer_rationale:
                if second_direction:
                    lines.append(f"Defer rationale for {second_direction}: {defer_rationale}.")
                else:
                    lines.append(f"Defer rationale: {defer_rationale}.")
            if risk:
                if top_direction:
                    lines.append(f"Risk/Kill criterion for {top_direction}: {risk}.")
                else:
                    lines.append(f"Risk/Kill criterion: {risk}.")
            return "\n".join(lines).strip()
        if family == "venue_aware_research_positioning":
            rows = payload.get("ranked_directions") or []
            lines: List[str] = []
            max_items = int(contract.get("max_items") or (len(rows) if rows else 2))
            for idx, row in enumerate(rows[:max_items], start=1):
                direction = self._canonicalize_text(str((row or {}).get("direction") or ""), pool)
                fit = normalize_ws((row or {}).get("venue_fit") or (row or {}).get("rationale") or "")
                evidence_ids = [str(x).strip() for x in ((row or {}).get("evidence_ids") or []) if str(x).strip()]
                if not direction:
                    continue
                suffix = f" Evidence anchors: {', '.join(evidence_ids[:3])}." if evidence_ids else ""
                lines.append(f"{idx}. {direction} — {fit}.{suffix}".replace("..", "."))
            positioning = self._canonicalize_text(str(payload.get("primary_positioning") or ""), pool)
            contribution_package = normalize_ws(payload.get("contribution_package") or "")
            venue_fit = normalize_ws(payload.get("venue_fit_rationale") or "")
            evaluation = normalize_ws(payload.get("evaluation_signature") or "")
            contrast = self._canonicalize_text(str(payload.get("nearby_but_wrong_positioning") or ""), pool)
            venue_name = normalize_ws(payload.get("target_venue") or target_venue)
            if positioning:
                lines.append(f"Positioning: {positioning}.")
            if contribution_package:
                lines.append(f"Package: {contribution_package}.")
            if venue_fit:
                prefix = f"Why {venue_name}: " if venue_name else "Why this venue: "
                lines.append(f"{prefix}{venue_fit}.")
            if evaluation:
                lines.append(f"Evaluation: {evaluation}.")
            if contrast:
                lines.append(f"Contrast: {contrast}.")
            return "\n".join(lines).strip()
        if family == "direction_forecasting":
            label = str(payload.get("trajectory_label") or "").strip().lower()
            primary_direction = self._canonicalize_text(str(payload.get("primary_direction") or ""), pool)
            secondary_directions = [
                self._canonicalize_text(str(x or ""), pool)
                for x in (payload.get("secondary_directions") or payload.get("next_directions") or [])
            ]
            secondary_directions = [x for x in secondary_directions if x and x != primary_direction]
            rationale = normalize_ws(payload.get("rationale") or "")
            calibration = normalize_ws(payload.get("calibration") or payload.get("uncertainty_notes") or "")
            evidence_ids = [str(x).strip() for x in (payload.get("evidence_ids") or []) if str(x).strip()]
            signal_rows = payload.get("supporting_signals") or payload.get("momentum_signals") or payload.get("recurring_signals") or []
            counter_rows = payload.get("counter_signals") or payload.get("friction_points") or payload.get("branch_divergences") or []
            signal_texts: List[str] = []
            for row in signal_rows[:3]:
                if isinstance(row, dict):
                    signal = normalize_ws(
                        (row or {}).get("signal")
                        or (row or {}).get("why_it_matters")
                        or (row or {}).get("shift")
                        or ""
                    )
                    row_ids = [str(x).strip() for x in (((row or {}).get("evidence_ids") or [])) if str(x).strip()]
                    if signal:
                        suffix = f" [{', '.join(row_ids[:2])}]" if row_ids else ""
                        signal_texts.append(f"{signal}{suffix}")
                elif normalize_ws(row):
                    signal_texts.append(normalize_ws(str(row)))
            counter_texts: List[str] = []
            for row in counter_rows[:2]:
                if isinstance(row, dict):
                    signal = normalize_ws(
                        (row or {}).get("signal")
                        or (row or {}).get("friction")
                        or (row or {}).get("variant")
                        or ""
                    )
                    impact = normalize_ws((row or {}).get("impact") or (row or {}).get("why_secondary") or "")
                    row_ids = [str(x).strip() for x in (((row or {}).get("evidence_ids") or [])) if str(x).strip()]
                    text = signal if not impact else f"{signal} ({impact})"
                    if text:
                        suffix = f" [{', '.join(row_ids[:2])}]" if row_ids else ""
                        counter_texts.append(f"{text}{suffix}")
                elif normalize_ws(row):
                    counter_texts.append(normalize_ws(str(row)))
            lines: List[str] = []
            if label:
                lines.append(f"Trajectory: {label}.")
            if primary_direction:
                suffix = f" [{', '.join(evidence_ids[:3])}]" if evidence_ids else ""
                lines.append(f"Primary next direction: {primary_direction}.{suffix}".replace(". [", " ["))
            elif secondary_directions:
                lines.append(f"Primary next direction: {secondary_directions[0]}.")
            if signal_texts:
                lines.append(f"Why this follows from pre-cutoff signals: {'; '.join(signal_texts)}.")
            if rationale:
                lines.append(f"Rationale: {rationale}.")
            if secondary_directions:
                lines.append(f"Secondary but weaker branches: {'; '.join(secondary_directions[:2])}.")
            if counter_texts:
                lines.append(f"Counter-signals: {'; '.join(counter_texts)}.")
            if calibration:
                lines.append(f"Calibration: {calibration}.")
            return " ".join(line.strip().rstrip(".") + "." for line in lines if normalize_ws(line)).strip()
        bottleneck_pool = pool.get("bottleneck_candidates") if isinstance(pool, dict) else pool
        opportunity_pool = pool.get("opportunity_candidates") if isinstance(pool, dict) else pool
        repaired_payload = self._repair_bottleneck_payload(task=task, payload=payload, label_state=pool) if isinstance(pool, dict) else payload
        bottleneck = self._canonicalize_text(str(repaired_payload.get("bottleneck") or ""), bottleneck_pool)
        opportunity = self._canonicalize_text(str(repaired_payload.get("opportunity") or ""), opportunity_pool)
        linkage = normalize_ws(repaired_payload.get("linkage") or "")
        evidence_ids = [str(x).strip() for x in (repaired_payload.get("evidence_ids") or []) if str(x).strip()]
        suffix = f" Evidence anchors: {', '.join(evidence_ids[:3])}." if evidence_ids else ""
        return f"A concrete unresolved bottleneck is {bottleneck}. The associated opportunity is {opportunity}. {linkage}.{suffix}".replace("..", ".").strip()

    def _score_candidate_run(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        chain: Dict[str, Any],
        projected: Dict[str, Any],
        trend: str,
        future: str,
    ) -> float:
        family = str(task.get("family") or "")
        contract = self._contract(task)
        payload = projected.get("final") or projected.get("draft") or {}
        chain_profiles = list(chain.get("profiles") or [])
        score = 0.0
        score += min(2.0, 0.5 * len(chain_profiles))
        score += min(0.8, len(normalize_ws(trend)) / 500.0)
        score += min(0.8, len(normalize_ws(future)) / 500.0)
        pool_labels = self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain)["planning_candidates"]
        forecast_labels = self._forecast_candidate_labels(task=task, candidate_pool=candidate_pool, chain=chain)
        if family == "strategic_research_planning":
            rows = payload.get("ranked_directions") or []
            score += 1.2 if len(rows) >= min(2, int(contract.get("max_items") or 3)) else 0.0
            score += min(1.0, len(rows) / max(1, int(contract.get("max_items") or 3)))
            hit = 0
            for row in rows:
                direction = str((row or {}).get("direction") or "")
                if self._canonicalize_text(direction, pool_labels):
                    hit += 1
            score += 0.4 * hit
            score += 0.35 if normalize_ws(payload.get("first_milestone") or "") else 0.0
            score += 0.35 if normalize_ws(payload.get("dependency_chain") or "") else 0.0
            score += 0.2 if normalize_ws(payload.get("defer_rationale") or "") else 0.0
            score += 0.2 if normalize_ws(payload.get("risk_or_kill_criterion") or "") else 0.0
        elif family == "venue_aware_research_positioning":
            rows = payload.get("ranked_directions") or []
            score += 1.0 if rows else 0.0
            score += 0.3 * min(3, len(rows))
            primary = normalize_ws(payload.get("primary_positioning") or "")
            if self._canonicalize_text(primary, pool_labels):
                score += 0.6
            score += 0.45 if normalize_ws(payload.get("contribution_package") or "") else 0.0
            score += 0.45 if normalize_ws(payload.get("venue_fit_rationale") or "") else 0.0
            score += 0.35 if normalize_ws(payload.get("evaluation_signature") or "") else 0.0
        elif family == "direction_forecasting":
            label = str(payload.get("trajectory_label") or "").strip().lower()
            score += 1.2 if label in {"accelerating", "fragmenting", "steady", "cooling"} else 0.0
            score += 0.9 if normalize_ws(payload.get("primary_direction") or "") else 0.0
            score += 0.2 * min(2, len(payload.get("secondary_directions") or payload.get("next_directions") or []))
            score += 0.25 * min(3, len(payload.get("supporting_signals") or payload.get("momentum_signals") or payload.get("recurring_signals") or []))
            score += 0.25 if normalize_ws(payload.get("calibration") or payload.get("uncertainty_notes") or "") else 0.0
            score += 0.15 if payload.get("evidence_ids") else 0.0
            primary = normalize_ws(payload.get("primary_direction") or "")
            canonical_primary = self._canonicalize_text(primary, forecast_labels) if primary else ""
            if primary and canonical_primary:
                score += 0.35
        else:
            bottleneck_state = self._bottleneck_label_state(task=task, candidate_pool=candidate_pool, chain=chain)
            bottleneck = normalize_ws(payload.get("bottleneck") or "")
            opportunity = normalize_ws(payload.get("opportunity") or "")
            score += 1.0 if normalize_ws(payload.get("bottleneck") or "") else 0.0
            score += 1.0 if normalize_ws(payload.get("opportunity") or "") else 0.0
            score += 0.6 if normalize_ws(payload.get("linkage") or "") else 0.0
            if bottleneck and self._canonicalize_text(bottleneck, bottleneck_state.get("bottleneck_candidates") or []):
                score += 0.4
            if opportunity and self._canonicalize_text(opportunity, bottleneck_state.get("opportunity_candidates") or []):
                score += 0.4
            if bottleneck and opportunity and not self._labels_too_similar(bottleneck, opportunity):
                score += 0.25
            if bottleneck and opportunity:
                for row in bottleneck_state.get("paired_unlocks") or []:
                    if _keyword_overlap_score((row or {}).get("bottleneck") or "", bottleneck) >= 0.55 and _keyword_overlap_score((row or {}).get("opportunity") or "", opportunity) >= 0.55:
                        score += 0.45
                        break
        return round(score, 4)

    def _candidate_anchor_count(self, task: Dict[str, Any]) -> int:
        return 5

    def _family_anchor_score(
        self,
        *,
        task: Dict[str, Any],
        anchor: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
    ) -> float:
        family = str(task.get("family") or "")
        text = "\n".join([str(anchor.get("title") or ""), str(anchor.get("abstract") or "")]).lower()
        score = 0.0
        score += 1.8 * float(anchor.get("pool_bonus") or 0.0)
        score += 1.2 * len(anchor.get("packet_match") or [])
        score += 0.5 * int(anchor.get("query_hit_count") or 0)
        score += 0.25 * max(0, 20 - int(anchor.get("best_rank") or 20))
        score += 0.4 * float((anchor.get("scores") or {}).get("combined_score") or 0.0)
        if family == "bottleneck_opportunity_discovery":
            for kw in ["limitation", "benchmark", "evaluation", "failure", "error", "challenge", "memory", "alignment"]:
                if kw in text:
                    score += 0.45
        elif family == "direction_forecasting":
            for kw in ["trend", "emerging", "survey", "benchmark", "evaluation", "scaling", "trajectory"]:
                if kw in text:
                    score += 0.35
            cutoff_dt = self._parse_date(str(task.get("time_cutoff") or ""))
            published_dt = self._parse_date(str(anchor.get("published_date") or ""))
            if cutoff_dt is not None and published_dt is not None:
                score += max(0.0, 1.5 - abs((cutoff_dt - published_dt).days) / 365.0)
        elif family == "strategic_research_planning":
            for kw in ["future work", "trade-off", "tradeoff", "efficiency", "scaling", "robust", "generalization", "benchmark"]:
                if kw in text:
                    score += 0.35
            if candidate_pool is not None:
                score += 0.15 * min(4, len(candidate_pool.packet_ids))
        elif family == "venue_aware_research_positioning":
            venue = _extract_target_venue(str(task.get("question") or ""))
            venue_bucket = _target_venue_bucket_from_name(venue)
            for kw in ["benchmark", "empirical", "evaluation", "ablation", "analysis", "human evaluation", "ranking"]:
                if kw in text:
                    score += 0.3
            if venue and venue.lower() in text:
                score += 0.8
            paper_bucket = _target_venue_bucket_from_name(anchor.get("venue") or "")
            if venue_bucket and paper_bucket in _compatible_venue_buckets(venue_bucket):
                score += 0.6
        return round(score, 4)

    def _support_queries_from_runs(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> List[str]:
        family = str(task.get("family") or "")
        focus = extract_focus_text(task)
        queries = [str(task.get("question") or ""), str(task.get("title") or ""), focus]
        for run in candidate_runs[: max(4, self._candidate_anchor_count(task))]:
            for profile in (run.get("chain", {}) or {}).get("profiles", [])[:5]:
                queries.extend(
                    [
                        str(profile.get("title") or ""),
                        str(profile.get("idea") or ""),
                        str(profile.get("entities") or ""),
                    ]
                )
        if candidate_pool is not None:
            for packet in candidate_pool.packets[:3]:
                queries.extend(
                    [
                        _humanize_label(packet.get("display_name") or packet.get("node_id") or ""),
                        str(packet.get("description") or ""),
                    ]
                )
                for row in (packet.get("emergent_descendants") or [])[:5]:
                    queries.append(_humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or ""))
                for row in (packet.get("top_limitations") or [])[:4]:
                    queries.append(str((row or {}).get("name") or ""))
                for row in (packet.get("top_future_work") or [])[:4]:
                    queries.append(str((row or {}).get("direction") or (row or {}).get("name") or ""))
        if family == "bottleneck_opportunity_discovery":
            queries.extend(
                [
                    f"{focus} limitation failure mode benchmark",
                    f"{focus} error analysis challenge",
                    f"{focus} evaluation gap future work",
                ]
            )
        elif family == "direction_forecasting":
            queries.extend(
                [
                    f"{focus} technical trajectory",
                    f"{focus} emerging direction evaluation",
                    f"{focus} benchmark trend scaling",
                ]
            )
        elif family == "strategic_research_planning":
            queries.extend(
                [
                    f"{focus} future work priority",
                    f"{focus} open problems trade-offs",
                    f"{focus} benchmark bottleneck opportunity",
                ]
            )
        elif family == "venue_aware_research_positioning":
            venue = _extract_target_venue(str(task.get("question") or ""))
            queries.extend(
                [
                    f"{focus} venue fit evaluation package",
                    f"{focus} reviewer expectations empirical study",
                    f"{focus} {venue} strong baseline ablation" if venue else f"{focus} top venue traction empirical study",
                ]
            )
        return dedupe([q for q in queries if normalize_ws(q)])[:16]

    def _collect_family_evidence(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        family = str(task.get("family") or "")
        paper_ids: List[str] = []
        for run in candidate_runs[: self._candidate_anchor_count(task)]:
            for profile in (run.get("chain", {}) or {}).get("profiles", [])[:5]:
                pid = str(profile.get("paper_id") or "").strip()
                if pid:
                    paper_ids.append(pid)
        if candidate_pool is not None and family in {"direction_forecasting", "strategic_research_planning", "venue_aware_research_positioning"}:
            for paper in candidate_pool.papers[:6]:
                pid = str(paper.get("paper_id") or "").strip()
                if pid:
                    paper_ids.append(pid)
        paper_ids = dedupe(paper_ids)[:14]
        paper_rows: List[Dict[str, Any]] = []
        for idx, paper_id in enumerate(paper_ids, start=1):
            paper = domain.get_paper(paper_id) or {}
            pub = paper.get("publication") or {}
            paper_rows.append(
                {
                    "evidence_id": f"P{idx}",
                    "paper_id": paper_id,
                    "title": str(paper.get("title") or ""),
                    "published_date": paper.get("published_date"),
                    "venue": pub.get("venue_name"),
                    "venue_bucket": _target_venue_bucket_from_name(pub.get("venue_name") or ""),
                    "citations": pub.get("citation_count"),
                    "abstract_snippet": clip_text(paper.get("abstract") or "", 700),
                }
            )

        queries = self._support_queries_from_runs(task=task, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        structure_rows: List[Dict[str, Any]] = []
        if paper_ids:
            structure_hits = merge_multi_query_results(
                domain.structure_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                queries,
                top_k_per_query=8,
                limit=10,
            )
            for idx, (doc, scores) in enumerate(structure_hits, start=1):
                structure_rows.append(
                    {
                        "evidence_id": f"T{idx}",
                        "paper_id": doc.paper_id,
                        "title": doc.title,
                        "problem_statement": clip_text(doc.meta.get("problem_statement"), 260),
                        "limitations": list(doc.meta.get("limitations") or [])[:4],
                        "future_work": list(doc.meta.get("future_work") or [])[:4],
                        "core_ideas": list(doc.meta.get("core_ideas") or [])[:4],
                        "snippet": clip_text(doc.text, 900),
                        "scores": scores,
                    }
                )
        page_rows: List[Dict[str, Any]] = []
        if paper_ids and family == "bottleneck_opportunity_discovery":
            page_hits = merge_multi_query_results(
                domain.pageindex_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                queries,
                top_k_per_query=8,
                limit=10,
            )
            for idx, (doc, scores) in enumerate(page_hits, start=1):
                page_rows.append(
                    {
                        "evidence_id": f"G{idx}",
                        "paper_id": doc.paper_id,
                        "title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "kind": doc.meta.get("kind"),
                        "snippet": clip_text(doc.text, 1000),
                        "scores": scores,
                    }
                )
        fulltext_rows: List[Dict[str, Any]] = []
        if paper_ids:
            section_hits = merge_multi_query_results(
                domain.section_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids),
                queries,
                top_k_per_query=8,
                limit=10,
            )
            for idx, (doc, scores) in enumerate(section_hits, start=1):
                fulltext_rows.append(
                    {
                        "evidence_id": f"F{idx}",
                        "paper_id": doc.paper_id,
                        "title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "level": doc.meta.get("level"),
                        "source_type": "kb_section",
                        "snippet": clip_text(doc.text, 1200),
                        "scores": scores,
                    }
                )
        packet_rows: List[Dict[str, Any]] = []
        if candidate_pool is not None:
            for idx, packet in enumerate(candidate_pool.packets[:4], start=1):
                packet_rows.append(
                    {
                        "evidence_id": f"N{idx}",
                        "packet_id": packet.get("packet_id"),
                        "display_name": _humanize_label(packet.get("display_name") or packet.get("node_id") or ""),
                        "description": packet.get("description"),
                        "snippet": _join_lines(
                            [
                                f"Display name: {_humanize_label(packet.get('display_name') or packet.get('node_id') or '')}",
                                f"Description: {packet.get('description') or ''}",
                                "Emergent descendants: "
                                + "; ".join(
                                    _humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or "")
                                    for row in (packet.get("emergent_descendants") or [])[:6]
                                    if _humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or "")
                                ),
                            ]
                        ),
                    }
                )
        return {
            "queries": queries,
            "papers": paper_rows,
            "structures": structure_rows,
            "pageindex": page_rows,
            "fulltext": fulltext_rows,
            "candidate_node_evidence": packet_rows,
        }

    def _aggregate_candidate_labels(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> List[str]:
        labels: List[str] = []
        if candidate_pool is not None:
            for packet in candidate_pool.packets[:4]:
                labels.append(_humanize_label(packet.get("display_name") or packet.get("node_id") or ""))
                for row in (packet.get("emergent_descendants") or [])[:8]:
                    labels.append(_humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or ""))
                for row in (packet.get("top_limitations") or [])[:6]:
                    labels.append(str((row or {}).get("name") or ""))
                for row in (packet.get("top_future_work") or [])[:6]:
                    labels.append(str((row or {}).get("direction") or (row or {}).get("name") or ""))
        for run in candidate_runs[: max(4, self._candidate_anchor_count(task))]:
            labels.extend(self._chain_candidate_labels(task=task, candidate_pool=candidate_pool, chain=run.get("chain") or {}).get("planning_candidates") or [])
        filtered = []
        for label in dedupe(labels):
            text = normalize_ws(label or "")
            if not text:
                continue
            if len(text) > 160:
                continue
            if text.count(":") >= 4:
                continue
            if "{background:" in text.lower():
                continue
            filtered.append(text)
        return filtered[:24]

    def _render_candidate_runs(self, candidate_runs: List[Dict[str, Any]], *, limit: int = 4) -> str:
        blocks: List[str] = []
        for idx, run in enumerate(candidate_runs[:limit], start=1):
            chain = run.get("chain") or {}
            profiles = list(chain.get("profiles") or [])
            paper_line = " -> ".join(
                f"{str(p.get('published_date') or '')[:10]} {str(p.get('title') or '')}".strip()
                for p in profiles[:5]
                if str(p.get("title") or "").strip()
            )
            idea_line = " | ".join(
                clip_text(str(p.get("idea") or ""), 180)
                for p in profiles[:4]
                if normalize_ws(p.get("idea") or "")
            )
            blocks.append(
                _join_lines(
                    [
                        f"Branch {idx}",
                        f"Anchor: {run.get('anchor', {}).get('title') or ''}",
                        f"Paper chain: {paper_line}",
                        f"Key ideas: {idea_line}",
                        f"Trend summary: {run.get('trend') or ''}",
                        f"Future summary: {run.get('future') or ''}",
                        f"Provisional answer: {run.get('answer') or ''}",
                    ]
                )
            )
        return "\n\n".join(blocks).strip()

    def _coi_family_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            obj = complete_json_object(
                self.main_client,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=1400,
                temperature=0.15,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            if isinstance(obj, dict) and obj:
                return obj
        except Exception:
            pass
        return fallback

    def _render_packet_context(self, candidate_pool: Optional[TaskCandidatePool], *, limit: int = 4) -> str:
        if candidate_pool is None:
            return ""
        blocks: List[str] = []
        for idx, packet in enumerate(candidate_pool.packets[:limit], start=1):
            descendants = [
                _humanize_label((row or {}).get("display_name") or (row or {}).get("node_id") or "")
                for row in (packet.get("emergent_descendants") or [])[:6]
            ]
            descendants = [x for x in descendants if x]
            limitations = [
                normalize_ws((row or {}).get("name") or "")
                for row in (packet.get("top_limitations") or [])[:5]
                if normalize_ws((row or {}).get("name") or "")
            ]
            future_work = [
                normalize_ws((row or {}).get("direction") or (row or {}).get("name") or "")
                for row in (packet.get("top_future_work") or [])[:5]
                if normalize_ws((row or {}).get("direction") or (row or {}).get("name") or "")
            ]
            blocks.append(
                _join_lines(
                    [
                        f"Packet {idx}: {_humanize_label(packet.get('display_name') or packet.get('node_id') or '')}",
                        f"Description: {packet.get('description') or ''}",
                        f"Historical bottlenecks: {'; '.join(limitations)}",
                        f"Emergent descendants: {'; '.join(descendants)}",
                        f"Historical future work: {'; '.join(future_work)}",
                    ]
                )
            )
        return "\n\n".join(blocks).strip()

    def _render_coi_backbone_state(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
        evidence: Dict[str, Any],
    ) -> str:
        run_blocks: List[str] = []
        family = str(task.get("family") or "")
        render_limit = max(4, self._candidate_anchor_count(task))
        for idx, run in enumerate(candidate_runs[:render_limit], start=1):
            chain = run.get("chain") or {}
            profiles = list(chain.get("profiles") or [])
            transition_line = " -> ".join(
                f"{str(row.get('published_date') or '')[:10]} {str(row.get('title') or '')}".strip()
                for row in profiles[:5]
                if normalize_ws(row.get("title") or "")
            )
            idea_line = " | ".join(
                clip_text(str(row.get("idea") or ""), 160)
                for row in profiles[:4]
                if normalize_ws(row.get("idea") or "")
            )
            run_blocks.append(
                _join_lines(
                    [
                        f"Branch {idx}",
                        f"Anchor: {(run.get('anchor') or {}).get('title') or ''}",
                        f"Paper transitions: {transition_line}",
                        f"Key ideas: {idea_line}",
                        f"Trend summary: {run.get('trend') or ''}",
                        f"Forward-looking hypothesis: {run.get('future') or ''}",
                        f"Provisional answer: {run.get('answer') or ''}",
                    ]
                )
            )
        counts = {
            "candidate_runs": len(candidate_runs),
            "packet_count": len((candidate_pool.packets if candidate_pool is not None else []) or []),
            "paper_evidence": len(evidence.get("papers") or []),
            "structure_evidence": len(evidence.get("structures") or []),
            "pageindex_evidence": len(evidence.get("pageindex") or []),
            "fulltext_evidence": len(evidence.get("fulltext") or []),
        }
        bottleneck_signal_state = ""
        if family == "bottleneck_opportunity_discovery":
            branch_lines = []
            total_limitations: Counter[str] = Counter()
            total_future: Counter[str] = Counter()
            total_core: Counter[str] = Counter()
            for idx, run in enumerate(candidate_runs[:render_limit], start=1):
                chain = run.get("chain") or {}
                signals = chain.get("bottleneck_signals") or {}
                branch_lines.append(f"Branch {idx}: {clip_text(signals.get('signal_summary_text') or '', 500)}")
                for row in signals.get("recurring_limitations") or []:
                    total_limitations[normalize_ws((row or {}).get("label") or "")] += int((row or {}).get("count") or 0)
                for row in signals.get("future_work_signals") or []:
                    total_future[normalize_ws((row or {}).get("label") or "")] += int((row or {}).get("count") or 0)
                for row in signals.get("core_idea_signals") or []:
                    total_core[normalize_ws((row or {}).get("label") or "")] += int((row or {}).get("count") or 0)
            bottleneck_signal_state = _join_lines(
                [
                    "Cross-branch bottleneck signal summary:",
                    f"Recurring limitations: {_render_counter_rows(_counter_rows(total_limitations)) or 'none'}",
                    f"Future-work unlock signals: {_render_counter_rows(_counter_rows(total_future)) or 'none'}",
                    f"Related core ideas: {_render_counter_rows(_counter_rows(total_core)) or 'none'}",
                    "\n".join(line for line in branch_lines if normalize_ws(line)),
                ]
            )
        venue_signal_state = ""
        if family == "venue_aware_research_positioning":
            branch_lines = []
            total_venues: Counter[str] = Counter()
            total_buckets: Counter[str] = Counter()
            total_packages: Counter[str] = Counter()
            total_evals: Counter[str] = Counter()
            compatible_hits = 0
            mismatched_hits = 0
            for idx, run in enumerate(candidate_runs[:render_limit], start=1):
                chain = run.get("chain") or {}
                signals = chain.get("venue_signals") or {}
                branch_lines.append(f"Branch {idx}: {clip_text(signals.get('signal_summary_text') or '', 500)}")
                for row in signals.get("chain_venue_counts") or []:
                    total_venues[normalize_ws((row or {}).get("label") or "")] += int((row or {}).get("count") or 0)
                for row in signals.get("chain_venue_bucket_counts") or []:
                    total_buckets[normalize_ws((row or {}).get("label") or "")] += int((row or {}).get("count") or 0)
                for row in signals.get("package_pattern_counts") or []:
                    total_packages[normalize_ws((row or {}).get("label") or "")] += int((row or {}).get("count") or 0)
                for row in signals.get("evaluation_pattern_counts") or []:
                    total_evals[normalize_ws((row or {}).get("label") or "")] += int((row or {}).get("count") or 0)
                compatible_hits += int(signals.get("compatible_bucket_hits") or 0)
                mismatched_hits += int(signals.get("mismatched_bucket_hits") or 0)
            venue_signal_state = _join_lines(
                [
                    "Cross-branch venue signal summary:",
                    f"Observed venues: {_render_counter_rows(_counter_rows(total_venues)) or 'none'}",
                    f"Observed venue buckets: {_render_counter_rows(_counter_rows(total_buckets)) or 'none'}",
                    f"Compatible vs mismatched branch hits: {compatible_hits} compatible / {mismatched_hits} mismatched",
                    f"Recurring packaging patterns: {_render_counter_rows(_counter_rows(total_packages)) or 'none'}",
                    f"Recurring evaluation patterns: {_render_counter_rows(_counter_rows(total_evals)) or 'none'}",
                    "\n".join(line for line in branch_lines if normalize_ws(line)),
                ]
            )
        return _join_lines(
            [
                f"Task family: {task.get('family')}",
                f"Question: {task.get('question')}",
                f"Backbone coverage: {json.dumps(counts, ensure_ascii=False)}",
                "",
                "Packet context:",
                self._render_packet_context(candidate_pool),
                "",
                "CoI candidate branches:",
                "\n\n".join(run_blocks).strip(),
                "",
                bottleneck_signal_state,
                "",
                venue_signal_state,
                "",
                "Evidence block:",
                self._render_evidence_block(evidence),
            ]
        ).strip()

    def _adapt_head_output(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
        evidence: Dict[str, Any],
        head_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        family = str(task.get("family") or "")
        contract = self._contract(task)
        contract_candidates = _task_candidate_directions(task)
        target_venue = _extract_target_venue(str(task.get("question") or ""))
        labels = self._aggregate_candidate_labels(task=task, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        forecast_labels = self._aggregate_forecast_labels(task=task, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        strategic_labels = self._strategic_candidate_labels(task=task, candidate_pool=candidate_pool, chain={})
        bottleneck_state = self._aggregate_bottleneck_label_state(task=task, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        if family == "bottleneck_opportunity_discovery":
            augmented_bottlenecks = list(bottleneck_state.get("bottleneck_candidates") or [])
            augmented_opportunities = list(bottleneck_state.get("opportunity_candidates") or [])
            augmented_pairs = list(bottleneck_state.get("paired_unlocks") or [])
            title_pool = list(bottleneck_state.get("title_like_blocks") or [])

            for row in (head_analysis.get("recurring_bottlenecks") or [])[:8]:
                name = _clean_topic_label((row or {}).get("name") or "")
                if name and not self._is_title_like_bottleneck_opportunity_label(name, title_pool):
                    augmented_bottlenecks.append(name)

            analysis_opportunity_names: List[str] = []
            for row in (head_analysis.get("opportunity_candidates") or [])[:8]:
                name = _clean_topic_label((row or {}).get("name") or "")
                if name and not self._is_title_like_bottleneck_opportunity_label(name, title_pool):
                    analysis_opportunity_names.append(name)
                    augmented_opportunities.append(name)

            selected_bottleneck = _clean_topic_label(head_analysis.get("selected_bottleneck") or "")
            selected_opportunity = _clean_topic_label(head_analysis.get("selected_opportunity") or "")
            if selected_bottleneck and not self._is_title_like_bottleneck_opportunity_label(selected_bottleneck, title_pool):
                augmented_bottlenecks.append(selected_bottleneck)
            if selected_opportunity and not self._is_title_like_bottleneck_opportunity_label(selected_opportunity, title_pool):
                augmented_opportunities.append(selected_opportunity)
            if selected_bottleneck and selected_opportunity:
                augmented_pairs.insert(
                    0,
                    {
                        "bottleneck": selected_bottleneck,
                        "opportunity": selected_opportunity,
                        "support_count": 3,
                        "paper_ids": [],
                    },
                )
            seen_pairs = set()
            merged_pairs: List[Dict[str, Any]] = []
            for row in augmented_pairs:
                left = _clean_topic_label((row or {}).get("bottleneck") or "")
                right = _clean_topic_label((row or {}).get("opportunity") or "")
                if not left or not right:
                    continue
                key = (_normalize_label(left), _normalize_label(right))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                merged_pairs.append(
                    {
                        "bottleneck": left,
                        "opportunity": right,
                        "support_count": int((row or {}).get("support_count") or 0),
                        "paper_ids": [str(x).strip() for x in ((row or {}).get("paper_ids") or []) if str(x).strip()],
                    }
                )

            bottleneck_state = {
                **bottleneck_state,
                "bottleneck_candidates": dedupe(augmented_bottlenecks)[:24],
                "opportunity_candidates": dedupe(augmented_opportunities)[:24],
                "paired_unlocks": merged_pairs[:12],
            }
        if contract_candidates:
            labels = dedupe([*contract_candidates, *labels])
        evidence_refs = []
        for group in ["papers", "structures", "pageindex", "fulltext"]:
            for row in (evidence.get(group) or [])[:6]:
                evidence_refs.append(
                    {
                        "evidence_id": row.get("evidence_id"),
                        "paper_id": row.get("paper_id"),
                        "title": row.get("title"),
                    }
                )
        if family == "strategic_research_planning":
            schema_hint = {
                "ranked_directions": [
                    {
                        "rank": 1,
                        "direction": "...",
                        "rationale": "...",
                        "evidence_ids": ["P1", "T1"],
                    }
                ],
                "first_milestone": "...",
                "dependency_chain": "...",
                "defer_rationale": "...",
                "risk_or_kill_criterion": "...",
            }
            family_rule = (
                f"Return up to {int(contract.get('max_items') or 3)} ranked directions. Each direction should be specific, non-redundant, and phrased as a real research agenda item. "
                "Also include first_milestone, dependency_chain, defer_rationale, and risk_or_kill_criterion. "
                "Those four fields must explicitly name the prioritized direction, and when there is an alternative candidate they must explain why that alternative is deferred."
            )
            if contract_candidates:
                family_rule += f" Use only these candidate directions: {json.dumps(contract_candidates, ensure_ascii=False)}."
            labels = contract_candidates or strategic_labels or labels
        elif family == "venue_aware_research_positioning":
            schema_hint = {
                "ranked_directions": [
                    {
                        "rank": 1,
                        "direction": "...",
                        "venue_fit": "...",
                        "evidence_ids": ["P1", "T1"],
                    }
                ],
                "primary_positioning": "...",
                "target_venue": target_venue or "...",
                "contribution_package": "...",
                "venue_fit_rationale": "...",
                "evaluation_signature": "...",
                "nearby_but_wrong_positioning": "...",
                "evidence_ids": ["P1", "T1"],
            }
            family_rule = "Return a venue-facing positioning judgment with an explicit contribution_package, venue_fit_rationale, evaluation_signature, and nearby_but_wrong_positioning."
            if target_venue:
                family_rule += f" Explicitly name {target_venue} in the reasoning."
            if contract_candidates:
                family_rule += f" If the task lists candidate directions, rank only these directions: {json.dumps(contract_candidates, ensure_ascii=False)}."
        elif family == "direction_forecasting":
            labels = forecast_labels or labels
            schema_hint = {
                "trajectory_label": "accelerating",
                "primary_direction": "...",
                "secondary_directions": ["..."],
                "supporting_signals": [
                    {"signal": "...", "evidence_ids": ["P1", "T1"]}
                ],
                "counter_signals": [
                    {"signal": "...", "impact": "...", "evidence_ids": ["F1"]}
                ],
                "rationale": "...",
                "calibration": "...",
                "evidence_ids": ["P1", "T1"],
            }
            family_rule = (
                "Return exactly one trajectory_label from accelerating, fragmenting, steady, cooling and exactly one primary_direction. "
                "You may include up to 2 secondary_directions, but they must be explicitly framed as weaker or contingent alternatives. "
                "supporting_signals must name 2-3 concrete pre-cutoff signals, frictions, or momentum cues with evidence ids. "
                "calibration must explain why the primary_direction is the mainline near-term expectation rather than a certainty."
            )
            if contract_candidates:
                family_rule += f" If the task lists candidate directions, choose the primary_direction from these candidates: {json.dumps(contract_candidates, ensure_ascii=False)}."
            else:
                family_rule += (
                    " Choose the primary_direction from the provided candidate labels whenever a good fit exists. "
                    "Prefer descendant-style or future-work-style labels over paper titles, benchmark names, or method names."
                )
        else:
            schema_hint = {
                "bottleneck": "...",
                "opportunity": "...",
                "linkage": "...",
                "evidence_ids": ["T1", "F1"],
            }
            family_rule = (
                "Return exactly one bottleneck and one opportunity. "
                "Keep the bottleneck close to recurring limitation labels from the evidence when possible. "
                "Choose the bottleneck from bottleneck_candidates when possible and the opportunity from opportunity_candidates when possible. "
                "Prioritize paired_unlocks when they clearly fit the evidence. "
                "The opportunity must directly respond to the bottleneck and should stay near supported future-work or descendant-style labels instead of copying a paper title or inventing a branded framework name. "
                "If the evidence only gives a paper title, rewrite it into the underlying reusable technical direction. "
                "Do not output the same or near-duplicate label for both fields."
            )
            labels = bottleneck_state
        label_prompt_block = labels if isinstance(labels, dict) else labels[:24]
        prompt = f"""You are the task-specific answer adapter that sits on top of a Chain-of-Ideas evidence backbone.

Task:
{task.get('question')}

Task family:
{family}

Family-specific analysis output:
{json.dumps(head_analysis, ensure_ascii=False, indent=2)}

Candidate labels:
{json.dumps(label_prompt_block, ensure_ascii=False, indent=2)}

Evidence refs:
{json.dumps(evidence_refs[:18], ensure_ascii=False, indent=2)}

Rules:
- Preserve the backbone reasoning and the family-specific analysis; do not invent a new line of argument.
- Keep wording technically specific and close to supported labels when possible.
- Avoid generic management language.
- {family_rule}
- Prefer milestone/dependency/defer/risk statements that clearly map evidence into an execution decision, not generic roadmap filler.
- Output JSON only.

Schema:
{json.dumps(schema_hint, ensure_ascii=False, indent=2)}
"""
        draft = complete_json_object(
            self.main_client,
            [
                {"role": "system", "content": "You are a precise benchmark answer adapter. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.1,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
        critique_prompt = f"""Review and repair the draft benchmark JSON so that it is specific, evidence-grounded, and family-aligned.

Task:
{task.get('question')}

Task family:
{family}

Family-specific analysis:
{json.dumps(head_analysis, ensure_ascii=False, indent=2)}

Draft JSON:
{json.dumps(draft, ensure_ascii=False, indent=2)}

Candidate labels:
{json.dumps(label_prompt_block, ensure_ascii=False, indent=2)}

Rules:
- Keep the draft faithful to the backbone analysis.
- Remove vague, duplicate, or overly broad items.
- Prefer technically discriminative labels over broad parent topics.
- For forecasting, do not output several co-equal future directions; force one mainline primary_direction and move the rest into secondary_directions or counter_signals.
- For strategic planning, make the milestone/dependency/defer/risk fields explicitly point to the chosen ranked direction and the deferred alternative.
- For bottleneck discovery, never leave the bottleneck blank and never use a raw paper title as the opportunity label; rewrite titles into reusable technical direction labels.
- Output repaired JSON only.
"""
        final = complete_json_object(
            self.main_client,
            [
                {"role": "system", "content": "You are a strict benchmark answer critic. Output JSON only."},
                {"role": "user", "content": critique_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=1200,
            temperature=0.0,
            timeout=180,
            transport_retries=2,
            max_parse_attempts=3,
        )
        if family == "bottleneck_opportunity_discovery":
            if isinstance(draft, dict):
                draft = self._align_bottleneck_payload_to_head_analysis(
                    task=task,
                    payload=draft,
                    label_state=bottleneck_state,
                    head_analysis=head_analysis,
                )
            if isinstance(final, dict):
                final = self._align_bottleneck_payload_to_head_analysis(
                    task=task,
                    payload=final,
                    label_state=bottleneck_state,
                    head_analysis=head_analysis,
                )
        return {"draft": draft, "final": final, "candidate_labels": labels}

    def _finalize_head_answer(
        self,
        *,
        task: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
        evidence: Dict[str, Any],
        head_analysis: Dict[str, Any],
        fallback_answer: str,
    ) -> Tuple[str, Dict[str, Any]]:
        adapted = self._adapt_head_output(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=head_analysis,
        )
        answer = self._verbalize_structured_answer(task=task, projected=adapted).strip()
        if not answer:
            answer = normalize_ws(fallback_answer)
        return answer, adapted

    def _run_bottleneck_head(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        evidence = self._collect_family_evidence(task=task, domain_id=domain_id, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        fallback_answer = candidate_runs[0].get("answer") if candidate_runs else ""
        backbone_state = self._render_coi_backbone_state(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
        )
        bottleneck_state = self._aggregate_bottleneck_label_state(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
        )
        bottleneck_signal_state = _join_lines(
            [
                clip_text((run.get("chain") or {}).get("bottleneck_signal_summary") or "", 700)
                for run in candidate_runs[: max(4, self._candidate_anchor_count(task))]
                if normalize_ws((run.get("chain") or {}).get("bottleneck_signal_summary") or "")
            ]
        )
        prompt = f"""You are the bottleneck-opportunity analysis head that sits on top of a Chain-of-Ideas evidence backbone.

Backbone state:
{backbone_state}

Auxiliary chain-stage bottleneck signal summaries:
{bottleneck_signal_state or "No bottleneck signal summary was available."}

Bottleneck candidate state:
{json.dumps(bottleneck_state, ensure_ascii=False, indent=2)}

Instructions:
- Preserve the Chain-of-Ideas reasoning and synthesize recurring signals across branches instead of writing an explicit branch-by-branch comparison.
- Extract recurring technical frictions, evaluation gaps, and unresolved trade-offs.
- Treat the auxiliary chain-stage bottleneck signals as corroborating hints only.
- Use them to preserve historically repeated limitation wording and to identify plausible near-term unlocks, but do not let them override a sharper branch-supported bottleneck or a more concrete provisional answer label.
- Then decide which bottleneck is the most central unresolved constraint.
- Finally identify which opportunity is genuinely opened if that bottleneck is addressed.
- Distinguish the bottleneck itself from its downstream opportunity.
- Prefer benchmark-facing wording that stays close to recurring limitation labels, packet descendants, or future-work phrasing already present in the evidence.
- Select `selected_bottleneck` from `bottleneck_candidates` when a close match exists.
- Select `selected_opportunity` from `opportunity_candidates` when a close match exists.
- Prefer a supported `paired_unlocks` pair over a free-form combination when both are available.
- Do not use a paper title, benchmark title, or branded method title as `selected_opportunity`; rewrite it into the underlying reusable technical direction label.
- Avoid inventing bespoke framework names or overly specific branded methods unless the benchmark contract itself uses that exact label as the reusable direction.
- Keep the opportunity to a one-step technical direction that would become viable after resolving the bottleneck, not a full productized system pitch.
- Do not output the same or near-duplicate label for both `selected_bottleneck` and `selected_opportunity`.

Return JSON only:
{{
  "recurring_bottlenecks": [
    {{"name": "...", "mechanism": "...", "evidence_ids": ["T1", "F1"]}}
  ],
  "opportunity_candidates": [
    {{"name": "...", "why_now": "...", "evidence_ids": ["P1", "T2"]}}
  ],
  "selected_bottleneck": "...",
  "selected_opportunity": "...",
  "linkage": "...",
  "strategic_implication": "..."
}}
"""
        fallback = {
            "recurring_bottlenecks": [],
            "opportunity_candidates": [],
            "selected_bottleneck": "",
            "selected_opportunity": "",
            "linkage": "",
            "strategic_implication": "",
        }
        analysis = self._coi_family_json(
            system_prompt="You are a precise Chain-of-Ideas benchmark synthesizer for bottleneck discovery. Output JSON only.",
            user_prompt=prompt,
            fallback=fallback,
        )
        answer, adapted = self._finalize_head_answer(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=analysis,
            fallback_answer=fallback_answer,
        )
        return answer, {"analysis": analysis, "adapted": adapted}, evidence

    def _run_direction_head(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        evidence = self._collect_family_evidence(task=task, domain_id=domain_id, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        fallback_answer = candidate_runs[0].get("answer") if candidate_runs else ""
        forecast_labels = self._aggregate_forecast_labels(task=task, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        backbone_state = self._render_coi_backbone_state(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
        )
        prompt = f"""You are the direction-forecasting head that sits on top of a Chain-of-Ideas evidence backbone.

Backbone state:
{backbone_state}

Candidate direction labels:
{json.dumps(forecast_labels[:24], ensure_ascii=False, indent=2)}

Instructions:
- Preserve the backbone reasoning and synthesize the recurring cross-branch signals for convergence, divergence, momentum, and recombination.
- Separate recurring momentum signals from enabling shifts and unresolved frictions.
- Choose exactly one mainline primary_direction that best reflects the strongest recurring signals across branches.
- If other plausible directions exist, place them under secondary_directions and explain why they remain weaker or more contingent.
- Make one trajectory call and explain why.
- Avoid generic trend language or laundry lists of unrelated directions; focus on technically discriminative directions.
- The output should be visibly grounded in pre-cutoff evidence, not just in abstract trend prose.
- When the candidate direction labels already contain a good descendant-style or future-work-style label, use that label for primary_direction instead of a paper title or method title.

Return JSON only:
{{
  "recurring_signals": [
    {{"signal": "...", "why_it_matters": "...", "branches": [1, 2], "evidence_ids": ["P1", "T1"]}}
  ],
  "branch_divergences": [
    {{"variant": "...", "why_secondary": "...", "evidence_ids": ["P2"]}}
  ],
  "friction_points": [
    {{"friction": "...", "evidence_ids": ["T2", "F1"]}}
  ],
  "trajectory_label": "accelerating",
  "primary_direction": "...",
  "secondary_directions": ["..."],
  "rationale": "...",
  "calibration": "...",
  "uncertainty_notes": "..."
}}
"""
        fallback = {
            "recurring_signals": [],
            "branch_divergences": [],
            "friction_points": [],
            "trajectory_label": "",
            "primary_direction": "",
            "secondary_directions": [],
            "rationale": "",
            "calibration": "",
            "uncertainty_notes": "",
        }
        analysis = self._coi_family_json(
            system_prompt="You are a precise Chain-of-Ideas benchmark synthesizer for trajectory forecasting. Output JSON only.",
            user_prompt=prompt,
            fallback=fallback,
        )
        answer, adapted = self._finalize_head_answer(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=analysis,
            fallback_answer=fallback_answer,
        )
        return answer, {"analysis": analysis, "adapted": adapted}, evidence

    def _run_planning_head(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        evidence = self._collect_family_evidence(task=task, domain_id=domain_id, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        fallback_answer = candidate_runs[0].get("answer") if candidate_runs else ""
        contract = self._contract(task)
        contract_candidates = _task_candidate_directions(task)
        max_items = int(contract.get("max_items") or 3)
        backbone_state = self._render_coi_backbone_state(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
        )
        prompt = f"""You are the strategic-planning head that sits on top of a Chain-of-Ideas evidence backbone.

Backbone state:
{backbone_state}

        Instructions:
- Preserve the backbone reasoning and translate the recurring cross-branch signals into a ranked research plan.
- Distinguish promising directions, crowded directions, and risky bets.
- Rank only directions that are technically actionable in the next six months.
- Use why-now value, tractability, and non-redundancy to determine rank.
- Provide one first_milestone, one dependency_chain, one defer_rationale, and one risk_or_kill_criterion that make the ordering executable.
- Prefer milestone and dependency statements that point to concrete technical work, not general program management.
- Make the milestone/dependency/defer/risk fields explicitly name the prioritized direction and, if present, the deferred alternative direction.
- Each ranked direction rationale should cite the specific evidence anchors that justify the ordering.
- Keep the final ranked list to at most {max_items} items.
{"- Only rank these explicit candidate directions: " + json.dumps(contract_candidates, ensure_ascii=False) if contract_candidates else ""}

Return JSON only:
{{
  "promising_directions": [
    {{"direction": "...", "why_now": "...", "evidence_ids": ["P1", "T1"]}}
  ],
  "crowded_or_saturated_areas": [
    {{"direction": "...", "reason": "..."}}
  ],
  "risky_bets": [
    {{"direction": "...", "risk": "..."}}
  ],
  "ranked_directions": [
    {{"rank": 1, "direction": "...", "rationale": "...", "evidence_ids": ["T1", "P2"]}}
  ],
  "first_milestone": "...",
  "dependency_chain": "...",
  "defer_rationale": "...",
  "risk_or_kill_criterion": "..."
}}
"""
        fallback = {
            "promising_directions": [],
            "crowded_or_saturated_areas": [],
            "risky_bets": [],
            "ranked_directions": [],
            "first_milestone": "",
            "dependency_chain": "",
            "defer_rationale": "",
            "risk_or_kill_criterion": "",
        }
        analysis = self._coi_family_json(
            system_prompt="You are a precise Chain-of-Ideas benchmark synthesizer for strategic research planning. Output JSON only.",
            user_prompt=prompt,
            fallback=fallback,
        )
        answer, adapted = self._finalize_head_answer(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=analysis,
            fallback_answer=fallback_answer,
        )
        return answer, {"analysis": analysis, "adapted": adapted}, evidence

    def _run_venue_head(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        candidate_pool: Optional[TaskCandidatePool],
        candidate_runs: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        evidence = self._collect_family_evidence(task=task, domain_id=domain_id, candidate_pool=candidate_pool, candidate_runs=candidate_runs)
        fallback_answer = candidate_runs[0].get("answer") if candidate_runs else ""
        contract = self._contract(task)
        contract_candidates = _task_candidate_directions(task)
        target_venue = _extract_target_venue(str(task.get("question") or ""))
        max_items = int(contract.get("max_items") or (len(contract_candidates) if contract_candidates else 2))
        backbone_state = self._render_coi_backbone_state(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
        )
        venue_signal_state = _join_lines(
            [
                clip_text((run.get("chain") or {}).get("venue_signal_summary") or "", 700)
                for run in candidate_runs[: max(4, self._candidate_anchor_count(task))]
                if normalize_ws((run.get("chain") or {}).get("venue_signal_summary") or "")
            ]
        )
        prompt = f"""You are the venue-aware positioning head that sits on top of a Chain-of-Ideas evidence backbone.

Backbone state:
{backbone_state}

Auxiliary chain-stage venue signal summaries:
{venue_signal_state or "No venue signal summary was available."}

Instructions:
- Preserve the backbone reasoning and use the recurring cross-branch signals to decide what contribution framing will travel best at the target venue.
- Treat the auxiliary chain-stage venue signals as supporting hints for contribution packaging and evaluation recipe only.
- Do not let those auxiliary summaries replace a more concrete branch-supported positioning label with a generic package category.
- Separate broad direction choice from venue-specific packaging requirements.
- State what contribution package, evaluation signature, and framing logic make the positioning credible.
- Include one nearby but weaker alternative positioning and explain why it is less suitable.
- Keep `primary_positioning` as a concrete technical framing, not an abstract package label such as `new_method`, `empirical_comparison`, `analysis_or_diagnosis`, or `benchmark_eval`.
- Keep the ranked list to at most {max_items} items.
{"- The target venue is " + target_venue + "." if target_venue else ""}
{"- Only rank these explicit candidate directions: " + json.dumps(contract_candidates, ensure_ascii=False) if contract_candidates else ""}

Return JSON only:
{{
  "ranked_directions": [
    {{"rank": 1, "direction": "...", "venue_fit": "...", "evidence_ids": ["P1", "T1"]}}
  ],
  "primary_positioning": "...",
  "target_venue": "{target_venue}",
  "contribution_package": "...",
  "venue_fit_rationale": "...",
  "evaluation_signature": "...",
  "nearby_but_wrong_positioning": "...",
  "evidence_ids": ["P1", "T1"]
}}
"""
        fallback = {
            "ranked_directions": [],
            "primary_positioning": "",
            "target_venue": target_venue,
            "contribution_package": "",
            "venue_fit_rationale": "",
            "evaluation_signature": "",
            "nearby_but_wrong_positioning": "",
            "evidence_ids": [],
        }
        analysis = self._coi_family_json(
            system_prompt="You are a precise Chain-of-Ideas benchmark synthesizer for venue-aware research positioning. Output JSON only.",
            user_prompt=prompt,
            fallback=fallback,
        )
        answer, adapted = self._finalize_head_answer(
            task=task,
            candidate_pool=candidate_pool,
            candidate_runs=candidate_runs,
            evidence=evidence,
            head_analysis=analysis,
            fallback_answer=fallback_answer,
        )
        return answer, {"analysis": analysis, "adapted": adapted}, evidence

    def build_queries(self, task: Dict[str, Any]) -> List[str]:
        cache_key = str(task.get("task_id") or "") or _cache_key("queries", task.get("title"), task.get("question"), task.get("family"))
        with self._cache_lock:
            cached_queries = list(self._task_query_cache.get(cache_key) or [])
        if cached_queries:
            return cached_queries
        family = str(task.get("family") or "")
        contract = self._contract(task)
        topic = extract_focus_text(task)
        if family == "venue_aware_research_positioning":
            topic = _extract_venue_topic_text(task) or str(contract.get("topic_text") or topic or "").strip()
        short_topic = short_focus_terms(topic) or topic
        title_focus = short_focus_terms(str(task.get("title") or "")) or str(task.get("title") or "")
        prompt = get_deep_search_query_prompt(topic=topic)
        try:
            response = self._complete(self.main_client, prompt, max_tokens=500, temperature=0.2)
            queries = parse_query_list(response)
        except Exception:
            queries = []
        if family == "venue_aware_research_positioning":
            queries = []
        fallbacks = [
            task.get("question") or "",
            task.get("title") or "",
            topic,
            short_topic,
            title_focus,
        ]
        contract_candidates = _task_candidate_directions(task)
        if contract_candidates:
            joined_candidates = " vs ".join(contract_candidates[:3])
            fallbacks.append(joined_candidates)
            for candidate in contract_candidates[:4]:
                fallbacks.append(candidate)
                if short_topic:
                    fallbacks.append(f"{candidate} {short_topic}")
            if family == "strategic_research_planning":
                for candidate in contract_candidates[:4]:
                    fallbacks.append(f"{candidate} open problems")
                    fallbacks.append(f"{candidate} unresolved bottlenecks")
                if len(contract_candidates) >= 2:
                    fallbacks.append(f"{contract_candidates[0]} vs {contract_candidates[1]} {short_topic}")
        if family == "bottleneck_opportunity_discovery":
            fallbacks += [f"{short_topic} limitation bottleneck", f"{short_topic} future work"]
        elif family == "direction_forecasting":
            fallbacks += [f"{short_topic} emerging direction", f"{short_topic} trend evaluation"]
        elif family == "strategic_research_planning":
            fallbacks += [f"{short_topic} open problems", f"{short_topic} future direction priority"]
        elif family == "venue_aware_research_positioning":
            venue = _extract_target_venue(str(task.get("question") or ""))
            venue_bucket = _target_venue_bucket_from_name(venue)
            fallbacks = [
                topic,
                short_topic,
                f"{short_topic} empirical evaluation benchmark analysis",
                f"{short_topic} method ablation package",
                f"{short_topic} contribution framing experimental setup",
            ]
            for candidate in contract_candidates[:4]:
                fallbacks.append(f"{candidate} {short_topic}")
                fallbacks.append(f"{candidate} evaluation benchmark analysis")
                fallbacks.append(f"{candidate} empirical package ablation")
            if venue_bucket in {"aaai", "iclr", "neurips", "icml"}:
                fallbacks.append(f"{short_topic} learning method empirical analysis")
            elif venue_bucket in {"emnlp", "acl", "naacl"}:
                fallbacks.append(f"{short_topic} language methodology evaluation analysis")
            elif venue_bucket:
                fallbacks.append(f"{short_topic} {venue_bucket} style empirical analysis")
        final_queries = dedupe([*queries, *fallbacks])[:8]
        with self._cache_lock:
            self._task_query_cache[cache_key] = list(final_queries)
        return final_queries

    def render_paper_content(self, *, domain_id: str, paper_id: str) -> str:
        cache_key = (str(domain_id or ""), str(paper_id or ""))
        with self._cache_lock:
            if cache_key in self._paper_content_cache:
                return self._paper_content_cache[cache_key]
        disk_key = _cache_key("paper_content_v1", domain_id, paper_id)
        cached_payload = self._read_cached_json("paper_content", disk_key)
        if isinstance(cached_payload, dict):
            cached_content = str(cached_payload.get("content") or "").strip()
            if cached_content:
                with self._cache_lock:
                    self._paper_content_cache[cache_key] = cached_content
                return cached_content
        domain = self.kb.domain(domain_id)
        paper = domain.get_paper(paper_id) or {}
        title = str(paper.get("title") or "")
        abstract = str(paper.get("abstract") or "")
        sections: List[Dict[str, Any]] = []
        if self.fulltext_cache is not None:
            cached = self.fulltext_cache.get_content(paper_id) or self.fulltext_cache.ensure_content(
                paper_id,
                allow_fetch=self.allow_fulltext_fetch,
            )
            if cached:
                sections = list(cached.get("sections") or [])
        if not sections:
            sections = list(domain.sections_by_paper.get(str(paper_id), []))
        parts = [f"Title: {title}", f"Abstract: {abstract}"]
        for idx, section in enumerate(sections[:12], start=1):
            section_title = str(section.get("title") or section.get("section_title") or f"Section {idx}")
            section_text = clip_text(section.get("text") or "", 2200)
            if not normalize_ws(section_text):
                continue
            parts.append(f"Section {idx}: {section_title}\n{section_text}")
        rendered = "\n\n".join(part for part in parts if normalize_ws(part))
        if rendered:
            with self._cache_lock:
                self._paper_content_cache[cache_key] = rendered
            self._write_cached_json(
                "paper_content",
                disk_key,
                {
                    "domain_id": domain_id,
                    "paper_id": paper_id,
                    "content": rendered,
                },
            )
        return rendered

    def extract_paper_profile(self, *, task: Dict[str, Any], domain_id: str, paper_id: str) -> Dict[str, Any]:
        focus_text = extract_focus_text(task)
        focus_hash = hashlib.sha1(focus_text.encode("utf-8")).hexdigest()[:16] if focus_text else "empty"
        cache_key = (str(domain_id or ""), str(paper_id or ""), focus_hash)
        domain = self.kb.domain(domain_id)
        paper = domain.get_paper(paper_id) or {}
        pub = paper.get("publication") or {}
        structure = domain.get_structure(paper_id) or {}
        venue_name = str(pub.get("venue_name") or "").strip()
        venue_bucket = _target_venue_bucket_from_name(venue_name)
        structure_limitations = [
            str((row or {}).get("name") or row or "").strip()
            for row in (structure.get("explicit_limitations") or [])
            if str((row or {}).get("name") or row or "").strip()
        ]
        structure_future_work = [
            str((row or {}).get("direction") or (row or {}).get("name") or row or "").strip()
            for row in (structure.get("future_work") or [])
            if str((row or {}).get("direction") or (row or {}).get("name") or row or "").strip()
        ]
        structure_core_ideas = [
            str((row or {}).get("name") or row or "").strip()
            for row in (structure.get("core_ideas") or [])
            if str((row or {}).get("name") or row or "").strip()
        ]
        problem_statement = str(structure.get("problem_statement") or "").strip()
        with self._cache_lock:
            if cache_key in self._paper_profile_cache:
                cached = dict(self._paper_profile_cache[cache_key])
                if venue_name and not normalize_ws(cached.get("venue") or ""):
                    cached["venue"] = venue_name
                if venue_bucket and not normalize_ws(cached.get("venue_bucket") or ""):
                    cached["venue_bucket"] = venue_bucket
                if pub.get("citation_count") is not None and cached.get("citation_count") is None:
                    cached["citation_count"] = pub.get("citation_count")
                if problem_statement and not normalize_ws(cached.get("problem_statement") or ""):
                    cached["problem_statement"] = problem_statement
                if structure_limitations and not list(cached.get("structure_limitations") or []):
                    cached["structure_limitations"] = list(structure_limitations)
                if structure_future_work and not list(cached.get("structure_future_work") or []):
                    cached["structure_future_work"] = list(structure_future_work)
                if structure_core_ideas and not list(cached.get("structure_core_ideas") or []):
                    cached["structure_core_ideas"] = list(structure_core_ideas)
                return cached
        disk_key = _cache_key("paper_profile_v2", domain_id, paper_id, focus_hash)
        cached_payload = self._read_cached_json("paper_profile", disk_key)
        if isinstance(cached_payload, dict) and str(cached_payload.get("paper_id") or "").strip():
            if venue_name and not normalize_ws(cached_payload.get("venue") or ""):
                cached_payload["venue"] = venue_name
            if venue_bucket and not normalize_ws(cached_payload.get("venue_bucket") or ""):
                cached_payload["venue_bucket"] = venue_bucket
            if pub.get("citation_count") is not None and cached_payload.get("citation_count") is None:
                cached_payload["citation_count"] = pub.get("citation_count")
            if problem_statement and not normalize_ws(cached_payload.get("problem_statement") or ""):
                cached_payload["problem_statement"] = problem_statement
            if structure_limitations and not list(cached_payload.get("structure_limitations") or []):
                cached_payload["structure_limitations"] = list(structure_limitations)
            if structure_future_work and not list(cached_payload.get("structure_future_work") or []):
                cached_payload["structure_future_work"] = list(structure_future_work)
            if structure_core_ideas and not list(cached_payload.get("structure_core_ideas") or []):
                cached_payload["structure_core_ideas"] = list(structure_core_ideas)
            with self._cache_lock:
                self._paper_profile_cache[cache_key] = dict(cached_payload)
            return dict(cached_payload)
        self._log(f"extract_profile paper_id={paper_id} title={str(paper.get('title') or '')[:120]}")
        content = self.render_paper_content(domain_id=domain_id, paper_id=paper_id)
        prompt = get_deep_reference_prompt(content, focus_text)
        response = self._complete(self.cheap_client, prompt, max_tokens=2200, temperature=0.1)
        profile = {
            "paper_id": paper_id,
            "title": str(paper.get("title") or ""),
            "abstract": str(paper.get("abstract") or ""),
            "published_date": str(paper.get("published_date") or paper.get("published") or ""),
            "idea": extract_tag(response, "idea").strip(),
            "experiment": extract_tag(response, "experiment").strip(),
            "entities": extract_tag(response, "entities").strip(),
            "references": parse_query_list(extract_tag(response, "references").strip()),
            "venue": venue_name,
            "venue_bucket": venue_bucket,
            "citation_count": pub.get("citation_count"),
            "problem_statement": problem_statement,
            "structure_limitations": structure_limitations,
            "structure_future_work": structure_future_work,
            "structure_core_ideas": structure_core_ideas,
            "raw": response,
        }
        with self._cache_lock:
            self._paper_profile_cache[cache_key] = dict(profile)
        self._write_cached_json("paper_profile", disk_key, profile)
        return profile

    def _build_chain_bottleneck_signals(
        self,
        *,
        task: Dict[str, Any],
        profiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        limitation_counter: Counter[str] = Counter()
        future_work_counter: Counter[str] = Counter()
        core_idea_counter: Counter[str] = Counter()
        problem_counter: Counter[str] = Counter()
        representative_papers: List[Dict[str, Any]] = []

        for profile in profiles:
            limitations = [
                _clean_topic_label(x)
                for x in (profile.get("structure_limitations") or [])
                if _clean_topic_label(x)
            ]
            future_work = [
                _clean_topic_label(x)
                for x in (profile.get("structure_future_work") or [])
                if _clean_topic_label(x)
            ]
            core_ideas = [
                _clean_topic_label(x)
                for x in (profile.get("structure_core_ideas") or [])
                if _clean_topic_label(x)
            ]
            problem_statement = clip_text(profile.get("problem_statement") or "", 220)
            for label in limitations[:4]:
                limitation_counter[label] += 1
            for label in future_work[:4]:
                future_work_counter[label] += 1
            for label in core_ideas[:4]:
                core_idea_counter[label] += 1
            if normalize_ws(problem_statement):
                problem_counter[normalize_ws(problem_statement)] += 1
            representative_papers.append(
                {
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "published_date": profile.get("published_date"),
                    "problem_statement": problem_statement,
                    "limitations": limitations[:3],
                    "future_work": future_work[:3],
                    "core_ideas": core_ideas[:3],
                }
            )

        recurring_limitations = _signal_rows(limitation_counter)
        future_work_signals = _signal_rows(future_work_counter)
        core_idea_signals = _signal_rows(core_idea_counter)
        recurring_problem_statements = _signal_rows(problem_counter, limit=4)
        representative_papers = representative_papers[:5]

        summary_lines = [
            f"Topic: {extract_focus_text(task) or 'unspecified'}",
            f"Recurring limitation signals: {_render_counter_rows(recurring_limitations) or 'none'}",
            f"Recurring future-work signals: {_render_counter_rows(future_work_signals) or 'none'}",
            f"Related core-idea signals: {_render_counter_rows(core_idea_signals) or 'none'}",
            f"Recurring problem statements: {_render_counter_rows(recurring_problem_statements, limit=2) or 'none'}",
        ]
        if representative_papers:
            rep_text = "; ".join(
                _join_lines(
                    [
                        f"{row.get('title')}",
                        f"problem={row.get('problem_statement') or 'n/a'}",
                        f"limitations={'; '.join(row.get('limitations') or []) or 'none'}",
                        f"future_work={'; '.join(row.get('future_work') or []) or 'none'}",
                    ]
                ).replace("\n", " | ")
                for row in representative_papers[:3]
            )
            summary_lines.append(f"Representative bottleneck papers: {rep_text}")
        return {
            "recurring_limitations": recurring_limitations,
            "future_work_signals": future_work_signals,
            "core_idea_signals": core_idea_signals,
            "recurring_problem_statements": recurring_problem_statements,
            "representative_papers": representative_papers,
            "signal_summary_text": " | ".join(line for line in summary_lines if normalize_ws(line)),
        }

    def _build_chain_venue_signals(
        self,
        *,
        task: Dict[str, Any],
        profiles: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        target_venue = _extract_target_venue(str(task.get("question") or ""))
        target_bucket = _target_venue_bucket_from_name(target_venue)
        compatible_buckets = set(_compatible_venue_buckets(target_bucket))
        chain_venue_counter: Counter[str] = Counter()
        chain_bucket_counter: Counter[str] = Counter()
        package_counter: Counter[str] = Counter()
        evaluation_counter: Counter[str] = Counter()
        compatible_bucket_hits = 0
        mismatched_bucket_hits = 0
        representative_papers: List[Dict[str, Any]] = []

        for profile in profiles:
            venue = normalize_ws(profile.get("venue") or "")
            bucket = normalize_ws(profile.get("venue_bucket") or _target_venue_bucket_from_name(venue))
            title = normalize_ws(profile.get("title") or "")
            text_blob = "\n".join(
                [
                    title,
                    str(profile.get("abstract") or ""),
                    str(profile.get("idea") or ""),
                    str(profile.get("experiment") or ""),
                    str(profile.get("entities") or ""),
                ]
            )
            package_hits = _detect_venue_package_patterns(text_blob)
            evaluation_hits = _detect_venue_evaluation_patterns(text_blob)
            for label in package_hits:
                package_counter[label] += 1
            for label in evaluation_hits:
                evaluation_counter[label] += 1
            if venue:
                chain_venue_counter[venue] += 1
            if bucket:
                chain_bucket_counter[bucket] += 1
                if target_bucket:
                    if bucket in compatible_buckets:
                        compatible_bucket_hits += 1
                    else:
                        mismatched_bucket_hits += 1
            representative_papers.append(
                {
                    "paper_id": profile.get("paper_id"),
                    "title": title,
                    "published_date": profile.get("published_date"),
                    "venue": venue,
                    "venue_bucket": bucket,
                    "matched_target_bucket": bool(target_bucket and bucket in compatible_buckets),
                    "package_patterns": package_hits[:3],
                    "evaluation_patterns": evaluation_hits[:3],
                }
            )

        chain_venue_counts = _counter_rows(chain_venue_counter)
        chain_bucket_counts = _counter_rows(chain_bucket_counter)
        package_pattern_counts = _counter_rows(package_counter)
        evaluation_pattern_counts = _counter_rows(evaluation_counter)
        representative_papers = representative_papers[:5]

        summary_lines = [
            f"Target venue: {target_venue or 'unspecified'}",
            f"Target venue bucket: {target_bucket or 'unknown'}",
            f"Observed venues in chain: {_render_counter_rows(chain_venue_counts) or 'none'}",
            f"Observed venue buckets in chain: {_render_counter_rows(chain_bucket_counts) or 'none'}",
        ]
        if target_bucket:
            summary_lines.append(
                f"Compatible vs mismatched bucket hits: {compatible_bucket_hits} compatible / {mismatched_bucket_hits} mismatched"
            )
        summary_lines.append(
            f"Recurring contribution packaging: {_render_counter_rows(package_pattern_counts) or 'none'}"
        )
        summary_lines.append(
            f"Recurring evaluation signals: {_render_counter_rows(evaluation_pattern_counts) or 'none'}"
        )
        if representative_papers:
            rep_text = "; ".join(
                _join_lines(
                    [
                        f"{row.get('title')} @ {row.get('venue') or 'unknown venue'}",
                        (
                            "target-compatible"
                            if row.get("matched_target_bucket")
                            else (f"bucket={row.get('venue_bucket')}" if row.get("venue_bucket") else "bucket=unknown")
                        ),
                        (
                            "signals="
                            + ",".join(
                                dedupe(
                                    [
                                        *list(row.get("package_patterns") or []),
                                        *list(row.get("evaluation_patterns") or []),
                                    ]
                                )[:3]
                            )
                        ),
                    ]
                ).replace("\n", " | ")
                for row in representative_papers[:3]
            )
            summary_lines.append(f"Representative papers: {rep_text}")
        return {
            "target_venue": target_venue,
            "target_venue_bucket": target_bucket,
            "chain_venue_counts": chain_venue_counts,
            "chain_venue_bucket_counts": chain_bucket_counts,
            "compatible_bucket_hits": compatible_bucket_hits,
            "mismatched_bucket_hits": mismatched_bucket_hits,
            "package_pattern_counts": package_pattern_counts,
            "evaluation_pattern_counts": evaluation_pattern_counts,
            "representative_papers": representative_papers,
            "signal_summary_text": " | ".join(line for line in summary_lines if normalize_ws(line)),
        }

    def judge_relevant(
        self,
        *,
        candidate_title: str,
        candidate_abstract: str,
        topic: str,
        current_title: str = "",
        current_idea: str = "",
        direction: str = "",
    ) -> bool:
        if current_title or current_idea or direction:
            prompt = f"""You are validating one step in a Chain-of-Ideas research trajectory.

Topic:
{topic}

Direction:
{direction or "same-line continuation"}

Current paper:
Title: {current_title}
Idea: {current_idea}

Candidate paper:
Title: {candidate_title}
Abstract: {candidate_abstract}

Return <relevant>1</relevant> only if the candidate is a plausible adjacent step in the same technical line of work.
Reject broad neighboring papers that share the topic but do not continue the same line.
Output only the XML tag.
"""
        else:
            prompt = get_deep_judge_relevant_prompt(candidate_title, candidate_abstract, topic)
        response = self._complete(self.main_client, prompt, max_tokens=300, temperature=0.0)
        return extract_tag(response, "relevant").strip() != "0"

    def _parse_date(self, text: str) -> Optional[datetime]:
        value = str(text or "").strip()
        if not value:
            return None
        candidates = [value[:10], value[:7], value[:4], value]
        for sample, fmt in (
            (candidates[0], "%Y-%m-%d"),
            (candidates[1], "%Y-%m"),
            (candidates[2], "%Y"),
        ):
            try:
                return datetime.strptime(sample, fmt)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(value[:10])
        except Exception:
            return None

    def _paper_pool_features(
        self,
        *,
        paper_id: str,
        candidate_pool: Optional[TaskCandidatePool] = None,
    ) -> Dict[str, Any]:
        packet_match: List[str] = []
        pool_score = {}
        in_pool = False
        if candidate_pool is not None:
            in_pool = paper_id in {str(row.get("paper_id") or "") for row in candidate_pool.papers}
            pool_score = candidate_pool.paper_scores.get(paper_id) or {}
            for packet in candidate_pool.packets:
                packet_name = str(packet.get("display_name") or packet.get("node_id") or "")
                rep_ids = {str(x.get("paper_id") or "") for x in (packet.get("historical_representative_papers") or [])}
                if paper_id in rep_ids:
                    packet_match.append(packet_name)
        return {
            "packet_match": packet_match,
            "pool_bonus": 1.0 if in_pool else 0.0,
            "pool_score": pool_score,
        }

    def _transition_queries(self, *, task: Dict[str, Any], current: Dict[str, Any], direction: str) -> List[str]:
        topic = extract_focus_text(task)
        title = normalize_ws(current.get("title") or "")
        idea = normalize_ws(current.get("idea") or "")
        entities = normalize_ws(current.get("entities") or "")
        refs = [normalize_ws(x) for x in (current.get("references") or []) if normalize_ws(x)]
        queries = [
            f"{topic} {title}",
            f"{topic} {idea}",
            f"{title} {idea}",
            f"{topic} {entities}",
        ]
        if direction == "backward":
            queries = [*refs[:3], *queries]
        else:
            queries = [*queries, *refs[:2]]
        return dedupe([q for q in queries if normalize_ws(q)])[:6]

    def _transition_query_text(self, *, task: Dict[str, Any], current: Dict[str, Any], direction: str) -> str:
        return _join_lines(
            [
                f"Topic: {extract_focus_text(task)}",
                f"Direction: {direction}",
                f"Current title: {current.get('title') or ''}",
                f"Current idea: {current.get('idea') or ''}",
                f"Current entities: {current.get('entities') or ''}",
                "Transition queries: " + " ; ".join(self._transition_queries(task=task, current=current, direction=direction)),
                "Reference hints: " + " ; ".join(str(x) for x in (current.get("references") or [])[:4] if str(x).strip()),
            ]
        )

    def _transition_temporal_score(self, candidate: Dict[str, Any]) -> float:
        value = float(candidate.get("temporal_priority") or -10**9)
        if value <= -(10**8):
            return 0.0
        return 1.0 / (1.0 + abs(value) / 365.0)

    def _prefilter_transition_candidates(
        self,
        *,
        task: Dict[str, Any],
        current: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        direction: str,
        limit: int = COI_TRANSITION_BATCH_SIZE,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        query_text = self._transition_query_text(task=task, current=current, direction=direction)
        candidate_texts = [
            _join_lines(
                [
                    str(cand.get("title") or ""),
                    clip_text(str(cand.get("abstract") or ""), 900),
                    " ".join(str(x) for x in (cand.get("packet_match") or [])[:3]),
                ]
            )
            for cand in candidates
        ]
        embed_vectors = self._embed_texts([query_text, *candidate_texts])
        query_vector = embed_vectors[0] if embed_vectors else None
        embed_scores = [_cosine_similarity(query_vector, vector) for vector in embed_vectors[1:]]
        retrieval_raw: List[float] = []
        query_raw: List[float] = []
        packet_raw: List[float] = []
        temporal_raw: List[float] = []
        keyword_raw: List[float] = []
        for cand in candidates:
            scores = cand.get("scores") or {}
            retrieval_raw.append(float(scores.get("combined_score") or 0.0))
            query_raw.append(min(1.0, float(cand.get("query_hit_count") or 0.0) / 3.0))
            packet_raw.append(min(1.0, float(cand.get("pool_bonus") or 0.0) + 0.35 * min(len(cand.get("packet_match") or []), 2)))
            temporal_raw.append(self._transition_temporal_score(cand))
            keyword_raw.append(_keyword_overlap_score(query_text, _join_lines([cand.get("title") or "", cand.get("abstract") or ""])))
        retrieval_scores = _normalize_scores(retrieval_raw)
        embed_scores_norm = [max(0.0, min(1.0, (score + 1.0) / 2.0)) for score in embed_scores] if any(embed_scores) else [0.0 for _ in candidates]
        scored: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            prefilter_score = (
                0.30 * embed_scores_norm[idx]
                + 0.24 * keyword_raw[idx]
                + 0.18 * retrieval_scores[idx]
                + 0.12 * packet_raw[idx]
                + 0.10 * query_raw[idx]
                + 0.06 * temporal_raw[idx]
            )
            enriched = dict(cand)
            enriched["prefilter"] = {
                "score": round(prefilter_score, 6),
                "embedding_similarity": round(embed_scores[idx], 6) if idx < len(embed_scores) else 0.0,
                "keyword_overlap": round(keyword_raw[idx], 6),
                "retrieval_score": round(retrieval_scores[idx], 6),
                "packet_score": round(packet_raw[idx], 6),
                "query_score": round(query_raw[idx], 6),
                "temporal_score": round(temporal_raw[idx], 6),
            }
            scored.append(enriched)
        scored.sort(
            key=lambda row: (
                float(((row.get("prefilter") or {}).get("score") or 0.0)),
                float(row.get("pool_bonus") or 0.0),
                len(row.get("packet_match") or []),
                int(row.get("query_hit_count") or 0),
                float(((row.get("scores") or {}).get("combined_score") or 0.0)),
            ),
            reverse=True,
        )
        return scored[: max(1, int(limit or COI_TRANSITION_BATCH_SIZE))]

    def _judge_transition_batch(
        self,
        *,
        task: Dict[str, Any],
        current: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        direction: str,
    ) -> Dict[str, Any]:
        if not candidates:
            return {}
        topic = extract_focus_text(task)
        candidate_rows = []
        for cand in candidates:
            prefilter = cand.get("prefilter") or {}
            candidate_rows.append(
                {
                    "paper_id": cand.get("paper_id"),
                    "title": cand.get("title"),
                    "published_date": cand.get("published_date"),
                    "abstract_snippet": clip_text(cand.get("abstract") or "", 520),
                    "packet_match": cand.get("packet_match") or [],
                    "query_hit_count": cand.get("query_hit_count") or 0,
                    "prefilter_hints": {
                        "score": prefilter.get("score"),
                        "keyword_overlap": prefilter.get("keyword_overlap"),
                        "embedding_similarity": prefilter.get("embedding_similarity"),
                    },
                }
            )
        prompt = f"""You are validating one transition step in a Chain-of-Ideas research trajectory.

Topic:
{topic}

Direction:
{direction}

Current paper:
Title: {current.get('title') or ''}
Idea: {current.get('idea') or ''}
Entities: {current.get('entities') or ''}
References: {json.dumps((current.get('references') or [])[:4], ensure_ascii=False)}

Candidate papers:
{json.dumps(candidate_rows, ensure_ascii=False, indent=2)}

Task:
- Pick the single best candidate that is a plausible adjacent step in the same technical line of work.
- Prefer concrete technical continuity over broad topical similarity.
- Use the prefilter hints only as soft signals.
- If none are truly adjacent, leave `selected_paper_id` empty and optionally name the least-bad `backup_paper_id`.

Output JSON only with this schema:
{{
  "selected_paper_id": "paper id or empty string",
  "backup_paper_id": "paper id or empty string",
  "reason": "short explanation"
}}
"""
        try:
            result = complete_json_object(
                self.cheap_client,
                [
                    {"role": "system", "content": "You are a strict research trajectory judge. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=900,
                temperature=0.0,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            if isinstance(result, dict):
                return result
        except Exception:
            return {}
        return {}

    def _select_transition_candidate(
        self,
        *,
        task: Dict[str, Any],
        current: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        direction: str,
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        short_candidates = self._prefilter_transition_candidates(task=task, current=current, candidates=candidates, direction=direction)
        self._log(
            f"judge_{direction}_batch current={str(current.get('paper_id') or '')} "
            f"candidate_count={len(candidates)} shortlisted={len(short_candidates)}"
        )
        judged = self._judge_transition_batch(task=task, current=current, candidates=short_candidates, direction=direction)
        chosen_ids = [
            str(judged.get("selected_paper_id") or "").strip(),
            str(judged.get("backup_paper_id") or "").strip(),
        ]
        for chosen_id in chosen_ids:
            if chosen_id:
                for cand in short_candidates:
                    if str(cand.get("paper_id") or "") == chosen_id:
                        return cand
        heuristic_fallback = None
        for cand in short_candidates:
            if heuristic_fallback is None:
                if (
                    float(cand.get("pool_bonus") or 0.0) > 0.0
                    or len(cand.get("packet_match") or []) > 0
                    or int(cand.get("query_hit_count") or 0) >= 2
                ):
                    heuristic_fallback = cand
        return heuristic_fallback or short_candidates[0]

    def search_anchor_papers(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        queries: List[str],
        candidate_pool: Optional[TaskCandidatePool] = None,
    ) -> List[Dict[str, Any]]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        anchor_map: Dict[str, Dict[str, Any]] = {}
        for query in queries[:6]:
            self._log(f"anchor_query={query[:160]}")
            rows = domain.paper_retriever(cutoff_date=cutoff_date).retrieve(query, top_k=20)
            for rank, (doc, scores) in enumerate(rows, start=1):
                paper = domain.get_paper(doc.paper_id) or {}
                features = self._paper_pool_features(paper_id=doc.paper_id, candidate_pool=candidate_pool)
                existing = anchor_map.get(doc.paper_id)
                record = {
                    "paper_id": doc.paper_id,
                    "title": doc.title,
                    "abstract": str(paper.get("abstract") or ""),
                    "published_date": str(paper.get("published_date") or ""),
                    "scores": dict(scores),
                    "packet_match": features["packet_match"],
                    "pool_bonus": features["pool_bonus"],
                    "query_hit_count": 1,
                    "best_rank": rank,
                }
                if existing is None:
                    anchor_map[doc.paper_id] = record
                else:
                    existing["query_hit_count"] = int(existing.get("query_hit_count") or 0) + 1
                    existing["best_rank"] = min(int(existing.get("best_rank") or rank), rank)
                    existing["scores"].update(scores)
                    existing["packet_match"] = dedupe([*(existing.get("packet_match") or []), *features["packet_match"]])
                    existing["pool_bonus"] = max(float(existing.get("pool_bonus") or 0.0), float(features["pool_bonus"] or 0.0))
        anchors = sorted(
            anchor_map.values(),
            key=lambda row: (
                len(row.get("packet_match") or []),
                int(row.get("query_hit_count") or 0),
                float(row.get("pool_bonus") or 0.0),
                float((row.get("scores") or {}).get("combined_score") or 0.0),
                str(row.get("published_date") or ""),
            ),
            reverse=True,
        )[: self.max_anchor_papers]
        if not anchors and candidate_pool is not None:
            for paper in candidate_pool.papers[: self.max_anchor_papers]:
                paper_id = str(paper.get("paper_id") or "")
                if not paper_id:
                    continue
                anchors.append(
                    {
                        "paper_id": paper_id,
                        "title": str(paper.get("title") or ""),
                        "abstract": str(paper.get("abstract") or ""),
                        "published_date": str(paper.get("published_date") or ""),
                        "scores": candidate_pool.paper_scores.get(paper_id) or {},
                        "packet_match": [],
                        "pool_bonus": 1.0,
                        "query_hit_count": 0,
                        "best_rank": 999,
                    }
                )
        return anchors

    def _search_candidate_papers(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        query: Any,
        seen_paper_ids: Iterable[str],
        current_published_date: str,
        direction: str,
        candidate_pool: Optional[TaskCandidatePool] = None,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        seen = {str(x) for x in seen_paper_ids}
        queries = [str(query)] if isinstance(query, str) else [str(x) for x in (query or []) if str(x).strip()]
        if not queries:
            return []
        rows = merge_multi_query_results(
            domain.paper_retriever(cutoff_date=cutoff_date),
            queries,
            top_k_per_query=max(top_k, 10),
            limit=max(24, top_k * 3),
        )
        out: List[Dict[str, Any]] = []
        current_dt = self._parse_date(current_published_date)
        for doc, scores in rows:
            if doc.paper_id in seen:
                continue
            paper = domain.get_paper(doc.paper_id) or {}
            published_date = str(paper.get("published_date") or "")
            if direction == "forward" and current_published_date and published_date and published_date < current_published_date:
                continue
            if direction == "backward" and current_published_date and published_date and published_date > current_published_date:
                continue
            features = self._paper_pool_features(paper_id=doc.paper_id, candidate_pool=candidate_pool)
            published_dt = self._parse_date(published_date)
            temporal_priority = -10**9
            if current_dt is not None and published_dt is not None:
                delta = (published_dt - current_dt).days
                if direction == "forward" and delta >= 0:
                    temporal_priority = -abs(delta)
                elif direction == "backward" and delta <= 0:
                    temporal_priority = -abs(delta)
            out.append(
                {
                    "paper_id": doc.paper_id,
                    "title": doc.title,
                    "abstract": str(paper.get("abstract") or ""),
                    "published_date": published_date,
                    "scores": scores,
                    "packet_match": features["packet_match"],
                    "pool_bonus": features["pool_bonus"],
                    "query_hit_count": len(scores.get("matched_queries") or []),
                    "temporal_priority": temporal_priority,
                }
            )
        out.sort(
            key=lambda row: (
                float(row.get("pool_bonus") or 0.0),
                len(row.get("packet_match") or []),
                int(row.get("query_hit_count") or 0),
                float(row.get("temporal_priority") or -10**9),
                float((row.get("scores") or {}).get("combined_score") or 0.0),
            ),
            reverse=True,
        )
        return out

    def build_faithful_chain(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        anchor: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool] = None,
    ) -> Dict[str, Any]:
        topic = extract_focus_text(task)
        self._log(f"build_chain anchor={anchor.get('paper_id')} title={str(anchor.get('title') or '')[:120]}")
        seen_paper_ids: List[str] = []
        profiles: List[Dict[str, Any]] = []
        chain_logs: List[Dict[str, Any]] = []

        def append_profile(profile: Dict[str, Any], position: str) -> None:
            seen_paper_ids.append(str(profile.get("paper_id") or ""))
            profiles.append(profile)
            chain_logs.append(
                {
                    "position": position,
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "published_date": profile.get("published_date"),
                }
            )

        anchor_profile = self.extract_paper_profile(task=task, domain_id=domain_id, paper_id=str(anchor.get("paper_id") or ""))
        append_profile(anchor_profile, "anchor")

        current = anchor_profile
        while len(profiles) < self.max_chain_length:
            query = self._transition_queries(task=task, current=current, direction="forward")
            candidates = self._search_candidate_papers(
                task=task,
                domain_id=domain_id,
                query=query,
                seen_paper_ids=seen_paper_ids,
                current_published_date=str(current.get("published_date") or ""),
                direction="forward",
                candidate_pool=candidate_pool,
            )
            picked = self._select_transition_candidate(task=task, current=current, candidates=candidates, direction="forward")
            if picked is None:
                break
            profile = self.extract_paper_profile(task=task, domain_id=domain_id, paper_id=str(picked.get("paper_id") or ""))
            append_profile(profile, "forward")
            current = profile

        current = anchor_profile
        pending_refs = list(current.get("references") or [])
        while len(profiles) < self.max_chain_length:
            query = [*pending_refs[:3], *self._transition_queries(task=task, current=current, direction="backward")]
            candidates = self._search_candidate_papers(
                task=task,
                domain_id=domain_id,
                query=query,
                seen_paper_ids=seen_paper_ids,
                current_published_date=str(current.get("published_date") or ""),
                direction="backward",
                candidate_pool=candidate_pool,
                top_k=6,
            )
            picked = self._select_transition_candidate(task=task, current=current, candidates=candidates, direction="backward")
            if picked is None:
                break
            profile = self.extract_paper_profile(task=task, domain_id=domain_id, paper_id=str(picked.get("paper_id") or ""))
            profiles = [profile] + profiles
            seen_paper_ids = [str(profile.get("paper_id") or "")] + seen_paper_ids
            chain_logs = [
                {
                    "position": "backward",
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "published_date": profile.get("published_date"),
                }
            ] + chain_logs
            current = profile
            pending_refs = list(profile.get("references") or [])

        profiles = sorted(profiles, key=lambda row: str(row.get("published_date") or ""))
        bottleneck_signals = self._build_chain_bottleneck_signals(task=task, profiles=profiles)
        venue_signals = self._build_chain_venue_signals(task=task, profiles=profiles)
        idea_chain_text = ""
        experiments: List[str] = []
        entities: List[str] = []
        years: List[str] = []
        evidence = {"papers": [], "fulltext": []}
        for idx, profile in enumerate(profiles):
            idea_chain_text += f"{idx}.Paper:{profile.get('title')} idea:{profile.get('idea')}\n \n"
            experiments.append(str(profile.get("experiment") or ""))
            entities.append(str(profile.get("entities") or ""))
            years.append(str(profile.get("published_date") or ""))
            evidence["papers"].append(
                {
                    "evidence_id": f"P{idx+1}",
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "published_date": profile.get("published_date"),
                    "venue": profile.get("venue"),
                    "venue_bucket": profile.get("venue_bucket"),
                    "abstract_snippet": clip_text(profile.get("abstract") or "", 700),
                }
            )
            content_text = self.render_paper_content(domain_id=domain_id, paper_id=str(profile.get("paper_id") or ""))
            evidence["fulltext"].append(
                {
                    "evidence_id": f"F{idx+1}",
                    "paper_id": profile.get("paper_id"),
                    "title": profile.get("title"),
                    "section_title": "paper_content",
                    "source_type": "coi_fulltext",
                    "snippet": clip_text(content_text, 1400),
                }
            )
        return {
            "profiles": profiles,
            "idea_chain_text": idea_chain_text.strip(),
            "experiments": experiments,
            "entities": entities,
            "years": years,
            "chain_logs": chain_logs,
            "evidence": evidence,
            "bottleneck_signals": bottleneck_signals,
            "bottleneck_signal_summary": bottleneck_signals.get("signal_summary_text") or "",
            "venue_signals": venue_signals,
            "venue_signal_summary": venue_signals.get("signal_summary_text") or "",
        }

    def summarize_entities(self, *, task: Dict[str, Any], entities: List[str]) -> str:
        normalized_entities = [normalize_ws(x) for x in (entities or []) if normalize_ws(x)]
        cache_key = _cache_key("entity_summary_v1", extract_focus_text(task), normalized_entities)
        with self._cache_lock:
            cached = self._entity_summary_cache.get(cache_key) or ""
        if cached:
            return cached
        disk_cached = self._read_cached_text("entity_summary", cache_key)
        if disk_cached:
            with self._cache_lock:
                self._entity_summary_cache[cache_key] = disk_cached
            return disk_cached
        prompt = f"""The current research topic is: {extract_focus_text(task)}. Please help me summarize and refine the following entities by merging, simplifying, or deleting them : {entities}
Please output strictly in the following format:
<entities>{{cleaned entities}}</entities>
"""
        response = self._complete(self.main_client, prompt, max_tokens=600, temperature=0.1)
        final_text = extract_tag(response, "entities").strip() or " ".join(entities[:4])
        with self._cache_lock:
            self._entity_summary_cache[cache_key] = final_text
        self._write_cached_text("entity_summary", cache_key, final_text, topic=extract_focus_text(task))
        return final_text

    def project_benchmark_answer(
        self,
        *,
        task: Dict[str, Any],
        chain_text: str,
        trend: str,
        future: str,
        human: str,
        evidence: Dict[str, Any],
    ) -> str:
        family = str(task.get("family") or "")
        family_instruction = {
            "direction_forecasting": "State one concrete next-step direction and one trajectory label (accelerating, fragmenting, steady, or cooling).",
            "bottleneck_opportunity_discovery": "State one concrete historical bottleneck and one concrete realized or plausible opportunity opened by addressing it.",
            "strategic_research_planning": "Provide a prioritized 2-4 item research plan for the next six months, with explicit ranking and rationale.",
        }.get(family, "Answer the task concretely.")
        prompt = f"""You are adapting a Chain-of-Ideas research result to an offline benchmark answer.

Task:
{task.get('question')}

CoI chain:
{chain_text}

Trend:
{trend}

Human reasoning:
{human}

Future direction:
{future}

Requirements:
- Use the Chain-of-Ideas reasoning above as the primary basis.
- Do not mention having access to post-cutoff papers.
- Keep the answer concise but technically specific.
- {family_instruction}
- When useful, cite chain evidence inline with ids like [P1] or [F2].

Output only the final answer.
"""
        return self._complete(self.main_client, prompt, max_tokens=1200, temperature=0.15).strip()

    def _anchor_branch_cache_key(self, *, task: Dict[str, Any], domain_id: str, anchor: Dict[str, Any]) -> str:
        return _cache_key(
            "anchor_branch_v1",
            str(task.get("task_id") or ""),
            str(task.get("family") or ""),
            str(domain_id or ""),
            str(task.get("time_cutoff") or ""),
            str(anchor.get("paper_id") or ""),
        )

    def _run_anchor_candidate(
        self,
        *,
        task: Dict[str, Any],
        domain_id: str,
        anchor: Dict[str, Any],
        candidate_pool: Optional[TaskCandidatePool],
    ) -> Dict[str, Any]:
        cache_key = self._anchor_branch_cache_key(task=task, domain_id=domain_id, anchor=anchor)
        cached_payload = self._read_cached_json("anchor_branch", cache_key)
        if isinstance(cached_payload, dict) and isinstance(cached_payload.get("chain"), dict):
            self._log(f"anchor_branch_cache_hit anchor={anchor.get('paper_id')}")
            return cached_payload

        chain = self.build_faithful_chain(task=task, domain_id=domain_id, anchor=anchor, candidate_pool=candidate_pool)
        self._log(f"chain_built anchor={anchor.get('paper_id')} profile_count={len(chain.get('profiles') or [])}")
        entities = self.summarize_entities(task=task, entities=chain.get("entities") or [])
        trend_prompt = get_deep_trend_idea_chains_prompt(chain.get("idea_chain_text") or "", entities, extract_focus_text(task))
        trend_raw = self._complete(self.main_client, trend_prompt, max_tokens=1600, temperature=0.2)
        trend = strip_xmlish_tags(extract_tag(trend_raw, "trend").strip() or trend_raw.strip())
        future_prompt = get_deep_generate_future_direciton_prompt(chain.get("idea_chain_text") or "", trend, extract_focus_text(task), entities)
        future_raw = self._complete(self.main_client, future_prompt, max_tokens=1400, temperature=0.2)
        future = strip_xmlish_tags(extract_tag(future_raw, "future").strip() or future_raw.strip())
        human = strip_xmlish_tags(extract_tag(future_raw, "human").strip())
        projected = self._project_structured_answer(
            task=task,
            chain_text=chain.get("idea_chain_text") or "",
            trend=trend,
            future=future,
            human=human,
            evidence=chain.get("evidence") or {},
            candidate_pool=candidate_pool,
            chain=chain,
        )
        answer = self._verbalize_structured_answer(task=task, projected=projected)
        run_score = self._score_candidate_run(
            task=task,
            candidate_pool=candidate_pool,
            chain=chain,
            projected=projected,
            trend=trend,
            future=future,
        )
        result = {
            "anchor": anchor,
            "chain": chain,
            "entities": entities,
            "trend": trend,
            "future": future,
            "human": human,
            "projected": projected,
            "run_score": run_score,
            "answer": answer,
        }
        self._write_cached_json("anchor_branch", cache_key, result)
        return result

    def gather_evidence(self, *, task: Dict[str, Any], domain_id: str, queries: List[str]) -> Dict[str, Any]:
        domain = self.kb.domain(domain_id)
        cutoff_date = str(task.get("time_cutoff") or "").strip() or None
        agent_queries = [str(x) for x in (queries or []) if str(x).strip()]
        hybrid_queries = build_hybrid_task_queries(task)
        merged_queries = dedupe([*agent_queries, *hybrid_queries])

        paper_hits = merge_retrieval_runs(
            [
                (
                    "agent",
                    merge_multi_query_results(
                        domain.paper_retriever(cutoff_date=cutoff_date),
                        agent_queries,
                        top_k_per_query=8,
                        limit=10,
                    ),
                ),
                (
                    "hybrid_rag",
                    merge_multi_query_results(
                        domain.paper_retriever(cutoff_date=cutoff_date),
                        hybrid_queries,
                        top_k_per_query=8,
                        limit=10,
                    ),
                ),
            ],
            limit=10,
        )
        paper_rows: List[Dict[str, Any]] = []
        paper_ids: List[str] = []
        for idx, (doc, scores) in enumerate(paper_hits, start=1):
            paper = domain.get_paper(doc.paper_id) or {}
            pub = paper.get("publication") or {}
            paper_ids.append(doc.paper_id)
            paper_rows.append(
                {
                    "evidence_id": f"P{idx}",
                    "paper_id": doc.paper_id,
                    "title": doc.title,
                    "published_date": paper.get("published_date"),
                    "venue": pub.get("venue_name"),
                    "citations": pub.get("citation_count"),
                    "abstract_snippet": clip_text((paper.get("abstract") or doc.text), 700),
                    "scores": scores,
                }
            )

        structure_hits = merge_multi_query_results(domain.structure_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids[:10]), merged_queries, top_k_per_query=6, limit=8)
        structures: Dict[str, Dict[str, Any]] = {}
        structure_rows: List[Dict[str, Any]] = []
        for idx, (doc, scores) in enumerate(structure_hits, start=1):
            row = {
                "evidence_id": f"T{idx}",
                "paper_id": doc.paper_id,
                "title": doc.title,
                "problem_statement": clip_text(doc.meta.get("problem_statement"), 260),
                "limitations": list(doc.meta.get("limitations") or [])[:4],
                "future_work": list(doc.meta.get("future_work") or [])[:4],
                "core_ideas": list(doc.meta.get("core_ideas") or [])[:4],
                "snippet": clip_text(doc.text, 900),
                "scores": scores,
            }
            structures[doc.paper_id] = row
            structure_rows.append(row)

        page_hits = merge_multi_query_results(domain.pageindex_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids[:10]), merged_queries, top_k_per_query=6, limit=8)
        page_rows: List[Dict[str, Any]] = []
        for idx, (doc, scores) in enumerate(page_hits, start=1):
            page_rows.append(
                {
                    "evidence_id": f"G{idx}",
                    "paper_id": doc.paper_id,
                    "title": doc.meta.get("paper_title") or doc.title,
                    "section_title": doc.meta.get("section_title"),
                    "kind": doc.meta.get("kind"),
                    "snippet": clip_text(doc.text, 900),
                    "scores": scores,
                }
            )

        fulltext_rows: List[Dict[str, Any]] = []
        if paper_ids:
            builtin_hits = merge_multi_query_results(
                domain.section_retriever(cutoff_date=cutoff_date, paper_ids=paper_ids[:10]),
                merged_queries,
                top_k_per_query=6,
                limit=8,
            )
            for doc, scores in builtin_hits:
                fulltext_rows.append(
                    {
                        "paper_id": doc.paper_id,
                        "title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "level": doc.meta.get("level"),
                        "source_type": "kb_section",
                        "snippet": clip_text(doc.text, 1200),
                        "scores": scores,
                    }
                )

        if self.fulltext_cache is not None and paper_ids:
            fulltext_queries = list(merged_queries)
            family = str(task.get("family") or "")
            if family == "bottleneck_opportunity_discovery":
                fulltext_queries.extend(
                    [
                        f"{extract_focus_text(task)} limitation failure challenge",
                        f"{extract_focus_text(task)} future work open problem metric benchmark",
                    ]
                )
            elif family == "direction_forecasting":
                fulltext_queries.extend(
                    [
                        f"{extract_focus_text(task)} future work next step direction",
                        f"{extract_focus_text(task)} benchmark evaluation limitation",
                    ]
                )
            section_retriever = self.fulltext_cache.build_section_retriever(
                paper_ids=paper_ids[:4],
                cutoff_date=cutoff_date,
                allow_fetch=self.allow_fulltext_fetch,
            )
            fulltext_hits = merge_multi_query_results(section_retriever, fulltext_queries, top_k_per_query=6, limit=10)
            for doc, scores in fulltext_hits:
                fulltext_rows.append(
                    {
                        "paper_id": doc.paper_id,
                        "title": doc.meta.get("paper_title") or doc.title,
                        "section_title": doc.meta.get("section_title"),
                        "level": doc.meta.get("level"),
                        "source_type": doc.meta.get("source_type"),
                        "snippet": clip_text(doc.text, 1200),
                        "scores": scores,
                    }
                )
        deduped_fulltext: List[Dict[str, Any]] = []
        seen_fulltext = set()
        for row in fulltext_rows:
            key = (
                str(row.get("paper_id") or ""),
                str(row.get("section_title") or ""),
                str(row.get("source_type") or ""),
            )
            if key in seen_fulltext:
                continue
            seen_fulltext.add(key)
            deduped_fulltext.append(row)
        fulltext_rows = []
        for idx, row in enumerate(
            sorted(
                deduped_fulltext,
                key=lambda item: float((item.get("scores") or {}).get("combined_score") or 0.0),
                reverse=True,
            )[:10],
            start=1,
        ):
            row = dict(row)
            row["evidence_id"] = f"F{idx}"
            fulltext_rows.append(row)

        return {
            "queries": merged_queries,
            "papers": paper_rows,
            "structures": structure_rows,
            "pageindex": page_rows,
            "fulltext": fulltext_rows,
            "structures_by_paper": structures,
        }

    def build_chain_cards(self, evidence: Dict[str, Any], *, max_papers: int = 4) -> Dict[str, Any]:
        papers = list(evidence.get("papers") or [])
        structures_by_paper = evidence.get("structures_by_paper") or {}
        fulltext_rows = list(evidence.get("fulltext") or [])
        selected = papers[:max_papers]
        selected.sort(key=lambda row: (str(row.get("published_date") or ""), -float((row.get("scores") or {}).get("combined_score") or 0.0)))

        cards: List[Dict[str, Any]] = []
        entity_pool: List[str] = []
        for idx, paper in enumerate(selected):
            structure = structures_by_paper.get(str(paper.get("paper_id") or "")) or {}
            limitations = list(structure.get("limitations") or [])[:3]
            future_work = list(structure.get("future_work") or [])[:3]
            core_ideas = list(structure.get("core_ideas") or [])[:3]
            fulltext_snippets = [
                clip_text(row.get("snippet") or "", 350)
                for row in fulltext_rows
                if str(row.get("paper_id") or "") == str(paper.get("paper_id") or "")
            ][:2]
            entity_pool.extend(limitations)
            entity_pool.extend(future_work)
            entity_pool.extend(core_ideas)
            cards.append(
                {
                    "order": idx,
                    "paper_id": paper.get("paper_id"),
                    "title": paper.get("title"),
                    "published_date": paper.get("published_date"),
                    "venue": paper.get("venue"),
                    "citations": paper.get("citations"),
                    "abstract_snippet": paper.get("abstract_snippet"),
                    "problem_statement": structure.get("problem_statement"),
                    "limitations": limitations,
                    "future_work": future_work,
                    "core_ideas": core_ideas,
                    "fulltext_snippets": fulltext_snippets,
                }
            )
        card_text = []
        for idx, card in enumerate(cards):
            card_text.append(
                _join_lines(
                    [
                        f"Paper {idx}",
                        f"Title: {card.get('title')}",
                        f"Published: {card.get('published_date')}",
                        f"Venue: {card.get('venue')}",
                        f"Citations: {card.get('citations')}",
                        f"Abstract: {card.get('abstract_snippet')}",
                        f"Problem: {card.get('problem_statement')}",
                        f"Core ideas: {'; '.join(card.get('core_ideas') or [])}",
                        f"Limitations: {'; '.join(card.get('limitations') or [])}",
                        f"Future work: {'; '.join(card.get('future_work') or [])}",
                        f"Fulltext highlights: {' || '.join(card.get('fulltext_snippets') or [])}",
                    ]
                )
            )
        return {
            "cards": cards,
            "card_text": "\n\n".join(card_text).strip(),
            "entities": dedupe(entity_pool)[:16],
        }

    def generate_trend(self, *, task: Dict[str, Any], chain: Dict[str, Any]) -> str:
        topic = extract_focus_text(task)
        prompt = get_deep_trend_idea_chains_prompt(chain.get("card_text") or "", chain.get("entities") or [], topic)
        try:
            response = self._complete(self.main_client, prompt, max_tokens=1200, temperature=0.2)
            trend = extract_tag(response, "trend").strip()
            if trend:
                return trend
        except Exception:
            response = ""
        fallback_prompt = f"""Summarize the historical progression of this topic using the paper chain below.

Topic: {topic}

Paper chain:
{chain.get('card_text')}

Write 3-5 concise sentences describing how the research moved from earlier work to later work. Focus on technical progression, not generic history.
"""
        try:
            return self._complete(self.main_client, fallback_prompt, max_tokens=700, temperature=0.2).strip()
        except Exception:
            return ""

    def generate_future(self, *, task: Dict[str, Any], chain: Dict[str, Any], trend: str) -> Dict[str, str]:
        topic = extract_focus_text(task)
        base_prompt = get_deep_generate_future_direciton_prompt(chain.get("card_text") or "", trend, topic, chain.get("entities") or [])
        prompt = f"""{base_prompt}

Return JSON only:
{{
  "human_reasoning": "2-4 concise sentences explaining why the direction follows from the historical chain",
  "future_direction": "one concrete next-step direction phrase plus one concise explanatory sentence"
}}
"""
        try:
            obj = complete_json_object(
                self.main_client,
                [
                    {"role": "system", "content": "You are a precise research agent. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=900,
                temperature=0.2,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            result = {
                "human_reasoning": str(obj.get("human_reasoning") or "").strip(),
                "future_direction": str(obj.get("future_direction") or "").strip(),
                "raw": json.dumps(obj, ensure_ascii=False),
            }
            if result["human_reasoning"] or result["future_direction"]:
                return result
        except Exception:
            response = ""
        fallback_prompt = f"""Infer one plausible next-step future research direction from the historical chain below.

Topic: {topic}

Historical chain:
{chain.get('card_text')}

Trend summary:
{trend}

Return JSON only:
{{
  "human_reasoning": "2-4 concise sentences about how the future direction follows from the prior chain",
  "future_direction": "one concrete direction phrase and a short explanation"
}}
"""
        try:
            obj = complete_json_object(
                self.main_client,
                [
                    {"role": "system", "content": "You are a precise research agent. Output JSON only."},
                    {"role": "user", "content": fallback_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=700,
                temperature=0.2,
                timeout=180,
                transport_retries=2,
                max_parse_attempts=3,
            )
            return {
                "human_reasoning": str(obj.get("human_reasoning") or "").strip(),
                "future_direction": str(obj.get("future_direction") or "").strip(),
                "raw": json.dumps(obj, ensure_ascii=False),
            }
        except Exception:
            return {"human_reasoning": "", "future_direction": "", "raw": response}

    def _render_evidence_block(self, evidence: Dict[str, Any]) -> str:
        parts: List[str] = []
        for row in (evidence.get("papers") or [])[:6]:
            parts.append(_join_lines([f"[{row['evidence_id']}] {row.get('title')}", f"Published: {row.get('published_date')}", f"Venue: {row.get('venue')}", f"Citations: {row.get('citations')}", row.get("abstract_snippet")]))
            parts.append("")
        for row in (evidence.get("structures") or [])[:5]:
            parts.append(_join_lines([f"[{row['evidence_id']}] {row.get('title')}", f"Problem: {row.get('problem_statement')}", f"Limitations: {'; '.join(row.get('limitations') or [])}", f"Future work: {'; '.join(row.get('future_work') or [])}", row.get("snippet")]))
            parts.append("")
        for row in (evidence.get("pageindex") or [])[:4]:
            parts.append(_join_lines([f"[{row['evidence_id']}] {row.get('title')} / {row.get('section_title')}", f"Kind: {row.get('kind')}", row.get("snippet")]))
            parts.append("")
        for row in (evidence.get("fulltext") or [])[:6]:
            parts.append(
                _join_lines(
                    [
                        f"[{row['evidence_id']}] {row.get('title')} / {row.get('section_title')}",
                        f"Source: {row.get('source_type')}",
                        row.get("snippet"),
                    ]
                )
            )
            parts.append("")
        return "\n".join(parts).strip()

    def draft_answer(self, *, task: Dict[str, Any], chain: Dict[str, Any], trend: str, future: Dict[str, str], evidence: Dict[str, Any]) -> str:
        family = str(task.get("family") or "")
        evidence_block = self._render_evidence_block(evidence)
        family_instruction = {
            "direction_forecasting": "State one concrete next-step direction and one trajectory label (accelerating, fragmenting, steady, or cooling).",
            "bottleneck_opportunity_discovery": "State one historically grounded bottleneck and one concrete opportunity that would open if it were addressed.",
            "strategic_research_planning": "Prioritize 2-4 directions for the next six months and justify the ranking.",
            "venue_aware_research_positioning": "State which direction is best positioned for the target venue, what contribution package fits that venue, and what evidence/evaluation package makes the fit credible.",
        }.get(family, "Answer the task concretely.")
        prompt = f"""You are CoI-Agent-Offline, an offline adaptation of Chain-of-Ideas Agent for benchmark answering.

Task metadata:
- Task ID: {task.get('task_id')}
- Family: {family}
- Domain: {task.get('domain')}
- Time cutoff: {task.get('time_cutoff')}
- Title: {task.get('title')}

Question:
{task.get('question')}

CoI-style research chain:
{chain.get('card_text')}

Inferred research trend:
{trend}

Inferred future direction:
{future.get('future_direction')}

Human-style reasoning trace:
{future.get('human_reasoning')}

Historical evidence (cutoff-safe only):
{evidence_block}

Requirements:
- Use only the historical evidence above.
- Do not claim direct access to post-cutoff papers.
- {family_instruction}
- When the evidence contains explicit limitation, problem, or future-work phrases, preserve that wording where possible instead of paraphrasing it away.
- Prefer concrete technical language over generic trend language.
- Cite evidence inline with ids like [P1], [T2], [G1] when useful.
- Keep the answer concise but substantive.
"""
        return self._complete(self.main_client, prompt, max_tokens=1200, temperature=0.2).strip()

    def critique_answer(self, *, task: Dict[str, Any], answer: str, trend: str, future: Dict[str, str], evidence: Dict[str, Any]) -> str:
        evidence_block = self._render_evidence_block(evidence)
        prompt = f"""You are the review agent in a Chain-of-Ideas style benchmark system.

Question:
{task.get('question')}

Current draft answer:
{answer}

Trend summary:
{trend}

Future direction summary:
{future.get('future_direction')}

Evidence:
{evidence_block}

Give a short critique focused on:
1. missing evidence grounding
2. vague or unsupported claims
3. task-family mismatch
4. weak prioritization / weak bottleneck-opportunity linkage / weak trajectory call

Output 3-5 concise bullet points only.
"""
        return self._complete(self.cheap_client, prompt, max_tokens=600, temperature=0.1).strip()

    def revise_answer(self, *, task: Dict[str, Any], draft: str, critique: str, evidence: Dict[str, Any]) -> str:
        evidence_block = self._render_evidence_block(evidence)
        prompt = f"""Revise the benchmark answer using the critique.

Question:
{task.get('question')}

Draft answer:
{draft}

Critique:
{critique}

Evidence:
{evidence_block}

Constraints:
- Keep only claims supported by historical evidence.
- Make the answer more specific and task-aligned.
- Keep inline evidence ids when useful.
- Output only the revised answer.
"""
        return self._complete(self.main_client, prompt, max_tokens=1200, temperature=0.15).strip()

    def run_task(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        return self.run_task_faithful(task=task, domain_id=domain_id)

    def run_task_faithful(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        self._log(f"run_task task_id={task.get('task_id')} family={task.get('family')} domain_id={domain_id}")
        family = str(task.get("family") or "")
        queries = self.build_queries(task)
        self._log(f"queries={queries}")
        candidate_pool = self.retrieval_adaptor.build_candidate_pool(task=task, domain_id=domain_id)
        self._log(
            f"candidate_pool packets={candidate_pool.packet_ids} paper_count={len(candidate_pool.papers)} "
            f"paper_ids={[str(row.get('paper_id')) for row in candidate_pool.papers[:8]]}"
        )
        anchors = self.search_anchor_papers(task=task, domain_id=domain_id, queries=queries, candidate_pool=candidate_pool)
        anchors = sorted(
            anchors,
            key=lambda row: self._family_anchor_score(task=task, anchor=row, candidate_pool=candidate_pool),
            reverse=True,
        )
        anchors = anchors[: max(1, self._candidate_anchor_count(task))]
        self._log(f"anchor_count={len(anchors)}")
        if not anchors:
            return {
                "answer": "",
                "queries": queries,
                "anchors": [],
                "chain": {},
                "trend": "",
                "future": {"future_direction": "", "human_reasoning": "", "raw": ""},
                "draft_answer": "",
                "critique": "",
                "retrieval_mode": "coi_offline_faithful",
                "evidence": {"papers": [], "fulltext": [], "structures": [], "pageindex": []},
                "diagnostics": {"retrieved_papers": 0, "retrieved_fulltext_sections": 0, "reflection_steps": 0, "revision_rounds": 0, "answer_changed_after_revision": False},
            }

        candidate_runs = []
        anchor_workers = min(
            len(anchors),
            max(1, int(os.environ.get("RTL_COI_ANCHOR_WORKERS", str(COI_DEFAULT_ANCHOR_WORKERS)) or COI_DEFAULT_ANCHOR_WORKERS)),
        )
        if anchor_workers <= 1 or len(anchors) <= 1:
            for anchor in anchors:
                result = self._run_anchor_candidate(task=task, domain_id=domain_id, anchor=anchor, candidate_pool=candidate_pool)
                self._log(
                    f"candidate_done anchor={anchor.get('paper_id')} profile_count={len((result.get('chain') or {}).get('profiles') or [])} "
                    f"trend_chars={len(str(result.get('trend') or ''))} future_chars={len(str(result.get('future') or ''))} "
                    f"answer_chars={len(str(result.get('answer') or ''))} run_score={result.get('run_score')}"
                )
                candidate_runs.append(result)
        else:
            self._log(f"anchor_parallel workers={anchor_workers} anchor_count={len(anchors)}")
            with ThreadPoolExecutor(max_workers=anchor_workers, thread_name_prefix="coi-anchor") as executor:
                future_map = {
                    executor.submit(
                        self._run_anchor_candidate,
                        task=task,
                        domain_id=domain_id,
                        anchor=anchor,
                        candidate_pool=candidate_pool,
                    ): anchor
                    for anchor in anchors
                }
                for future in as_completed(future_map):
                    anchor = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self._log(f"anchor_failed anchor={anchor.get('paper_id')} error={exc}")
                        continue
                    self._log(
                        f"candidate_done anchor={anchor.get('paper_id')} profile_count={len((result.get('chain') or {}).get('profiles') or [])} "
                        f"trend_chars={len(str(result.get('trend') or ''))} future_chars={len(str(result.get('future') or ''))} "
                        f"answer_chars={len(str(result.get('answer') or ''))} run_score={result.get('run_score')}"
                    )
                    candidate_runs.append(result)
        if not candidate_runs:
            return {
                "answer": "",
                "queries": queries,
                "anchors": anchors,
                "chain": {},
                "trend": "",
                "future": {"future_direction": "", "human_reasoning": "", "raw": ""},
                "draft_answer": "",
                "critique": "",
                "retrieval_mode": "coi_offline_faithful",
                "evidence": {"papers": [], "fulltext": [], "structures": [], "pageindex": []},
                "diagnostics": {"retrieved_papers": 0, "retrieved_fulltext_sections": 0, "reflection_steps": 0, "revision_rounds": 0, "answer_changed_after_revision": False},
            }
        candidate_runs.sort(
            key=lambda item: (
                float(item.get("run_score") or 0.0),
                len(item["chain"].get("profiles") or []),
                len(normalize_ws(item.get("trend") or "")),
            ),
            reverse=True,
        )
        if family == "direction_forecasting":
            final_answer, family_head, evidence = self._run_direction_head(
                task=task,
                domain_id=domain_id,
                candidate_pool=candidate_pool,
                candidate_runs=candidate_runs,
            )
        elif family == "strategic_research_planning":
            final_answer, family_head, evidence = self._run_planning_head(
                task=task,
                domain_id=domain_id,
                candidate_pool=candidate_pool,
                candidate_runs=candidate_runs,
            )
        elif family == "venue_aware_research_positioning":
            final_answer, family_head, evidence = self._run_venue_head(
                task=task,
                domain_id=domain_id,
                candidate_pool=candidate_pool,
                candidate_runs=candidate_runs,
            )
        else:
            final_answer, family_head, evidence = self._run_bottleneck_head(
                task=task,
                domain_id=domain_id,
                candidate_pool=candidate_pool,
                candidate_runs=candidate_runs,
            )
        best = candidate_runs[0]
        evidence = dict(evidence or {})
        evidence.setdefault("structures", [])
        evidence.setdefault("pageindex", [])
        evidence.setdefault("candidate_node_evidence", [])
        raw_result = {
            "answer": final_answer or best["answer"],
            "queries": queries,
            "anchors": anchors,
            "chain": {
                "idea_chain_text": best["chain"].get("idea_chain_text"),
                "profiles": best["chain"].get("profiles"),
                "chain_logs": best["chain"].get("chain_logs"),
                "entities": best["entities"],
                "experiments": best["chain"].get("experiments"),
                "years": best["chain"].get("years"),
                "bottleneck_signals": best["chain"].get("bottleneck_signals"),
                "bottleneck_signal_summary": best["chain"].get("bottleneck_signal_summary"),
                "venue_signals": best["chain"].get("venue_signals"),
                "venue_signal_summary": best["chain"].get("venue_signal_summary"),
            },
            "candidate_chains": [
                {
                    "anchor": run.get("anchor"),
                    "run_score": run.get("run_score"),
                    "trend": run.get("trend"),
                    "future": run.get("future"),
                    "answer": run.get("answer"),
                    "chain_logs": (run.get("chain") or {}).get("chain_logs"),
                    "profiles": (run.get("chain") or {}).get("profiles"),
                    "bottleneck_signals": (run.get("chain") or {}).get("bottleneck_signals"),
                    "bottleneck_signal_summary": (run.get("chain") or {}).get("bottleneck_signal_summary"),
                    "venue_signals": (run.get("chain") or {}).get("venue_signals"),
                    "venue_signal_summary": (run.get("chain") or {}).get("venue_signal_summary"),
                }
                for run in candidate_runs[: self._candidate_anchor_count(task)]
            ],
            "trend": best["trend"],
            "future": {"future_direction": best["future"], "human_reasoning": best["human"], "raw": ""},
            "draft_answer": self._verbalize_structured_answer(task=task, projected={"final": best.get("projected", {}).get("draft"), "candidate_labels": best.get("projected", {}).get("candidate_labels") or []}),
            "critique": json.dumps(family_head or best.get("projected", {}).get("final") or {}, ensure_ascii=False),
            "family_head": family_head,
            "retrieval_mode": "coi_offline_faithful_v3: query -> family_packet_pool -> family_anchor_selection -> multi_chain_coi_backbone -> family_analysis_head -> shared_family_final_renderer_v2",
            "evidence": evidence,
            "diagnostics": {
                "selected_packet_ids": candidate_pool.packet_ids,
                "candidate_pool_papers": [str(row.get("paper_id") or "") for row in candidate_pool.papers[:12]],
                "retrieved_papers": len(evidence.get("papers") or []),
                "retrieved_structures": len(evidence.get("structures") or []),
                "retrieved_pageindex": len(evidence.get("pageindex") or []),
                "retrieved_fulltext_sections": len(evidence.get("fulltext") or []),
                "reflection_steps": 2,
                "revision_rounds": 1,
                "answer_changed_after_revision": normalize_ws(final_answer or best["answer"]) != normalize_ws(self._verbalize_structured_answer(task=task, projected={"final": best.get("projected", {}).get("draft"), "candidate_labels": best.get("projected", {}).get("candidate_labels") or []})),
                "candidate_run_score": float(best.get("run_score") or 0.0),
                "candidate_run_count": len(candidate_runs),
            },
        }
        return apply_shared_final_adapter_to_trace_result(
            self.main_client,
            public_task=task,
            trace_result=raw_result,
        )

    def run_task_linear_chain(self, *, task: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        queries = self.build_queries(task)
        evidence = self.gather_evidence(task=task, domain_id=domain_id, queries=queries)
        chain = self.build_chain_cards(evidence)
        trend = self.generate_trend(task=task, chain=chain)
        future = self.generate_future(task=task, chain=chain, trend=trend)
        draft = self.draft_answer(task=task, chain=chain, trend=trend, future=future, evidence=evidence)
        critique = self.critique_answer(task=task, answer=draft, trend=trend, future=future, evidence=evidence)
        final_answer = self.revise_answer(task=task, draft=draft, critique=critique, evidence=evidence)
        return {
            "answer": final_answer,
            "queries": queries,
            "evidence": evidence,
            "chain": chain,
            "trend": trend,
            "future": future,
            "draft_answer": draft,
            "critique": critique,
            "retrieval_mode": "coi_query_generation -> offline_hybrid_retrieval -> chain_cards -> trend_future_reasoning -> review_revision",
            "diagnostics": {
                "retrieved_papers": len(evidence.get("papers") or []),
                "retrieved_structures": len(evidence.get("structures") or []),
                "retrieved_pageindex": len(evidence.get("pageindex") or []),
                "retrieved_fulltext_sections": len(evidence.get("fulltext") or []),
                "reflection_steps": 1,
                "revision_rounds": 1,
                "answer_changed_after_revision": final_answer.strip() != draft.strip(),
            },
        }
