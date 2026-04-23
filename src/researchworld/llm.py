from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from json_repair import repair_json as _repair_json_with_pkg
except Exception:  # pragma: no cover
    _repair_json_with_pkg = None


@dataclass
class OpenAICompatConfig:
    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120


@dataclass
class OpenAICompatEmbeddingConfig:
    model_name: str
    api_key: str
    base_url: str
    timeout: int = 120
    dimensions: Optional[int] = None


def _load_openai_compat_section(path: str | Path, section_name: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle) or {}
    section = obj.get(section_name) or {}
    if not isinstance(section, dict):
        raise ValueError(f"Invalid {section_name} section in {path}")
    return section


def load_openai_compat_config(path: str | Path) -> OpenAICompatConfig:
    def expand(value: Any) -> str:
        return os.path.expandvars(str(value or ""))
    llm = _load_openai_compat_section(path, "llm")
    return OpenAICompatConfig(
        model_name=expand(llm.get("model_name")),
        api_key=expand(llm.get("api_key")),
        base_url=expand(llm.get("base_url")).rstrip("/"),
        temperature=float(llm.get("temperature", 0.0)),
        max_tokens=int(llm.get("max_tokens", 4096)),
        timeout=int(llm.get("timeout", 120)),
    )


def load_openai_compat_embedding_config(path: str | Path) -> OpenAICompatEmbeddingConfig:
    def expand(value: Any) -> str:
        return os.path.expandvars(str(value or ""))
    embedding = _load_openai_compat_section(path, "embedding")
    dimensions_raw = embedding.get("dimensions")
    return OpenAICompatEmbeddingConfig(
        model_name=expand(embedding.get("model_name")),
        api_key=expand(embedding.get("api_key")),
        base_url=expand(embedding.get("base_url")).rstrip("/"),
        timeout=int(embedding.get("timeout", 120)),
        dimensions=(int(dimensions_raw) if dimensions_raw not in (None, "") else None),
    )


class OpenAICompatChatClient:
    def __init__(self, config: OpenAICompatConfig):
        self.config = config

    def chat(self, messages: List[Dict[str, str]], **overrides: Any) -> Dict[str, Any]:
        payload = {
            "model": overrides.get("model", self.config.model_name),
            "messages": messages,
            "temperature": overrides.get("temperature", self.config.temperature),
            "max_tokens": overrides.get("max_tokens", self.config.max_tokens),
        }
        if "response_format" in overrides:
            payload["response_format"] = overrides["response_format"]

        request = urllib.request.Request(
            url=f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )

        max_attempts = int(overrides.pop("transport_retries", 1))
        retry_delay = float(overrides.pop("retry_delay", 1.5))
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            try:
                with urllib.request.urlopen(request, timeout=overrides.get("timeout", self.config.timeout)) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"LLM HTTPError {exc.code}: {body}")
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"LLM URLError: {exc}")
            if attempt + 1 < max_attempts:
                time.sleep(retry_delay * (attempt + 1))
        assert last_error is not None
        raise last_error

    def complete_text(self, messages: List[Dict[str, str]], **overrides: Any) -> str:
        response = self.chat(messages, **overrides)
        try:
            content = response["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(str(text))
                    elif isinstance(item, str):
                        parts.append(item)
                return "".join(parts)
            return str(content or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {response}") from exc


class OpenAICompatEmbeddingClient:
    def __init__(self, config: OpenAICompatEmbeddingConfig):
        self.config = config

    def embed(self, inputs: List[str], **overrides: Any) -> List[List[float]]:
        max_attempts = int(overrides.pop("transport_retries", 1))
        retry_delay = float(overrides.pop("retry_delay", 1.5))
        model_name = overrides.get("model", self.config.model_name)
        dimensions = overrides.get("dimensions", self.config.dimensions)
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            dimension_candidates = [dimensions]
            if dimensions is not None:
                dimension_candidates.append(None)
            for dim in dimension_candidates:
                try:
                    payload: Dict[str, Any] = {
                        "model": model_name,
                        "input": inputs,
                    }
                    if dim is not None:
                        payload["dimensions"] = int(dim)
                    request = urllib.request.Request(
                        url=f"{self.config.base_url}/embeddings",
                        data=json.dumps(payload).encode("utf-8"),
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {self.config.api_key}",
                        },
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=overrides.get("timeout", self.config.timeout)) as response:
                        obj = json.loads(response.read().decode("utf-8"))
                    data = obj.get("data")
                    if not isinstance(data, list):
                        raise RuntimeError(f"Unexpected embedding response shape: {obj}")
                    vectors: List[List[float]] = []
                    for row in data:
                        if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                            raise RuntimeError(f"Unexpected embedding item shape: {row}")
                        vectors.append([float(x) for x in row["embedding"]])
                    if len(vectors) != len(inputs):
                        raise RuntimeError(f"Embedding result count mismatch: expected {len(inputs)} got {len(vectors)}")
                    return vectors
                except urllib.error.HTTPError as exc:
                    body = exc.read().decode("utf-8", errors="replace")
                    if dim is not None and "does not support matryoshka representation" in body.lower():
                        last_error = RuntimeError("Embedding endpoint rejected dimensions; retrying without dimensions.")
                        continue
                    last_error = RuntimeError(f"Embedding HTTPError {exc.code}: {body}")
                    break
                except urllib.error.URLError as exc:
                    last_error = RuntimeError(f"Embedding URLError: {exc}")
                    break
                except Exception as exc:
                    last_error = exc
                    break
            if attempt + 1 < max_attempts:
                time.sleep(retry_delay * (attempt + 1))
        assert last_error is not None
        raise last_error


class FallbackOpenAICompatChatClient:
    def __init__(
        self,
        primary: OpenAICompatChatClient,
        fallback: Optional[OpenAICompatChatClient] = None,
        *,
        fallback_on: tuple[type[BaseException], ...] = (
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            RuntimeError,
        ),
    ):
        self.primary = primary
        self.fallback = fallback
        self.fallback_on = fallback_on
        self.config = primary.config

    def _should_fallback(self, exc: BaseException) -> bool:
        if not isinstance(exc, self.fallback_on):
            return False
        text = str(exc).lower()
        fallback_markers = [
            "timed out",
            "timeout",
            "rate limit",
            "429",
            "connection reset",
            "remote end closed connection",
            "temporarily unavailable",
            "bad gateway",
            "502",
            "503",
            "504",
            "urlerror",
            "httperror",
        ]
        return any(marker in text for marker in fallback_markers) or isinstance(exc, (TimeoutError, urllib.error.HTTPError, urllib.error.URLError))

    def chat(self, messages: List[Dict[str, str]], **overrides: Any) -> Dict[str, Any]:
        try:
            return self.primary.chat(messages, **overrides)
        except Exception as exc:
            if self.fallback is None or not self._should_fallback(exc):
                raise
            return self.fallback.chat(messages, **overrides)

    def complete_text(self, messages: List[Dict[str, str]], **overrides: Any) -> str:
        response = self.chat(messages, **overrides)
        try:
            content = response["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(str(text))
                    elif isinstance(item, str):
                        parts.append(item)
                return "".join(parts)
            return str(content or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {response}") from exc


CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def _balance_truncated_json_dict(text: str) -> str:
    candidate = text.rstrip()
    if not candidate.startswith("{"):
        return text
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    if in_string:
        candidate += '"'
    while stack:
        candidate += stack.pop()
    return candidate


def _repair_common_json_issues(text: str) -> str:
    candidate = text
    # Fix malformed object keys like: " "rationale": ...
    candidate = re.sub(r'"\s+"([A-Za-z0-9_\-]+)"\s*:', r'"\1":', candidate)
    # Remove trailing commas before object/array close.
    candidate = re.sub(r",(\s*[}\]])", r"\1", candidate)
    return candidate


def extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("LLM returned empty text")

    fenced = CODE_FENCE_RE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for start in range(len(stripped)):
        if stripped[start] != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    repaired = _balance_truncated_json_dict(stripped[stripped.find("{") :]) if "{" in stripped else stripped
    repaired = _repair_common_json_issues(repaired)
    if repaired:
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    if _repair_json_with_pkg is not None:
        try:
            repaired_pkg = _repair_json_with_pkg(stripped, return_objects=False)
            if repaired_pkg:
                obj = json.loads(repaired_pkg)
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass
    raise ValueError(f"Could not find a JSON object in LLM output: {text[:400]}")


def complete_json_object(
    client: OpenAICompatChatClient,
    messages: List[Dict[str, str]],
    *,
    max_parse_attempts: int = 3,
    repair_instruction: str | None = None,
    **overrides: Any,
) -> Dict[str, Any]:
    convo = [dict(m) for m in messages]
    last_error: Exception | None = None
    last_text = ""
    base_instruction = repair_instruction or (
        "Your previous response was not valid JSON. Return exactly one valid JSON object only. "
        "Do not wrap it in markdown. Do not add commentary. Preserve the intended schema."
    )
    for attempt in range(max_parse_attempts):
        last_text = client.complete_text(convo, **overrides)
        try:
            return extract_json_object(last_text)
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= max_parse_attempts:
                break
            convo = convo + [
                {"role": "assistant", "content": last_text},
                {"role": "user", "content": base_instruction},
            ]
    assert last_error is not None
    raise ValueError(f"{last_error}; last_text={last_text[:400]}")
