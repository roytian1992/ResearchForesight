from __future__ import annotations

import re
from typing import Iterable, List


TOKEN_MAP = {
    "llm": "LLM",
    "llms": "LLMs",
    "rag": "RAG",
    "qa": "QA",
    "cot": "chain-of-thought",
    "rl": "reinforcement learning",
    "rft": "reinforcement fine-tuning",
    "rlhf": "RLHF",
    "dpo": "DPO",
    "ppo": "PPO",
    "grpo": "GRPO",
    "sft": "supervised fine-tuning",
    "peft": "parameter-efficient fine-tuning",
    "lora": "LoRA",
    "api": "API",
    "gui": "GUI",
    "ui": "UI",
    "multimodal": "multimodal",
}

DROP_TOKENS = {
    "general",
    "general_purpose",
}

MULTISPACE_RE = re.compile(r"\s+")


def snake_to_text(value: str) -> str:
    text = str(value or "").strip().strip("/")
    if not text:
        return ""
    text = text.split("/")[-1]
    tokens = [tok for tok in text.split("_") if tok]
    out: List[str] = []
    for tok in tokens:
        lowered = tok.lower()
        if lowered in DROP_TOKENS:
            continue
        out.append(TOKEN_MAP.get(lowered, tok))
    text = " ".join(out)
    text = text.replace(" - ", "-")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_display_name(display_name: str) -> str:
    text = str(display_name or "").strip()
    if not text:
        return ""
    if "_" in text or "/" in text:
        text = snake_to_text(text)
    return MULTISPACE_RE.sub(" ", text).strip()


def title_case_phrase(text: str) -> str:
    raw = normalize_display_name(text)
    if not raw:
        return raw
    words = []
    for word in raw.split():
        if word.isupper() or any(ch.isupper() for ch in word[1:]):
            words.append(word)
        elif word.lower() in {"for", "and", "of", "in", "with", "to", "on"}:
            words.append(word.lower())
        else:
            words.append(word.lower())
    phrase = " ".join(words).strip()
    if phrase:
        phrase = phrase[0].upper() + phrase[1:]
    return phrase


def sentence_case_phrase(text: str) -> str:
    phrase = title_case_phrase(text)
    if phrase:
        return phrase[0].lower() + phrase[1:]
    return phrase


def public_topic_from_packet(packet: dict) -> str:
    display = normalize_display_name(packet.get("display_name") or "")
    description = str(packet.get("description") or "").strip()
    if display and display.lower() not in {"other", "unknown"}:
        return sentence_case_phrase(display)
    if description:
        return description.rstrip(".")
    node_id = str(packet.get("node_id") or "")
    return sentence_case_phrase(snake_to_text(node_id))


def public_descendant_names(rows: Iterable[dict], limit: int = 4) -> List[str]:
    out: List[str] = []
    seen = set()
    for row in rows:
        name = normalize_display_name(row.get("display_name") or row.get("node_id") or "")
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sentence_case_phrase(name))
        if len(out) >= limit:
            break
    return out


def clean_surface_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("_", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()
