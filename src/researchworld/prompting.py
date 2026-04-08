from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml


_VAR_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _declared_names(items: Any) -> Set[str]:
    names: Set[str] = set()
    if not isinstance(items, list):
        return names
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def _required_names(items: Any) -> Set[str]:
    names: Set[str] = set()
    if not isinstance(items, list):
        return names
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name.strip() and item.get("required") is True:
            names.add(name.strip())
    return names


def _safe_replace(template: str, declared: Set[str], values: Dict[str, Any]) -> str:
    string_values = {key: _stringify(val) for key, val in values.items() if key in declared}

    def repl(match: re.Match) -> str:
        name = match.group(1)
        if name not in declared:
            return match.group(0)
        return string_values.get(name, "")

    return _VAR_PATTERN.sub(repl, template)


@dataclass
class PromptSpec:
    id: str
    template: str
    task_variables: List[Dict[str, Any]]
    static_variables: List[Dict[str, Any]]
    raw: Dict[str, Any]


class YAMLPromptLoader:
    def __init__(self, prompt_dir: str | Path, global_static: Optional[Dict[str, Any]] = None):
        self.prompt_dir = Path(prompt_dir)
        if not self.prompt_dir.exists():
            raise FileNotFoundError(f"Prompt dir not found: {self.prompt_dir}")
        self.global_static = global_static or {}

    def load(self, prompt_id: str) -> PromptSpec:
        rel = prompt_id.strip()
        if not rel:
            raise ValueError("prompt_id must be non-empty")
        candidates = [rel]
        if not rel.endswith(".yaml") and not rel.endswith(".yml"):
            candidates = [f"{rel}.yaml", f"{rel}.yml"]
        for candidate in candidates:
            path = self.prompt_dir / candidate
            if path.exists() and path.is_file():
                with open(path, "r", encoding="utf-8") as handle:
                    obj = yaml.safe_load(handle) or {}
                if not isinstance(obj, dict):
                    raise ValueError(f"Prompt YAML must be a mapping: {path}")
                template = obj.get("template")
                if not isinstance(template, str) or not template.strip():
                    raise ValueError(f"Prompt missing template: {path}")
                return PromptSpec(
                    id=str(obj.get("id") or path.stem),
                    template=template,
                    task_variables=obj.get("task_variables") or [],
                    static_variables=obj.get("static_variables") or [],
                    raw=obj,
                )
        raise FileNotFoundError(f"Prompt not found for id={prompt_id} under {self.prompt_dir}")

    def render(
        self,
        prompt_id: Union[str, PromptSpec],
        *,
        task_values: Optional[Dict[str, Any]] = None,
        static_values: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> str:
        spec = self.load(prompt_id) if isinstance(prompt_id, str) else prompt_id
        task_values = task_values or {}
        static_values = static_values or {}

        declared = _declared_names(spec.task_variables) | _declared_names(spec.static_variables)
        required = _required_names(spec.task_variables) | _required_names(spec.static_variables)
        merged = dict(self.global_static)
        merged.update(static_values)
        merged.update(task_values)

        if strict:
            missing = sorted(name for name in required if name not in merged)
            if missing:
                raise ValueError(f"Missing prompt variables: {missing}")

        return _safe_replace(spec.template, declared=declared, values=merged)
