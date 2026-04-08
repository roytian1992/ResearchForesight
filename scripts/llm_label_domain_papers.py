from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.analytics import TAXONOMY_KEYS
from researchworld.config import load_yaml
from researchworld.corpus import iter_jsonl
from researchworld.domains import load_domain_config, resolve_domain_bundle
from researchworld.llm import OpenAICompatChatClient, complete_json_object, load_openai_compat_config
from researchworld.prompting import YAMLPromptLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM scope screening and taxonomy annotation for a target domain.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "benchmark.yaml"), help="Benchmark config YAML.")
    parser.add_argument("--domain-config", required=True, help="Domain pipeline config YAML.")
    parser.add_argument("--input", default="", help="Override candidate-paper JSONL.")
    parser.add_argument("--output", default="", help="Override label JSONL output.")
    parser.add_argument("--error-output", default="", help="Override error JSONL output.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of papers to annotate. 0 means all.")
    parser.add_argument("--resume", action="store_true", help="Skip paper_ids already present in the output file.")
    parser.add_argument("--workers", type=int, default=0, help="Concurrent LLM workers. 0 uses domain config default.")
    parser.add_argument("--max-retries", type=int, default=-1, help="Retries per paper after the first failure.")
    parser.add_argument("--candidate-tier", default="", choices=["", "any", "core_candidate", "review_candidate"], help="Optional candidate-tier filter.")
    parser.add_argument("--progress-every", type=int, default=20, help="Print progress every N completed papers.")
    return parser.parse_args()


def format_taxonomy_guide(path: Path) -> str:
    obj = load_yaml(path)
    dimensions = obj.get("dimensions") or {}
    lines: List[str] = []
    for dim_name, dim_obj in dimensions.items():
        lines.append(f"[{dim_name}] {dim_obj.get('description', '')}".strip())
        labels = dim_obj.get("labels") or {}
        for label_id, label_obj in labels.items():
            lines.append(f"- {label_id}: {label_obj.get('description', '')}".strip())
        lines.append("")
    return "\n".join(lines).strip()


def allowed_labels_from_taxonomy(path: Path) -> Dict[str, Set[str]]:
    obj = load_yaml(path)
    dims = obj.get("dimensions") or {}
    return {key: set((dims.get(key) or {}).get("labels", {}).keys()) for key in TAXONOMY_KEYS}


def load_seen_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    seen = set()
    for row in iter_jsonl(path):
        paper_id = row.get("paper_id")
        if isinstance(paper_id, str) and paper_id:
            seen.add(paper_id)
    return seen


def validate_label(obj: Dict, allowed_labels: Dict[str, Set[str]]) -> None:
    required = {"paper_id", "scope_decision", "confidence", "reasoning", "evidence_phrases", "taxonomy"}
    missing = sorted(required - set(obj.keys()))
    if missing:
        raise ValueError(f"Missing keys: {missing}")
    if obj["scope_decision"] not in {"core_domain", "adjacent", "out_of_scope"}:
        raise ValueError(f"Invalid scope_decision: {obj['scope_decision']}")
    if not isinstance(obj["confidence"], (int, float)):
        raise ValueError("confidence must be numeric")
    taxonomy = obj.get("taxonomy")
    if not isinstance(taxonomy, dict):
        raise ValueError("taxonomy must be an object")
    for key in TAXONOMY_KEYS:
        if key not in taxonomy:
            raise ValueError(f"taxonomy missing {key}")
        values = taxonomy[key]
        if not isinstance(values, list):
            raise ValueError(f"taxonomy[{key}] must be a list")
        invalid = sorted(v for v in values if v not in allowed_labels[key])
        if invalid:
            raise ValueError(f"Invalid labels for {key}: {invalid}")
    if obj["scope_decision"] == "core_domain":
        if not taxonomy["task_settings"]:
            raise ValueError("core_domain paper must have at least one task_settings label")
        allows_empty_method_modules = "benchmark_or_evaluation_suite" in set(taxonomy["task_settings"] or [])
        if not taxonomy["method_modules"] and not allows_empty_method_modules:
            raise ValueError("core_domain paper must have at least one method_modules label")


def sanitize_label(obj: Dict, paper_id: str, allowed_labels: Dict[str, Set[str]]) -> Dict:
    cleaned = dict(obj)
    cleaned["paper_id"] = paper_id
    scope_decision = str(cleaned.get("scope_decision") or "").strip()
    scope_aliases = {
        "core_agent": "core_domain",
        "core_paper": "core_domain",
        "in_scope": "core_domain",
    }
    cleaned["scope_decision"] = scope_aliases.get(scope_decision, scope_decision)
    confidence = cleaned.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    cleaned["confidence"] = max(0.0, min(1.0, float(confidence)))

    taxonomy = cleaned.get("taxonomy")
    if not isinstance(taxonomy, dict):
        taxonomy = {}
    taxonomy_aliases = {
        "task_settings": {
            "test_time_scaling_integration": "test_time_scaling_with_post_training",
        },
    }
    normalized_taxonomy = {}
    for key in TAXONOMY_KEYS:
        raw_values = taxonomy.get(key) or []
        if not isinstance(raw_values, list):
            raw_values = []
        seen = set()
        valid_values = []
        for value in raw_values:
            if not isinstance(value, str):
                continue
            value = taxonomy_aliases.get(key, {}).get(value, value)
            if value not in allowed_labels[key] or value in seen:
                continue
            seen.add(value)
            valid_values.append(value)
        normalized_taxonomy[key] = valid_values
    if cleaned.get("scope_decision") != "out_of_scope" and not normalized_taxonomy["reliability_safety"]:
        normalized_taxonomy["reliability_safety"] = ["none_explicit"]
    cleaned["taxonomy"] = normalized_taxonomy
    return cleaned


def prepare_jobs(input_path: Path, *, seen_ids: Set[str], limit: int, candidate_tier: str) -> List[Dict]:
    jobs: List[Dict] = []
    for row in iter_jsonl(input_path):
        paper_id = row.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id or paper_id in seen_ids:
            continue
        if candidate_tier and candidate_tier != "any" and row.get("candidate_tier") != candidate_tier:
            continue
        jobs.append(row)
        if limit and len(jobs) >= limit:
            break
    return jobs


def annotate_one(
    row: Dict,
    *,
    prompt_id: str,
    prompt_loader: YAMLPromptLoader,
    llm: OpenAICompatChatClient,
    prompt_static: Dict[str, str],
    allowed_labels: Dict[str, Set[str]],
    request_timeout: int,
    temperature: float,
    max_retries: int,
) -> Tuple[bool, Dict]:
    paper_id = row["paper_id"]
    prompt = prompt_loader.render(
        prompt_id,
        task_values={
            "paper_id": paper_id,
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "categories": ", ".join(row.get("categories") or []),
            "published": row.get("published", ""),
        },
        static_values=prompt_static,
    )

    raw_text = ""
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            obj = complete_json_object(
                llm,
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=request_timeout,
                max_parse_attempts=3,
            )
            obj = sanitize_label(obj, paper_id, allowed_labels)
            validate_label(obj, allowed_labels)
            return True, obj
        except Exception as exc:
            last_error = str(exc)
            if attempt == max_retries:
                break

    return False, {"paper_id": paper_id, "error": last_error, "raw_text": raw_text}


def main() -> None:
    args = parse_args()
    benchmark_cfg = load_yaml(args.config)
    domain_cfg = load_domain_config(args.domain_config)
    domain = resolve_domain_bundle(ROOT, benchmark_cfg, domain_cfg)
    project = benchmark_cfg.get("project") or {}
    annotation_cfg = domain["annotation"]

    llm_cfg_path = (ROOT / project["llm_config_path"]).resolve()
    input_path = Path(args.input) if args.input else ROOT / annotation_cfg["input_path"]
    output_path = Path(args.output) if args.output else ROOT / annotation_cfg["output_path"]
    error_path = Path(args.error_output) if args.error_output else ROOT / annotation_cfg["error_path"]
    worker_count = args.workers or int(annotation_cfg.get("workers", 1))
    max_retries = args.max_retries if args.max_retries >= 0 else int(annotation_cfg.get("max_retries", 0))
    candidate_tier = args.candidate_tier or str(annotation_cfg.get("candidate_tier", "any"))
    progress_every = max(1, args.progress_every)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    taxonomy_guide = format_taxonomy_guide(domain["taxonomy_path"])
    allowed_labels = allowed_labels_from_taxonomy(domain["taxonomy_path"])
    loader = YAMLPromptLoader(ROOT / "prompts")
    llm = OpenAICompatChatClient(load_openai_compat_config(llm_cfg_path))
    prompt_static = dict(annotation_cfg.get("prompt_static") or {})
    prompt_static["taxonomy_guide"] = taxonomy_guide

    seen_ids = load_seen_ids(output_path) if args.resume else set()
    jobs = prepare_jobs(input_path, seen_ids=seen_ids, limit=args.limit, candidate_tier=candidate_tier)
    total = len(jobs)
    success = 0
    completed = 0
    started_at = time.time()

    with open(output_path, "a", encoding="utf-8") as out_handle, open(error_path, "a", encoding="utf-8") as err_handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, worker_count)) as executor:
            future_to_paper_id = {
                executor.submit(
                    annotate_one,
                    row,
                    prompt_id=domain["prompt_id"],
                    prompt_loader=loader,
                    llm=llm,
                    prompt_static=prompt_static,
                    allowed_labels=allowed_labels,
                    request_timeout=int(annotation_cfg.get("request_timeout", 180)),
                    temperature=float(annotation_cfg.get("temperature", 0.0)),
                    max_retries=max_retries,
                ): row["paper_id"]
                for row in jobs
            }
            for future in concurrent.futures.as_completed(future_to_paper_id):
                completed += 1
                ok, payload = future.result()
                if ok:
                    out_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    success += 1
                else:
                    err_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                out_handle.flush()
                err_handle.flush()
                if completed % progress_every == 0 or completed == total:
                    elapsed = max(time.time() - started_at, 1e-6)
                    rate = completed / elapsed
                    print(
                        f"[{domain['domain_id']}] Progress: {completed}/{total} "
                        f"success={success} errors={completed - success} rate={rate:.2f} papers/s"
                    )

    print(f"Domain: {domain['domain_id']}")
    print(f"Processed: {total}")
    print(f"Succeeded: {success}")
    print(f"Output: {output_path}")
    print(f"Errors: {error_path}")


if __name__ == "__main__":
    main()
