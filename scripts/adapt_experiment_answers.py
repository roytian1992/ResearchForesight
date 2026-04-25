from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.answer_adapter import apply_shared_final_adapter, shared_adapter_name
from researchworld.corpus import iter_jsonl
from researchworld.llm import FallbackOpenAICompatChatClient, OpenAICompatChatClient, load_openai_compat_config
from researchworld.refined_release import load_task_refined_public_by_id


def build_client(primary_config: Path, fallback_config: str) -> FallbackOpenAICompatChatClient:
    primary = OpenAICompatChatClient(load_openai_compat_config(primary_config))
    fallback = None
    if str(fallback_config or "").strip():
        fallback_path = Path(fallback_config)
        if fallback_path.exists():
            fallback = OpenAICompatChatClient(load_openai_compat_config(fallback_path))
    return FallbackOpenAICompatChatClient(primary, fallback)


def method_label(row: dict) -> str:
    return str(
        row.get("method_key")
        or row.get("method_name")
        or row.get("agent")
        or row.get("baseline")
        or "unknown"
    )


def adapt_one(
    row: dict,
    *,
    public_task: dict,
    primary_config: Path,
    fallback_config: str,
) -> dict:
    client = build_client(primary_config, fallback_config)
    return apply_shared_final_adapter(client, public_task=public_task, result_row=row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply shared final adapter to an experiment result file.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--adapter-llm-config", default="configs/llm/qwen3_235b_8002.local.yaml")
    parser.add_argument("--adapter-fallback-llm-config", default="")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    release_dir = Path(args.release_dir)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    public_by_id = load_task_refined_public_by_id(release_dir)

    primary_config = Path(args.adapter_llm_config)
    fallback_config = str(args.adapter_fallback_llm_config or "")

    rows = list(iter_jsonl(results_path))
    if args.task_limit is not None:
        rows = rows[: args.task_limit]

    completed = {}
    if args.resume and output_path.exists():
        for row in iter_jsonl(output_path):
            task_id = str(row.get("task_id") or "").strip()
            if task_id:
                completed[task_id] = row

    release_task_ids = set(public_by_id)
    result_task_ids = [str(row.get("task_id") or "").strip() for row in rows]
    bad_task_ids = [task_id or "<missing-task-id>" for task_id in result_task_ids if not task_id or task_id not in release_task_ids]
    if bad_task_ids:
        raise SystemExit(f"results contain task IDs not present in release: count={len(bad_task_ids)} first={bad_task_ids[:5]}")
    remaining = [row for row in rows if str(row.get("task_id") or "").strip() not in completed]
    write_mode = "a" if args.resume and output_path.exists() else "w"

    with output_path.open(write_mode, encoding="utf-8") as handle:
        if args.workers <= 1:
            client = build_client(primary_config, fallback_config)
            for idx, row in enumerate(remaining, start=1):
                task_id = str(row.get("task_id") or "")
                public_task = public_by_id.get(task_id)
                if public_task is None:
                    continue
                print(f"[adapter] {idx}/{len(remaining)} {task_id} family={row.get('family')} method={method_label(row)}", flush=True)
                out_row = apply_shared_final_adapter(client, public_task=public_task, result_row=row)
                handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                handle.flush()
        else:
            jobs = []
            for idx, row in enumerate(remaining, start=1):
                task_id = str(row.get("task_id") or "")
                public_task = public_by_id.get(task_id)
                if public_task is None:
                    continue
                jobs.append((idx, row, public_task))
                print(f"[adapter-queue] {idx}/{len(remaining)} {task_id} family={row.get('family')} method={method_label(row)}", flush=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                future_map = {
                    executor.submit(
                        adapt_one,
                        row,
                        public_task=public_task,
                        primary_config=primary_config,
                        fallback_config=fallback_config,
                    ): (idx, row)
                    for idx, row, public_task in jobs
                }
                for future in concurrent.futures.as_completed(future_map):
                    idx, row = future_map[future]
                    task_id = str(row.get("task_id") or "")
                    out_row = future.result()
                    print(f"[adapter-done] {idx}/{len(remaining)} {task_id} family={row.get('family')} method={method_label(row)}", flush=True)
                    handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    handle.flush()

    summary = {
        "input_results_jsonl": str(results_path),
        "output_results_jsonl": str(output_path),
        "task_count": len(list(iter_jsonl(output_path))),
        "adapter": shared_adapter_name(),
        "workers": args.workers,
    }
    (output_path.parent / "adapter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
