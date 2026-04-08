from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply shared final adapter to an experiment result file.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--adapter-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--adapter-fallback-llm-config", default="configs/llm/qwen_235b.local.yaml")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    release_dir = Path(args.release_dir)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    public_by_id = {row["task_id"]: row for row in iter_jsonl(release_dir / "tasks.jsonl")}

    primary = OpenAICompatChatClient(load_openai_compat_config(Path(args.adapter_llm_config)))
    fallback = None
    if str(args.adapter_fallback_llm_config or "").strip():
        fallback = OpenAICompatChatClient(load_openai_compat_config(Path(args.adapter_fallback_llm_config)))
    client = FallbackOpenAICompatChatClient(primary, fallback)

    rows = list(iter_jsonl(results_path))
    if args.task_limit is not None:
        rows = rows[: args.task_limit]

    completed = {}
    if args.resume and output_path.exists():
        for row in iter_jsonl(output_path):
            task_id = str(row.get("task_id") or "").strip()
            if task_id:
                completed[task_id] = row

    remaining = [row for row in rows if str(row.get("task_id") or "").strip() not in completed]
    write_mode = "a" if args.resume and output_path.exists() else "w"

    with output_path.open(write_mode, encoding="utf-8") as handle:
        for idx, row in enumerate(remaining, start=1):
            task_id = str(row.get("task_id") or "")
            public_task = public_by_id.get(task_id)
            if public_task is None:
                continue
            print(f"[adapter] {idx}/{len(remaining)} {task_id} family={row.get('family')} method={row.get('agent') or row.get('baseline')}", flush=True)
            out_row = apply_shared_final_adapter(client, public_task=public_task, result_row=row)
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            handle.flush()

    summary = {
        "input_results_jsonl": str(results_path),
        "output_results_jsonl": str(output_path),
        "task_count": len(list(iter_jsonl(output_path))),
        "adapter": shared_adapter_name(),
    }
    (output_path.parent / "adapter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
