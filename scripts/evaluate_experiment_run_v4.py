from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.experiment_eval_v4 import (
    build_experiment_result_row_v4,
    evaluate_evidence_traceability_v4,
    summarize_results_v4,
    write_dimension_csv_v4,
    write_main_table_csv_v4,
)
from researchworld.llm import FallbackOpenAICompatChatClient, OpenAICompatChatClient, load_openai_compat_config
from researchworld.refined_release import load_release_public_by_id


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate an experiment run under benchmark v4.')
    parser.add_argument('--results-jsonl', required=True)
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--judge-llm-config', default='configs/llm/qwen_235b.local.yaml')
    parser.add_argument('--judge-fallback-llm-config', default='configs/llm/mimo_pro.local.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-id', default='')
    parser.add_argument('--task-limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    release_dir = Path(args.release_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    public_by_id = load_release_public_by_id(release_dir)
    judge_primary = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))
    judge_fallback = None
    fallback_path = Path(args.judge_fallback_llm_config) if str(args.judge_fallback_llm_config or '').strip() else None
    if fallback_path and fallback_path.exists():
        judge_fallback = OpenAICompatChatClient(load_openai_compat_config(fallback_path))
    judge_client = FallbackOpenAICompatChatClient(judge_primary, judge_fallback)
    run_id = args.run_id.strip() or results_path.parent.name or results_path.stem

    result_rows = list(iter_jsonl(results_path))
    if args.task_limit is not None:
        result_rows = result_rows[: args.task_limit]
    missing_task_ids = [
        str(row.get('task_id') or '')
        for row in result_rows
        if str(row.get('task_id') or '') not in public_by_id
    ]
    if missing_task_ids:
        raise SystemExit(f"results contain task ids not present in release public data: count={len(missing_task_ids)} first={missing_task_ids[:5]}")

    out_jsonl = output_dir / 'results_eval_v4.jsonl'
    outputs = []
    completed_task_ids = set()
    if args.resume and out_jsonl.exists():
        with out_jsonl.open('r', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                outputs.append(obj)
                task_id = str(obj.get('task_id') or '').strip()
                if task_id:
                    completed_task_ids.add(task_id)
    if completed_task_ids:
        result_rows = [row for row in result_rows if str(row.get('task_id') or '') not in completed_task_ids]

    write_mode = 'a' if args.resume and out_jsonl.exists() else 'w'
    with out_jsonl.open(write_mode, encoding='utf-8') as handle:
        for idx, row in enumerate(result_rows, start=1):
            task_id = str(row.get('task_id') or '')
            public_task = public_by_id.get(task_id)
            family = str(public_task.get('family') or row.get('family') or '')
            answer = str(row.get('answer') or '')
            print(f"[eval_v4] {idx}/{len(result_rows)} {task_id} family={family} method={row.get('agent') or row.get('baseline')}", flush=True)
            evidence_traceability_eval = evaluate_evidence_traceability_v4(
                judge_client,
                public_task=public_task,
                family=family,
                candidate_answer=answer,
                result_row=row,
            )
            out_row = build_experiment_result_row_v4(
                run_id=run_id,
                public_task=public_task,
                result_row=row,
                evidence_traceability_eval=evidence_traceability_eval,
            )
            outputs.append(out_row)
            handle.write(json.dumps(out_row, ensure_ascii=False) + '\n')
            handle.flush()

    summary = summarize_results_v4(outputs)
    summary.update({
        'run_id': run_id,
        'input_results_jsonl': str(results_path),
        'output_results_jsonl': str(out_jsonl),
    })
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_main_table_csv_v4(outputs, output_dir / 'main_results.csv')
    write_dimension_csv_v4(outputs, output_dir / 'dimension_breakdown.csv')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
