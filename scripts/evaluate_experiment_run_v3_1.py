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
from researchworld.experiment_eval_v3 import infer_domain_id
from researchworld.experiment_eval_v3_1 import (
    build_experiment_result_row_v3_1,
    evaluate_component_task_fulfillment_v3_1,
    evaluate_strategic_intelligence_v3_1,
    summarize_results_v3_1,
    write_main_table_csv_v3_1,
)
from researchworld.factscore_eval_v3 import FactScoreV3Config, evaluate_answer_factscore_v3
from researchworld.future_alignment_eval_v3_1 import FutureAlignmentV3_1Config, evaluate_future_alignment_v3_1
from researchworld.llm import FallbackOpenAICompatChatClient, OpenAICompatChatClient, load_openai_compat_config
from researchworld.offline_kb import OfflineKnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate an experiment run under benchmark v3.1.')
    parser.add_argument('--results-jsonl', required=True)
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--history-kb-dir', required=True)
    parser.add_argument('--future-kb-dir', required=True)
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

    public_by_id = {row['task_id']: row for row in iter_jsonl(release_dir / 'tasks.jsonl')}
    hidden_by_id = {row['task_id']: row for row in iter_jsonl(release_dir / 'tasks_hidden_eval_v3_1.jsonl')}
    history_kb = OfflineKnowledgeBase(Path(args.history_kb_dir))
    future_kb = OfflineKnowledgeBase(Path(args.future_kb_dir))
    judge_primary = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_llm_config)))
    judge_fallback = None
    if str(args.judge_fallback_llm_config or '').strip():
        judge_fallback = OpenAICompatChatClient(load_openai_compat_config(Path(args.judge_fallback_llm_config)))
    judge_client = FallbackOpenAICompatChatClient(judge_primary, judge_fallback)
    fact_cfg = FactScoreV3Config()
    future_cfg = FutureAlignmentV3_1Config()
    run_id = args.run_id.strip() or results_path.parent.name or results_path.stem

    result_rows = list(iter_jsonl(results_path))
    if args.task_limit is not None:
        result_rows = result_rows[: args.task_limit]

    out_jsonl = output_dir / 'results_eval_v3_1.jsonl'
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
            hidden_row = hidden_by_id.get(task_id)
            if not public_task or not hidden_row:
                continue
            row = dict(row)
            row['domain_id'] = infer_domain_id(row)
            print(f"[eval_v3_1] {idx}/{len(result_rows)} {task_id} domain={row['domain_id']} family={row.get('family')}", flush=True)
            fact_eval = evaluate_answer_factscore_v3(
                history_kb=history_kb,
                future_kb=future_kb,
                judge_client=judge_client,
                result_row=row,
                gt_row=hidden_row,
                cfg=fact_cfg,
            )
            task_fulfillment_eval = evaluate_component_task_fulfillment_v3_1(
                judge_client,
                public_task=public_task,
                hidden_row=hidden_row,
                candidate_answer=str(row.get('answer') or ''),
            )
            strategic_eval = evaluate_strategic_intelligence_v3_1(
                judge_client,
                public_task=public_task,
                hidden_row=hidden_row,
                candidate_answer=str(row.get('answer') or ''),
            )
            future_alignment_eval = evaluate_future_alignment_v3_1(
                future_kb=future_kb,
                judge_client=judge_client,
                public_task=public_task,
                result_row=row,
                hidden_row=hidden_row,
                cfg=future_cfg,
            )
            out_row = build_experiment_result_row_v3_1(
                run_id=run_id,
                public_task=public_task,
                hidden_row=hidden_row,
                result_row=row,
                fact_eval=fact_eval,
                task_fulfillment_eval=task_fulfillment_eval,
                strategic_eval=strategic_eval,
                future_alignment_eval=future_alignment_eval,
            )
            outputs.append(out_row)
            handle.write(json.dumps(out_row, ensure_ascii=False) + '\n')
            handle.flush()

    summary = summarize_results_v3_1(outputs)
    summary.update({
        'run_id': run_id,
        'input_results_jsonl': str(results_path),
        'output_results_jsonl': str(out_jsonl),
    })
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_main_table_csv_v3_1(outputs, output_dir / 'main_results.csv')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
