from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.baseline_runner import PUBLIC_DOMAIN_TO_ID, load_release_tasks
from researchworld.llm import FallbackOpenAICompatChatClient, OpenAICompatChatClient, load_openai_compat_config
from researchworld.offline_kb import OfflineKnowledgeBase
from researchworld.researchagent_offline import ResearchAgentOffline, safe_run_task


def main() -> None:
    parser = argparse.ArgumentParser(description='Run ResearchAgent-Offline on ResearchInsightBenchmark tasks.')
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--kb-dir', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--reasoning-llm-config', default='../ResearchTrajectoryLab/configs/llm/qwen_235b.local.yaml')
    parser.add_argument('--render-llm-config', default='../ResearchTrajectoryLab/configs/llm/qwen_235b.local.yaml')
    parser.add_argument('--fallback-llm-config', default='../ResearchTrajectoryLab/configs/llm/mimo_pro.local.yaml')
    parser.add_argument('--task-limit', type=int, default=None)
    parser.add_argument('--domains', nargs='*', default=None)
    parser.add_argument('--families', nargs='*', default=None)
    parser.add_argument('--task-ids-file', default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--pipeline-style', default='aggressive', choices=['aggressive', 'lite'])
    parser.add_argument('--render-passes', type=int, default=1)
    args = parser.parse_args()

    release_dir = Path(args.release_dir)
    kb_dir = Path(args.kb_dir) if args.kb_dir else (release_dir / 'kb')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fallback_path = Path(args.fallback_llm_config) if str(args.fallback_llm_config or '').strip() else None
    fallback_client = OpenAICompatChatClient(load_openai_compat_config(fallback_path)) if (fallback_path and fallback_path.exists()) else None
    reasoning_client = FallbackOpenAICompatChatClient(
        OpenAICompatChatClient(load_openai_compat_config(Path(args.reasoning_llm_config))),
        fallback_client,
    )
    render_client = FallbackOpenAICompatChatClient(
        OpenAICompatChatClient(load_openai_compat_config(Path(args.render_llm_config))),
        fallback_client,
    )

    agent = ResearchAgentOffline(
        kb=OfflineKnowledgeBase(kb_dir),
        reasoning_client=reasoning_client,
        render_client=render_client,
        iterations=args.iterations,
        pipeline_style=args.pipeline_style,
        render_passes=args.render_passes,
    )

    tasks = load_release_tasks(release_dir)
    allowed_task_ids = None
    if args.task_ids_file:
        allowed_task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding='utf-8').splitlines()
            if line.strip()
        }

    domain_filter = set(args.domains) if args.domains else None
    family_filter = set(args.families) if args.families else None
    rows = []
    for task in tasks:
        task_id = str(task.get('task_id') or '')
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        domain_id = PUBLIC_DOMAIN_TO_ID.get(str(task.get('domain') or '').strip())
        if not domain_id:
            continue
        if domain_filter and domain_id not in domain_filter:
            continue
        if family_filter and str(task.get('family') or '') not in family_filter:
            continue
        rows.append((task, domain_id))
    if args.task_limit is not None:
        rows = rows[: args.task_limit]

    results_path = out_dir / 'results.jsonl'
    outputs = []
    completed_task_ids = set()
    if args.resume and results_path.exists():
        with results_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                outputs.append(row)
                task_id = str(row.get('task_id') or '').strip()
                if task_id:
                    completed_task_ids.add(task_id)
        if completed_task_ids:
            rows = [(task, domain_id) for task, domain_id in rows if str(task.get('task_id') or '') not in completed_task_ids]

    mode = 'a' if args.resume and results_path.exists() else 'w'
    with results_path.open(mode, encoding='utf-8') as handle:
        for idx, (task, domain_id) in enumerate(rows, start=1):
            print(f"[ResearchAgent-Offline] {idx}/{len(rows)} {task['task_id']} domain={domain_id} family={task['family']}", flush=True)
            result = safe_run_task(agent, task=task, domain_id=domain_id)
            row = {
                'task_id': task['task_id'],
                'family': task['family'],
                'domain': task['domain'],
                'domain_id': domain_id,
                'agent': 'ResearchAgent-Offline',
                'title': task['title'],
                'question': task['question'],
                'time_cutoff': task['time_cutoff'],
                'answer': result.get('answer', ''),
                'trace': result,
            }
            if str(result.get('raw_answer') or '').strip():
                row['raw_answer'] = result.get('raw_answer')
            if isinstance(result.get('adapter'), dict) and result.get('adapter'):
                row['adapter'] = result.get('adapter')
            outputs.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()

    summary = {
        'agent': 'ResearchAgent-Offline',
        'task_count': len(outputs),
        'results_path': str(results_path),
    }
    with (out_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
