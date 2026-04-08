from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.experiment_eval_v3_1 import summarize_results_v3_1, write_main_table_csv_v3_1


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description='Parallel wrapper for evaluate_experiment_run_v3_1.py')
    parser.add_argument('--results-jsonl', required=True)
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--history-kb-dir', required=True)
    parser.add_argument('--future-kb-dir', required=True)
    parser.add_argument('--judge-llm-config', default='configs/llm/qwen_235b.local.yaml')
    parser.add_argument('--judge-fallback-llm-config', default='configs/llm/mimo_pro.local.yaml')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--run-id', default='')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    results_path = Path(args.results_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_root = output_dir / 'chunks'
    log_root = output_dir / 'logs'
    chunk_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(results_path))
    worker_rows: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, args.workers))]
    for idx, row in enumerate(rows):
        worker_rows[idx % len(worker_rows)].append(row)

    processes: List[subprocess.Popen[str]] = []
    for worker_idx, chunk in enumerate(worker_rows):
        chunk_file = chunk_root / f'chunk_{worker_idx:02d}.jsonl'
        _write_jsonl(chunk_file, chunk)
        worker_out = output_dir / f'worker_{worker_idx:02d}'
        if worker_out.exists() and not args.resume:
            shutil.rmtree(worker_out)
        worker_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / 'scripts' / 'evaluate_experiment_run_v3_1.py'),
            '--results-jsonl', str(chunk_file),
            '--release-dir', args.release_dir,
            '--history-kb-dir', args.history_kb_dir,
            '--future-kb-dir', args.future_kb_dir,
            '--judge-llm-config', args.judge_llm_config,
            '--judge-fallback-llm-config', args.judge_fallback_llm_config,
            '--output-dir', str(worker_out),
            '--run-id', f"{args.run_id or output_dir.name}_w{worker_idx:02d}",
        ]
        if args.resume:
            cmd.append('--resume')
        log_path = log_root / f'worker_{worker_idx:02d}.log'
        log_f = log_path.open('a' if args.resume else 'w', encoding='utf-8')
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, text=True)
        proc._log_f = log_f  # type: ignore[attr-defined]
        processes.append(proc)
        print(f'[parallel-eval-v3.1] launched worker={worker_idx} tasks={len(chunk)} pid={proc.pid}', flush=True)

    failures = []
    for worker_idx, proc in enumerate(processes):
        code = proc.wait()
        proc._log_f.close()  # type: ignore[attr-defined]
        print(f'[parallel-eval-v3.1] worker={worker_idx} exit={code}', flush=True)
        if code != 0:
            failures.append(worker_idx)

    if failures:
        raise SystemExit(f'workers failed: {failures}')

    merged_by_task: Dict[str, Dict[str, Any]] = {}
    for worker_idx in range(len(worker_rows)):
        worker_out = output_dir / f'worker_{worker_idx:02d}' / 'results_eval_v3_1.jsonl'
        if not worker_out.exists():
            continue
        for row in _load_jsonl(worker_out):
            task_id = str(row.get('task_id') or '')
            if task_id:
                merged_by_task[task_id] = row

    merged_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for row in rows:
        task_id = str(row.get('task_id') or '')
        if task_id in merged_by_task:
            merged_rows.append(merged_by_task[task_id])
        else:
            missing.append(task_id)

    if missing:
        raise SystemExit(f'missing merged eval rows for {len(missing)} tasks, first={missing[:5]}')

    merged_path = output_dir / 'results_eval_v3_1.jsonl'
    _write_jsonl(merged_path, merged_rows)
    summary = summarize_results_v3_1(merged_rows)
    summary.update({
        'run_id': args.run_id.strip() or output_dir.name,
        'input_results_jsonl': str(results_path),
        'output_results_jsonl': str(merged_path),
        'workers': len(worker_rows),
    })
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_main_table_csv_v3_1(merged_rows, output_dir / 'main_results.csv')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
