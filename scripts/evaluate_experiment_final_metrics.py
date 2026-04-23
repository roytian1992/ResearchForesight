from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.corpus import iter_jsonl
from researchworld.experiment_eval_aux import (
    build_aux_result_row,
    evaluate_family_auxiliary,
    summarize_aux_results,
    write_aux_csv,
    write_rubric_breakdown_csv_aux,
)
from researchworld.experiment_eval_v3 import infer_domain_id
from researchworld.experiment_eval_v3_1 import (
    build_experiment_result_row_v3_1,
    summarize_results_v3_1,
    write_breakdown_csv_v3_1,
    write_main_table_csv_v3_1,
)
from researchworld.experiment_eval_v4 import (
    build_experiment_result_row_v4,
    evaluate_evidence_traceability_v4,
    summarize_results_v4,
    write_dimension_csv_v4,
    write_main_table_csv_v4,
)
from researchworld.factscore_eval_v3 import FactScoreV3Config, evaluate_answer_factscore_v3
from researchworld.future_alignment_eval_v3_1 import FutureAlignmentV3_1Config, evaluate_future_alignment_v3_1
from researchworld.llm import FallbackOpenAICompatChatClient, OpenAICompatChatClient, load_openai_compat_config
from researchworld.offline_kb import OfflineKnowledgeBase

METRIC_BUNDLES = ('v31', 'v4', 'aux')

METRIC_SUBDIRS = {
    'v31': 'eval_v31',
    'v4': 'eval_v4',
    'aux': 'eval_aux',
}

METRIC_RESULT_FILENAMES = {
    'v31': 'results_eval_v3_1.jsonl',
    'v4': 'results_eval_v4.jsonl',
    'aux': 'results_eval_aux.jsonl',
}

METRIC_ALIASES = {
    'all': {'v31', 'v4', 'aux'},
    'primary': {'v31', 'v4'},
    'factuality': {'v31'},
    'fact': {'v31'},
    'future_alignment': {'v31'},
    'future': {'v31'},
    'traceability': {'v4'},
    'evidence_traceability': {'v4'},
    'v31': {'v31'},
    'v4': {'v4'},
    'aux': {'aux'},
    'family_aux': {'aux'},
}


def _parse_metrics(raw_values: Sequence[str]) -> tuple[List[str], List[str]]:
    selected: Set[str] = set()
    requested: List[str] = []
    for raw in raw_values:
        parts = [part.strip().lower() for part in str(raw or '').split(',') if part.strip()]
        for part in parts:
            requested.append(part)
            bundles = METRIC_ALIASES.get(part)
            if not bundles:
                valid = ', '.join(sorted(METRIC_ALIASES))
                raise SystemExit(f'unknown metric selection "{part}". valid values: {valid}')
            selected.update(bundles)
    if not selected:
        selected = {'v31', 'v4', 'aux'}
        requested = ['all']
    ordered = [bundle for bundle in METRIC_BUNDLES if bundle in selected]
    return ordered, requested


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def _metric_dir(root: Path, bundle: str) -> Path:
    return root / METRIC_SUBDIRS[bundle]


def _result_path(root: Path, bundle: str) -> Path:
    return _metric_dir(root, bundle) / METRIC_RESULT_FILENAMES[bundle]


def _build_judge_client(primary_config: Path, fallback_config: str) -> FallbackOpenAICompatChatClient:
    judge_primary = OpenAICompatChatClient(load_openai_compat_config(primary_config))
    judge_fallback = None
    if str(fallback_config or '').strip():
        judge_fallback = OpenAICompatChatClient(load_openai_compat_config(Path(fallback_config)))
    return FallbackOpenAICompatChatClient(judge_primary, judge_fallback)


def _task_id(row: Dict[str, Any]) -> str:
    return str(row.get('task_id') or '').strip()


def _load_completed_task_ids(path: Path) -> tuple[List[Dict[str, Any]], Set[str]]:
    rows = _load_jsonl(path)
    return rows, {_task_id(row) for row in rows if _task_id(row)}


def _write_metric_outputs(bundle: str, rows: List[Dict[str, Any]], output_root: Path, run_id: str, input_results_jsonl: Path) -> Dict[str, Any]:
    metric_dir = _metric_dir(output_root, bundle)
    metric_dir.mkdir(parents=True, exist_ok=True)
    result_path = _result_path(output_root, bundle)
    _write_jsonl(result_path, rows)
    if bundle == 'v31':
        summary = summarize_results_v3_1(rows)
        summary.update({
            'run_id': run_id,
            'input_results_jsonl': str(input_results_jsonl),
            'output_results_jsonl': str(result_path),
        })
        (metric_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        write_main_table_csv_v3_1(rows, metric_dir / 'main_results.csv')
        write_breakdown_csv_v3_1(rows, metric_dir / 'metric_breakdown.csv')
        return summary
    if bundle == 'v4':
        summary = summarize_results_v4(rows)
        summary.update({
            'run_id': run_id,
            'input_results_jsonl': str(input_results_jsonl),
            'output_results_jsonl': str(result_path),
        })
        (metric_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        write_main_table_csv_v4(rows, metric_dir / 'main_results.csv')
        write_dimension_csv_v4(rows, metric_dir / 'dimension_breakdown.csv')
        return summary
    summary = summarize_aux_results(rows)
    summary.update({
        'run_id': run_id,
        'input_results_jsonl': str(input_results_jsonl),
        'output_results_jsonl': str(result_path),
    })
    (metric_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_aux_csv(rows, metric_dir / 'aux_results.csv')
    write_rubric_breakdown_csv_aux(rows, metric_dir / 'dimension_breakdown.csv')
    return summary


def _evaluate_rows(
    *,
    rows: List[Dict[str, Any]],
    selected_bundles: Sequence[str],
    release_dir: Path,
    history_kb_dir: Path,
    future_kb_dir: Path,
    judge_client: FallbackOpenAICompatChatClient,
    output_root: Path,
    run_id: str,
    results_path: Path,
    resume: bool,
) -> Dict[str, Dict[str, Any]]:
    public_by_id = {row['task_id']: row for row in iter_jsonl(release_dir / 'tasks.jsonl')}
    hidden_by_id = {}
    if any(bundle in {'v31', 'aux'} for bundle in selected_bundles):
        hidden_by_id = {row['task_id']: row for row in iter_jsonl(release_dir / 'tasks_hidden_eval_v3_1.jsonl')}

    history_kb = None
    future_kb = None
    fact_cfg = None
    future_cfg = None
    if 'v31' in selected_bundles:
        history_kb = OfflineKnowledgeBase(history_kb_dir)
        future_kb = OfflineKnowledgeBase(future_kb_dir)
        fact_cfg = FactScoreV3Config()
        future_cfg = FutureAlignmentV3_1Config()

    existing_rows_by_bundle: Dict[str, List[Dict[str, Any]]] = {}
    completed_by_bundle: Dict[str, Set[str]] = {}
    handles: Dict[str, Any] = {}
    for bundle in selected_bundles:
        metric_dir = _metric_dir(output_root, bundle)
        metric_dir.mkdir(parents=True, exist_ok=True)
        existing_rows, completed_ids = _load_completed_task_ids(_result_path(output_root, bundle)) if resume else ([], set())
        existing_rows_by_bundle[bundle] = existing_rows
        completed_by_bundle[bundle] = completed_ids
        mode = 'a' if resume and completed_ids else 'w'
        handles[bundle] = _result_path(output_root, bundle).open(mode, encoding='utf-8')

    outputs_by_bundle: Dict[str, List[Dict[str, Any]]] = {bundle: list(existing_rows_by_bundle[bundle]) for bundle in selected_bundles}
    try:
        for idx, source_row in enumerate(rows, start=1):
            task_id = _task_id(source_row)
            if not task_id:
                continue
            public_task = public_by_id.get(task_id)
            if not public_task:
                continue
            hidden_row = hidden_by_id.get(task_id) if hidden_by_id else None
            row = dict(source_row)
            row['domain_id'] = infer_domain_id(row)
            pending_bundles = [bundle for bundle in selected_bundles if task_id not in completed_by_bundle[bundle]]
            if not pending_bundles:
                continue
            family = str(public_task.get('family') or row.get('family') or '')
            print(f"[eval_final] {idx}/{len(rows)} {task_id} family={family} bundles={','.join(pending_bundles)}", flush=True)

            if 'v31' in pending_bundles:
                if hidden_row is None or history_kb is None or future_kb is None or fact_cfg is None or future_cfg is None:
                    raise RuntimeError(f'missing v31 evaluation dependencies for task {task_id}')
                fact_eval = evaluate_answer_factscore_v3(
                    history_kb=history_kb,
                    future_kb=future_kb,
                    judge_client=judge_client,
                    result_row=row,
                    gt_row=hidden_row,
                    cfg=fact_cfg,
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
                    future_alignment_eval=future_alignment_eval,
                )
                outputs_by_bundle['v31'].append(out_row)
                handles['v31'].write(json.dumps(out_row, ensure_ascii=False) + '\n')
                handles['v31'].flush()

            if 'v4' in pending_bundles:
                evidence_traceability_eval = evaluate_evidence_traceability_v4(
                    judge_client,
                    public_task=public_task,
                    family=family,
                    candidate_answer=str(row.get('answer') or ''),
                    result_row=row,
                )
                out_row = build_experiment_result_row_v4(
                    run_id=run_id,
                    public_task=public_task,
                    result_row=row,
                    evidence_traceability_eval=evidence_traceability_eval,
                )
                outputs_by_bundle['v4'].append(out_row)
                handles['v4'].write(json.dumps(out_row, ensure_ascii=False) + '\n')
                handles['v4'].flush()

            if 'aux' in pending_bundles:
                if hidden_row is None:
                    raise RuntimeError(f'missing aux hidden row for task {task_id}')
                family_aux_eval = evaluate_family_auxiliary(
                    judge_client,
                    public_task=public_task,
                    hidden_row=hidden_row,
                    result_row=row,
                )
                out_row = build_aux_result_row(
                    run_id=run_id,
                    public_task=public_task,
                    hidden_row=hidden_row,
                    result_row=row,
                    family_aux_eval=family_aux_eval,
                )
                outputs_by_bundle['aux'].append(out_row)
                handles['aux'].write(json.dumps(out_row, ensure_ascii=False) + '\n')
                handles['aux'].flush()
    finally:
        for handle in handles.values():
            handle.close()

    summaries: Dict[str, Dict[str, Any]] = {}
    for bundle in selected_bundles:
        summaries[bundle] = _write_metric_outputs(bundle, outputs_by_bundle[bundle], output_root, run_id, results_path)
    return summaries


def _split_rows(rows: List[Dict[str, Any]], workers: int) -> List[List[Dict[str, Any]]]:
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, workers))]
    for idx, row in enumerate(rows):
        buckets[idx % len(buckets)].append(row)
    return buckets


def _worker_output_root(output_dir: Path, worker_idx: int) -> Path:
    return output_dir / f'worker_{worker_idx:02d}'


def _spawn_parallel_workers(args: argparse.Namespace, rows: List[Dict[str, Any]], selected_bundles: Sequence[str]) -> None:
    output_dir = Path(args.output_dir)
    chunk_root = output_dir / 'chunks'
    log_root = output_dir / 'logs'
    chunk_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    worker_rows = _split_rows(rows, args.workers)
    processes: List[subprocess.Popen[str]] = []

    for worker_idx, chunk in enumerate(worker_rows):
        chunk_file = chunk_root / f'chunk_{worker_idx:02d}.jsonl'
        _write_jsonl(chunk_file, chunk)
        worker_out = _worker_output_root(output_dir, worker_idx)
        if worker_out.exists() and not args.resume:
            shutil.rmtree(worker_out)
        worker_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--results-jsonl', str(chunk_file),
            '--release-dir', args.release_dir,
            '--output-dir', str(worker_out),
            '--run-id', f"{args.run_id or output_dir.name}_w{worker_idx:02d}",
            '--metrics', ','.join(selected_bundles),
            '--workers', '1',
            '--judge-llm-config', args.judge_llm_config,
            '--judge-fallback-llm-config', args.judge_fallback_llm_config,
            '--history-kb-dir', args.history_kb_dir,
            '--future-kb-dir', args.future_kb_dir,
            '--_worker-mode',
        ]
        if args.resume:
            cmd.append('--resume')
        log_path = log_root / f'worker_{worker_idx:02d}.log'
        log_f = log_path.open('a' if args.resume else 'w', encoding='utf-8')
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log_f, stderr=subprocess.STDOUT, text=True)
        proc._log_f = log_f  # type: ignore[attr-defined]
        processes.append(proc)
        print(f"[eval_final_parallel] launched worker={worker_idx} tasks={len(chunk)} pid={proc.pid}", flush=True)

    failures = []
    for worker_idx, proc in enumerate(processes):
        code = proc.wait()
        proc._log_f.close()  # type: ignore[attr-defined]
        print(f"[eval_final_parallel] worker={worker_idx} exit={code}", flush=True)
        if code != 0:
            failures.append(worker_idx)
    if failures:
        raise SystemExit(f'workers failed: {failures}')


def _merge_parallel_outputs(output_dir: Path, rows: List[Dict[str, Any]], selected_bundles: Sequence[str], run_id: str, input_results_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    for bundle in selected_bundles:
        merged_by_task: Dict[str, Dict[str, Any]] = {}
        for worker_idx in range(1000):
            worker_root = _worker_output_root(output_dir, worker_idx)
            if not worker_root.exists():
                break
            for row in _load_jsonl(_result_path(worker_root, bundle)):
                task_id = _task_id(row)
                if task_id:
                    merged_by_task[task_id] = row

        merged_rows: List[Dict[str, Any]] = []
        missing: List[str] = []
        for row in rows:
            task_id = _task_id(row)
            if task_id in merged_by_task:
                merged_rows.append(merged_by_task[task_id])
            else:
                missing.append(task_id)
        if missing:
            raise SystemExit(f'missing merged eval rows for bundle={bundle} count={len(missing)} first={missing[:5]}')
        summaries[bundle] = _write_metric_outputs(bundle, merged_rows, output_dir, run_id, input_results_jsonl)
    return summaries


def _build_root_summary(
    *,
    output_dir: Path,
    run_id: str,
    selected_bundles: Sequence[str],
    requested_metrics: Sequence[str],
    args: argparse.Namespace,
    summaries: Dict[str, Dict[str, Any]],
) -> None:
    root_summary = {
        'run_id': run_id,
        'release_dir': str(args.release_dir),
        'results_jsonl': str(args.results_jsonl),
        'history_kb_dir': str(args.history_kb_dir),
        'future_kb_dir': str(args.future_kb_dir),
        'requested_metrics': list(requested_metrics),
        'resolved_metric_bundles': list(selected_bundles),
        'workers': int(args.workers),
        'metric_outputs': {bundle: str(_metric_dir(output_dir, bundle)) for bundle in selected_bundles},
        'metric_summaries': summaries,
        'bundle_resolution_note': {
            'factuality': 'runs inside eval_v31 together with future_alignment',
            'future_alignment': 'runs inside eval_v31 together with factuality',
            'traceability': 'runs inside eval_v4',
            'family_aux': 'runs inside eval_aux',
        },
    }
    (output_dir / 'summary.json').write_text(json.dumps(root_summary, ensure_ascii=False, indent=2), encoding='utf-8')


def _resolve_default_kb_dir(release_dir: Path, provided: str, suffix: str) -> Path:
    return Path(provided) if str(provided or '').strip() else release_dir / suffix


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified final-metrics evaluator for ResearchForesight.')
    parser.add_argument('--results-jsonl', required=True)
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--metrics', action='append', default=[])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--judge-llm-config', default='configs/llm/qwen_235b.local.yaml')
    parser.add_argument('--judge-fallback-llm-config', default='configs/llm/mimo_pro.local.yaml')
    parser.add_argument('--history-kb-dir', default='')
    parser.add_argument('--future-kb-dir', default='')
    parser.add_argument('--run-id', default='')
    parser.add_argument('--task-limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--_worker-mode', action='store_true')
    args = parser.parse_args()

    args.release_dir = str(Path(args.release_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    release_dir = Path(args.release_dir)
    selected_bundles, requested_metrics = _parse_metrics(args.metrics)
    results_path = Path(args.results_jsonl)
    run_id = args.run_id.strip() or output_dir.name or results_path.stem
    args.history_kb_dir = str(_resolve_default_kb_dir(release_dir, args.history_kb_dir, 'kb'))
    args.future_kb_dir = str(_resolve_default_kb_dir(release_dir, args.future_kb_dir, 'future_kb'))

    rows = list(iter_jsonl(results_path))
    if args.task_limit is not None:
        rows = rows[: args.task_limit]

    if 'v31' in selected_bundles:
        if not Path(args.history_kb_dir).exists():
            raise SystemExit(f'history kb dir not found: {args.history_kb_dir}')
        if not Path(args.future_kb_dir).exists():
            raise SystemExit(f'future kb dir not found: {args.future_kb_dir}')

    if args._worker_mode or args.workers <= 1:
        judge_client = _build_judge_client(Path(args.judge_llm_config), args.judge_fallback_llm_config)
        summaries = _evaluate_rows(
            rows=rows,
            selected_bundles=selected_bundles,
            release_dir=release_dir,
            history_kb_dir=Path(args.history_kb_dir),
            future_kb_dir=Path(args.future_kb_dir),
            judge_client=judge_client,
            output_root=output_dir,
            run_id=run_id,
            results_path=results_path,
            resume=args.resume,
        )
        _build_root_summary(
            output_dir=output_dir,
            run_id=run_id,
            selected_bundles=selected_bundles,
            requested_metrics=requested_metrics,
            args=args,
            summaries=summaries,
        )
        print(json.dumps({'run_id': run_id, 'selected_bundles': selected_bundles, 'output_dir': str(output_dir)}, ensure_ascii=False, indent=2))
        return

    _spawn_parallel_workers(args, rows, selected_bundles)
    summaries = _merge_parallel_outputs(output_dir, rows, selected_bundles, run_id, results_path)
    _build_root_summary(
        output_dir=output_dir,
        run_id=run_id,
        selected_bundles=selected_bundles,
        requested_metrics=requested_metrics,
        args=args,
        summaries=summaries,
    )
    print(json.dumps({'run_id': run_id, 'selected_bundles': selected_bundles, 'output_dir': str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
