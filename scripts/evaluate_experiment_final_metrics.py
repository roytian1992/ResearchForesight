from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

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
from researchworld.factscore_eval_v5 import evaluate_answer_factscore_v5
from researchworld.future_alignment_eval_v3_1 import FutureAlignmentV3_1Config, evaluate_future_alignment_v3_1
from researchworld.future_alignment_eval_v5 import evaluate_future_alignment_v5
from researchworld.llm import (
    FallbackOpenAICompatChatClient,
    OpenAICompatChatClient,
    OpenAICompatEmbeddingClient,
    load_openai_compat_config,
    load_openai_compat_embedding_config,
)
from researchworld.offline_kb import OfflineKnowledgeBase
from researchworld.research_judgment_eval_v8 import (
    build_research_judgment_result_row_v8,
    evaluate_research_judgment_v8,
    summarize_research_judgment_results_v8,
    write_dimension_csv_v8,
    write_main_table_csv_v8,
)
from researchworld.research_judgment_rubrics import default_evaluation_rubric
from researchworld.refined_release import load_task_refined_public_by_id, load_task_refined_views

METRIC_BUNDLES = ('v6', 'v5', 'v4', 'aux', 'v31')

METRIC_SUBDIRS = {
    'v6': 'eval_v6',
    'v5': 'eval_v5',
    'v31': 'eval_v31',
    'v4': 'eval_v4',
    'aux': 'eval_aux',
}

METRIC_RESULT_FILENAMES = {
    'v6': 'results_eval_v6.jsonl',
    'v5': 'results_eval_v5.jsonl',
    'v31': 'results_eval_v3_1.jsonl',
    'v4': 'results_eval_v4.jsonl',
    'aux': 'results_eval_aux.jsonl',
}

METRIC_ALIASES = {
    'all': {'v6', 'v5', 'v4', 'aux'},
    'primary': {'v6', 'v5', 'v4'},
    'answer_quality': {'v6'},
    'judgment': {'v6'},
    'research_judgment': {'v6'},
    'rubric': {'v6'},
    'v6': {'v6'},
    'factuality': {'v5'},
    'fact': {'v5'},
    'future_alignment': {'v5'},
    'future': {'v5'},
    'v5': {'v5'},
    'traceability': {'v4'},
    'evidence_traceability': {'v4'},
    'v31': {'v31'},
    'legacy_v31': {'v31'},
    'v4': {'v4'},
    'aux': {'aux'},
    'family_aux': {'aux'},
}

V6_EVAL_CACHE_VERSION = 'v8_research_judgment_strict_deliberative_decision_20260430'
V5_EVAL_CACHE_VERSION = 'v5_task_json_only_family_fact_claims_20260430a'


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
        selected = {'v6', 'v5', 'v4', 'aux'}
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


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _v5_cache_key(*, public_task: Dict[str, Any], eval_row: Dict[str, Any], answer: str) -> str:
    payload = {
        'version': V5_EVAL_CACHE_VERSION,
        'task_id': str(public_task.get('task_id') or eval_row.get('task_id') or ''),
        'family': str(public_task.get('family') or eval_row.get('family') or ''),
        'question': public_task.get('question'),
        'answer': answer,
        'claim_bank': eval_row.get('claim_bank') or [],
        'future_alignment_targets': eval_row.get('future_alignment_targets') or {},
        'temporal_policy': eval_row.get('temporal_policy') or {},
    }
    return _sha256_text(_stable_json(payload))


def _v6_cache_key(*, public_task: Dict[str, Any], eval_row: Dict[str, Any], answer: str) -> str:
    payload = {
        'version': V6_EVAL_CACHE_VERSION,
        'task_id': str(public_task.get('task_id') or eval_row.get('task_id') or ''),
        'family': str(public_task.get('family') or eval_row.get('family') or ''),
        'question': public_task.get('question'),
        'answer_contract': public_task.get('answer_contract') or {},
        'answer': answer,
        'gold_answer': eval_row.get('gold_answer'),
        'expected_answer_points': eval_row.get('expected_answer_points') or [],
        'evaluation_rubric': eval_row.get('evaluation_rubric') or default_evaluation_rubric(
            str(public_task.get('family') or eval_row.get('family') or '')
        ),
        'component_targets': eval_row.get('component_targets') or {},
        'future_alignment_targets': eval_row.get('future_alignment_targets') or {},
        'temporal_policy': eval_row.get('temporal_policy') or {},
    }
    return _sha256_text(_stable_json(payload))


def _v5_cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / cache_key[:2] / f'{cache_key}.json'


def _read_v5_cache(cache_dir: Optional[Path], cache_key: str) -> Optional[Dict[str, Any]]:
    if cache_dir is None:
        return None
    path = _v5_cache_path(cache_dir, cache_key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _write_v5_cache(cache_dir: Optional[Path], cache_key: str, payload: Dict[str, Any]) -> None:
    if cache_dir is None:
        return
    path = _v5_cache_path(cache_dir, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def _metric_dir(root: Path, bundle: str) -> Path:
    return root / METRIC_SUBDIRS[bundle]


def _result_path(root: Path, bundle: str) -> Path:
    return _metric_dir(root, bundle) / METRIC_RESULT_FILENAMES[bundle]


def _build_judge_client(primary_config: Path, fallback_config: str) -> FallbackOpenAICompatChatClient:
    judge_primary = OpenAICompatChatClient(load_openai_compat_config(primary_config))
    judge_fallback = None
    fallback_path = Path(fallback_config) if str(fallback_config or '').strip() else None
    if fallback_path and fallback_path.exists():
        judge_fallback = OpenAICompatChatClient(load_openai_compat_config(fallback_path))
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
    if bundle == 'v6':
        summary = summarize_research_judgment_results_v8(rows)
        summary.update({
            'run_id': run_id,
            'input_results_jsonl': str(input_results_jsonl),
            'output_results_jsonl': str(result_path),
        })
        (metric_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        write_main_table_csv_v8(rows, metric_dir / 'main_results.csv')
        write_dimension_csv_v8(rows, metric_dir / 'dimension_breakdown.csv')
        return summary
    if bundle in {'v5', 'v31'}:
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
    future_kb_dir: Optional[Path],
    judge_client: FallbackOpenAICompatChatClient,
    embedding_client: Optional[OpenAICompatEmbeddingClient],
    output_root: Path,
    run_id: str,
    results_path: Path,
    resume: bool,
    eval_cache_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    public_by_id = load_task_refined_public_by_id(release_dir)
    eval_by_id = {}
    if any(bundle in {'v6', 'v5', 'v31', 'aux'} for bundle in selected_bundles):
        _, eval_by_id = load_task_refined_views(release_dir)

    history_kb = None
    future_kb = None
    fact_cfg = None
    future_cfg = None
    if 'v31' in selected_bundles:
        history_kb = OfflineKnowledgeBase(history_kb_dir)
        future_kb = OfflineKnowledgeBase(future_kb_dir) if future_kb_dir is not None else None
        fact_cfg = FactScoreV3Config()
        future_cfg = FutureAlignmentV3_1Config()
    fact_v5_cfg = FactScoreV3Config() if 'v5' in selected_bundles else None
    future_v5_cfg = FutureAlignmentV3_1Config() if 'v5' in selected_bundles else None

    seen_source_ids: Set[str] = set()
    bad_source_ids: List[str] = []
    duplicate_source_ids: List[str] = []
    for source_row in rows:
        task_id = _task_id(source_row)
        if not task_id or task_id not in public_by_id:
            bad_source_ids.append(task_id or '<missing-task-id>')
            continue
        if task_id in seen_source_ids:
            duplicate_source_ids.append(task_id)
        seen_source_ids.add(task_id)
    if bad_source_ids:
        raise RuntimeError(f'results contain task IDs not present in release: count={len(bad_source_ids)} first={bad_source_ids[:5]}')
    if duplicate_source_ids:
        raise RuntimeError(f'results contain duplicate task IDs: count={len(duplicate_source_ids)} first={duplicate_source_ids[:5]}')
    if eval_by_id:
        missing_eval_ids = [task_id for task_id in seen_source_ids if task_id not in eval_by_id]
        if missing_eval_ids:
            raise RuntimeError(f'results contain task IDs not present in task_refined eval data: count={len(missing_eval_ids)} first={missing_eval_ids[:5]}')

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
            eval_row = eval_by_id.get(task_id) if eval_by_id else None
            row = dict(source_row)
            row['domain_id'] = infer_domain_id(row)
            pending_bundles = [bundle for bundle in selected_bundles if task_id not in completed_by_bundle[bundle]]
            if not pending_bundles:
                continue
            family = str(public_task.get('family') or row.get('family') or '')
            print(f"[eval_final] {idx}/{len(rows)} {task_id} family={family} bundles={','.join(pending_bundles)}", flush=True)

            if 'v6' in pending_bundles:
                if eval_row is None:
                    raise RuntimeError(f'missing v6 evaluation dependencies for task {task_id}')
                answer = str(row.get('answer') or '')
                cache_key = _v6_cache_key(public_task=public_task, eval_row=eval_row, answer=answer)
                cached_v6 = _read_v5_cache(eval_cache_dir, cache_key)
                if cached_v6:
                    judgment_eval = cached_v6['judgment_eval']
                    cache_status = 'hit'
                else:
                    judgment_eval = evaluate_research_judgment_v8(
                        judge_client=judge_client,
                        public_task=public_task,
                        hidden_row=eval_row,
                        result_row=row,
                    )
                    _write_v5_cache(
                        eval_cache_dir,
                        cache_key,
                        {
                            'cache_version': V6_EVAL_CACHE_VERSION,
                            'task_id': task_id,
                            'answer_sha256': _sha256_text(answer),
                            'judgment_eval': judgment_eval,
                        },
                    )
                    cache_status = 'miss'
                out_row = build_research_judgment_result_row_v8(
                    run_id=run_id,
                    public_task=public_task,
                    result_row=row,
                    judgment_eval=judgment_eval,
                )
                out_row.setdefault('metadata', {})['eval_cache'] = {
                    'enabled': eval_cache_dir is not None,
                    'status': cache_status,
                    'cache_key': cache_key if eval_cache_dir is not None else '',
                    'cache_version': V6_EVAL_CACHE_VERSION,
                }
                outputs_by_bundle['v6'].append(out_row)
                handles['v6'].write(json.dumps(out_row, ensure_ascii=False) + '\n')
                handles['v6'].flush()

            if 'v5' in pending_bundles:
                if eval_row is None or fact_v5_cfg is None or future_v5_cfg is None:
                    raise RuntimeError(f'missing v5 evaluation dependencies for task {task_id}')
                answer = str(row.get('answer') or '')
                cache_key = _v5_cache_key(public_task=public_task, eval_row=eval_row, answer=answer)
                cached_v5 = _read_v5_cache(eval_cache_dir, cache_key)
                if cached_v5:
                    fact_eval = cached_v5['fact_eval']
                    future_alignment_eval = cached_v5['future_alignment_eval']
                    cache_status = 'hit'
                else:
                    fact_eval = evaluate_answer_factscore_v5(
                        judge_client=judge_client,
                        result_row=row,
                        gt_row=eval_row,
                        cfg=fact_v5_cfg,
                    )
                    future_alignment_eval = evaluate_future_alignment_v5(
                        judge_client=judge_client,
                        embedding_client=embedding_client,
                        public_task=public_task,
                        result_row=row,
                        hidden_row=eval_row,
                        cfg=future_v5_cfg,
                    )
                    _write_v5_cache(
                        eval_cache_dir,
                        cache_key,
                        {
                            'cache_version': V5_EVAL_CACHE_VERSION,
                            'task_id': task_id,
                            'answer_sha256': _sha256_text(answer),
                            'fact_eval': fact_eval,
                            'future_alignment_eval': future_alignment_eval,
                        },
                    )
                    cache_status = 'miss'
                out_row = build_experiment_result_row_v3_1(
                    run_id=run_id,
                    public_task=public_task,
                    hidden_row=eval_row,
                    result_row=row,
                    fact_eval=fact_eval,
                    future_alignment_eval=future_alignment_eval,
                )
                out_row.setdefault('metadata', {})['schema_version'] = 'v5_task_json_only'
                out_row.setdefault('metadata', {})['evaluator_scope'] = 'task_refined_json_only_no_kb'
                out_row.setdefault('metadata', {})['eval_cache'] = {
                    'enabled': eval_cache_dir is not None,
                    'status': cache_status,
                    'cache_key': cache_key if eval_cache_dir is not None else '',
                    'cache_version': V5_EVAL_CACHE_VERSION,
                }
                outputs_by_bundle['v5'].append(out_row)
                handles['v5'].write(json.dumps(out_row, ensure_ascii=False) + '\n')
                handles['v5'].flush()

            if 'v31' in pending_bundles:
                if eval_row is None or history_kb is None or fact_cfg is None or future_cfg is None:
                    raise RuntimeError(f'missing v31 evaluation dependencies for task {task_id}')
                fact_eval = evaluate_answer_factscore_v3(
                    history_kb=history_kb,
                    future_kb=future_kb,
                    judge_client=judge_client,
                    result_row=row,
                    gt_row=eval_row,
                    cfg=fact_cfg,
                )
                future_alignment_eval = evaluate_future_alignment_v3_1(
                    future_kb=future_kb,
                    judge_client=judge_client,
                    public_task=public_task,
                    result_row=row,
                    hidden_row=eval_row,
                    cfg=future_cfg,
                )
                out_row = build_experiment_result_row_v3_1(
                    run_id=run_id,
                    public_task=public_task,
                    hidden_row=eval_row,
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
                if eval_row is None:
                    raise RuntimeError(f'missing task_refined eval row for task {task_id}')
                family_aux_eval = evaluate_family_auxiliary(
                    judge_client,
                    public_task=public_task,
                    hidden_row=eval_row,
                    result_row=row,
                )
                out_row = build_aux_result_row(
                    run_id=run_id,
                    public_task=public_task,
                    hidden_row=eval_row,
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
            '--_worker-mode',
        ]
        if str(args.embedding_config or '').strip():
            cmd.extend(['--embedding-config', args.embedding_config])
        if 'v31' in selected_bundles and str(args.kb_dir or '').strip():
            cmd.extend(['--kb-dir', args.kb_dir])
        if 'v31' in selected_bundles and str(args.history_kb_dir or '').strip():
            cmd.extend(['--history-kb-dir', args.history_kb_dir])
        if 'v31' in selected_bundles and str(args.future_kb_dir or '').strip():
            cmd.extend(['--future-kb-dir', args.future_kb_dir])
        if args.resume:
            cmd.append('--resume')
        if str(args.eval_cache_dir or '').strip():
            cmd.extend(['--eval-cache-dir', args.eval_cache_dir])
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
        'kb_dir': str(args.kb_dir),
        'history_kb_dir': str(args.history_kb_dir),
        'future_kb_dir': str(args.future_kb_dir or ''),
        'embedding_config': str(args.embedding_config or ''),
        'eval_cache_dir': str(args.eval_cache_dir or ''),
        'requested_metrics': list(requested_metrics),
        'resolved_metric_bundles': list(selected_bundles),
        'workers': int(args.workers),
        'metric_outputs': {bundle: str(_metric_dir(output_dir, bundle)) for bundle in selected_bundles},
        'metric_summaries': summaries,
        'bundle_resolution_note': {
            'research_judgment': 'runs inside eval_v6 directory for backward file-layout compatibility, but scores with research_judgment v8 strict deliberative decision rubric; uses task_refined.jsonl reference fields only and does not open KB',
            'factuality': 'runs inside eval_v5 together with future_alignment; eval_v5 uses task_refined.jsonl only and does not open KB',
            'future_alignment': 'runs inside eval_v5 together with factuality; eval_v5 uses task_refined.jsonl only and does not open KB',
            'legacy_v31': 'legacy evaluator that may use offline KB for fact evidence retrieval',
            'traceability': 'runs inside eval_v4',
            'family_aux': 'runs inside eval_aux',
        },
    }
    (output_dir / 'summary.json').write_text(json.dumps(root_summary, ensure_ascii=False, indent=2), encoding='utf-8')


def _resolve_eval_kb_dirs(release_dir: Path, kb_dir: str, history_kb_dir: str, future_kb_dir: str) -> tuple[Path, Path, Optional[Path]]:
    base_kb = Path(kb_dir) if str(kb_dir or '').strip() else release_dir / 'kb'
    history_kb = Path(history_kb_dir) if str(history_kb_dir or '').strip() else base_kb
    future_kb = Path(future_kb_dir) if str(future_kb_dir or '').strip() else None
    return base_kb, history_kb, future_kb


def main() -> None:
    parser = argparse.ArgumentParser(description='Unified final-metrics evaluator for ResearchForesight.')
    parser.add_argument('--results-jsonl', required=True)
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--metrics', action='append', default=[])
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--judge-llm-config', default='configs/llm/qwen3_235b_8002.local.yaml')
    parser.add_argument('--judge-fallback-llm-config', default='')
    parser.add_argument('--embedding-config', default='')
    parser.add_argument('--kb-dir', default='')
    parser.add_argument('--history-kb-dir', default='')
    parser.add_argument('--future-kb-dir', default='')
    parser.add_argument('--run-id', default='')
    parser.add_argument('--task-limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-cache-dir', default='results/eval_cache/refined422_v6_v5')
    parser.add_argument('--_worker-mode', action='store_true')
    args = parser.parse_args()

    args.release_dir = str(Path(args.release_dir))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    release_dir = Path(args.release_dir)
    selected_bundles, requested_metrics = _parse_metrics(args.metrics)
    results_path = Path(args.results_jsonl)
    run_id = args.run_id.strip() or output_dir.name or results_path.stem
    if 'v31' in selected_bundles:
        kb_dir, history_kb_dir, future_kb_dir = _resolve_eval_kb_dirs(
            release_dir,
            args.kb_dir,
            args.history_kb_dir,
            args.future_kb_dir,
        )
        args.kb_dir = str(kb_dir)
        args.history_kb_dir = str(history_kb_dir)
        args.future_kb_dir = str(future_kb_dir) if future_kb_dir is not None else ''
    else:
        args.kb_dir = ''
        args.history_kb_dir = ''
        args.future_kb_dir = ''

    rows = list(iter_jsonl(results_path))
    if args.task_limit is not None:
        rows = rows[: args.task_limit]

    if 'v31' in selected_bundles:
        if not Path(args.history_kb_dir).exists():
            raise SystemExit(f'history kb dir not found: {args.history_kb_dir}')
        if args.future_kb_dir and not Path(args.future_kb_dir).exists():
            raise SystemExit(f'future kb dir not found: {args.future_kb_dir}')

    if args._worker_mode or args.workers <= 1:
        judge_client = _build_judge_client(Path(args.judge_llm_config), args.judge_fallback_llm_config)
        embedding_path = Path(args.embedding_config) if str(args.embedding_config or '').strip() else None
        embedding_client = (
            OpenAICompatEmbeddingClient(load_openai_compat_embedding_config(embedding_path))
            if embedding_path and embedding_path.exists()
            else None
        )
        summaries = _evaluate_rows(
            rows=rows,
            selected_bundles=selected_bundles,
            release_dir=release_dir,
            history_kb_dir=Path(args.history_kb_dir),
            future_kb_dir=Path(args.future_kb_dir) if args.future_kb_dir else None,
            judge_client=judge_client,
            embedding_client=embedding_client,
            output_root=output_dir,
            run_id=run_id,
            results_path=results_path,
            resume=args.resume,
            eval_cache_dir=Path(args.eval_cache_dir) if str(args.eval_cache_dir or '').strip() else None,
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
