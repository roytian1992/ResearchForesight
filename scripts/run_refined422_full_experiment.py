from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.refined_release import load_task_refined_public_tasks


METHOD_SPECS = {
    "researchagent": {
        "label": "ResearchAgent-Offline",
        "script": "scripts/run_researchagent_offline.py",
        "result_name": "results.jsonl",
    },
    "aris": {
        "label": "ARIS-Offline",
        "script": "scripts/run_aris_offline.py",
        "result_name": "results.jsonl",
    },
    "coi": {
        "label": "CoI-Agent-Offline",
        "script": "scripts/run_coi_agent_offline_sharded.py",
        "result_name": "results_merged.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean task_refined 422-task experiment end to end.")
    parser.add_argument("--release-dir", default="data/releases/researchforesight_refined_422")
    parser.add_argument("--output-dir", default="results/refined422_full_clean_contract_20260425")
    parser.add_argument("--methods", nargs="*", default=["researchagent", "aris", "coi"], choices=sorted(METHOD_SPECS))
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--llm-config", default="configs/llm/qwen3_235b_8002.local.yaml")
    parser.add_argument("--embedding-config", default="configs/embedding/bge_m3.local.yaml")
    parser.add_argument("--coi-fulltext-cache-root", default="tmp/coi_fulltext_seed")
    parser.add_argument("--allow-fulltext-fetch", action="store_true")
    parser.add_argument("--max-restarts-per-shard", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--skip-existing-eval", action="store_true")
    return parser.parse_args()


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def log(root: Path, message: str) -> None:
    line = f"[{time.strftime('%F %T')}] {message}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "orchestrator.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def build_shards(task_ids: List[str], num_shards: int) -> List[List[str]]:
    shards = [[] for _ in range(max(1, num_shards))]
    for idx, task_id in enumerate(task_ids):
        shards[idx % len(shards)].append(task_id)
    return shards


def launch(cmd: List[str], *, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=handle,
        stderr=handle,
        text=True,
        start_new_session=True,
    )
    proc._log_handle = handle  # type: ignore[attr-defined]
    return proc


def close_proc(proc: subprocess.Popen) -> None:
    handle = getattr(proc, "_log_handle", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def completed_ids(path: Path) -> set[str]:
    return {str(row.get("task_id") or "") for row in read_jsonl(path) if str(row.get("task_id") or "")}


def merge_shard_results(method_dir: Path, shard_dirs: List[Path], task_ids: List[str]) -> Path:
    by_task: Dict[str, Dict[str, Any]] = {}
    for shard_dir in shard_dirs:
        for row in read_jsonl(shard_dir / "results.jsonl"):
            task_id = str(row.get("task_id") or "")
            if task_id:
                by_task[task_id] = row
    missing = [task_id for task_id in task_ids if task_id not in by_task]
    if missing:
        raise RuntimeError(f"missing merged method rows: count={len(missing)} first={missing[:5]}")
    rows = [by_task[task_id] for task_id in task_ids]
    merged_path = method_dir / "results.jsonl"
    write_jsonl(merged_path, rows)
    write_json(
        method_dir / "summary_merged.json",
        {"task_count": len(rows), "results_path": str(merged_path), "missing_count": 0},
    )
    return merged_path


def run_sharded_method(
    *,
    output_root: Path,
    method: str,
    release_dir: Path,
    kb_dir: Path,
    llm_config: str,
    task_ids: List[str],
    num_shards: int,
    max_restarts: int,
    poll_seconds: int,
) -> Path:
    spec = METHOD_SPECS[method]
    method_dir = output_root / method
    shards_dir = method_dir / "shards"
    shards = build_shards(task_ids, num_shards)
    shard_dirs: List[Path] = []
    procs: Dict[int, subprocess.Popen] = {}
    restarts: Dict[int, int] = {}

    for shard_idx, shard_task_ids in enumerate(shards):
        shard_dir = shards_dir / f"shard_{shard_idx:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_dirs.append(shard_dir)
        task_file = shard_dir / "task_ids.txt"
        task_file.write_text("\n".join(shard_task_ids) + ("\n" if shard_task_ids else ""), encoding="utf-8")
        if not shard_task_ids:
            continue
        cmd = [
            sys.executable,
            "-u",
            spec["script"],
            "--release-dir",
            str(release_dir),
            "--kb-dir",
            str(kb_dir),
            "--output-dir",
            str(shard_dir),
            "--task-ids-file",
            str(task_file),
            "--resume",
        ]
        if method == "researchagent":
            cmd.extend(
                [
                    "--reasoning-llm-config",
                    llm_config,
                    "--render-llm-config",
                    llm_config,
                    "--iterations",
                    "3",
                    "--pipeline-style",
                    "aggressive",
                    "--render-passes",
                    "1",
                ]
            )
        elif method == "aris":
            cmd.extend(["--answer-llm-config", llm_config, "--critic-llm-config", llm_config])
        proc = launch(cmd, log_path=shard_dir / "run.log")
        procs[shard_idx] = proc
        restarts[shard_idx] = 0
        log(output_root, f"launch method={method} shard={shard_idx} pid={proc.pid} tasks={len(shard_task_ids)}")

    while procs:
        for shard_idx, proc in list(procs.items()):
            shard_dir = shard_dirs[shard_idx]
            expected = len(shards[shard_idx])
            count = len(completed_ids(shard_dir / "results.jsonl"))
            rc = proc.poll()
            if count >= expected:
                close_proc(proc)
                procs.pop(shard_idx, None)
                log(output_root, f"done method={method} shard={shard_idx} rows={count}/{expected}")
                continue
            if rc is None:
                continue
            close_proc(proc)
            restarts[shard_idx] += 1
            log(output_root, f"exit method={method} shard={shard_idx} rc={rc} rows={count}/{expected} restart={restarts[shard_idx]}")
            if restarts[shard_idx] > max_restarts:
                raise RuntimeError(f"{method} shard {shard_idx} exceeded restart limit")
            task_file = shard_dir / "task_ids.txt"
            old_cmd = list(proc.args)  # type: ignore[arg-type]
            new_proc = launch(old_cmd, log_path=shard_dir / "run.log")
            procs[shard_idx] = new_proc
            log(output_root, f"restart method={method} shard={shard_idx} pid={new_proc.pid}")
        if procs:
            time.sleep(poll_seconds)

    return merge_shard_results(method_dir, shard_dirs, task_ids)


def run_coi_method(
    *,
    output_root: Path,
    release_dir: Path,
    kb_dir: Path,
    llm_config: str,
    embedding_config: str,
    fulltext_cache_root: str,
    num_shards: int,
    allow_fulltext_fetch: bool,
    poll_seconds: int,
) -> Path:
    method_dir = output_root / "coi"
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_coi_agent_offline_sharded.py",
        "--release-dir",
        str(release_dir),
        "--kb-dir",
        str(kb_dir),
        "--output-dir",
        str(method_dir),
        "--main-llm-config",
        llm_config,
        "--cheap-llm-config",
        llm_config,
        "--embedding-config",
        embedding_config,
        "--fulltext-cache-root",
        fulltext_cache_root,
        "--num-shards",
        str(num_shards),
        "--poll-seconds",
        str(poll_seconds),
    ]
    if allow_fulltext_fetch:
        cmd.append("--allow-fulltext-fetch")
    log(output_root, "launch method=coi supervisor")
    proc = launch(cmd, log_path=method_dir / "coi_supervisor.outer.log")
    rc = proc.wait()
    close_proc(proc)
    if rc != 0:
        raise RuntimeError(f"CoI sharded supervisor failed with rc={rc}")
    merged_path = method_dir / "results_merged.jsonl"
    if not merged_path.exists():
        raise FileNotFoundError(merged_path)
    log(output_root, f"done method=coi rows={len(read_jsonl(merged_path))}")
    return merged_path


def run_eval(
    *,
    output_root: Path,
    method: str,
    release_dir: Path,
    results_path: Path,
    llm_config: str,
    workers: int,
    skip_existing: bool,
) -> None:
    eval_dir = output_root / method / "final_metrics"
    summary_path = eval_dir / "summary.json"
    if skip_existing and summary_path.exists():
        log(output_root, f"skip existing eval method={method} output={eval_dir}")
        return
    cmd = [
        sys.executable,
        "-u",
        "scripts/evaluate_experiment_final_metrics.py",
        "--results-jsonl",
        str(results_path),
        "--release-dir",
        str(release_dir),
        "--output-dir",
        str(eval_dir),
        "--metrics",
        "all",
        "--workers",
        str(workers),
        "--judge-llm-config",
        llm_config,
        "--resume",
    ]
    log(output_root, f"launch eval method={method} workers={workers}")
    proc = launch(cmd, log_path=eval_dir / "eval.outer.log")
    rc = proc.wait()
    close_proc(proc)
    if rc != 0:
        raise RuntimeError(f"final metrics failed for {method} with rc={rc}")
    log(output_root, f"done eval method={method} output={eval_dir}")


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    kb_dir = release_dir / "kb"
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = load_task_refined_public_tasks(release_dir)
    task_ids = [str(row.get("task_id") or "") for row in tasks if str(row.get("task_id") or "")]
    write_json(
        output_root / "experiment_manifest.json",
        {
            "release_dir": str(release_dir),
            "kb_dir": str(kb_dir),
            "task_count": len(task_ids),
            "methods": list(args.methods),
            "num_shards": args.num_shards,
            "eval_workers": args.eval_workers,
            "llm_config": args.llm_config,
            "embedding_config": args.embedding_config,
            "started_at": time.strftime("%F %T"),
        },
    )
    log(output_root, f"start release={release_dir} tasks={len(task_ids)} methods={','.join(args.methods)}")

    for method in args.methods:
        if method == "coi":
            results_path = run_coi_method(
                output_root=output_root,
                release_dir=release_dir,
                kb_dir=kb_dir,
                llm_config=args.llm_config,
                embedding_config=args.embedding_config,
                fulltext_cache_root=args.coi_fulltext_cache_root,
                num_shards=args.num_shards,
                allow_fulltext_fetch=bool(args.allow_fulltext_fetch),
                poll_seconds=args.poll_seconds,
            )
        else:
            results_path = run_sharded_method(
                output_root=output_root,
                method=method,
                release_dir=release_dir,
                kb_dir=kb_dir,
                llm_config=args.llm_config,
                task_ids=task_ids,
                num_shards=args.num_shards,
                max_restarts=args.max_restarts_per_shard,
                poll_seconds=args.poll_seconds,
            )
        run_eval(
            output_root=output_root,
            method=method,
            release_dir=release_dir,
            results_path=results_path,
            llm_config=args.llm_config,
            workers=args.eval_workers,
            skip_existing=bool(args.skip_existing_eval),
        )

    log(output_root, "completed all requested methods")


if __name__ == "__main__":
    main()
