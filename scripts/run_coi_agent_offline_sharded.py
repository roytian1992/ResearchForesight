from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.baseline_runner import load_release_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoI-Agent-Offline in parallel shards with restart and merge.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--kb-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--main-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--cheap-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--fallback-llm-config", default="configs/llm/qwen_235b.local.yaml")
    parser.add_argument("--fulltext-cache-root", default="tmp/coi_fulltext_seed")
    parser.add_argument("--allow-fulltext-fetch", action="store_true")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--resume-seed-results", default="", help="Optional existing results.jsonl to reuse completed tasks from a previous run.")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--restart-delay", type=int, default=10)
    parser.add_argument("--max-restarts-per-shard", type=int, default=50)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def build_shards(task_ids: List[str], num_shards: int) -> List[List[str]]:
    shards = [[] for _ in range(max(1, num_shards))]
    for idx, task_id in enumerate(task_ids):
        shards[idx % len(shards)].append(task_id)
    return shards


def launch_shard(
    *,
    shard_idx: int,
    task_ids_file: Path,
    shard_dir: Path,
    log_path: Path,
    release_dir: Path,
    kb_dir: Path,
    main_llm_config: str,
    cheap_llm_config: str,
    fallback_llm_config: str,
    fulltext_cache_root: str,
    allow_fulltext_fetch: bool,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_coi_agent_offline.py",
        "--release-dir",
        str(release_dir),
        "--kb-dir",
        str(kb_dir),
        "--output-dir",
        str(shard_dir),
        "--main-llm-config",
        main_llm_config,
        "--cheap-llm-config",
        cheap_llm_config,
        "--fallback-llm-config",
        fallback_llm_config,
        "--task-ids-file",
        str(task_ids_file),
        "--fulltext-cache-root",
        fulltext_cache_root,
    ]
    if allow_fulltext_fetch:
        cmd.append("--allow-fulltext-fetch")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=handle,
        stderr=handle,
        env={**os.environ, "PYTHONPATH": "src", "RTL_VERBOSE_COI": "1"},
        start_new_session=True,
        text=True,
    )
    proc._log_handle = handle  # type: ignore[attr-defined]
    return proc


def close_proc_handle(proc: subprocess.Popen) -> None:
    handle = getattr(proc, "_log_handle", None)
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def shard_results_count(shard_dir: Path) -> Tuple[int, str | None]:
    rows = read_jsonl(shard_dir / "results.jsonl")
    if not rows:
        return 0, None
    return len(rows), str(rows[-1].get("task_id") or "") or None


def shard_done(shard_dir: Path, expected_count: int) -> bool:
    summary_path = shard_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        return int(obj.get("task_count") or 0) >= expected_count
    except Exception:
        return False


def merge_results(output_dir: Path, seed_rows: List[Dict], shard_dirs: List[Path]) -> Dict:
    merged: Dict[str, Dict] = {}
    for row in seed_rows:
        merged[str(row.get("task_id") or "")] = row
    for shard_dir in shard_dirs:
        for row in read_jsonl(shard_dir / "results.jsonl"):
            merged[str(row.get("task_id") or "")] = row
    merged_rows = [merged[k] for k in sorted(merged)]
    merged_path = output_dir / "results_merged.jsonl"
    with merged_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "agent": "CoI-Agent-Offline",
        "task_count": len(merged_rows),
        "results_path": str(merged_path),
        "seed_reused": len(seed_rows),
        "shards": len(shard_dirs),
    }
    write_json(output_dir / "summary_merged.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    kb_dir = Path(args.kb_dir) if args.kb_dir else (release_dir / "kb")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = output_dir / "supervisor.log"
    supervisor_status = output_dir / "supervisor_status.jsonl"

    def log(msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}"
        with supervisor_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def status(**kwargs) -> None:
        row = {"time": time.strftime("%F %T"), **kwargs}
        with supervisor_status.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    tasks = load_release_tasks(release_dir)
    all_task_ids = [str(row.get("task_id") or "") for row in tasks if str(row.get("task_id") or "").strip()]
    auto_seed_path = output_dir / "results_merged.jsonl"
    seed_path = Path(args.resume_seed_results) if args.resume_seed_results else (auto_seed_path if auto_seed_path.exists() else None)
    seed_rows = read_jsonl(seed_path) if seed_path else []
    completed_seed_ids = {str(row.get("task_id") or "") for row in seed_rows}
    pending_task_ids = [task_id for task_id in all_task_ids if task_id not in completed_seed_ids]

    log(
        f"total_tasks={len(all_task_ids)} seed_completed={len(completed_seed_ids)} pending={len(pending_task_ids)} "
        f"shards={args.num_shards} seed_path={str(seed_path) if seed_path else ''}"
    )
    status(
        event="start",
        total_tasks=len(all_task_ids),
        seed_completed=len(completed_seed_ids),
        pending=len(pending_task_ids),
        seed_path=(str(seed_path) if seed_path else ""),
    )

    shard_lists = build_shards(pending_task_ids, args.num_shards)
    shard_dirs: List[Path] = []
    procs: Dict[int, subprocess.Popen] = {}
    restart_counts: Dict[int, int] = {}
    expected_counts: Dict[int, int] = {}

    for shard_idx, shard_task_ids in enumerate(shard_lists):
        shard_dir = output_dir / f"shard_{shard_idx:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_dirs.append(shard_dir)
        expected_counts[shard_idx] = len(shard_task_ids)
        task_ids_file = shard_dir / "task_ids.txt"
        task_ids_file.write_text("\n".join(shard_task_ids) + ("\n" if shard_task_ids else ""), encoding="utf-8")
        if not shard_task_ids:
            write_json(
                shard_dir / "summary.json",
                {"agent": "CoI-Agent-Offline", "task_count": 0, "total_requested_tasks": 0, "results_path": str(shard_dir / "results.jsonl")},
            )
            continue
        proc = launch_shard(
            shard_idx=shard_idx,
            task_ids_file=task_ids_file,
            shard_dir=shard_dir,
            log_path=shard_dir / "run.log",
            release_dir=release_dir,
            kb_dir=kb_dir,
            main_llm_config=args.main_llm_config,
            cheap_llm_config=args.cheap_llm_config,
            fallback_llm_config=args.fallback_llm_config,
            fulltext_cache_root=args.fulltext_cache_root,
            allow_fulltext_fetch=bool(args.allow_fulltext_fetch),
        )
        procs[shard_idx] = proc
        restart_counts[shard_idx] = 0
        log(f"launch shard={shard_idx} pid={proc.pid} tasks={len(shard_task_ids)}")
        status(event="launch", shard=shard_idx, pid=proc.pid, tasks=len(shard_task_ids))

    try:
        while True:
            all_done = True
            for shard_idx, shard_dir in enumerate(shard_dirs):
                expected = expected_counts[shard_idx]
                count, last_task = shard_results_count(shard_dir)
                done_flag = shard_done(shard_dir, expected)
                proc = procs.get(shard_idx)
                rc = proc.poll() if proc is not None else None
                status(event="heartbeat", shard=shard_idx, pid=(proc.pid if proc else None), rc=rc, results_lines=count, last_task=last_task, expected=expected, done=done_flag)
                if done_flag:
                    if proc is not None and rc is None:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        except Exception:
                            pass
                    close_proc_handle(proc) if proc is not None else None
                    procs[shard_idx] = None  # type: ignore[assignment]
                    continue
                all_done = False
                if proc is None:
                    continue
                if rc is not None:
                    close_proc_handle(proc)
                    restart_counts[shard_idx] += 1
                    log(f"shard_exit shard={shard_idx} pid={proc.pid} rc={rc} results={count}/{expected} last_task={last_task}")
                    status(event="shard_exit", shard=shard_idx, pid=proc.pid, rc=rc, results_lines=count, expected=expected, last_task=last_task)
                    if restart_counts[shard_idx] > args.max_restarts_per_shard:
                        raise RuntimeError(f"shard {shard_idx} exceeded restart limit")
                    time.sleep(args.restart_delay)
                    task_ids_file = shard_dir / "task_ids.txt"
                    new_proc = launch_shard(
                        shard_idx=shard_idx,
                        task_ids_file=task_ids_file,
                        shard_dir=shard_dir,
                        log_path=shard_dir / "run.log",
                        release_dir=release_dir,
                        kb_dir=kb_dir,
                        main_llm_config=args.main_llm_config,
                        cheap_llm_config=args.cheap_llm_config,
                        fallback_llm_config=args.fallback_llm_config,
                        fulltext_cache_root=args.fulltext_cache_root,
                        allow_fulltext_fetch=bool(args.allow_fulltext_fetch),
                    )
                    procs[shard_idx] = new_proc
                    log(f"restart shard={shard_idx} pid={new_proc.pid} restart_count={restart_counts[shard_idx]}")
                    status(event="restart", shard=shard_idx, pid=new_proc.pid, restart_count=restart_counts[shard_idx])
            if all_done:
                break
            merge_results(output_dir, seed_rows, shard_dirs)
            time.sleep(args.poll_seconds)
    finally:
        for proc in procs.values():
            if proc is None:
                continue
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
            close_proc_handle(proc)

    summary = merge_results(output_dir, seed_rows, shard_dirs)
    log(f"completed merged_task_count={summary['task_count']}")
    status(event="completed", merged_task_count=summary["task_count"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
