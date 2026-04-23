from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.baseline_runner import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline non-agent baselines on RTL benchmark v2.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--baseline", required=True, choices=["hybrid", "pageindex", "native"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--answer-llm-config", default="configs/llm/mimo_flash.local.yaml")
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--task-ids-file", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    task_ids = None
    if args.task_ids_file:
        task_ids = {
            line.strip()
            for line in Path(args.task_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    summary = run_baseline(
        release_dir=Path(args.release_dir),
        baseline_name=args.baseline,
        output_dir=Path(args.output_dir),
        answer_llm_config=Path(args.answer_llm_config),
        judge_llm_config=None if args.skip_judge else (Path(args.judge_llm_config) if args.judge_llm_config else None),
        task_limit=args.task_limit,
        domain_filter=set(args.domains) if args.domains else None,
        family_filter=set(args.families) if args.families else None,
        task_ids=task_ids,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
