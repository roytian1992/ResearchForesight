from __future__ import annotations

import argparse
import json
from pathlib import Path

from researchworld.baseline_runner import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline non-agent baselines on RTL benchmark v2.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--baseline", required=True, choices=["hybrid", "pageindex"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--answer-llm-config", default="configs/llm/mimo_flash.local.yaml")
    parser.add_argument("--judge-llm-config", default="configs/llm/mimo_pro.local.yaml")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--task-limit", type=int, default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    args = parser.parse_args()

    summary = run_baseline(
        release_dir=Path(args.release_dir),
        baseline_name=args.baseline,
        output_dir=Path(args.output_dir),
        answer_llm_config=Path(args.answer_llm_config),
        judge_llm_config=None if args.skip_judge else (Path(args.judge_llm_config) if args.judge_llm_config else None),
        task_limit=args.task_limit,
        domain_filter=set(args.domains) if args.domains else None,
        family_filter=set(args.families) if args.families else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
