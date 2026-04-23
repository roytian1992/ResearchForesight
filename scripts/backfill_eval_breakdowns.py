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
from researchworld.experiment_eval_aux import summarize_aux_results, write_aux_csv, write_rubric_breakdown_csv_aux
from researchworld.experiment_eval_v3_1 import summarize_results_v3_1, write_breakdown_csv_v3_1, write_main_table_csv_v3_1
from researchworld.experiment_eval_v4 import summarize_results_v4, write_dimension_csv_v4, write_main_table_csv_v4


def _existing_meta(output_dir: Path) -> dict:
    summary_path = output_dir / 'summary.json'
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _rewrite_v3_1(output_dir: Path) -> None:
    rows = list(iter_jsonl(output_dir / 'results_eval_v3_1.jsonl'))
    summary = summarize_results_v3_1(rows)
    old = _existing_meta(output_dir)
    for key in ['run_id', 'input_results_jsonl', 'output_results_jsonl', 'workers']:
        if key in old:
            summary[key] = old[key]
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_main_table_csv_v3_1(rows, output_dir / 'main_results.csv')
    write_breakdown_csv_v3_1(rows, output_dir / 'metric_breakdown.csv')


def _rewrite_v4(output_dir: Path) -> None:
    rows = list(iter_jsonl(output_dir / 'results_eval_v4.jsonl'))
    summary = summarize_results_v4(rows)
    old = _existing_meta(output_dir)
    for key in ['run_id', 'input_results_jsonl', 'output_results_jsonl', 'workers']:
        if key in old:
            summary[key] = old[key]
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_main_table_csv_v4(rows, output_dir / 'main_results.csv')
    write_dimension_csv_v4(rows, output_dir / 'dimension_breakdown.csv')


def _rewrite_aux(output_dir: Path) -> None:
    rows = list(iter_jsonl(output_dir / 'results_eval_aux.jsonl'))
    summary = summarize_aux_results(rows)
    old = _existing_meta(output_dir)
    for key in ['run_id', 'input_results_jsonl', 'output_results_jsonl', 'workers']:
        if key in old:
            summary[key] = old[key]
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_aux_csv(rows, output_dir / 'aux_results.csv')
    write_rubric_breakdown_csv_aux(rows, output_dir / 'dimension_breakdown.csv')


def main() -> None:
    parser = argparse.ArgumentParser(description='Backfill summary.json and dimension/component breakdown files from existing eval jsonl outputs.')
    parser.add_argument('--eval-dir', action='append', required=True, help='Eval output directory such as .../eval_v31 or .../eval_aux')
    args = parser.parse_args()

    for raw_dir in args.eval_dir:
        output_dir = Path(raw_dir)
        if not output_dir.exists():
            raise FileNotFoundError(output_dir)
        if (output_dir / 'results_eval_v3_1.jsonl').exists():
            _rewrite_v3_1(output_dir)
            print(f'[backfill] updated v3.1 breakdowns in {output_dir}')
            continue
        if (output_dir / 'results_eval_v4.jsonl').exists():
            _rewrite_v4(output_dir)
            print(f'[backfill] updated v4 breakdowns in {output_dir}')
            continue
        if (output_dir / 'results_eval_aux.jsonl').exists():
            _rewrite_aux(output_dir)
            print(f'[backfill] updated aux breakdowns in {output_dir}')
            continue
        raise FileNotFoundError(f'No supported eval jsonl found in {output_dir}')


if __name__ == '__main__':
    main()
