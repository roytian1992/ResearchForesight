from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from researchworld.eval_v3_1 import build_hidden_eval_v3_1_row


def iter_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build benchmark hidden-eval v3.1 with component targets and future alignment.')
    parser.add_argument('--release-dir', required=True)
    parser.add_argument('--hidden-v3', default='')
    parser.add_argument('--trace', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--manifest', default='')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release_dir = Path(args.release_dir)
    hidden_v3_path = Path(args.hidden_v3) if args.hidden_v3 else release_dir / 'tasks_hidden_eval_v3.jsonl'
    trace_path = Path(args.trace) if args.trace else release_dir / 'tasks_build_trace.jsonl'
    output_path = Path(args.output) if args.output else release_dir / 'tasks_hidden_eval_v3_1.jsonl'
    manifest_path = Path(args.manifest) if args.manifest else release_dir / 'tasks_hidden_eval_v3_1_manifest.json'

    trace_by_id = {row['task_id']: row for row in iter_jsonl(trace_path)}
    rows = []
    family_counts = Counter()
    domain_counts = Counter()
    component_counts = defaultdict(Counter)
    future_unit_counts = Counter()

    for hidden_row in iter_jsonl(hidden_v3_path):
        task_id = hidden_row['task_id']
        trace_row = trace_by_id[task_id]
        row = build_hidden_eval_v3_1_row(hidden_row, trace_row)
        rows.append(row)
        family = str(row.get('family') or '')
        family_counts[family] += 1
        domain_counts[str(row.get('domain') or '')] += 1
        for component in ((row.get('component_targets') or {}).get('components') or []):
            component_counts[family][str(component.get('name') or '')] += 1
        future_unit_counts[family] += len(((row.get('future_alignment_targets') or {}).get('alignment_units') or []))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    manifest = {
        'release_dir': str(release_dir),
        'input_hidden_v3': str(hidden_v3_path),
        'input_trace': str(trace_path),
        'output': str(output_path),
        'task_count': len(rows),
        'family_counts': dict(family_counts),
        'domain_counts': dict(domain_counts),
        'component_coverage': {family: dict(counter) for family, counter in component_counts.items()},
        'future_unit_counts': dict(future_unit_counts),
        'notes': [
            'v3.1 is additive on top of v3 hidden eval.',
            'It adds component_targets for component-aware task-fulfillment evaluation.',
            'It adds future_alignment_targets for realized-future alignment evaluation.',
        ],
    }
    dump_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
