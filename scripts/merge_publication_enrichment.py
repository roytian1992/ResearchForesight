from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with open(path, 'r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description='Merge publication enrichment sidecar into paper rows.')
    parser.add_argument('--input', required=True, help='Base paper JSONL.')
    parser.add_argument('--enrichment', required=True, help='Publication enrichment JSONL keyed by paper_id.')
    parser.add_argument('--output', required=True, help='Merged output JSONL.')
    args = parser.parse_args()

    enrich_by_id: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(Path(args.enrichment)):
        paper_id = row.get('paper_id')
        if isinstance(paper_id, str) and paper_id:
            enrich_by_id[paper_id] = row

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    matched = 0
    with open(output_path, 'w', encoding='utf-8') as out:
        for row in iter_jsonl(Path(args.input)):
            count += 1
            paper_id = row.get('paper_id')
            if isinstance(paper_id, str) and paper_id in enrich_by_id:
                payload = dict(row)
                enrichment = dict(enrich_by_id[paper_id])
                for key in ['paper_id', 'source_paper_id', 'title', 'published', 'authors']:
                    enrichment.pop(key, None)
                payload['publication_enrichment'] = enrichment
                out.write(json.dumps(payload, ensure_ascii=False) + '\n')
                matched += 1
            else:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
    print({'input_count': count, 'merged_count': matched, 'output': str(output_path)})


if __name__ == '__main__':
    main()
