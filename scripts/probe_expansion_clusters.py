from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open('r', encoding='utf-8') if line.strip()]


def norm(text: Any) -> str:
    return str(text or '').lower()


def main() -> None:
    cfg = load_yaml(ROOT / 'configs' / 'domain_expansion_clusters.yaml')
    tasks = load_jsonl(ROOT / 'data' / 'releases' / 'benchmark_v3_20260408_expanded' / 'tasks.jsonl')
    task_texts = [f"{row.get('domain','')} || {row.get('family','')} || {row.get('title','')} || {row.get('question','')}".lower() for row in tasks]

    outputs = []
    for cluster in cfg.get('clusters') or []:
        domain_id = str(cluster['domain_id'])
        papers = load_jsonl(ROOT / 'data' / 'releases' / 'benchmark_v3_20260408_expanded' / 'kb' / 'domains' / domain_id / 'papers.jsonl')
        include_keywords = [str(x).lower() for x in (cluster.get('include_keywords') or [])]
        query_terms = [str(x).lower() for x in (cluster.get('query_terms') or [])]
        matched_papers = 0
        title_hits = Counter()
        for row in papers:
            title = norm(row.get('title'))
            abstract = norm(row.get('abstract'))
            text = title + ' ' + abstract
            matched = False
            for kw in include_keywords + query_terms:
                if kw and kw in text:
                    matched = True
                    title_hits[kw] += 1
            if matched:
                matched_papers += 1
        current_task_hits = 0
        current_titles = []
        for row, text in zip(tasks, task_texts):
            if row.get('domain') == _public_domain(domain_id):
                if any(kw in text for kw in include_keywords[:6]):
                    current_task_hits += 1
                    if len(current_titles) < 8:
                        current_titles.append(str(row.get('title') or ''))
        outputs.append(
            {
                'cluster_id': cluster['cluster_id'],
                'domain_id': domain_id,
                'display_name': cluster['display_name'],
                'matched_papers_estimate': matched_papers,
                'current_task_hits_estimate': current_task_hits,
                'top_keyword_hits': title_hits.most_common(10),
                'sample_existing_titles': current_titles,
                'target_new_tasks_range': cluster.get('target_new_tasks_range'),
            }
        )
    out = ROOT / 'tmp' / 'expansion_cluster_probe' / 'probe_results.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def _public_domain(domain_id: str) -> str:
    return {
        'llm_agent': 'LLM agents',
        'llm_finetuning_post_training': 'LLM fine-tuning and post-training',
        'rag_and_retrieval_structuring': 'RAG and retrieval structuring',
        'visual_generative_modeling_and_diffusion': 'Visual generative modeling and diffusion',
    }[domain_id]


if __name__ == '__main__':
    main()
