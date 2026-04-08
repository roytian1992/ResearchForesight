from __future__ import annotations

import subprocess
import sys
from datetime import datetime


def ts() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def run(cmd: list[str]) -> None:
    print(f"[RUN {ts()}] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[OK  {ts()}] {' '.join(cmd[:2])}", flush=True)


def main() -> None:
    py = sys.executable

    print(f"[START {ts()}] queued domain publication enrichment", flush=True)

    run(
        [
            py,
            "-u",
            "scripts/enrich_publication_semanticscholar.py",
            "--input",
            "data/domains/llm_finetuning_post_training/interim/papers_merged.jsonl",
            "--output",
            "data/domains/llm_finetuning_post_training/interim/publication_enrichment.semanticscholar.all.jsonl",
            "--summary-output",
            "data/domains/llm_finetuning_post_training/interim/publication_enrichment.semanticscholar.all.summary.json",
            "--batch-size",
            "500",
            "--sleep",
            "0.1",
            "--sort-by-published",
            "--resume",
        ]
    )
    run(
        [
            py,
            "scripts/merge_publication_enrichment.py",
            "--input",
            "data/domains/llm_finetuning_post_training/interim/papers_merged.jsonl",
            "--enrichment",
            "data/domains/llm_finetuning_post_training/interim/publication_enrichment.semanticscholar.all.jsonl",
            "--output",
            "data/domains/llm_finetuning_post_training/interim/papers_merged.publication_enriched.semanticscholar.jsonl",
        ]
    )
    run(
        [
            py,
            "scripts/merge_publication_enrichment.py",
            "--input",
            "data/domains/llm_finetuning_post_training/clean/core_papers.jsonl",
            "--enrichment",
            "data/domains/llm_finetuning_post_training/interim/publication_enrichment.semanticscholar.all.jsonl",
            "--output",
            "data/domains/llm_finetuning_post_training/clean/core_papers.publication_enriched.semanticscholar.jsonl",
        ]
    )

    run(
        [
            py,
            "-u",
            "scripts/enrich_publication_semanticscholar.py",
            "--input",
            "data/domains/llm_agent/interim/papers_merged.jsonl",
            "--output",
            "data/domains/llm_agent/interim/publication_enrichment.semanticscholar.all.jsonl",
            "--summary-output",
            "data/domains/llm_agent/interim/publication_enrichment.semanticscholar.all.summary.json",
            "--batch-size",
            "500",
            "--sleep",
            "0.1",
            "--sort-by-published",
            "--resume",
        ]
    )
    run(
        [
            py,
            "scripts/merge_publication_enrichment.py",
            "--input",
            "data/domains/llm_agent/interim/papers_merged.jsonl",
            "--enrichment",
            "data/domains/llm_agent/interim/publication_enrichment.semanticscholar.all.jsonl",
            "--output",
            "data/domains/llm_agent/interim/papers_merged.publication_enriched.semanticscholar.jsonl",
        ]
    )

    print(f"[DONE  {ts()}] queued domain publication enrichment", flush=True)


if __name__ == "__main__":
    main()
