from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT.parent / "ResearchWorld"


LEGACY_INPUTS = [
    LEGACY_ROOT / "data" / "raw" / "metadata.jsonl",
    LEGACY_ROOT / "data" / "raw" / "paper_metadata_llm_agent.jsonl",
    LEGACY_ROOT / "data" / "raw" / "paper_metadata_llm_post_training.jsonl",
    LEGACY_ROOT / "data" / "raw" / "paper_metadata_rag.jsonl",
]


POSITIVE_RE = re.compile(
    r"(diffusion|latent diffusion|stable diffusion|score-based|consistency model|"
    r"flow matching|rectified flow|diffusion transformer|\bdit\b|text-to-image|text to image|"
    r"image generation|image synthesis|image editing|instruction-based image editing|"
    r"text-guided image editing|text-to-video|text to video|video generation|video synthesis|"
    r"image-to-video|image to video|3d generation|3d diffusion|view synthesis|multi-view generation|"
    r"multiview generation|subject-driven generation|personalization|customization|"
    r"controllable generation|visual generation)",
    re.I,
)

NEGATIVE_RE = re.compile(
    r"(protein diffusion|molecule diffusion|molecular generation|drug discovery|"
    r"graph diffusion|social diffusion|time series|traffic prediction|weather forecasting|"
    r"recommendation|music generation|audio generation|speech generation|audio synthesis|"
    r"trajectory diffusion|robot control|planning diffusion|policy diffusion)",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract visual generative / diffusion papers from legacy raw pools.")
    parser.add_argument(
        "--current",
        default=str(ROOT / "data" / "raw" / "paper_metadata_visual_generative_modeling_and_diffusion.jsonl"),
        help="Current visual raw file used for deduplication.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "raw" / "paper_metadata_visual_generative_modeling_and_diffusion_legacy_supplement.jsonl"),
        help="Output supplemental JSONL.",
    )
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-03-01")
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if raw:
                try:
                    yield json.loads(raw)
                except Exception:
                    continue


def canonical_id(raw: str) -> str:
    text = str(raw or "").strip()
    return text.split("v")[0]


def keep_record(row: dict, *, start: str, end: str) -> bool:
    text = f"{row.get('title', '')}\n{row.get('abstract', '')}"
    published = str(row.get("published") or "")[:10]
    if published and not (start <= published < end):
        return False
    if not POSITIVE_RE.search(text):
        return False
    if NEGATIVE_RE.search(text):
        return False
    return True


def main() -> None:
    args = parse_args()
    current_path = Path(args.current)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    current_ids: set[str] = set()
    if current_path.exists():
        for row in iter_jsonl(current_path):
            current_ids.add(canonical_id(row.get("id")))

    kept: dict[str, dict] = {}
    stats = {"scanned": 0, "kept": 0}
    per_source: dict[str, int] = {}

    for path in LEGACY_INPUTS:
        source_kept = 0
        if not path.exists():
            continue
        for row in iter_jsonl(path):
            stats["scanned"] += 1
            cid = canonical_id(row.get("id"))
            if not cid or cid in current_ids or cid in kept:
                continue
            if not keep_record(row, start=args.start, end=args.end):
                continue
            new_row = dict(row)
            new_row["legacy_supplement_source"] = str(path)
            kept[cid] = new_row
            source_kept += 1
        per_source[str(path)] = source_kept

    rows = sorted(kept.values(), key=lambda r: (str(r.get("published") or ""), str(r.get("id") or "")))
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats["kept"] = len(rows)
    print(json.dumps({"output": str(output_path), **stats, "per_source": per_source}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
