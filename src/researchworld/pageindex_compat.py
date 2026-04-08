from __future__ import annotations

import re
from typing import Dict, List

from researchworld.content import safe_text


SECTION_KIND_PATTERNS = [
    ("abstract", [r"\babstract\b"]),
    ("introduction", [r"\bintro", r"\boverview\b", r"\bbackground\b", r"\bmotivation\b", r"\bpreliminar"]),
    ("related_work", [r"related work", r"prior work", r"\bliterature review\b", r"\bprevious work\b"]),
    (
        "method",
        [
            r"\bmethod",
            r"\bapproach",
            r"\bframework",
            r"\bmodel\b",
            r"\balgorithm\b",
            r"\barchitecture\b",
            r"\bdesign\b",
            r"\bmodule\b",
            r"\bpipeline\b",
            r"\bsystem model\b",
            r"\bproblem formulation\b",
            r"\bsolution\b",
        ],
    ),
    ("data", [r"\bdataset", r"\bdata\b", r"\bcorpus\b", r"\bcollection\b", r"\bresource\b"]),
    (
        "experiment",
        [
            r"\bexperiment",
            r"\bevaluation",
            r"\bresults?\b",
            r"\bsetup\b",
            r"\bbenchmark\b",
            r"\bimplementation details\b",
            r"\bempirical\b",
            r"\bcase study\b",
        ],
    ),
    ("analysis", [r"\banalysis", r"\bablation", r"\berror\b", r"\bfindings?\b", r"\bdiscussion of results\b"]),
    ("discussion", [r"\bdiscussion", r"\bimplication", r"\binsight", r"\bobservations?\b"]),
    ("limitation", [r"\blimitation", r"\bconstraints", r"\bthreats?\b", r"\bfailure", r"\bchallenge", r"\bweakness"]),
    ("future_work", [r"future work", r"next step", r"open question", r"\bconclusion", r"\bconcluding remarks\b"]),
    ("appendix", [r"\bappendix", r"\bsupplement"]),
]

KIND_PRIORITY = {
    "abstract": 0,
    "introduction": 1,
    "method": 2,
    "experiment": 3,
    "analysis": 4,
    "discussion": 5,
    "limitation": 6,
    "future_work": 7,
    "related_work": 8,
    "data": 9,
    "appendix": 10,
    "other": 99,
}


SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[ivxlcdm]+(?:-[a-z])?"
    r"|[a-z]\."
    r"|\d+(?:\.\d+)*"
    r"|[a-z]\)"
    r")[\s:.\-]+",
    re.IGNORECASE,
)


def normalize_section_title(title: str) -> str:
    normalized = str(title or "").strip()
    previous = None
    while previous != normalized:
        previous = normalized
        normalized = re.sub(r"\b([A-Z])\s+([A-Z][a-z]+)\b", r"\1\2", normalized)
        normalized = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", normalized)
        normalized = re.sub(r"\s*-\s*", "-", normalized)
        normalized = SECTION_PREFIX_RE.sub("", normalized).strip()
    return normalized or str(title or "").strip()


def classify_section_kind(title: str) -> str:
    lowered = normalize_section_title(title).lower()
    for kind, patterns in SECTION_KIND_PATTERNS:
        if any(re.search(pattern, lowered) for pattern in patterns):
            return kind
    return "other"


def build_pageindex_tree(content_row: Dict) -> Dict:
    nodes: List[Dict] = []
    path_titles: Dict[int, str] = {}
    for idx, section in enumerate(content_row.get("sections") or []):
        title = str(section.get("title") or f"Section {idx + 1}")
        level = int(section.get("level") or 1)
        path_titles[level] = title
        for stale in [key for key in path_titles if key > level]:
            path_titles.pop(stale, None)
        section_path = " > ".join(path_titles[key] for key in sorted(path_titles))
        text = str(section.get("text") or "")
        kind = classify_section_kind(title)
        nodes.append(
            {
                "node_id": str(section.get("section_id") or f"sec_{idx + 1}"),
                "title": title,
                "normalized_title": normalize_section_title(title),
                "level": level,
                "kind": kind,
                "section_path": section_path,
                "summary": safe_text(text, limit=320),
                "text": text,
            }
        )

    outline = [
        {
            "node_id": node["node_id"],
            "title": node["title"],
            "level": node["level"],
            "kind": node["kind"],
            "section_path": node["section_path"],
            "summary": node["summary"],
        }
        for node in nodes
    ]
    return {
        "paper_id": content_row.get("paper_id"),
        "source_type": content_row.get("source_type"),
        "source_url": content_row.get("source_url"),
        "paper_title": content_row.get("title"),
        "node_count": len(nodes),
        "outline": outline,
        "nodes": nodes,
    }


def select_nodes_for_structure_extraction(index_row: Dict, *, max_nodes: int = 6) -> List[Dict]:
    nodes = list(index_row.get("nodes") or [])
    paper_title = str(index_row.get("paper_title") or "").strip().lower()
    nodes = [
        node
        for node in nodes
        if not (
            paper_title
            and str(node.get("normalized_title") or node.get("title") or "").strip().lower() == paper_title
        )
    ]
    nodes.sort(
        key=lambda node: (
            KIND_PRIORITY.get(str(node.get("kind") or "other"), 99),
            int(node.get("level") or 9),
            -len(str(node.get("text") or "")),
        )
    )
    selected: List[Dict] = []
    seen_titles = set()
    preferred_kinds = [
        "abstract",
        "introduction",
        "method",
        "data",
        "experiment",
        "analysis",
        "limitation",
        "future_work",
        "discussion",
    ]
    kind_counts: Dict[str, int] = {}

    def append_node(node: Dict) -> None:
        kind = str(node.get("kind") or "other")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        selected.append(
            {
                "node_id": node.get("node_id"),
                "title": node.get("title"),
                "normalized_title": node.get("normalized_title"),
                "section_path": node.get("section_path"),
                "kind": node.get("kind"),
                "summary": node.get("summary"),
                "text": safe_text(node.get("text") or "", limit=1000),
            }
        )

    for preferred_kind in preferred_kinds:
        for node in nodes:
            title = str(node.get("title") or "")
            kind = str(node.get("kind") or "other")
            if kind != preferred_kind or title in seen_titles:
                continue
            seen_titles.add(title)
            append_node(node)
            break
        if len(selected) >= max_nodes:
            return selected

    for node in nodes:
        title = str(node.get("title") or "")
        kind = str(node.get("kind") or "other")
        if title in seen_titles:
            continue
        if kind_counts.get(kind, 0) >= 2:
            continue
        seen_titles.add(title)
        append_node(node)
        if len(selected) >= max_nodes:
            break
    return selected
