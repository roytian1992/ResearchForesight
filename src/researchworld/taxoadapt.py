from __future__ import annotations

import concurrent.futures
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from researchworld.benchmark import parse_published_date
from researchworld.corpus import iter_jsonl
from researchworld.llm import OpenAICompatChatClient, complete_json_object


DEFAULT_DIMENSIONS: List[Dict[str, str]] = [
    {
        "id": "tasks",
        "display_name": "Tasks",
        "definition": (
            "Problem settings or objectives addressed by the paper. "
            "This includes concrete tasks, goals, operating settings, and benchmark problem formulations."
        ),
    },
    {
        "id": "methodologies",
        "display_name": "Methodologies",
        "definition": (
            "Techniques, model families, algorithms, architectural patterns, training recipes, "
            "retrieval pipelines, agent loops, or optimization strategies proposed or emphasized by the paper."
        ),
    },
    {
        "id": "datasets",
        "display_name": "Datasets",
        "definition": (
            "Datasets, benchmarks, evaluation suites, corpora, structured resources, or data-construction "
            "artifacts introduced or substantially curated by the paper."
        ),
    },
    {
        "id": "evaluation_methods",
        "display_name": "Evaluation Methods",
        "definition": (
            "Evaluation protocols, metrics, benchmark methodologies, stress tests, ablation frameworks, "
            "or analysis methods used to assess research systems."
        ),
    },
    {
        "id": "real_world_domains",
        "display_name": "Real-World Domains",
        "definition": (
            "Applied settings, user populations, industries, scientific domains, enterprise contexts, "
            "or operational environments in which the research is deployed or evaluated."
        ),
    },
]


def slugify(text: str) -> str:
    value = str(text or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def truncate_text(text: str, limit: int = 1200) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def citation_count(row: Dict[str, Any]) -> int:
    enrichment = row.get("publication_enrichment") or {}
    value = enrichment.get("preferred_cited_by_count")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


@dataclass
class DimensionSpec:
    id: str
    display_name: str
    definition: str


@dataclass
class TaxonomyNode:
    node_id: str
    label: str
    display_name: str
    description: str
    dimension_id: str
    level: int
    created_year: Any
    source: str
    parent_id: Optional[str]
    child_ids: List[str] = field(default_factory=list)
    paper_ids: Set[str] = field(default_factory=set)
    last_width_attempt_year: Optional[int] = None
    last_depth_attempt_year: Optional[int] = None
    created_time_slice: Optional[str] = None


def load_scope_map(path: Path) -> Dict[str, str]:
    scope_map: Dict[str, str] = {}
    for row in iter_jsonl(path):
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id:
            scope_map[paper_id] = str(row.get("scope_decision") or "")
    return scope_map


def load_domain_papers(
    papers_path: Path,
    *,
    scope_labels_path: Optional[Path] = None,
    allowed_scope: str = "core_domain",
    year_sequence: Optional[Sequence[int]] = None,
    time_slices: Optional[Sequence[Dict[str, Any]]] = None,
    max_papers_per_year: int = 0,
    min_abstract_chars: int = 40,
) -> List[Dict[str, Any]]:
    scope_map = load_scope_map(scope_labels_path) if scope_labels_path and scope_labels_path.exists() else {}
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    normalized_slices: List[Tuple[str, date, date]] = []
    if time_slices:
        for item in time_slices:
            slice_id = str(item["slice_id"])
            start = parse_published_date(item["start"])
            end = parse_published_date(item["end"])
            if start is None or end is None:
                raise ValueError(f"Invalid time slice boundaries: {item}")
            normalized_slices.append((slice_id, start, end))

    for row in iter_jsonl(papers_path):
        paper_id = str(row.get("paper_id") or "").strip()
        if not paper_id:
            continue
        if scope_map and scope_map.get(paper_id) != allowed_scope:
            continue
        abstract = str(row.get("abstract") or "").strip()
        if len(abstract) < min_abstract_chars:
            continue
        published_date = parse_published_date(row.get("published"))
        if published_date is None:
            continue
        year = int(published_date.year)
        if year_sequence and year not in set(year_sequence):
            continue

        normalized = dict(row)
        normalized["published_date"] = published_date.isoformat()
        normalized["chronology_year"] = year
        if normalized_slices:
            slice_id = None
            for candidate_id, start, end in normalized_slices:
                if start <= published_date < end:
                    slice_id = candidate_id
                    break
            if slice_id is None:
                continue
            normalized["chronology_slice"] = slice_id
        normalized["preferred_cited_by_count"] = citation_count(normalized)
        grouped[year].append(normalized)

    ordered_rows: List[Dict[str, Any]] = []
    for year in sorted(grouped):
        rows = grouped[year]
        rows.sort(
            key=lambda row: (
                -int(row.get("preferred_cited_by_count") or 0),
                row.get("published_date") or "",
                row.get("paper_id") or "",
            )
        )
        if max_papers_per_year > 0:
            rows = rows[:max_papers_per_year]
        rows.sort(key=lambda row: (row.get("published_date") or "", row.get("paper_id") or ""))
        ordered_rows.extend(rows)
    return ordered_rows


class TemporalTaxoAdaptRunner:
    def __init__(
        self,
        *,
        project_root: Path,
        llm: OpenAICompatChatClient,
        domain_id: str,
        topic: str,
        papers: Sequence[Dict[str, Any]],
        output_dir: Path,
        dimensions: Sequence[Dict[str, str]] | Sequence[DimensionSpec] | None = None,
        year_sequence: Sequence[int] = (2023, 2024, 2025, 2026),
        time_sequence: Optional[Sequence[str]] = None,
        init_levels: int = 1,
        max_depth: int = 3,
        max_density: int = 40,
        max_children_per_node: int = 5,
        bootstrap_paper_sample_size: int = 20,
        width_paper_sample_size: int = 40,
        depth_paper_sample_size: int = 40,
        candidate_label_top_k: int = 20,
        min_candidate_votes: int = 2,
        workers: int = 6,
        request_timeout: int = 240,
        max_retries: int = 2,
        temperature_bootstrap: float = 0.1,
        temperature_routing: float = 0.0,
        temperature_classification: float = 0.0,
        temperature_expansion: float = 0.4,
        temperature_clustering: float = 0.2,
    ) -> None:
        self.project_root = project_root
        self.llm = llm
        self.domain_id = domain_id
        self.topic = topic
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        parsed_dimensions: List[DimensionSpec] = []
        for dim in dimensions or DEFAULT_DIMENSIONS:
            if isinstance(dim, DimensionSpec):
                parsed_dimensions.append(dim)
            else:
                parsed_dimensions.append(
                    DimensionSpec(
                        id=str(dim["id"]),
                        display_name=str(dim.get("display_name") or dim["id"]),
                        definition=str(dim["definition"]),
                    )
                )
        self.dimensions = parsed_dimensions
        self.dimension_map = {dim.id: dim for dim in self.dimensions}
        if time_sequence:
            self.time_sequence = [str(value) for value in time_sequence]
        else:
            self.time_sequence = [str(int(year)) for year in year_sequence]
        self.year_sequence = list(self.time_sequence)
        self.init_levels = max(1, int(init_levels))
        self.max_depth = max(1, int(max_depth))
        self.max_density = max(1, int(max_density))
        self.max_children_per_node = max(1, int(max_children_per_node))
        self.bootstrap_paper_sample_size = max(1, int(bootstrap_paper_sample_size))
        self.width_paper_sample_size = max(1, int(width_paper_sample_size))
        self.depth_paper_sample_size = max(1, int(depth_paper_sample_size))
        self.candidate_label_top_k = max(1, int(candidate_label_top_k))
        self.min_candidate_votes = max(1, int(min_candidate_votes))
        self.workers = max(1, int(workers))
        self.request_timeout = max(30, int(request_timeout))
        self.max_retries = max(0, int(max_retries))
        self.temperature_bootstrap = float(temperature_bootstrap)
        self.temperature_routing = float(temperature_routing)
        self.temperature_classification = float(temperature_classification)
        self.temperature_expansion = float(temperature_expansion)
        self.temperature_clustering = float(temperature_clustering)

        self.papers = list(papers)
        self.paper_map: Dict[str, Dict[str, Any]] = {str(row["paper_id"]): dict(row) for row in self.papers}
        self.papers_by_time: Dict[str, List[str]] = defaultdict(list)
        for row in self.papers:
            time_label = str(row.get("chronology_slice") or row.get("chronology_year"))
            self.papers_by_time[time_label].append(str(row["paper_id"]))
        self.papers_by_year = self.papers_by_time

        self.nodes: Dict[str, TaxonomyNode] = {}
        self.roots: Dict[str, str] = {}
        self.paper_dimension_membership: Dict[str, Set[str]] = defaultdict(set)
        self.dimension_label_inventory: Dict[str, Set[str]] = defaultdict(set)
        self.year_summaries: List[Dict[str, Any]] = []
        self.trace_by_time: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"width_expansions": [], "depth_expansions": []})
        self.trace_by_year = self.trace_by_time
        self.progress_log_every = 25

        self._initialize_roots()

    def _log(self, message: str) -> None:
        print(f"[TaxoAdapt][{self.domain_id}] {message}", flush=True)

    def _initialize_roots(self) -> None:
        root_label = slugify(self.topic)
        for dim in self.dimensions:
            node_id = f"{dim.id}/{root_label}"
            node = TaxonomyNode(
                node_id=node_id,
                label=root_label,
                display_name=self.topic,
                description=f"Root topic for {self.topic} under the {dim.display_name} dimension.",
                dimension_id=dim.id,
                level=0,
                created_year=self.time_sequence[0],
                created_time_slice=self.time_sequence[0],
                source="root",
                parent_id=None,
            )
            self.nodes[node_id] = node
            self.roots[dim.id] = node_id
            self.dimension_label_inventory[dim.id].add(node.label)

    def run(self) -> Dict[str, Any]:
        manifest = {
            "domain_id": self.domain_id,
            "topic": self.topic,
            "year_sequence": list(self.year_sequence),
            "time_sequence": list(self.time_sequence),
            "paper_count": len(self.papers),
            "settings": {
                "init_levels": self.init_levels,
                "max_depth": self.max_depth,
                "max_density": self.max_density,
                "max_children_per_node": self.max_children_per_node,
                "bootstrap_paper_sample_size": self.bootstrap_paper_sample_size,
                "width_paper_sample_size": self.width_paper_sample_size,
                "depth_paper_sample_size": self.depth_paper_sample_size,
                "candidate_label_top_k": self.candidate_label_top_k,
                "min_candidate_votes": self.min_candidate_votes,
                "workers": self.workers,
                "request_timeout": self.request_timeout,
                "max_retries": self.max_retries,
            },
            "dimensions": [dim.__dict__ for dim in self.dimensions],
        }
        (self.output_dir / "manifest.json").write_text(json_dumps(manifest), encoding="utf-8")
        self._log(
            f"manifest written: papers={len(self.papers)} slices={len(self.time_sequence)} "
            f"workers={self.workers} output_dir={self.output_dir}"
        )

        for time_label in self.time_sequence:
            new_paper_ids = list(self.papers_by_time.get(time_label) or [])
            self._log(f"slice {time_label} start: new_papers={len(new_paper_ids)}")
            if not new_paper_ids:
                self._write_time_snapshot(time_label, route_map={})
                self._log(f"slice {time_label} done: no new papers")
                continue

            route_map = self._route_papers_to_dimensions(new_paper_ids)
            self._bootstrap_missing_roots(time_label, route_map)
            self._ingest_year(time_label, route_map)
            self._write_time_snapshot(time_label, route_map=route_map)
            self._log(f"slice {time_label} done: node_count={len(self.nodes)}")

        final_summary = {
            "domain_id": self.domain_id,
            "topic": self.topic,
            "paper_count": len(self.papers),
            "years": self.year_summaries,
            "node_count": len(self.nodes),
            "dimension_node_counts": {
                dim.id: len([node for node in self.nodes.values() if node.dimension_id == dim.id])
                for dim in self.dimensions
            },
        }
        (self.output_dir / "final_summary.json").write_text(json_dumps(final_summary), encoding="utf-8")
        self._log(f"final summary written: node_count={len(self.nodes)}")
        return final_summary

    def _ingest_year(self, year: Any, route_map: Dict[str, List[str]]) -> None:
        for dim in self.dimensions:
            root_id = self.roots[dim.id]
            root_new_ids = [paper_id for paper_id, dims in route_map.items() if dim.id in dims]
            if not root_new_ids:
                continue
            self.nodes[root_id].paper_ids.update(root_new_ids)
            self._process_node(root_id, set(root_new_ids), year, reset_subtree=False)

    def _bootstrap_missing_roots(self, year: Any, route_map: Dict[str, List[str]]) -> None:
        for dim in self.dimensions:
            root_id = self.roots[dim.id]
            root = self.nodes[root_id]
            if root.child_ids:
                continue
            seed_ids = [paper_id for paper_id, dims in route_map.items() if dim.id in dims]
            if not seed_ids:
                continue
            self._bootstrap_children(root_id, seed_ids, year)

    def _all_existing_labels(self, dimension_id: str) -> List[str]:
        return sorted(self.dimension_label_inventory.get(dimension_id) or set())

    def _pick_sample_papers(self, paper_ids: Iterable[str], limit: int) -> List[Dict[str, Any]]:
        rows = [self.paper_map[paper_id] for paper_id in paper_ids if paper_id in self.paper_map]
        rows.sort(
            key=lambda row: (
                -int(row.get("preferred_cited_by_count") or 0),
                row.get("published_date") or "",
                row.get("paper_id") or "",
            )
        )
        return rows[:limit]

    def _sample_text_block(self, paper_ids: Iterable[str], limit: int) -> str:
        samples = self._pick_sample_papers(paper_ids, limit)
        lines: List[str] = []
        for row in samples:
            lines.append(
                (
                    f"- paper_id: {row['paper_id']}\n"
                    f"  title: {truncate_text(row.get('title', ''), 240)}\n"
                    f"  abstract: {truncate_text(row.get('abstract', ''), 600)}"
                )
            )
        return "\n".join(lines) if lines else "- none"

    def _node_path(self, node_id: str) -> str:
        path: List[str] = []
        current = self.nodes[node_id]
        while current is not None:
            path.append(current.display_name)
            current = self.nodes[current.parent_id] if current.parent_id else None
        path.reverse()
        return " -> ".join(path)

    def _call_json(self, prompt: str, *, temperature: float) -> Dict[str, Any]:
        last_error = ""
        raw_text = ""
        for _ in range(self.max_retries + 1):
            try:
                obj = complete_json_object(
                    self.llm,
                    [{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=self.request_timeout,
                    max_parse_attempts=3,
                )
                return obj
            except Exception as exc:  # pragma: no cover - runtime I/O path
                last_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(f"LLM JSON call failed after retries: {last_error}; raw={raw_text[:500]}")

    def _parallel_json_map(
        self,
        prompts: Sequence[Tuple[str, str]],
        *,
        temperature: float,
        stage_label: str = "llm_batch",
    ) -> Dict[str, Dict[str, Any]]:
        if not prompts:
            return {}

        def run_one(item: Tuple[str, str]) -> Tuple[str, Dict[str, Any]]:
            key, prompt = item
            return key, self._call_json(prompt, temperature=temperature)

        outputs: Dict[str, Dict[str, Any]] = {}
        total = len(prompts)
        completed = 0
        self._log(f"{stage_label} start: prompts={total} workers={self.workers}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_map = {executor.submit(run_one, item): item[0] for item in prompts}
            for future in concurrent.futures.as_completed(future_map):
                key, obj = future.result()
                outputs[key] = obj
                completed += 1
                if completed % self.progress_log_every == 0 or completed == total:
                    self._log(f"{stage_label} progress: {completed}/{total}")
        self._log(f"{stage_label} done: outputs={len(outputs)}")
        return outputs

    def _dimension_routing_prompt(self, paper: Dict[str, Any]) -> str:
        lines = []
        for dim in self.dimensions:
            lines.append(f"- {dim.id}: {dim.definition}")
        definitions = "\n".join(lines)
        keys = ",\n  ".join([f'"{dim.id}": true/false' for dim in self.dimensions])
        return f"""
You are classifying a research paper into multiple taxonomy dimensions.

Domain topic: {self.topic}

Dimension definitions:
{definitions}

Paper:
- paper_id: {paper["paper_id"]}
- title: {truncate_text(paper.get("title", ""), 320)}
- abstract: {truncate_text(paper.get("abstract", ""), 1200)}

Rules:
- A paper may belong to multiple dimensions.
- Be conservative for niche dimensions, but do not leave a paper completely unassigned if one dimension is clearly central.
- Return JSON only.

Output schema:
{{
  {keys}
}}
""".strip()

    def _route_papers_to_dimensions(self, paper_ids: Sequence[str]) -> Dict[str, List[str]]:
        prompts = [(paper_id, self._dimension_routing_prompt(self.paper_map[paper_id])) for paper_id in paper_ids]
        outputs = self._parallel_json_map(
            prompts,
            temperature=self.temperature_routing,
            stage_label=f"route_dimensions[{len(paper_ids)}]",
        )
        route_map: Dict[str, List[str]] = {}
        for paper_id in paper_ids:
            obj = outputs.get(paper_id) or {}
            dims: List[str] = []
            for dim in self.dimensions:
                if bool(obj.get(dim.id)):
                    dims.append(dim.id)
            if not dims:
                dims = ["tasks"]
            route_map[paper_id] = dims
            self.paper_dimension_membership[paper_id].update(dims)
        return route_map

    def _bootstrap_prompt(self, node: TaxonomyNode, paper_ids: Sequence[str]) -> str:
        dim = self.dimension_map[node.dimension_id]
        samples = self._sample_text_block(paper_ids, self.bootstrap_paper_sample_size)
        return f"""
You are building a taxonomy for a research corpus using the dimension "{dim.display_name}".

Domain topic: {self.topic}
Current node path: {self._node_path(node.node_id)}
Current node description: {node.description}
Dimension definition: {dim.definition}

Representative papers currently under this node:
{samples}

Task:
Propose up to {self.max_children_per_node} child categories that should directly organize this node.

Requirements:
- Children must be mutually distinct and comparable in granularity.
- Children must be specific to the current node and grounded in the representative papers.
- Use concise noun phrases for labels.
- Avoid duplicating any existing taxonomy labels for this dimension.

Existing labels in this dimension:
{", ".join(self._all_existing_labels(node.dimension_id)) or "none"}

Return JSON only in the following format:
{{
  "children": [
    {{
      "label": "child label",
      "description": "one-sentence description"
    }}
  ]
}}
""".strip()

    def _bootstrap_children(self, node_id: str, paper_ids: Sequence[str], year: Any) -> bool:
        node = self.nodes[node_id]
        self._log(f"bootstrap start: node={node_id} papers={len(paper_ids)} slice={year}")
        obj = self._call_json(self._bootstrap_prompt(node, paper_ids), temperature=self.temperature_bootstrap)
        children = obj.get("children") or []
        added = 0
        for child in children:
            if self._add_child_node(
                parent_id=node_id,
                raw_label=str(child.get("label") or ""),
                description=str(child.get("description") or ""),
                year=year,
                source="bootstrap",
            ):
                added += 1
        self._log(f"bootstrap done: node={node_id} added_children={added}")
        return added > 0

    def _add_child_node(
        self,
        *,
        parent_id: str,
        raw_label: str,
        description: str,
        year: Any,
        source: str,
    ) -> Optional[str]:
        parent = self.nodes[parent_id]
        label = slugify(raw_label)
        if not label:
            return None
        if label in self.dimension_label_inventory[parent.dimension_id]:
            return None
        child_id = f"{parent_id}/{label}"
        if child_id in self.nodes:
            return None
        child = TaxonomyNode(
            node_id=child_id,
            label=label,
            display_name=str(raw_label).strip() or label.replace("_", " "),
            description=str(description or "").strip() or f"{raw_label} under {parent.display_name}.",
            dimension_id=parent.dimension_id,
            level=parent.level + 1,
            created_year=year,
            created_time_slice=str(year),
            source=source,
            parent_id=parent_id,
        )
        self.nodes[child_id] = child
        parent.child_ids.append(child_id)
        self.dimension_label_inventory[parent.dimension_id].add(label)
        return child_id

    def _classification_prompt(self, node: TaxonomyNode, paper: Dict[str, Any]) -> str:
        options = []
        for child_id in node.child_ids:
            child = self.nodes[child_id]
            options.append(f'- "{child.label}": {child.display_name}. {child.description}')
        return f"""
You are assigning a paper to child categories in a research taxonomy.

Domain topic: {self.topic}
Dimension: {self.dimension_map[node.dimension_id].display_name}
Dimension definition: {self.dimension_map[node.dimension_id].definition}
Current node path: {self._node_path(node.node_id)}
Current node description: {node.description}

Paper:
- paper_id: {paper["paper_id"]}
- title: {truncate_text(paper.get("title", ""), 320)}
- abstract: {truncate_text(paper.get("abstract", ""), 1200)}

Allowed child labels:
{chr(10).join(options)}

Rules:
- Output only exact labels from the allowed child labels.
- Multi-label output is allowed only when the paper clearly belongs to multiple child categories.
- If no child is a tight fit, output an empty list.
- Return JSON only.

Output schema:
{{
  "labels": ["exact_child_label_1", "exact_child_label_2"]
}}
""".strip()

    def _classify_into_children(self, node_id: str, paper_ids: Iterable[str]) -> Dict[str, List[str]]:
        node = self.nodes[node_id]
        if not node.child_ids:
            return {}
        paper_ids = [paper_id for paper_id in paper_ids if paper_id in self.paper_map]
        if not paper_ids:
            return {}

        prompts = [
            (paper_id, self._classification_prompt(node, self.paper_map[paper_id]))
            for paper_id in paper_ids
        ]
        outputs = self._parallel_json_map(
            prompts,
            temperature=self.temperature_classification,
            stage_label=f"classify[{node_id}][{len(paper_ids)}]",
        )
        child_label_to_id = {self.nodes[child_id].label: child_id for child_id in node.child_ids}

        assignments: Dict[str, List[str]] = {}
        for paper_id in paper_ids:
            obj = outputs.get(paper_id) or {}
            labels = obj.get("labels") or []
            if not isinstance(labels, list):
                labels = []
            assigned_ids: List[str] = []
            seen: Set[str] = set()
            for label in labels:
                if not isinstance(label, str):
                    continue
                child_id = child_label_to_id.get(slugify(label))
                if child_id and child_id not in seen:
                    assigned_ids.append(child_id)
                    seen.add(child_id)
            assignments[paper_id] = assigned_ids
        return assignments

    def _width_candidate_prompt(self, node: TaxonomyNode, paper: Dict[str, Any]) -> str:
        sibling_lines = []
        for child_id in node.child_ids:
            child = self.nodes[child_id]
            sibling_lines.append(f'- "{child.label}": {child.display_name}. {child.description}')
        return f"""
You are proposing a new sibling taxonomy category.

Domain topic: {self.topic}
Dimension: {self.dimension_map[node.dimension_id].display_name}
Current node path: {self._node_path(node.node_id)}
Current node description: {node.description}

Existing child labels at this level:
{chr(10).join(sibling_lines)}

Paper:
- paper_id: {paper["paper_id"]}
- title: {truncate_text(paper.get("title", ""), 320)}
- abstract: {truncate_text(paper.get("abstract", ""), 1200)}

Task:
Suggest one new child label that should be a sibling of the existing child labels and that best captures this paper if none of the current children fit tightly.

Rules:
- The proposed label must be at the same granularity as the existing children.
- Use a concise noun phrase.
- Do not reuse an existing child label.
- Return JSON only.

Output schema:
{{
  "label": "new_sibling_label"
}}
""".strip()

    def _depth_candidate_prompt(self, node: TaxonomyNode, paper: Dict[str, Any]) -> str:
        return f"""
You are proposing a more specific child taxonomy category.

Domain topic: {self.topic}
Dimension: {self.dimension_map[node.dimension_id].display_name}
Current node path: {self._node_path(node.node_id)}
Current node description: {node.description}

Paper:
- paper_id: {paper["paper_id"]}
- title: {truncate_text(paper.get("title", ""), 320)}
- abstract: {truncate_text(paper.get("abstract", ""), 1200)}

Task:
Suggest one child label that is strictly more specific than the current node and that would directly fall under this node.

Rules:
- The label must be a child of the current node, not a sibling of the current node.
- Use a concise noun phrase.
- Return JSON only.

Output schema:
{{
  "label": "new_child_label"
}}
""".strip()

    def _cluster_prompt(
        self,
        *,
        node: TaxonomyNode,
        candidate_counts: Dict[str, int],
        mode: str,
    ) -> str:
        existing_children = []
        for child_id in node.child_ids:
            child = self.nodes[child_id]
            existing_children.append(f'- "{child.label}": {child.display_name}. {child.description}')
        relation = "new sibling categories under the current node" if mode == "width" else "new child categories below the current node"
        return f"""
You are clustering candidate taxonomy labels for {relation}.

Domain topic: {self.topic}
Dimension: {self.dimension_map[node.dimension_id].display_name}
Current node path: {self._node_path(node.node_id)}
Current node description: {node.description}

Existing labels in this dimension:
{", ".join(self._all_existing_labels(node.dimension_id)) or "none"}

Existing direct children of the current node:
{chr(10).join(existing_children) if existing_children else "- none"}

Candidate labels with vote counts:
{json.dumps(candidate_counts, ensure_ascii=False)}

Task:
Merge synonymous or overlapping candidate labels and output up to {self.max_children_per_node} high-quality taxonomy labels.

Rules:
- Labels must be concise noun phrases.
- Labels must be mutually distinct and comparable in granularity.
- Do not output any label that already exists in this dimension.
- Return JSON only.

Output schema:
{{
  "children": [
    {{
      "label": "taxonomy_label",
      "description": "one-sentence description",
      "covered_candidates": ["candidate_a", "candidate_b"]
    }}
  ]
}}
""".strip()

    def _count_candidate_labels(
        self,
        *,
        node: TaxonomyNode,
        paper_ids: Iterable[str],
        mode: str,
    ) -> Dict[str, int]:
        sample_size = self.width_paper_sample_size if mode == "width" else self.depth_paper_sample_size
        sample_rows = self._pick_sample_papers(paper_ids, sample_size)
        if not sample_rows:
            return {}

        prompts = []
        for row in sample_rows:
            prompt = self._width_candidate_prompt(node, row) if mode == "width" else self._depth_candidate_prompt(node, row)
            prompts.append((str(row["paper_id"]), prompt))
        outputs = self._parallel_json_map(
            prompts,
            temperature=self.temperature_expansion,
            stage_label=f"candidate_labels[{mode}][{node.node_id}][{len(prompts)}]",
        )

        counts: Counter[str] = Counter()
        existing_labels = self.dimension_label_inventory[node.dimension_id]
        for paper_id, _ in prompts:
            obj = outputs.get(paper_id) or {}
            label = slugify(str(obj.get("label") or ""))
            if not label or label in existing_labels:
                continue
            counts[label] += 1
        filtered = {label: count for label, count in counts.most_common(self.candidate_label_top_k) if count >= self.min_candidate_votes}
        return filtered

    def _expand_width(self, node_id: str, year: Any) -> bool:
        node = self.nodes[node_id]
        if node.last_width_attempt_year == year:
            return False
        node.last_width_attempt_year = year
        unlabeled_ids = self._unlabeled_paper_ids(node_id)
        if len(unlabeled_ids) <= self.max_density or not node.child_ids:
            return False
        self._log(f"width expansion check: node={node_id} unlabeled={len(unlabeled_ids)}")
        candidate_counts = self._count_candidate_labels(node=node, paper_ids=unlabeled_ids, mode="width")
        if not candidate_counts:
            self._log(f"width expansion skipped: node={node_id} no_candidates")
            return False
        obj = self._call_json(self._cluster_prompt(node=node, candidate_counts=candidate_counts, mode="width"), temperature=self.temperature_clustering)
        children = obj.get("children") or []
        added_children: List[str] = []
        for child in children:
            child_id = self._add_child_node(
                parent_id=node_id,
                raw_label=str(child.get("label") or ""),
                description=str(child.get("description") or ""),
                year=year,
                source="width_expansion",
            )
            if child_id:
                added_children.append(child_id)
        if added_children:
            self.trace_by_time[str(year)]["width_expansions"].append(
                {
                    "node_id": node_id,
                    "node_path": self._node_path(node_id),
                    "new_children": [
                        {
                            "node_id": child_id,
                            "label": self.nodes[child_id].label,
                            "display_name": self.nodes[child_id].display_name,
                        }
                        for child_id in added_children
                    ],
                    "candidate_counts": candidate_counts,
                    "unlabeled_count": len(unlabeled_ids),
                }
            )
            self._log(f"width expansion added: node={node_id} children={len(added_children)}")
        return bool(added_children)

    def _expand_depth(self, node_id: str, year: Any) -> bool:
        node = self.nodes[node_id]
        if node.last_depth_attempt_year == year:
            return False
        node.last_depth_attempt_year = year
        if node.child_ids or node.level >= self.max_depth or len(node.paper_ids) <= self.max_density:
            return False
        self._log(f"depth expansion check: node={node_id} papers={len(node.paper_ids)}")
        candidate_counts = self._count_candidate_labels(node=node, paper_ids=node.paper_ids, mode="depth")
        if not candidate_counts:
            self._log(f"depth expansion skipped: node={node_id} no_candidates")
            return False
        obj = self._call_json(self._cluster_prompt(node=node, candidate_counts=candidate_counts, mode="depth"), temperature=self.temperature_clustering)
        children = obj.get("children") or []
        added_children: List[str] = []
        for child in children:
            child_id = self._add_child_node(
                parent_id=node_id,
                raw_label=str(child.get("label") or ""),
                description=str(child.get("description") or ""),
                year=year,
                source="depth_expansion",
            )
            if child_id:
                added_children.append(child_id)
        if added_children:
            self.trace_by_time[str(year)]["depth_expansions"].append(
                {
                    "node_id": node_id,
                    "node_path": self._node_path(node_id),
                    "new_children": [
                        {
                            "node_id": child_id,
                            "label": self.nodes[child_id].label,
                            "display_name": self.nodes[child_id].display_name,
                        }
                        for child_id in added_children
                    ],
                    "paper_count": len(node.paper_ids),
                    "candidate_counts": candidate_counts,
                }
            )
            self._log(f"depth expansion added: node={node_id} children={len(added_children)}")
        return bool(added_children)

    def _clear_descendant_papers(self, node_id: str) -> None:
        queue = list(self.nodes[node_id].child_ids)
        while queue:
            child_id = queue.pop()
            child = self.nodes[child_id]
            child.paper_ids = set()
            queue.extend(child.child_ids)

    def _process_node(
        self,
        node_id: str,
        active_paper_ids: Set[str],
        year: Any,
        *,
        reset_subtree: bool,
    ) -> None:
        node = self.nodes[node_id]
        if reset_subtree:
            self._clear_descendant_papers(node_id)
            active_paper_ids = set(node.paper_ids)

        if not node.child_ids:
            if self._expand_depth(node_id, year):
                self._process_node(node_id, set(node.paper_ids), year, reset_subtree=True)
            return

        assignments = self._classify_into_children(node_id, active_paper_ids)
        grouped: Dict[str, Set[str]] = defaultdict(set)
        for paper_id, child_ids in assignments.items():
            for child_id in child_ids:
                grouped[child_id].add(paper_id)
                self.nodes[child_id].paper_ids.add(paper_id)

        if self._expand_width(node_id, year):
            self._process_node(node_id, set(node.paper_ids), year, reset_subtree=True)
            return

        for child_id, new_ids in grouped.items():
            self._process_node(child_id, new_ids, year, reset_subtree=False)

        for child_id in node.child_ids:
            child = self.nodes[child_id]
            if child.child_ids:
                continue
            if len(child.paper_ids) > self.max_density and child.level < self.max_depth:
                self._process_node(child_id, set(child.paper_ids), year, reset_subtree=False)

    def _unlabeled_paper_ids(self, node_id: str) -> Set[str]:
        node = self.nodes[node_id]
        covered: Set[str] = set()
        for child_id in node.child_ids:
            covered.update(self.nodes[child_id].paper_ids)
        return set(node.paper_ids) - covered

    def _paper_leaf_assignments(self, paper_id: str, dimension_id: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for node in self.nodes.values():
            if node.dimension_id != dimension_id:
                continue
            if paper_id not in node.paper_ids:
                continue
            child_hits = any(paper_id in self.nodes[child_id].paper_ids for child_id in node.child_ids)
            if child_hits:
                continue
            results.append(
                {
                    "node_id": node.node_id,
                    "label": node.label,
                    "display_name": node.display_name,
                    "path": self._node_path(node.node_id),
                    "level": node.level,
                }
            )
        results.sort(key=lambda row: (row["level"], row["path"], row["label"]))
        return results

    def _flatten_nodes(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for node_id in sorted(self.nodes):
            node = self.nodes[node_id]
            rows.append(
                {
                    "node_id": node.node_id,
                    "dimension_id": node.dimension_id,
                    "label": node.label,
                    "display_name": node.display_name,
                    "description": node.description,
                    "level": node.level,
                    "created_year": node.created_year,
                    "created_time_slice": node.created_time_slice,
                    "source": node.source,
                    "parent_id": node.parent_id,
                    "child_ids": list(node.child_ids),
                    "paper_count": len(node.paper_ids),
                }
            )
        return rows

    def _node_to_nested(self, node_id: str) -> Dict[str, Any]:
        node = self.nodes[node_id]
        sample_rows = self._pick_sample_papers(node.paper_ids, 5)
        return {
            "node_id": node.node_id,
            "label": node.label,
            "display_name": node.display_name,
            "description": node.description,
            "level": node.level,
            "created_year": node.created_year,
            "created_time_slice": node.created_time_slice,
            "source": node.source,
            "paper_count": len(node.paper_ids),
            "sample_papers": [
                {
                    "paper_id": row["paper_id"],
                    "title": row.get("title"),
                    "published_date": row.get("published_date"),
                    "preferred_cited_by_count": row.get("preferred_cited_by_count"),
                }
                for row in sample_rows
            ],
            "children": [self._node_to_nested(child_id) for child_id in sorted(node.child_ids)],
        }

    def _write_jsonl(self, path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_time_snapshot(self, time_label: str, route_map: Dict[str, List[str]]) -> None:
        year_dir = self.output_dir / str(time_label)
        year_dir.mkdir(parents=True, exist_ok=True)

        cumulative_paper_ids = [
            paper_id
            for current_label in self.time_sequence
            if self.time_sequence.index(current_label) <= self.time_sequence.index(time_label)
            for paper_id in self.papers_by_time.get(current_label, [])
        ]
        cumulative_paper_ids = [paper_id for paper_id in cumulative_paper_ids if paper_id in self.paper_map]

        flat_nodes = self._flatten_nodes()
        (year_dir / "taxonomy_nodes.json").write_text(json_dumps(flat_nodes), encoding="utf-8")

        nested = {
            dim.id: self._node_to_nested(self.roots[dim.id])
            for dim in self.dimensions
        }
        (year_dir / "taxonomy_snapshot.json").write_text(json_dumps(nested), encoding="utf-8")

        assignment_rows: List[Dict[str, Any]] = []
        for paper_id in cumulative_paper_ids:
            paper = self.paper_map[paper_id]
            dimension_membership = sorted(self.paper_dimension_membership.get(paper_id) or [])
            dimension_assignments = {
                dim_id: self._paper_leaf_assignments(paper_id, dim_id)
                for dim_id in dimension_membership
            }
            assignment_rows.append(
                {
                    "paper_id": paper_id,
                    "title": paper.get("title"),
                    "published_date": paper.get("published_date"),
                    "chronology_year": paper.get("chronology_year"),
                    "chronology_slice": paper.get("chronology_slice"),
                    "preferred_cited_by_count": paper.get("preferred_cited_by_count"),
                    "dimension_membership": dimension_membership,
                    "dimension_assignments": dimension_assignments,
                }
            )
        self._write_jsonl(year_dir / "paper_assignments.jsonl", assignment_rows)

        route_rows = [
            {
                "paper_id": paper_id,
                "dimension_membership": sorted(route_map.get(paper_id) or []),
            }
            for paper_id in sorted(route_map)
        ]
        self._write_jsonl(year_dir / "paper_routes.new.jsonl", route_rows)

        summary = {
            "time_slice": time_label,
            "new_paper_count": len(route_map),
            "cumulative_paper_count": len(cumulative_paper_ids),
            "node_count": len(self.nodes),
            "dimension_stats": {
                dim.id: {
                    "node_count": len([node for node in self.nodes.values() if node.dimension_id == dim.id]),
                    "root_paper_count": len(self.nodes[self.roots[dim.id]].paper_ids),
                    "max_depth_observed": max(
                        [node.level for node in self.nodes.values() if node.dimension_id == dim.id] or [0]
                    ),
                }
                for dim in self.dimensions
            },
            "width_expansions": self.trace_by_time[str(time_label)]["width_expansions"],
            "depth_expansions": self.trace_by_time[str(time_label)]["depth_expansions"],
        }
        if str(time_label).isdigit():
            summary["year"] = int(str(time_label))
        (year_dir / "summary.json").write_text(json_dumps(summary), encoding="utf-8")
        self.year_summaries.append(summary)
        self._log(
            f"snapshot written: slice={time_label} cumulative_papers={len(cumulative_paper_ids)} "
            f"node_count={len(self.nodes)}"
        )

    def _write_year_snapshot(self, year: Any, route_map: Dict[str, List[str]]) -> None:
        self._write_time_snapshot(str(year), route_map)
