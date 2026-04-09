from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS_DIR = ROOT / "scripts"
import sys

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import researchworld.benchmark_v2 as benchmark_v2
import researchworld.family_pipelines as family_pipelines
from build_family_task_candidates_v3 import (
    build_bottleneck_candidate,
    build_direction_candidate,
    build_planning_candidate,
)


FAMILY_BUILDERS = {
    "bottleneck_opportunity_discovery": (
        family_pipelines.build_opportunity_packets,
        build_bottleneck_candidate,
    ),
    "direction_forecasting": (
        family_pipelines.build_chain_packets,
        build_direction_candidate,
    ),
    "strategic_research_planning": (
        family_pipelines.build_planning_packets,
        build_planning_candidate,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cluster-focused expansion candidates on top of the existing benchmark backbone.")
    parser.add_argument(
        "--trajectorylab-root",
        default="/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab",
    )
    parser.add_argument("--clusters-config", default=str(ROOT / "configs" / "domain_expansion_clusters.yaml"))
    parser.add_argument("--release-trace", default=str(ROOT / "data" / "releases" / "benchmark_v3_20260408_expanded" / "tasks_build_trace.jsonl"))
    parser.add_argument("--out-dir", default=str(ROOT / "tmp" / "cluster_expansion_v1"))
    parser.add_argument("--clusters", default="")
    parser.add_argument("--max-seeds-per-cluster-family", type=int, default=18)
    parser.add_argument("--seed-cap-multiplier", type=float, default=2.0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def norm(text: Any) -> str:
    return str(text or "").lower().strip()


def phrase_tokens(text: Any) -> List[str]:
    raw = norm(text).replace("-", " ").replace("_", " ")
    tokens = [tok for tok in raw.split() if len(tok) >= 3 and tok not in {"and", "for", "with", "the"}]
    return tokens


def soft_phrase_match(text: str, phrase: str) -> bool:
    phrase_n = norm(phrase)
    if not phrase_n:
        return False
    if phrase_n in text:
        return True
    tokens = phrase_tokens(phrase_n)
    return bool(tokens) and all(tok in text for tok in tokens)


def packet_text(packet: Dict[str, Any]) -> str:
    titles = " ".join(str(x.get("title") or "") for x in (packet.get("historical_representative_papers") or [])[:8])
    desc_titles = []
    for key in ("emergent_descendants",):
        for row in (packet.get(key) or [])[:8]:
            desc_titles.append(str(row.get("display_name") or ""))
            desc_titles.append(str(row.get("description") or ""))
    return " || ".join(
        [
            str(packet.get("display_name") or ""),
            str(packet.get("description") or ""),
            str(packet.get("node_id") or ""),
            str(packet.get("dimension_id") or ""),
            titles,
            " ".join(desc_titles),
        ]
    ).lower()


def packet_cluster_match(packet: Dict[str, Any], cluster: Dict[str, Any]) -> Dict[str, Any]:
    text = packet_text(packet)
    query_hits = [q for q in (cluster.get("query_terms") or []) if soft_phrase_match(text, str(q))]
    keyword_hits = [k for k in (cluster.get("include_keywords") or []) if soft_phrase_match(text, str(k))]
    # Quality-first gating:
    # - accept if at least one explicit multi-word query phrase matches
    # - or if at least two cluster keywords match
    accepted = bool(query_hits) or len(set(keyword_hits)) >= 2
    historical_stats = packet.get("historical_stats") or {}
    score = (
        120.0 * len(set(query_hits))
        + 25.0 * len(set(keyword_hits))
        + min(80.0, float(packet.get("planning_priority_score") or 0.0))
        + min(80.0, float(packet.get("direction_score") or 0.0) / 4.0)
        + min(80.0, float(packet.get("bottleneck_pressure_score") or 0.0))
        + min(20.0, float(historical_stats.get("paper_count") or 0.0) / 30.0)
    )
    return {
        "accepted": accepted,
        "query_hits": sorted(set(query_hits)),
        "keyword_hits": sorted(set(keyword_hits)),
        "score": round(score, 4),
        "historical_paper_count": int(historical_stats.get("paper_count") or 0),
    }


def load_release_coverage(path: Path) -> set[Tuple[str, str, str]]:
    covered = set()
    for row in load_jsonl(path):
        seed = row.get("seed") or {}
        node_id = str(seed.get("node_id") or "")
        if node_id:
            covered.add((str(row.get("domain") or ""), str(row.get("family") or ""), node_id))
    return covered


def patch_data_roots(trajectorylab_root: Path) -> None:
    benchmark_v2.ROOT = trajectorylab_root
    family_pipelines.ROOT = trajectorylab_root


def select_clusters(cfg: Dict[str, Any], selected_ids: set[str]) -> List[Dict[str, Any]]:
    rows = list(cfg.get("clusters") or [])
    if not selected_ids:
        return rows
    selected = [row for row in rows if str(row.get("cluster_id")) in selected_ids]
    return sorted(selected, key=lambda row: int(row.get("priority") or 999))


def family_seed_cap(cluster: Dict[str, Any], args: argparse.Namespace) -> int:
    target = cluster.get("target_new_tasks_range") or [6, 10]
    hi = int(target[-1] or 10)
    return min(args.max_seeds_per_cluster_family, max(4, int(round(hi * args.seed_cap_multiplier))))


def choose_packets_for_cluster(
    *,
    all_packets: List[Dict[str, Any]],
    cluster: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    domain_id = str(cluster.get("domain_id") or "")
    ranked: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    skipped: List[Dict[str, Any]] = []
    for packet in all_packets:
        if str(packet.get("domain") or "") != domain_id:
            continue
        match = packet_cluster_match(packet, cluster)
        if not match["accepted"]:
            continue
        ranked.append((float(match["score"]), packet, match))
    ranked.sort(
        key=lambda item: (
            -item[0],
            -int((item[1].get("historical_stats") or {}).get("paper_count") or 0),
            str(item[1].get("node_id") or ""),
        )
    )
    chosen = []
    seen_nodes = set()
    for _, packet, match in ranked:
        node_id = str(packet.get("node_id") or "")
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        out = dict(packet)
        out["_cluster_match"] = match
        chosen.append(out)
    return chosen, skipped


def annotate_candidate(row: Dict[str, Any], cluster: Dict[str, Any], packet: Dict[str, Any], family: str) -> Dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    cluster_id = str(cluster.get("cluster_id") or "")
    out["task_id"] = f"{cluster_id}::{row['task_id']}"
    public_metadata = dict(out.get("public_metadata") or {})
    public_metadata["cluster_id"] = cluster_id
    public_metadata["cluster_display_name"] = cluster.get("display_name")
    out["public_metadata"] = public_metadata
    support_context = dict(out.get("support_context") or {})
    support_context["cluster_focus"] = {
        "cluster_id": cluster_id,
        "cluster_display_name": cluster.get("display_name"),
        "query_hits": (packet.get("_cluster_match") or {}).get("query_hits") or [],
        "keyword_hits": (packet.get("_cluster_match") or {}).get("keyword_hits") or [],
    }
    out["support_context"] = support_context
    quality_signals = dict(out.get("quality_signals") or {})
    quality_signals["cluster_id"] = cluster_id
    quality_signals["cluster_match_score"] = float((packet.get("_cluster_match") or {}).get("score") or 0.0)
    quality_signals["cluster_family_source"] = family
    out["quality_signals"] = quality_signals
    out["cluster_id"] = cluster_id
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectorylab_root = Path(args.trajectorylab_root)
    patch_data_roots(trajectorylab_root)

    cfg = load_yaml(Path(args.clusters_config))
    selected_ids = {x.strip() for x in args.clusters.split(",") if x.strip()}
    clusters = select_clusters(cfg, selected_ids)

    all_packets = load_json(trajectorylab_root / "data" / "support_packets" / "all_node_support_packets.json")
    release_coverage = load_release_coverage(Path(args.release_trace))

    manifest: Dict[str, Any] = {
        "trajectorylab_root": str(trajectorylab_root),
        "cluster_count": len(clusters),
        "clusters": [],
        "family_counts": Counter(),
        "candidate_count": 0,
        "deferred_families": ["venue_aware_research_positioning"],
    }
    all_candidates: List[Dict[str, Any]] = []

    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id") or "")
        preferred_families = [str(x) for x in (cluster.get("preferred_families") or [])]
        candidate_packets, _ = choose_packets_for_cluster(all_packets=all_packets, cluster=cluster)
        cluster_cap = family_seed_cap(cluster, args)

        cluster_summary = {
            "cluster_id": cluster_id,
            "domain_id": cluster.get("domain_id"),
            "display_name": cluster.get("display_name"),
            "matched_seed_packets": len(candidate_packets),
            "preferred_families": preferred_families,
            "family_candidates": {},
            "top_seed_samples": [],
        }

        for packet in candidate_packets[:8]:
            cluster_summary["top_seed_samples"].append(
                {
                    "node_id": packet.get("node_id"),
                    "display_name": packet.get("display_name"),
                    "match": packet.get("_cluster_match"),
                }
            )

        for family in preferred_families:
            if family not in FAMILY_BUILDERS:
                cluster_summary["family_candidates"][family] = {
                    "status": "deferred",
                    "reason": "derived after base-family curation",
                    "candidate_count": 0,
                }
                continue

            builder, candidate_builder = FAMILY_BUILDERS[family]
            family_packets_input = []
            for packet in candidate_packets:
                covered_key = (str(packet.get("domain") or ""), family, str(packet.get("node_id") or ""))
                if covered_key in release_coverage:
                    continue
                family_packets_input.append(packet)
                if len(family_packets_input) >= cluster_cap:
                    break

            built_packets = builder(domain=str(cluster.get("domain_id")), packets=family_packets_input)
            family_rows = []
            for built in built_packets:
                seed_packet_id = str(built.get("seed_packet_id") or "")
                source_packet = next((x for x in family_packets_input if str(x.get("packet_id") or "") == seed_packet_id), None)
                if source_packet is None and family_packets_input:
                    source_packet = next((x for x in family_packets_input if str(x.get("node_id") or "") == str(built.get("seed_node_id") or "")), None)
                source_packet = source_packet or {}
                row = candidate_builder(built)
                family_rows.append(annotate_candidate(row, cluster, source_packet, family))

            cluster_summary["family_candidates"][family] = {
                "status": "built",
                "seed_packets_selected": len(family_packets_input),
                "packet_rows_built": len(built_packets),
                "candidate_count": len(family_rows),
            }
            manifest["family_counts"][family] += len(family_rows)
            all_candidates.extend(family_rows)
            dump_jsonl(out_dir / "clusters" / cluster_id / f"{family}.jsonl", family_rows)

        manifest["clusters"].append(cluster_summary)

    all_candidates.sort(key=lambda row: (str(row.get("domain") or ""), str(row.get("family") or ""), str(row.get("task_id") or "")))
    manifest["family_counts"] = dict(manifest["family_counts"])
    manifest["candidate_count"] = len(all_candidates)

    dump_json(out_dir / "manifest.json", manifest)
    dump_jsonl(out_dir / "all_candidates.jsonl", all_candidates)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
