from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate round-robin pairwise results for benchmark v3.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collapse_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    winners = [str(row.get("winner_method") or "tie") for row in rows]
    non_ties = [w for w in winners if w != "tie"]
    if not non_ties:
        final_winner = "tie"
    elif len(set(non_ties)) == 1:
        final_winner = non_ties[0]
    else:
        final_winner = "tie"
    confidences = [float(row.get("confidence") or 0.0) for row in rows]
    row0 = rows[0]
    return {
        "comparison_key": row0["comparison_key"],
        "task_id": row0["task_id"],
        "family": row0["family"],
        "domain": row0["domain"],
        "methods": sorted({str(row0["method_a"]), str(row0["method_b"])}),
        "instance_count": len(rows),
        "winner_method": final_winner,
        "mean_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        "unstable": len(set(non_ties)) > 1,
    }


def bt_scores(methods: List[str], rows: List[Dict[str, Any]], iters: int = 200, eps: float = 1e-9) -> Dict[str, float]:
    idx = {m: i for i, m in enumerate(methods)}
    n = len(methods)
    w = [[0.0] * n for _ in range(n)]
    mtx = [[0.0] * n for _ in range(n)]
    for row in rows:
        a, b = row["methods"]
        i, j = idx[a], idx[b]
        winner = str(row.get("winner_method") or "tie")
        if winner == a:
            w[i][j] += 1.0
        elif winner == b:
            w[j][i] += 1.0
        else:
            w[i][j] += 0.5
            w[j][i] += 0.5
        mtx[i][j] += 1.0
        mtx[j][i] += 1.0
    pi = [1.0] * n
    for _ in range(iters):
        new_pi = pi[:]
        for i in range(n):
            wi = sum(w[i][j] for j in range(n) if j != i)
            denom = 0.0
            for j in range(n):
                if i == j or mtx[i][j] <= 0:
                    continue
                denom += mtx[i][j] / max(pi[i] + pi[j], eps)
            if wi > 0 and denom > 0:
                new_pi[i] = wi / denom
        s = sum(new_pi)
        if s > 0:
            new_pi = [max(x / s, eps) for x in new_pi]
        delta = max(abs(new_pi[i] - pi[i]) for i in range(n))
        pi = new_pi
        if delta < 1e-8:
            break
    mean_log = sum(math.log(x) for x in pi) / n
    return {methods[i]: round(math.log(pi[i]) - mean_log, 4) for i in range(n)}


def pair_matrix(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    stats: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(lambda: {"a_win": 0.0, "b_win": 0.0, "tie": 0.0, "count": 0.0})
    for row in rows:
        a, b = row["methods"]
        key = (a, b)
        winner = str(row.get("winner_method") or "tie")
        rec = stats[key]
        rec["count"] += 1.0
        if winner == a:
            rec["a_win"] += 1.0
        elif winner == b:
            rec["b_win"] += 1.0
        else:
            rec["tie"] += 1.0
    out: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (a, b), rec in sorted(stats.items()):
        count = max(rec["count"], 1.0)
        out[a][b] = {
            "win_rate": round(rec["a_win"] / count, 4),
            "loss_rate": round(rec["b_win"] / count, 4),
            "tie_rate": round(rec["tie"] / count, 4),
            "count": int(rec["count"]),
        }
        out[b][a] = {
            "win_rate": round(rec["b_win"] / count, 4),
            "loss_rate": round(rec["a_win"] / count, 4),
            "tie_rate": round(rec["tie"] / count, 4),
            "count": int(rec["count"]),
        }
    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    methods = sorted({m for row in rows for m in row["methods"]})
    return {
        "comparison_count": len(rows),
        "method_count": len(methods),
        "methods": methods,
        "bt_scores": bt_scores(methods, rows),
        "pair_matrix": pair_matrix(rows),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = list(iter_jsonl(input_path))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[str(row["comparison_key"])].append(row)
    collapsed = [collapse_group(rows) for _, rows in sorted(grouped.items())]

    overall = summarize(collapsed)
    by_family = {key: summarize([row for row in collapsed if row["family"] == key]) for key in sorted({row["family"] for row in collapsed})}
    by_domain = {key: summarize([row for row in collapsed if row["domain"] == key]) for key in sorted({row["domain"] for row in collapsed})}
    summary = {
        "input_results": str(input_path),
        "raw_instance_count": len(raw_rows),
        "collapsed_comparison_count": len(collapsed),
        "overall": overall,
        "family_summary": by_family,
        "domain_summary": by_domain,
    }
    (output_dir / "pairwise_collapsed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in collapsed),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
