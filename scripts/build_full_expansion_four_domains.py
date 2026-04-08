from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'tmp' / 'full_expansion_four_domains'

DOMAIN_NAME = {
    'llm_agent': 'LLM agents',
    'llm_finetuning_post_training': 'LLM fine-tuning and post-training',
    'rag_and_retrieval_structuring': 'RAG and retrieval structuring',
    'visual_generative_modeling_and_diffusion': 'Visual generative modeling and diffusion',
}
DOMAIN_ORDER = [
    'llm_agent',
    'llm_finetuning_post_training',
    'rag_and_retrieval_structuring',
    'visual_generative_modeling_and_diffusion',
]
FAMILY_ORDER = [
    'bottleneck_opportunity_discovery',
    'direction_forecasting',
    'strategic_research_planning',
    'venue_aware_research_positioning',
]
STRONG_THRESHOLDS = {
    'bottleneck_opportunity_discovery': 0.90,
    'direction_forecasting': 0.95,
    'strategic_research_planning': 0.90,
}
STRONG_CAPS = {
    'bottleneck_opportunity_discovery': 6,
    'direction_forecasting': 5,
    'strategic_research_planning': 4,
}
VENUE_THRESHOLDS = {
    'venue_aware_direction_forecast': 0.95,
    'venue_targeted_planning': 0.90,
}
RAG_OLD_PUBLIC_IDS = {'RTLv3-0158', 'RTLv3-0159', 'RTLv3-0165', 'RTLv3-0166'}


def candidate_quality_judge(row: Dict[str, Any]) -> Dict[str, Any]:
    return (row.get('candidate_quality_judge') or row.get('judge') or {})


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.open('r', encoding='utf-8')]


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def title_case(text: Any) -> str:
    raw = str(text or '').replace('_', ' ').strip()
    return raw[:1].upper() + raw[1:] if raw else ''


def likely_bucket_and_venue(stats: Dict[str, Any]) -> Tuple[str, str, Dict[str, int]]:
    top_buckets = dict(stats.get('top_venue_buckets') or {})
    conf_buckets = {str(k): int(v) for k, v in top_buckets.items() if str(k) not in {'other', 'unknown'} and int(v or 0) > 0}
    likely_bucket = ''
    if conf_buckets:
        likely_bucket = max(conf_buckets.items(), key=lambda kv: (kv[1], kv[0]))[0]
    top_venues = dict(stats.get('top_venues') or {})
    likely_venue = ''
    if likely_bucket:
        venue_rows = [
            (name, int(count or 0))
            for name, count in top_venues.items()
            if str(name) not in {'arXiv.org', 'unknown'} and int(count or 0) > 0
        ]
        if venue_rows:
            likely_venue = sorted(venue_rows, key=lambda kv: (-kv[1], kv[0]))[0][0]
    return likely_bucket, likely_venue, conf_buckets


def family_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    gt = row.get('ground_truth') or {}
    family = row.get('family')
    if family == 'direction_forecasting':
        return gt.get('future_half_stats') or {}
    if family == 'strategic_research_planning':
        return gt.get('target_window_stats') or {}
    if family == 'bottleneck_opportunity_discovery':
        return gt.get('future_half_stats') or {}
    return {}


def selection_rank(row: Dict[str, Any]) -> Tuple[float, int, float, int, int, str]:
    judge = candidate_quality_judge(row)
    stats = family_stats(row)
    hist = (row.get('support_context') or {}).get('historical_stats') or {}
    return (
        float(judge.get('overall_score') or 0.0),
        int(stats.get('top_conf_count') or 0),
        float(stats.get('top_conf_share') or 0.0),
        int(stats.get('paper_count') or 0),
        int(hist.get('paper_count') or 0),
        str(row.get('title') or ''),
    )


def public_deliverable_spec(family: str, subtype: str | None = None) -> Dict[str, Any]:
    base = {
        'format': 'free_form_research_analysis',
        'requirements': [
            'Use only evidence available up to the cutoff unless the benchmark setting explicitly provides an offline knowledge base.',
            'State a concrete conclusion rather than vague trend language.',
            'Support the conclusion with literature-based reasoning.',
        ],
    }
    if family == 'direction_forecasting' or subtype == 'venue_aware_direction_forecast':
        base['requirements'] += [
            'Name one specific next-step direction and characterize the trajectory.',
            'Identify one likely top-tier venue bucket for that direction.',
        ]
    elif family == 'strategic_research_planning' and subtype == 'comparative_opportunity_prioritization':
        base['requirements'] += [
            'Choose one direction over the alternative rather than hedging.',
            'Justify the comparative priority with evidence-based reasoning.',
            'Explain the trade-off that makes the other option less strategically attractive in the same window.',
        ]
    else:
        base['requirements'] += [
            'Select and justify a small ranked set of priority directions, or identify a focused bottleneck-opportunity argument when the task calls for it.',
            'Make the reasoning structure explicit rather than giving an unstructured list.',
        ]
    return base


def make_public_from_internal(row: Dict[str, Any], task_id: str, family_override: str | None = None, subtype_override: str | None = None) -> Dict[str, Any]:
    family = family_override or row.get('family')
    subtype = subtype_override if subtype_override is not None else row.get('subtype')
    return {
        'task_id': task_id,
        'family': family,
        'subtype': subtype,
        'domain': DOMAIN_NAME.get(str(row.get('domain')), str(row.get('domain'))),
        'horizon': row.get('horizon', 'half_year'),
        'title': row.get('title'),
        'question': row.get('question') or row.get('draft_question'),
        'time_cutoff': ((row.get('time_context') or {}).get('history_end')) or '2025-08-31',
        'deliverable_spec': public_deliverable_spec(family, subtype),
    }


def make_internal_addition(row: Dict[str, Any], public_task_id: str, family_override: str | None = None, subtype_override: str | None = None) -> Dict[str, Any]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    out['task_id'] = public_task_id
    if family_override is not None:
        out['family'] = family_override
    if subtype_override is not None:
        out['subtype'] = subtype_override
    out['public_task_id'] = public_task_id
    return out


def venue_direction_question(topic_title: str) -> str:
    return (
        f'Based on scholarly literature available before September 1, 2025, identify one concrete next-step direction within {topic_title} '
        f'that is most likely to gain traction in top-tier AI venues during the subsequent six-month period. Also identify the most likely venue bucket '
        f'(for example AAAI-like, EMNLP-like, ICLR-like, or similar top-tier venues) where that traction would appear. '
        f'Your answer must be justified only with pre-cutoff evidence.'
    )


def venue_planning_question(topic_title: str, bucket: str) -> str:
    bucket_label = f'{bucket}-like' if bucket else 'top-tier'
    return (
        f'A research team wants to maximize its relevance for {bucket_label} venues in the next submission cycle. '
        f'Based only on literature available before September 1, 2025, which one or two next-step directions in {topic_title} should be prioritized, '
        f'and what evidence-based rationale supports that ranking?'
    )


def make_extracted_venue_task(public_row: Dict[str, Any], hidden_row: Dict[str, Any], trace_row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    subtype = str(public_row.get('subtype'))
    internal = {
        'task_id': public_row.get('task_id'),
        'family': 'venue_aware_research_positioning',
        'subtype': subtype,
        'domain': hidden_row.get('domain'),
        'horizon': public_row.get('horizon', 'half_year'),
        'title': public_row.get('title'),
        'question': public_row.get('question'),
        'gold_answer': hidden_row.get('gold_answer'),
        'expected_answer_points': hidden_row.get('expected_answer_points') or [],
        'ground_truth': hidden_row.get('ground_truth') or {},
        'support_context': trace_row.get('support_context') or {},
        'time_context': trace_row.get('time_context') or {'history_end': public_row.get('time_cutoff', '2025-08-31')},
        'seed': trace_row.get('seed'),
        'public_metadata': hidden_row.get('public_metadata') or {},
        'quality_signals': {
            **(trace_row.get('quality_signals') or {}),
            'venue_family_status': 'extracted_from_old_release',
        },
        'source_task_id': public_row.get('task_id'),
    }
    public = dict(public_row)
    public['family'] = 'venue_aware_research_positioning'
    return public, internal


def make_new_venue_task_from_source(source_row: Dict[str, Any], public_task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    family = source_row.get('family')
    topic_title = str((source_row.get('public_metadata') or {}).get('topic_title') or (source_row.get('public_metadata') or {}).get('topic') or '')
    gt = json.loads(json.dumps(source_row.get('ground_truth') or {}, ensure_ascii=False))
    support_context = json.loads(json.dumps(source_row.get('support_context') or {}, ensure_ascii=False))
    time_context = json.loads(json.dumps(source_row.get('time_context') or {}, ensure_ascii=False))
    subtype = 'venue_aware_direction_forecast' if family == 'direction_forecasting' else 'venue_targeted_planning'
    if family == 'direction_forecasting':
        stats = gt.get('future_half_stats') or {}
        likely_bucket, likely_venue, top_conf_buckets = likely_bucket_and_venue(stats)
        predicted_direction = title_case((gt.get('future_terminal') or {}).get('display_name') or (((source_row.get('public_metadata') or {}).get('future_themes') or [''])[0]))
        gt['venue_forecast'] = {
            'likely_bucket': likely_bucket,
            'likely_venue': likely_venue,
            'future_top_conf_count': int(stats.get('top_conf_count') or 0),
            'future_top_conf_share': float(stats.get('top_conf_share') or 0.0),
            'top_venue_buckets': top_conf_buckets,
        }
        title = f'Forecasting Top-Venue Traction in {topic_title}'
        question = venue_direction_question(topic_title)
        gold_answer = (
            f'The strongest venue-aware forecast is {predicted_direction}, and the most likely top-tier venue bucket is {likely_bucket}. '
            f'This follows from the pre-cutoff record: the topic already showed enough methodological maturity to support a concrete next step, '
            f'its follow-on trajectory was active rather than flat, and its evaluation or application profile aligned with the style of {likely_bucket}-like venues.'
        )
        expected = [
            'Identifies one concrete next-step direction rather than a broad topic area.',
            'Names one likely top-tier venue bucket and links it to the direction with evidence-based reasoning.',
            'Justifies the forecast using pre-cutoff signals such as historical maturity, methodological branching, evaluation emphasis, or venue profile.',
        ]
        support_context['venue_forecast'] = gt['venue_forecast']
    else:
        stats = gt.get('target_window_stats') or {}
        likely_bucket, likely_venue, top_conf_buckets = likely_bucket_and_venue(stats)
        direction_records = list(gt.get('direction_records') or [])
        direction_records.sort(key=lambda row: int((row or {}).get('future_paper_count') or 0), reverse=True)
        top_directions = [title_case((row or {}).get('display_name') or '') for row in direction_records if title_case((row or {}).get('display_name') or '')]
        if not top_directions:
            top_directions = [title_case(x) for x in ((source_row.get('public_metadata') or {}).get('future_themes') or []) if title_case(x)]
        top_directions = top_directions[:2]
        rank_block = '; then '.join(top_directions) if len(top_directions) > 1 else (top_directions[0] if top_directions else 'the strongest emerging directions')
        gt['target_venue_bucket'] = likely_bucket
        gt['target_venue_name'] = likely_venue
        gt['venue_forecast'] = {
            'likely_bucket': likely_bucket,
            'likely_venue': likely_venue,
            'future_top_conf_count': int(stats.get('top_conf_count') or 0),
            'future_top_conf_share': float(stats.get('top_conf_share') or 0.0),
            'top_venue_buckets': top_conf_buckets,
        }
        title = f'Venue-Targeted Prioritization of Research Directions in {topic_title}'
        question = venue_planning_question(topic_title, likely_bucket)
        gold_answer = (
            f'For a team targeting {likely_bucket}-like venues, the highest-priority directions should be {rank_block}. '
            f'These directions best match the pre-cutoff evidence: they sit closest to the strongest emergent descendants, '
            f'they align with the topic\'s future-work and evaluation signals, and they are the directions most compatible with the realized {likely_bucket}-weighted venue mix.'
        )
        expected = [
            'Produces a ranked research plan rather than an unstructured list.',
            'Targets the venue bucket named in the question and links the ranking to that venue style.',
            'Uses pre-cutoff evidence such as emergent descendants, historical future-work signals, and venue profile to justify the ranking.',
        ]
        support_context['target_venue_bucket'] = likely_bucket
        support_context['target_venue_name'] = likely_venue
    internal = {
        'task_id': public_task_id,
        'family': 'venue_aware_research_positioning',
        'subtype': subtype,
        'domain': source_row.get('domain'),
        'horizon': source_row.get('horizon', 'half_year'),
        'title': title,
        'question': question,
        'gold_answer': gold_answer,
        'expected_answer_points': expected,
        'ground_truth': gt,
        'support_context': support_context,
        'time_context': time_context or {'history_end': '2025-08-31', 'future_window': '2025-09-01_to_2026-02-28'},
        'seed': source_row.get('seed'),
        'public_metadata': {
            **(source_row.get('public_metadata') or {}),
            'task_variant': 'venue_aware_research_positioning',
        },
        'quality_signals': {
            **(source_row.get('quality_signals') or {}),
            'venue_family_status': 'new_addition',
        },
        'source_task_id': source_row.get('task_id'),
    }
    public = {
        'task_id': public_task_id,
        'family': 'venue_aware_research_positioning',
        'subtype': subtype,
        'domain': DOMAIN_NAME[str(source_row.get('domain'))],
        'horizon': source_row.get('horizon', 'half_year'),
        'title': title,
        'question': question,
        'time_cutoff': ((time_context or {}).get('history_end')) or '2025-08-31',
        'deliverable_spec': public_deliverable_spec('venue_aware_research_positioning', subtype),
    }
    return public, internal


def planning_score_from_row(row: Dict[str, Any]) -> float:
    stats = (row.get('ground_truth') or {}).get('target_window_stats') or {}
    hist = (row.get('support_context') or {}).get('historical_stats') or {}
    return round(
        float(stats.get('planning_priority_score') or row.get('planning_priority_score') or 0.0)
        + 2.0 * float(stats.get('trend_signal') or row.get('trend_signal') or 0.0)
        + 2.0 * float(stats.get('top_conf_share') or 0.0)
        + min(3.0, float(stats.get('paper_count') or 0) / 30.0)
        + min(2.0, float(hist.get('paper_count') or 0) / 150.0),
        4,
    )


def path_lcp(a: str, b: str) -> int:
    sa = str(a or '').split('/')
    sb = str(b or '').split('/')
    n = 0
    for x, y in zip(sa, sb):
        if x != y:
            break
        n += 1
    return n


def topic_display_name(row: Dict[str, Any]) -> str:
    meta = row.get('public_metadata') or {}
    text = meta.get('topic_title') or meta.get('topic') or row.get('title') or ''
    return title_case(text)


def compact_candidate_stats(row: Dict[str, Any]) -> Dict[str, Any]:
    hist = (row.get('support_context') or {}).get('historical_stats') or {}
    fut = (row.get('ground_truth') or {}).get('target_window_stats') or {}
    return {
        'historical_paper_count': int(hist.get('paper_count') or 0),
        'historical_top_conf_share': float(hist.get('top_conf_share') or 0.0),
        'future_paper_count': int(fut.get('paper_count') or 0),
        'future_top_conf_count': int(fut.get('top_conf_count') or 0),
        'future_top_conf_share': float(fut.get('top_conf_share') or 0.0),
        'planning_score': planning_score_from_row(row),
    }


def build_planning_node_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'domain': str(row.get('domain')),
        'node_id': str((row.get('seed') or {}).get('node_id') or ''),
        'packet_id': str((row.get('seed') or {}).get('packet_id') or ''),
        'display_name': topic_display_name(row),
        'description': str(((row.get('support_context') or {}).get('node_description')) or ((row.get('public_metadata') or {}).get('topic_title')) or ''),
        'stats': compact_candidate_stats(row),
        'source_row': row,
        'score': planning_score_from_row(row),
    }


def rank_comparative_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    diff = abs(float(a['score']) - float(b['score']))
    min_score = min(float(a['score']), float(b['score']))
    max_score = max(float(a['score']), float(b['score']))
    common = path_lcp(a['node_id'], b['node_id'])
    diff_pref = -abs(diff - 1.25)
    future_sum = float(a['stats']['future_paper_count']) + float(b['stats']['future_paper_count'])
    return (float(common), min_score, diff_pref, max_score, future_sum)


def select_comparative_pairs(node_records: List[Dict[str, Any]], target_count: int = 11) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pairs: List[Tuple[Tuple[float, float, float, float, float], Dict[str, Any], Dict[str, Any]]] = []
    for i in range(len(node_records)):
        for j in range(i + 1, len(node_records)):
            a, b = node_records[i], node_records[j]
            common = path_lcp(a['node_id'], b['node_id'])
            if common < 3:
                continue
            if abs(float(a['score']) - float(b['score'])) < 0.05:
                continue
            pairs.append((rank_comparative_pair(a, b), a, b))
    pairs.sort(key=lambda x: x[0], reverse=True)
    for max_occ in (3, 4, 5, 99):
        chosen: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        usage: Counter[str] = Counter()
        seen = set()
        for _, a, b in pairs:
            key = tuple(sorted((a['node_id'], b['node_id'])))
            if key in seen:
                continue
            if usage[a['node_id']] >= max_occ or usage[b['node_id']] >= max_occ:
                continue
            chosen.append((a, b))
            seen.add(key)
            usage[a['node_id']] += 1
            usage[b['node_id']] += 1
            if len(chosen) >= target_count:
                return chosen
        if chosen:
            best = chosen
    return best if 'best' in locals() else []


def make_comparative_task(a: Dict[str, Any], b: Dict[str, Any], public_task_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    winner, loser = (a, b) if float(a['score']) >= float(b['score']) else (b, a)
    winner_name = winner['display_name']
    loser_name = loser['display_name']
    domain_label = DOMAIN_NAME[str(a['domain'])]
    question = (
        f"Based on scholarly literature available before September 1, 2025, which research direction should be prioritized over the next six months in the {domain_label} domain: "
        f"{a['display_name']} or {b['display_name']}? "
        f"Your answer must justify the choice using only pre-cutoff evidence, including technical maturity, unresolved bottlenecks, emerging momentum, and likely downstream leverage."
    )
    gold_answer = (
        f"The stronger priority is {winner_name}. Relative to {loser_name}, it entered the cutoff with a stronger combination of historical maturity, "
        f"post-cutoff trajectory strength, and research leverage. Historically, {winner_name} had {winner['stats']['historical_paper_count']} papers before the cutoff, compared with "
        f"{loser['stats']['historical_paper_count']} for {loser_name}. In the subsequent six-month window, {winner_name} was associated with "
        f"{winner['stats']['future_paper_count']} papers versus {loser['stats']['future_paper_count']} for {loser_name}. "
        f"A strong answer should therefore prioritize {winner_name} while explicitly explaining the trade-off that makes {loser_name} less strategically attractive under the same cutoff-bound evidence."
    )
    public = {
        'task_id': public_task_id,
        'family': 'strategic_research_planning',
        'subtype': 'comparative_opportunity_prioritization',
        'domain': DOMAIN_NAME[str(a['domain'])],
        'horizon': 'half_year',
        'title': f'Comparative Prioritization: {winner_name} vs. {loser_name}',
        'question': question,
        'time_cutoff': '2025-08-31',
        'deliverable_spec': public_deliverable_spec('strategic_research_planning', 'comparative_opportunity_prioritization'),
    }
    internal = {
        'task_id': public_task_id,
        'family': 'strategic_research_planning',
        'subtype': 'comparative_opportunity_prioritization',
        'domain': str(a['domain']),
        'horizon': 'half_year',
        'title': public['title'],
        'question': question,
        'gold_answer': gold_answer,
        'expected_answer_points': [
            f'Chooses one direction rather than hedging, and makes the comparative priority explicit between {a["display_name"]} and {b["display_name"]}.',
            'Justifies the choice using pre-cutoff evidence about technical maturity, bottlenecks, research momentum, or venue/impact trajectory.',
            'Explains the trade-off: why the deprioritized option is less strategically attractive in the same time window rather than merely describing the winner in isolation.',
        ],
        'time_context': {
            'history_end': '2025-08-31',
            'future_window': '2025-09-01_to_2026-02-28',
        },
        'seed': {
            'pair_node_ids': [a['node_id'], b['node_id']],
            'pair_packet_ids': [a['packet_id'], b['packet_id']],
        },
        'support_context': {
            'candidate_a': {
                'display_name': a['display_name'],
                'description': a['description'],
                'stats': a['stats'],
            },
            'candidate_b': {
                'display_name': b['display_name'],
                'description': b['description'],
                'stats': b['stats'],
            },
        },
        'ground_truth': {
            'winner_node_id': winner['node_id'],
            'winner_display_name': winner_name,
            'loser_node_id': loser['node_id'],
            'loser_display_name': loser_name,
            'winner_score': float(winner['score']),
            'loser_score': float(loser['score']),
            'winner_stats': winner['stats'],
            'loser_stats': loser['stats'],
        },
        'quality_signals': {
            'pair_strength': abs(float(a['score']) - float(b['score'])),
            'comparative_source': 'generic_domain_comparative',
        },
    }
    return public, internal


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    release_public = load_jsonl(ROOT / 'data' / 'releases' / 'benchmark_v3_20260407_venue' / 'tasks.jsonl')
    release_public_by_id = {row['task_id']: row for row in release_public}
    release_hidden = {row['task_id']: row for row in load_jsonl(ROOT / 'data' / 'releases' / 'benchmark_v3_20260407_venue' / 'tasks_hidden_eval.jsonl')}
    release_trace = {row['task_id']: row for row in load_jsonl(ROOT / 'data' / 'releases' / 'benchmark_v3_20260407_venue' / 'tasks_build_trace.jsonl')}
    all_candidates = load_jsonl(ROOT / 'data' / 'task_candidates_v3_20260402' / 'all_candidates.judged.recovered.jsonl')

    release_covered = {
        (row['domain'], row['family'], str((row.get('seed') or {}).get('node_id') or ''))
        for row in release_trace.values()
    }

    candidate_rows = defaultdict(list)
    for row in all_candidates:
        if candidate_quality_judge(row).get('decision') != 'accept':
            continue
        candidate_rows[str(row.get('domain'))].append(row)

    rag_strong = load_jsonl(ROOT / 'tmp' / 'rag_expansion_complete' / 'strong_additions_internal.jsonl')
    public_rows_final: List[Dict[str, Any]] = []
    internal_additions: List[Dict[str, Any]] = []
    ordinary_additions_public: List[Dict[str, Any]] = []
    venue_additions_public: List[Dict[str, Any]] = []
    extracted_venue_public: List[Dict[str, Any]] = []
    comparative_public: List[Dict[str, Any]] = []

    old_venue_source_nodes = defaultdict(lambda: {'venue_aware_direction_forecast': set(), 'venue_targeted_planning': set()})
    old_venue_counts = defaultdict(lambda: Counter())

    # Step 1: rebuild base release public rows, reassign old venue tasks to standalone family.
    for row in release_public:
        subtype = str(row.get('subtype') or '')
        if 'venue' in subtype:
            public2, internal2 = make_extracted_venue_task(row, release_hidden[row['task_id']], release_trace[row['task_id']])
            public_rows_final.append(public2)
            extracted_venue_public.append(public2)
            internal_additions.append(internal2)
            domain_code = internal2['domain']
            old_venue_counts[domain_code][subtype] += 1
            seed_node = str((internal2.get('seed') or {}).get('node_id') or '')
            if seed_node:
                old_venue_source_nodes[domain_code][subtype].add(seed_node)
        else:
            public_rows_final.append(dict(row))

    next_exp_id = 1000

    def new_task_id(prefix: str = 'RTLv3-EXP') -> str:
        nonlocal next_exp_id
        tid = f'{prefix}-{next_exp_id:04d}'
        next_exp_id += 1
        return tid

    # Step 2: ordinary strong additions.
    selected_strong_by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    # generic domains
    for domain in DOMAIN_ORDER:
        if domain == 'rag_and_retrieval_structuring':
            for row in rag_strong:
                selected_strong_by_domain[domain].append(row)
            continue
        per_family_pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        seen_nodes: Dict[str, set[str]] = defaultdict(set)
        for row in candidate_rows[domain]:
            family = str(row.get('family'))
            if family not in STRONG_THRESHOLDS:
                continue
            score = float((candidate_quality_judge(row).get('overall_score')) or 0.0)
            if score < STRONG_THRESHOLDS[family]:
                continue
            node_id = str((row.get('seed') or {}).get('node_id') or '')
            if (domain, family, node_id) in release_covered:
                continue
            if node_id and node_id in seen_nodes[family]:
                continue
            per_family_pool[family].append(row)
        for family, items in per_family_pool.items():
            items.sort(key=selection_rank, reverse=True)
            chosen_nodes = set()
            chosen = []
            for row in items:
                node_id = str((row.get('seed') or {}).get('node_id') or '')
                if node_id and node_id in chosen_nodes:
                    continue
                chosen.append(row)
                if node_id:
                    chosen_nodes.add(node_id)
                if len(chosen) >= STRONG_CAPS[family]:
                    break
            selected_strong_by_domain[domain].extend(chosen)

    # Add ordinary strong additions to outputs.
    for domain in DOMAIN_ORDER:
        for row in selected_strong_by_domain[domain]:
            task_id = new_task_id('RTLv3-EXP')
            public = make_public_from_internal(row, task_id)
            internal = make_internal_addition(row, task_id)
            public_rows_final.append(public)
            ordinary_additions_public.append(public)
            internal_additions.append(internal)

    # Step 3: comparative additions, now applied to all four domains with the same logic.
    planning_node_pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task_id, trace_row in release_trace.items():
        if trace_row.get('family') != 'strategic_research_planning':
            continue
        public_row = release_public_by_id.get(task_id) or {}
        if 'venue' in str(public_row.get('subtype') or ''):
            continue
        hidden_row = release_hidden.get(task_id) or {}
        merged = {
            'task_id': task_id,
            'family': 'strategic_research_planning',
            'domain': trace_row.get('domain'),
            'title': hidden_row.get('title'),
            'seed': trace_row.get('seed'),
            'ground_truth': trace_row.get('ground_truth'),
            'support_context': trace_row.get('support_context'),
            'time_context': trace_row.get('time_context'),
            'public_metadata': hidden_row.get('public_metadata'),
        }
        node_id = str((merged.get('seed') or {}).get('node_id') or '')
        if node_id:
            planning_node_pool[str(merged.get('domain'))].append(build_planning_node_record(merged))
    for domain in DOMAIN_ORDER:
        for row in selected_strong_by_domain[domain]:
            if row.get('family') != 'strategic_research_planning':
                continue
            planning_node_pool[domain].append(build_planning_node_record(row))
    for domain in DOMAIN_ORDER:
        dedup: Dict[str, Dict[str, Any]] = {}
        for rec in planning_node_pool[domain]:
            node_id = rec['node_id']
            if not node_id:
                continue
            prev = dedup.get(node_id)
            if prev is None or float(rec['score']) > float(prev['score']):
                dedup[node_id] = rec
        chosen_pairs = select_comparative_pairs(list(dedup.values()), target_count=11)
        for a, b in chosen_pairs:
            task_id = new_task_id('RTLv3-EXP-COMP')
            public, internal = make_comparative_task(a, b, task_id)
            public_rows_final.append(public)
            comparative_public.append(public)
            internal_additions.append(internal)

    # Step 4: strict-max venue family completion for all four domains.
    for domain in DOMAIN_ORDER:
        for subtype in ('venue_aware_direction_forecast', 'venue_targeted_planning'):
            source_family = 'direction_forecasting' if subtype == 'venue_aware_direction_forecast' else 'strategic_research_planning'
            used_nodes = set(old_venue_source_nodes[domain][subtype])
            pool = []
            for row in candidate_rows[domain]:
                if row.get('family') != source_family:
                    continue
                score = float((candidate_quality_judge(row).get('overall_score')) or 0.0)
                if score < VENUE_THRESHOLDS[subtype]:
                    continue
                stats = family_stats(row)
                bucket, _, _ = likely_bucket_and_venue(stats)
                if not bucket or int(stats.get('top_conf_count') or 0) <= 0:
                    continue
                node_id = str((row.get('seed') or {}).get('node_id') or '')
                if node_id and node_id in used_nodes:
                    continue
                pool.append(row)
            pool.sort(key=selection_rank, reverse=True)
            chosen_nodes = set()
            for row in pool:
                node_id = str((row.get('seed') or {}).get('node_id') or '')
                if node_id and (node_id in used_nodes or node_id in chosen_nodes):
                    continue
                if node_id:
                    chosen_nodes.add(node_id)
                task_id = new_task_id('RTLv3-EXP-VENUE')
                public, internal = make_new_venue_task_from_source(row, task_id)
                public_rows_final.append(public)
                venue_additions_public.append(public)
                internal_additions.append(internal)

    # Summaries.
    domain_counts = defaultdict(Counter)
    subtype_counts = defaultdict(Counter)
    for row in public_rows_final:
        domain_display = str(row.get('domain'))
        domain_counts[domain_display][str(row.get('family'))] += 1
        subtype_counts[domain_display][str(row.get('subtype'))] += 1

    summary = {
        'base_release_total': len(release_public),
        'ordinary_strong_additions': len(ordinary_additions_public),
        'comparative_additions': len(comparative_public),
        'new_venue_family_additions': len(venue_additions_public),
        'final_total': len(public_rows_final),
        'per_domain_family_counts': {domain: dict(domain_counts[domain]) for domain in sorted(domain_counts)},
        'per_domain_subtype_counts': {domain: dict(subtype_counts[domain]) for domain in sorted(subtype_counts)},
        'per_domain_totals': {domain: int(sum(counter.values())) for domain, counter in domain_counts.items()},
    }

    dump_json(OUT_DIR / 'summary.json', summary)
    dump_jsonl(OUT_DIR / 'expanded_tasks_public.jsonl', public_rows_final)
    dump_jsonl(OUT_DIR / 'expanded_internal_additions.jsonl', internal_additions)
    dump_jsonl(OUT_DIR / 'ordinary_strong_additions_public.jsonl', ordinary_additions_public)
    dump_jsonl(OUT_DIR / 'venue_family_new_additions_public.jsonl', venue_additions_public)
    dump_jsonl(OUT_DIR / 'comparative_additions_public.jsonl', comparative_public)

    lines = []
    lines.append('# Full Four-Domain Expansion Summary\n')
    lines.append(f"- base release total: **{len(release_public)}**")
    lines.append(f"- ordinary strong additions: **{len(ordinary_additions_public)}**")
    lines.append(f"- comparative additions: **{len(comparative_public)}**")
    lines.append(f"- new venue-family additions: **{len(venue_additions_public)}**")
    lines.append(f"- final total: **{len(public_rows_final)}**\n")
    lines.append('## Per-domain totals\n')
    lines.append('| Domain | Total | Bottleneck | Forecasting | Planning | Venue-aware |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for domain_code in DOMAIN_ORDER:
        domain_display = DOMAIN_NAME[domain_code]
        counts = domain_counts[domain_display]
        lines.append(
            f"| {domain_display} | {sum(counts.values())} | {counts.get('bottleneck_opportunity_discovery',0)} | {counts.get('direction_forecasting',0)} | {counts.get('strategic_research_planning',0)} | {counts.get('venue_aware_research_positioning',0)} |"
        )
    lines.append('\n## Added task counts by source\n')
    lines.append('| Domain | Ordinary strong | Comparative | New venue |')
    lines.append('|---|---:|---:|---:|')
    ordinary_by_domain = Counter(row['domain'] for row in ordinary_additions_public)
    venue_by_domain = Counter(row['domain'] for row in venue_additions_public)
    comparative_by_domain = Counter(row['domain'] for row in comparative_public)
    for domain_code in DOMAIN_ORDER:
        domain_display = DOMAIN_NAME[domain_code]
        lines.append(
            f"| {domain_display} | {ordinary_by_domain.get(domain_display,0)} | {comparative_by_domain.get(domain_display,0)} | {venue_by_domain.get(domain_display,0)} |"
        )
    lines.append('\n## Venue-aware subtype counts after completion\n')
    lines.append('| Domain | Venue-aware direction forecast | Venue-targeted planning |')
    lines.append('|---|---:|---:|')
    for domain_code in DOMAIN_ORDER:
        domain_display = DOMAIN_NAME[domain_code]
        sc = subtype_counts[domain_display]
        lines.append(
            f"| {domain_display} | {sc.get('venue_aware_direction_forecast',0)} | {sc.get('venue_targeted_planning',0)} |"
        )
    lines.append('\n## Ordinary strong additions preview\n')
    lines.append('| Domain | Family | Title |')
    lines.append('|---|---|---|')
    for row in ordinary_additions_public[:80]:
        lines.append(f"| {row['domain']} | {row['family']} | {row['title']} |")
    (OUT_DIR / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
