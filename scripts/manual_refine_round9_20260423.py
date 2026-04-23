import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
TASK_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
LOG_PATH = ROOT / "docs" / "benchmark_refine_log_20260423_round9.md"


def paper(pid, title, note):
    return {
        "paper_id": pid,
        "title": title,
        "url": f"https://arxiv.org/abs/{pid}",
        "note": note,
    }


UPDATES = {
    "RTLv3-0405": {
        "question": "Using literature available before September 2025 on diffusion models that try to share representations across modalities, identify one concrete unresolved bottleneck that repeatedly constrained progress. Then explain which specific research opportunity would most likely become viable if that bottleneck were addressed in the following six months. Ground your answer in recurring limitations, failure cases, or explicit future-work signals from the historical literature.",
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The original prompt was too broad for the evidence actually available. It now states more clearly that the task is about shared cross-modal representations in diffusion frameworks.",
            "edits": [
                "Narrowed 'cross-modal diffusion frameworks' to work that attempts to share representations across modalities.",
                "Removed some of the generic template wording so the bottleneck target is clearer."
            ],
        },
        "ground_truth": {
            "future_supporting_papers": [
                paper("2509.04406", "Few-step Flow for 3D Generation via Marginal-Data Transport Distillation", "Supports the downstream opportunity for transferable few-step generation beyond 2D images."),
                paper("2511.12072", "ProAV-DiT: A Projected Latent Diffusion Transformer for Efficient Synchronized Audio-Video Generation", "Direct support for the post-cutoff move toward unified latent spaces across modalities."),
                paper("2511.07067", "RaLD: Generating High-Resolution 3D Radar Point Clouds with Latent Diffusion", "Supports extension of latent diffusion to non-image sensing modalities.")
            ]
        },
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "The task is now cleaner and the future papers are better matched, but the evidence still spans heterogeneous modalities and does not support a fully crisp bottleneck-to-opportunity chain.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Rewrote the prompt so it matches the evidence around shared cross-modal representations.",
                "Replaced a weaker radar-heavy future set with a better-matched paper on unified audio-video latent modeling, while keeping the task provisional."
            ],
        },
    },
    "RTLv3-0504": {
        "question": "Using literature available before 2025-11-30, identify the most consequential unresolved bottleneck in biomedical retrieval-augmented systems that must combine heterogeneous clinical or biomedical evidence sources. Then explain which concrete research opportunity would most likely open if that bottleneck were addressed over the next three months. Ground the answer in recurring limitations or failure evidence from the historical literature.",
        "gold_answer": "A defensible bottleneck in biomedical retrieval augmentation was the inability to combine heterogeneous evidence sources while preserving biomedical structure. Historical work such as MedGraphRAG, OpenTCM, and CLI-RAG shows that biomedical systems often need to reconcile controlled vocabularies, knowledge graphs, free text, and longitudinal clinical evidence, yet most pipelines still treat retrieval as a loose collection of documents rather than a structured evidence synthesis problem. If that bottleneck were reduced, the clearest near-term opportunity would be structure-aware biomedical GraphRAG systems that explicitly model ontology links, evidence hierarchy, or temporal clinical structure during retrieval and reranking. The post-cutoff literature supports this more directly than before: early 2026 work introduced evidence-graded graph RAG in sports rehabilitation, ontology-aligned clinical KG-RAG construction, and event- and time-aware EHR-RAG for long-horizon clinical prediction.",
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The question now states the heterogeneity problem directly instead of the broader and less specific phrase 'domain adaptation'.",
            "edits": [
                "Made the target problem explicit: combining heterogeneous biomedical evidence sources.",
                "Removed some ambiguity around what kind of retrieval augmentation counts as in-scope."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2408.04187", "Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation", "Direct historical support for graph-structured biomedical retrieval and controlled-vocabulary grounding."),
                paper("2504.20118", "OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis", "Supports biomedical GraphRAG and terminology integration as a core difficulty."),
                paper("2507.06715", "CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs", "Supports clinically structured retrieval rather than flat document-only retrieval.")
            ],
            "future_supporting_papers": [
                paper("2601.00216", "From Evidence-Based Medicine to Knowledge Graph: Retrieval-Augmented Generation for Sports Rehabilitation and a Domain Benchmark", "Direct support for structure-aware graph RAG with evidence hierarchy and domain adaptation."),
                paper("2601.01844", "Clinical Knowledge Graph Construction and Evaluation with Multi-LLMs via Retrieval-Augmented Generation", "Direct support for ontology-aligned clinical KG-RAG."),
                paper("2601.21340", "EHR-RAG: Bridging Long-Horizon Structured Electronic Health Records and Large Language Models via Enhanced Retrieval-Augmented Generation", "Supports structure-aware retrieval over longitudinal clinical evidence.")
            ]
        },
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "After additional manual paper search, this task now has a coherent historical bottleneck and a direct early-2026 follow-on opportunity wave.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Reframed the bottleneck from vague biomedical adaptation to structured synthesis over heterogeneous biomedical evidence.",
                "Added direct post-cutoff support from graph- and structure-aware biomedical RAG papers, which is enough for an upgrade."
            ],
        },
    },
    "RTLv3-0505": {
        "question": "Using literature available before 2025-12-01, identify one concrete unresolved bottleneck in dialogue systems that need to stay consistent with both a user's persona and externally retrieved facts. Then explain which research opportunity would most plausibly open if that bottleneck were addressed within the following three months. Ground your answer in recurring limitations or failure evidence from the historical literature.",
        "gold_answer": "A defensible bottleneck in persona-plus-fact dialogue grounding is that systems still struggle to jointly select the right knowledge source, retrieve the right evidence, and keep the final response consistent with both the user profile and the retrieved facts. Historical work such as UniMS-RAG, citation-enhanced chatbots, and domain-grounded conversational QA makes clear that wrong retrieval, weak attribution, or source conflict quickly contaminates the response. If that bottleneck were reduced, the clearest opportunity would be multi-source persona-conditioned dialogue systems with explicit evidence attribution and consistency checking rather than simple one-shot persona injection. No tightly matched within-window follow-on paper was found, so this task remains provisional despite the cleaner formulation.",
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The revised prompt states the actual technical setting more naturally: dialogue must satisfy both persona consistency and fact grounding.",
            "edits": [
                "Replaced the more awkward phrase 'knowledge grounding for dialogue systems that integrate personas and external facts' with a more direct formulation.",
                "Made the target failure mode about joint consistency across persona and evidence."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2401.13256", "UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems", "Direct historical support for multi-source personalized dialogue retrieval and response consistency."),
                paper("2402.16063", "Citation-Enhanced Generation for LLM-based Chatbots", "Supports evidence attribution as a core grounding problem."),
                paper("2310.09536", "CarExpert: Leveraging Large Language Models for In-Car Conversational Question Answering", "Supports domain-grounded conversational QA under external-knowledge constraints.")
            ],
            "future_supporting_papers": []
        },
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "The task is now better specified, but no closely matched post-cutoff within-window paper was found to support a confident upgrade.",
            "historical_support": "strong",
            "future_support": "weak",
            "changes": [
                "Rewrote the task around joint persona-consistency and fact-grounding rather than a vaguer dialogue-grounding label.",
                "Removed weakly matched future papers instead of pretending the near-term opportunity was directly realized."
            ],
        },
    },
    "RTLv3-0516": {
        "question": "Using literature available before 2025-11-30, identify the most consequential unresolved bottleneck in retrieval-augmented tutoring or educational dialogue systems. Then explain which concrete research opportunity would most likely open if that bottleneck were addressed over the next three months. Ground the answer in recurring limitations or failure evidence from the historical literature.",
        "gold_answer": "A defensible bottleneck in retrieval-augmented educational dialogue is the difficulty of combining factual grounding with learner-state awareness. Historical work such as TutorLLM, RAG-PRISM, and tutoring-assessment papers shows that retrieval can improve factual support, but it does not by itself decide how much help a learner needs, which misconception should be addressed first, or how feedback should be adapted to the learner's current state. If that bottleneck were reduced, the clearest opportunity would be retrieval-augmented tutoring systems that explicitly condition retrieval and response planning on a learner model or knowledge-tracing signal. There is weak within-window support from higher-education RAG chatbot deployments, but not enough to claim a clean research wave, so the task remains provisional.",
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The prompt now names the actual target area more naturally: retrieval-augmented tutoring or educational dialogue systems.",
            "edits": [
                "Replaced the more awkward 'educational dialogue generation' wording with tutoring-oriented dialogue language.",
                "Made the task focus on learner-state-aware retrieval and response planning."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2502.15709", "TutorLLM: Customizing Learning Recommendations with Knowledge Tracing and Retrieval-Augmented Generation", "Direct historical support for learner-state-aware retrieval in education."),
                paper("2509.00646", "RAG-PRISM: A Personalized, Rapid, and Immersive Skill Mastery Framework with Adaptive Retrieval-Augmented Tutoring", "Supports adaptive retrieval-augmented tutoring before the cutoff."),
                paper("2402.14594", "Improving Assessment of Tutoring Practices using Retrieval-Augmented Generation", "Supports the distinction between retrieval quality and pedagogical adaptation.")
            ],
            "future_supporting_papers": [
                paper("2601.14265", "From Textbook to Talkbot: A Case Study of a Greek-Language RAG-Based Chatbot in Higher Education", "Weak but in-window support for curriculum-grounded educational RAG deployment.")
            ]
        },
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "The task is now better aligned with tutoring-specific evidence, but the post-cutoff support is still too application-heavy for an upgrade.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Rewrote the prompt around tutoring dialogue and learner-state awareness.",
                "Replaced an out-of-window future paper with a weaker but in-window educational RAG paper, keeping the task provisional."
            ],
        },
    },
}


def merge_ground_truth(current, patch):
    gt = deepcopy(current) if current else {}
    for key, value in patch.items():
        gt[key] = value
    return gt


def main():
    rows = []
    counts_before = {}
    counts_after = {}
    touched = []

    for raw in TASK_PATH.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        before = row.get("answer_audit", {}).get("status")
        counts_before[before] = counts_before.get(before, 0) + 1

        tid = row["task_id"]
        if tid in UPDATES:
            patch = UPDATES[tid]
            for key in ["question", "gold_answer", "question_quality", "answer_audit"]:
                if key in patch:
                    row[key] = patch[key]
            if "ground_truth" in patch:
                row["ground_truth"] = merge_ground_truth(row.get("ground_truth"), patch["ground_truth"])
            review = deepcopy(row.get("review_metadata", {}))
            review["refined_at"] = "2026-04-23"
            review["review_depth"] = "manual_round9_web_checked"
            review["used_web_verification"] = True
            review["round9_manual_refine"] = True
            row["review_metadata"] = review
            touched.append((tid, row["answer_audit"]["status"], row["answer_audit"]["historical_support"], row["answer_audit"]["future_support"]))

        after = row.get("answer_audit", {}).get("status")
        counts_after[after] = counts_after.get(after, 0) + 1
        rows.append(row)

    TASK_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    lines = [
        "# Benchmark Refine Log 20260423 Round 9",
        "",
        "## Scope",
        "",
        "- This round focuses on the last four weak items after round 8.",
        "- Goal: either upgrade items with newly found direct evidence, or make the remaining provisional items more honest and better specified.",
        "- Method: targeted manual paper search plus question/gold/evidence replacement where the earlier wording was too broad.",
        "",
        "## Files Updated",
        "",
        "- `benchmark_release/task_refined.jsonl`",
        "- `docs/benchmark_refine_log_20260423_round9.md`",
        "- `scripts/manual_refine_round9_20260423.py`",
        "",
        "## Status Counts",
        "",
        f"- Before: {counts_before}",
        f"- After: {counts_after}",
        "",
        "## Manually Re-audited Tasks",
        "",
    ]
    for tid, status, historical, future in touched:
        lines.append(f"- `{tid}`: {status} (historical={historical}, future={future})")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `RTLv3-0504` was upgraded after adding direct early-2026 biomedical GraphRAG / EHR-RAG follow-ons.",
            "- `RTLv3-0505` and `RTLv3-0516` were kept provisional deliberately; weak or out-of-window future papers were removed instead of overstating support.",
            "- `RTLv3-0405` remains provisional because the evidence still spans multiple loosely connected modality families even after narrowing the prompt.",
        ]
    )

    LOG_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
