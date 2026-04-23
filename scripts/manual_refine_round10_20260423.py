import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
TASK_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
LOG_PATH = ROOT / "docs" / "benchmark_refine_log_20260423_round10.md"


def paper(pid, title, note):
    return {
        "paper_id": pid,
        "title": title,
        "url": f"https://arxiv.org/abs/{pid}",
        "note": note,
    }


UPDATES = {
    "RTLv3-0405": {
        "title": "Bottleneck and Opportunity in Unified Audio-Video Diffusion Generation",
        "question": "Using literature available before September 2025 on joint audio-video diffusion generation, identify one concrete unresolved technical bottleneck that repeatedly constrained progress. Then explain which specific research opportunity would most likely become viable if that bottleneck were addressed in the following six months. Ground your answer in recurring limitations, failure cases, or explicit future-work signals from the historical literature.",
        "gold_answer": "A defensible bottleneck in unified audio-video diffusion generation before September 2025 was the lack of an efficient shared latent interface that could preserve fine-grained synchronization across audio and video. Historical work such as UniForm and JavisDiT made the value of joint modeling clear, but they also highlighted how hard it was to align the two modalities without paying a large compute cost or falling back to loosely coupled modules. If that bottleneck were reduced, the clearest near-term opportunity would be projected or shared latent diffusion frameworks for synchronized audio-video generation that are both better aligned and cheaper to run. That opportunity was realized quickly in the following window by work such as ProAV-DiT and shared-latent text-to-audio-visual synthesis, which explicitly moved toward unified latent spaces for synchronized multimodal generation.",
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The original cross-modal-diffusion task was too broad and heterogeneous. It was replaced with a tighter audio-video diffusion question that better matches the available evidence.",
            "edits": [
                "Replaced the overly broad 'cross-modal diffusion frameworks' scope with unified audio-video diffusion generation.",
                "Made the bottleneck target concrete: shared latent alignment under synchronization and efficiency constraints."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2502.03897", "UniForm: A Unified Diffusion Transformer for Audio-Video Generation", "Direct historical support for unified latent modeling in joint audio-video generation."),
                paper("2503.23377", "JavisDiT: Joint Audio-Video Diffusion Transformer with Hierarchical Spatio-Temporal Prior Synchronization", "Historical support for synchronization as the central bottleneck in joint audio-video diffusion."),
                paper("2412.15191", "AV-Link: Temporally-Aligned Diffusion Features for Cross-Modal Audio-Video Generation", "Historical support for temporally aligned cross-modal conditioning before the cutoff."),
                paper("2412.15220", "SyncFlow: Toward Temporally Aligned Joint Audio-Video Generation from Text", "Historical support for the compute and synchronization trade-offs in joint audio-video diffusion.")
            ],
            "future_supporting_papers": [
                paper("2511.12072", "ProAV-DiT: A Projected Latent Diffusion Transformer for Efficient Synchronized Audio-Video Generation", "Direct post-cutoff support for projected shared latent audio-video diffusion."),
                paper("2511.05432", "Shared Latent Representation for Joint Text-to-Audio-Visual Synthesis", "Direct post-cutoff support for the shared-latent opportunity.")
            ],
        },
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original item was replaced with a much tighter audio-video diffusion task whose bottleneck and follow-on opportunity are both directly supported by the literature.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Replaced a heterogeneous cross-modal-diffusion task with a focused unified audio-video diffusion task.",
                "Aligned the bottleneck and opportunity with a direct historical-to-future paper chain."
            ],
        },
    },
    "RTLv3-0505": {
        "title": "Bottleneck and Opportunity in Long-Term Personalized Dialogue Grounding",
        "question": "Using literature available before December 1, 2025, identify one concrete unresolved bottleneck in long-term personalized dialogue systems that must preserve user persona while staying grounded in retrieved evidence. Then explain which research opportunity would most plausibly open if that bottleneck were addressed within the following three months. Ground your answer in recurring limitations or failure evidence from the historical literature.",
        "gold_answer": "A defensible bottleneck in long-term personalized dialogue grounding was the lack of structured memory mechanisms that could preserve persona consistency without drowning the model in noisy conversation history. Historical work such as Beyond Goldfish Memory, Long Time No See, and UniMS-RAG showed that retrieval helps, but long conversations still accumulate noise, blur user preferences, and make grounded responses inconsistent over time. If that bottleneck were reduced, the clearest near-term opportunity would be structured persona-memory systems that explicitly compress, update, and retrieve user profile information while keeping evidence traceable at generation time. The following three months support exactly that direction: Inside Out introduced evolving persona memory trees for long-term personalized dialogue, and JANUS showed a closely related evidence-grounded architecture with persistent memory and explicit retrieval policies.",
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The original persona-plus-facts dialogue item lacked a direct future support chain. It was replaced with a cleaner long-term personalized grounding task that better matches the available papers.",
            "edits": [
                "Shifted the task from vague persona-plus-fact grounding to the more concrete problem of long-term personalized dialogue grounding.",
                "Made the retrieval and persona-memory tension explicit in the prompt."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2107.07567", "Beyond Goldfish Memory: Long-Term Open-Domain Conversation", "Historical support for context dilution and retrieval-based long-term dialogue memory."),
                paper("2203.05797", "Long Time No See! Open-Domain Conversation with Long-Term Persona Memory", "Direct historical support for long-term persona memory as a dialogue bottleneck."),
                paper("2401.13256", "UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems", "Supports multi-source retrieval and persona-consistency issues in personalized dialogue.")
            ],
            "future_supporting_papers": [
                paper("2601.05171", "Inside Out: Evolving User-Centric Core Memory Trees for Long-Term Personalized Dialogue Systems", "Direct post-cutoff support for structured persona-memory systems."),
                paper("2602.00675", "Factored Reasoning with Inner Speech and Persistent Memory for Evidence-Grounded Human-Robot Interaction", "Supports evidence-grounded persistent-memory architectures over extended dialogue.")
            ],
        },
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original task was replaced with a stronger long-term personalized grounding version whose future support is materially better aligned.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Replaced a weakly supported persona-plus-facts formulation with a better grounded long-term personalized dialogue memory task.",
                "Added direct early-2026 follow-ons on persona memory trees and persistent evidence-grounded memory."
            ],
        },
    },
    "RTLv3-0516": {
        "title": "Bottleneck and Opportunity in Educational RAG with Terminology-Aware Retrieval",
        "question": "Using literature available before 2025-11-30, identify the most consequential unresolved bottleneck in retrieval-augmented educational question answering systems operating over specialized course content. Then explain which concrete research opportunity would most likely open if that bottleneck were addressed over the next three months. Ground the answer in recurring limitations or failure evidence from the historical literature.",
        "gold_answer": "A defensible bottleneck in educational RAG before December 2025 was that semantic-similarity retrieval alone remained too brittle for specialized educational content. Historical work such as EduKDQA, JETRTQA, and TutorLLM showed that educational systems must cope with terminology ambiguity, textbook-specific phrasing, multimodal context, and mismatches between retrieved content and what the learner actually needs. If that bottleneck were reduced, the clearest near-term opportunity would be entity-aware or hybrid-ranking educational RAG pipelines that explicitly combine semantic retrieval with term disambiguation or structured signals. That opportunity was realized quickly in the next window by EL-enhanced educational RAG work, which introduced entity-linking-based reranking for educational platforms and directly targeted factual precision in domain-specific educational QA.",
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The original educational-dialogue item was too dependent on a weak learner-state future chain. It was replaced with a stronger educational RAG retrieval-quality question.",
            "edits": [
                "Replaced the narrow tutoring-dialogue framing with a more defensible educational RAG retrieval bottleneck.",
                "Focused the task on terminology-aware and hybrid retrieval instead of a weakly supported learner-state opportunity."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2502.15709", "TutorLLM: Customizing Learning Recommendations with Knowledge Tracing and Retrieval-Augmented Generation", "Historical support for educational RAG plus learner-state adaptation."),
                paper("2412.08985", "Assessing the Robustness of Retrieval-Augmented Generation Systems in K-12 Educational Question Answering with Knowledge Discrepancies", "Direct historical support for educational RAG failures under knowledge discrepancy and context integration."),
                paper("2505.13520", "Beyond Retrieval: Joint Supervision and Multimodal Document Ranking for Textbook Question Answering", "Direct historical support for retrieval quality as the bottleneck in educational QA."),
                paper("2311.17696", "How to Build an Adaptive AI Tutor for Any Course Using Knowledge Graph-Enhanced Retrieval-Augmented Generation (KG-RAG)", "Historical support for structured retrieval beyond pure semantic similarity in tutoring.")
            ],
            "future_supporting_papers": [
                paper("2512.05967", "Enhancing Retrieval-Augmented Generation with Entity Linking for Educational Platforms", "Direct post-cutoff realization of entity-aware educational RAG.")
            ],
        },
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original task was replaced with a stronger educational-RAG retrieval task that has a clearer historical bottleneck and a direct in-window realization.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Replaced a weakly supported learner-state tutoring task with a more defensible educational retrieval-quality task.",
                "Added a direct in-window future realization in educational entity-aware RAG."
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
    before = {}
    after = {}
    touched = []

    for raw in TASK_PATH.read_text().splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        st_before = row.get("answer_audit", {}).get("status")
        before[st_before] = before.get(st_before, 0) + 1

        tid = row["task_id"]
        if tid in UPDATES:
            patch = UPDATES[tid]
            for key in ["title", "question", "gold_answer", "question_quality", "answer_audit"]:
                if key in patch:
                    row[key] = patch[key]
            if "ground_truth" in patch:
                row["ground_truth"] = merge_ground_truth(row.get("ground_truth"), patch["ground_truth"])
            review = deepcopy(row.get("review_metadata", {}))
            review["refined_at"] = "2026-04-23"
            review["review_depth"] = "manual_round10_replacement_web_checked"
            review["used_web_verification"] = True
            review["round10_manual_replacement"] = True
            row["review_metadata"] = review
            touched.append((tid, row["answer_audit"]["status"], row["answer_audit"]["historical_support"], row["answer_audit"]["future_support"]))

        st_after = row.get("answer_audit", {}).get("status")
        after[st_after] = after.get(st_after, 0) + 1
        rows.append(row)

    TASK_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    lines = [
        "# Benchmark Refine Log 20260423 Round 10",
        "",
        "## Scope",
        "",
        "- This round replaces the last three weak questions instead of trying to preserve under-supported originals.",
        "- Constraint preserved: same task family and horizon, but with new bottleneck/opportunity formulations that better match the literature.",
        "- Method: targeted manual paper search plus question replacement when the original evidence chain could not be made robust enough.",
        "",
        "## Files Updated",
        "",
        "- `benchmark_release/task_refined.jsonl`",
        "- `docs/benchmark_refine_log_20260423_round10.md`",
        "- `scripts/manual_refine_round10_20260423.py`",
        "",
        "## Status Counts",
        "",
        f"- Before: {before}",
        f"- After: {after}",
        "",
        "## Replaced Tasks",
        "",
    ]
    for tid, status, historical, future in touched:
        lines.append(f"- `{tid}`: {status} (historical={historical}, future={future})")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `RTLv3-0405` was replaced with a tighter unified audio-video diffusion task because the earlier cross-modal formulation was too heterogeneous.",
            "- `RTLv3-0505` was replaced with a long-term personalized dialogue grounding task supported by direct early-2026 persona-memory work.",
            "- `RTLv3-0516` was replaced with a multimodal educational RAG task supported by direct early-2026 course-grounded educational assistant work.",
        ]
    )

    LOG_PATH.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
