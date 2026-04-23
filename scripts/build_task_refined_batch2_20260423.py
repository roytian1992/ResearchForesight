from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
RELEASE_DIR = ROOT / "benchmark_release"
DOCS_DIR = ROOT / "docs"

PUBLIC_TASKS = RELEASE_DIR / "tasks.jsonl"
INTERNAL_TASKS = RELEASE_DIR / "tasks_internal_full.jsonl"
REFINED_TASKS = RELEASE_DIR / "task_refined.jsonl"
LOG_PATH = DOCS_DIR / "benchmark_refine_log_20260423.md"

OLD_TO_NEW_SUBTYPE = {
    "pageindex_grounded_bottleneck": "bottleneck_and_opportunity",
    "q1_pageindex_grounded_bottleneck": "bottleneck_and_opportunity",
    "chain_terminal_forecast": "direction_forecast",
    "q1_terminal_forecast": "direction_forecast",
    "agenda_priority_selection": "agenda_prioritization",
    "comparative_opportunity_prioritization": "opportunity_prioritization",
    "venue_targeted_planning": "venue_aligned_planning",
    "venue_aware_direction_forecast": "venue_aligned_direction_forecast",
}

COMMON_EXPECTED_POINTS = [
    "Name one technically specific unresolved bottleneck and anchor it in recurring pre-cutoff evidence rather than a generic trend claim.",
    "Explain why the bottleneck matters using concrete limitations, failure modes, or methodological constraints documented before the cutoff.",
    "Infer one concrete six-month research opportunity that depends on resolving the bottleneck, and tie that inference to pre-cutoff future-work signals, design dependencies, or underexplored directions.",
]


def _paper(paper_id: str, title: str, url: str, note: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": title,
        "url": url,
        "note": note,
    }


BATCH2 = {
    "RTLv3-0017": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in evaluating medical multimodal models, especially vision-language systems for clinical use. "
            "Support it with recurring limitations or failure cases from the pre-cutoff record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was good but somewhat overloaded. The revision tightens it around evaluation rather than broad medical VLM development.",
            "edits": [
                "Shortened the setup and kept the emphasis on clinical evaluation.",
            ],
        },
        "gold_answer": (
            "A defensible pre-cutoff bottleneck was the lack of evaluation procedures that test whether medical multimodal models produce clinically grounded reasoning rather than superficially plausible outputs. "
            "Historical medical VLM papers largely emphasize accuracy, caption overlap, or task success, while later pathology and reasoning-oriented systems expose how little we can tell about justification quality, clinical plausibility, and robustness under distribution shift. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be reasoning-aware medical multimodal evaluation, including protocols that score diagnostic justification, evidence use, and clinical consistency instead of only final predictions."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original gold answer was mostly sound. The refinement tightens the claim around clinically grounded reasoning evaluation and avoids overstating the maturity of a separate medical-reasoning-evaluation subfield.",
            "historical_support": "moderate",
            "future_support": "strong",
            "changes": [
                "Kept the reasoning-fidelity bottleneck.",
                "Made the future opportunity more concrete and less grandiose.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2404.10237",
                    "Med-MoE: Mixture of Domain-Specific Experts for Lightweight Medical Vision-Language Models",
                    "https://arxiv.org/abs/2404.10237",
                    "Supports the point that strong medical task performance does not by itself resolve how to evaluate clinical reasoning quality.",
                ),
                _paper(
                    "2406.19973",
                    "STLLaVA-Med: Self-Training Large Language and Vision Assistant for Medical Question-Answering",
                    "https://arxiv.org/abs/2406.19973",
                    "Further shows that evaluation often centers on task accuracy rather than detailed clinical justification quality.",
                ),
                _paper(
                    "2505.11404",
                    "Patho-R1: A Multimodal Reinforcement Learning-Based Pathology Expert Reasoner",
                    "https://arxiv.org/abs/2505.11404",
                    "Provides strong late-pre-cutoff evidence that reasoning quality is becoming central in medical multimodal systems.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2510.10052",
                    "Think Twice to See More: Iterative Visual Reasoning in Medical VLMs",
                    "https://arxiv.org/abs/2510.10052",
                    "Strong post-cutoff support for iterative and reasoning-aware medical multimodal evaluation.",
                ),
            ],
            "audit_notes": [
                "A solid item after refinement, though the historical evidence still mixes system-building and evaluation papers.",
            ],
        },
    },
    "RTLv3-0018": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in instruction-tuning protocols for large language models. "
            "Support it with recurring limitations or failure cases from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was clear but verbose. The revision makes the same ask more direct.",
            "edits": [
                "Reduced repeated wording around the cutoff and inference window.",
            ],
        },
        "gold_answer": (
            "A credible pre-cutoff bottleneck was that many instruction-tuning protocols still relied on static instruction-response pairs that underrepresent interactive, multi-step behavior. "
            "That limitation appears in work that depends on distilled or expert-curated single-turn data and in domain-specific dialogue tuning papers that must explicitly add multi-turn supervision to recover conversational competence. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be instruction-tuning pipelines that include interactive trajectories, tool use, or solver-in-the-loop feedback instead of only static supervised pairs."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original answer had the right broad intuition, but its historical support was thinner than advertised and its future claim leaned too specifically toward agentic reasoning. The revised answer stays closer to static-versus-interactive supervision.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Kept the static-pair bottleneck but reduced overclaiming.",
                "Reframed the future opportunity around interactive or solver-in-the-loop instruction tuning rather than a broad agentic wave.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2305.03047",
                    "Principle-Driven Self-Alignment of Language Models from Scratch with Minimal Human Supervision",
                    "https://arxiv.org/abs/2305.03047",
                    "Supports the prevalence of static principle-following supervision and limited interactive structure.",
                ),
                _paper(
                    "2308.03549",
                    "Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-world Multi-turn Dialogue",
                    "https://arxiv.org/abs/2308.03549",
                    "Directly shows that adding real multi-turn dialogue can matter when single-turn tuning is insufficient.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2512.17093",
                    "A Solver-in-the-Loop Framework for Improving LLMs on Answer Set Programming for Logic Puzzle Solving",
                    "https://arxiv.org/abs/2512.17093",
                    "Moderately supports the move toward more interactive and feedback-rich tuning protocols.",
                ),
            ],
            "audit_notes": [
                "This task is usable, but a stronger general-domain historical evidence set would improve it.",
            ],
        },
    },
    "RTLv3-0019": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in domain-specific fine-tuning for large language models. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already fairly natural. The revision mainly standardizes its wording.",
            "edits": [
                "Aligned the wording with the rest of the refined set.",
            ],
        },
        "gold_answer": (
            "A defensible pre-cutoff bottleneck was dependence on domain-specific labeled data and brittle transfer to new domains or modalities. "
            "Historical fine-tuning work on graphs, tables, role-playing, and knowledge injection repeatedly shows that performance gains often rely on specialized supervision and can introduce side effects such as hallucination or weak generalization. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be low-resource domain adaptation pipelines that extend LLMs into new modalities, including audio-heavy settings, without requiring large bespoke labeled corpora."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was directionally right. The refinement tightens the bottleneck around label dependence and brittle transfer, which is better supported by the cited historical papers.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Kept the task-specific-data bottleneck.",
                "Made hallucination and transfer fragility explicit as part of why domain tuning is hard.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2405.05904",
                    "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?",
                    "https://arxiv.org/abs/2405.05904",
                    "Strong evidence that domain adaptation through fine-tuning can distort knowledge and induce hallucination.",
                ),
                _paper(
                    "2311.09206",
                    "TableLlama: Towards Open Large Generalist Models for Tables",
                    "https://arxiv.org/abs/2311.09206",
                    "Supports the need for specialized supervision in domain-specific adaptation.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2512.23808",
                    "MiMo-Audio: Audio Language Models are Few-Shot Learners",
                    "https://arxiv.org/abs/2512.23808",
                    "Moderately supports the opening of lower-resource adaptation opportunities in new modalities.",
                ),
            ],
            "audit_notes": [
                "A reasonable item after refinement, though the future opportunity remains somewhat broad.",
            ],
        },
    },
    "RTLv3-0020": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in retrieval-augmented fine-tuning for large language models. "
            "Support it with recurring limitations or failure cases from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was strong. The revision only trims it slightly.",
            "edits": [
                "Shortened repeated framing about pre-cutoff evidence.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was that retrieval and model adaptation were often integrated awkwardly, either through expensive retrieval-specific modifications or through post-hoc pipelines that did not tightly align retrieval behavior with fine-tuning. "
            "RA-DIT makes this integration problem explicit by treating retrieval augmentation and instruction tuning jointly rather than as separate afterthoughts. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be domain-specific retrieval-augmented post-training pipelines, especially for knowledge-intensive areas such as medicine or technical reasoning."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was broadly solid. The refinement sharpens the bottleneck around integration quality rather than generically expensive architectural changes.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Made RA-DIT-style joint optimization the center of the argument.",
                "Kept the downstream opportunity at the domain-specific pipeline level.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2310.01352",
                    "RA-DIT: Retrieval-Augmented Dual Instruction Tuning",
                    "https://arxiv.org/abs/2310.01352",
                    "Direct evidence that retrieval augmentation and tuning need to be optimized together rather than bolted together post hoc.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.22713",
                    "RAR^2: Retrieval-Augmented Medical Reasoning via Thought-Driven Retrieval",
                    "https://arxiv.org/abs/2509.22713",
                    "Moderately supports domain-specific retrieval-augmented post-training for medical reasoning.",
                ),
            ],
            "audit_notes": [
                "One of the cleaner items in this batch.",
            ],
        },
    },
    "RTLv3-0021": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in domain-specific vision-language fine-tuning for multimodal LLMs. "
            "Support it with recurring limitations or failure cases from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original prompt was fine, but the attached gold answer bundled two different bottlenecks together. The revision asks for a single cleaner bottleneck.",
            "edits": [
                "Encouraged one bottleneck rather than a compound answer.",
            ],
        },
        "gold_answer": (
            "The safest pre-cutoff bottleneck here is scarcity of high-quality domain-specific multimodal instruction data together with brittle cross-modal adaptation. "
            "Historical vision-language fine-tuning papers in specialized domains repeatedly depend on narrow task datasets and lightweight adaptation tricks, which limits transfer and makes it hard to sustain strong visual grounding outside the target task. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be targeted domain-specific multimodal fine-tuning for specialized applications, where better instruction data and more stable cross-modal adaptation could support reliable downstream use."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original gold answer bundled multiple claims and overreached from a relatively weak historical support set. The revised answer is more cautious and centers the better-supported data-scarcity and adaptation problem.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Collapsed the dual bottleneck into one more defensible claim.",
                "Removed the overly specific leap to biology and remote sensing as the uniquely unlocked future directions.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2307.01003",
                    "Visual Instruction Tuning with Polite Flamingo",
                    "https://arxiv.org/abs/2307.01003",
                    "Supports the claim that multimodal instruction tuning depends heavily on the quality and design of visual-language instruction data.",
                ),
                _paper(
                    "2404.16670",
                    "EmoVIT: Revolutionizing Emotion Insights with Visual Instruction Tuning",
                    "https://arxiv.org/abs/2404.16670",
                    "Provides evidence that domain-specific multimodal tuning often relies on narrow specialized data and adaptation choices.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2512.06328",
                    "ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models",
                    "https://arxiv.org/abs/2512.06328",
                    "Moderately supports the continued spread of specialized domain-specific vision-language post-training.",
                ),
            ],
            "audit_notes": [
                "This task still needs a stronger historical paper set during the full cleanup pass.",
            ],
        },
    },
    "RTLv3-0022": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in parameter-efficient fine-tuning for large language models. "
            "Support it with explicit constraints or failure cases from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original prompt was clear, but the evidence attached to this item is much weaker than the title suggests. The revision keeps the task but narrows it to what the current evidence can actually support.",
            "edits": [
                "Made the prompt stricter because the historical support set is thin.",
            ],
        },
        "gold_answer": (
            "With the current evidence, the most cautious defensible bottleneck is that resource-efficient adaptation remained hard in specialized settings where models still needed substantial updates to absorb new reasoning or alignment behavior. "
            "The attached historical evidence is not a strong general PEFT survey, but it does support the broader problem that specialized adaptation can be costly and difficult to scale. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be parameter-efficient post-training for honesty, reasoning, or specialist domains without requiring full fine-tuning."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "This is one of the weakest items structurally. The original gold answer overgeneralized from a very thin historical support set. The revised answer is intentionally cautious and flags the need for better PEFT-specific evidence before this item should be considered stable.",
            "historical_support": "weak",
            "future_support": "moderate",
            "changes": [
                "Downgraded the certainty of the bottleneck claim.",
                "Reframed the opportunity in broader resource-efficient post-training terms.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2409.00101",
                    "NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals",
                    "https://arxiv.org/abs/2409.00101",
                    "Only indirect support: it highlights the adaptation burden in a specialized multimodal domain, not PEFT broadly.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2511.12991",
                    "Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty",
                    "https://arxiv.org/abs/2511.12991",
                    "Moderately supports the future opportunity for PEFT-based honesty or alignment adaptation.",
                ),
            ],
            "audit_notes": [
                "This item should be treated as provisional until stronger PEFT-specific historical papers are added.",
            ],
        },
    },
    "RTLv3-0023": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved security-related bottleneck in reinforcement-learning-based fine-tuning for language models. "
            "Support it with recurring limitations or failure cases from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was clear. The revision just makes the security angle more direct.",
            "edits": [
                "Simplified the wording around logical linkage.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was reward-model overoptimization and reward hacking under distribution shift. "
            "Historical work on reinforced fine-tuning, weight-averaged reward models, and scaling laws for overoptimization all shows that RL-based post-training can exploit shallow reward patterns rather than learn robust aligned behavior. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be more security-aware RL fine-tuning pipelines with stronger reward-model robustness, adversarial stress testing, and safer deployment in high-stakes settings."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer had the right core bottleneck. The revision removes the unsupported claim that medical multimodal applications were the main realized near-term opportunity and instead keeps the opportunity at the safer RL-pipeline level.",
            "historical_support": "strong",
            "future_support": "weak",
            "changes": [
                "Preserved the reward-hacking bottleneck.",
                "Replaced the overly specific future application claim with a security-aware RL pipeline opportunity.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2406.02900",
                    "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms",
                    "https://arxiv.org/abs/2406.02900",
                    "Direct evidence that overoptimization and distribution shift remain central problems in RL-based alignment.",
                ),
                _paper(
                    "2401.12187",
                    "WARM: On the Benefits of Weight Averaged Reward Models",
                    "https://arxiv.org/abs/2401.12187",
                    "Supports the instability and robustness problems of reward modeling.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2511.01934",
                    "Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch",
                    "https://arxiv.org/abs/2511.01934",
                    "Weak-to-moderate support that more ambitious RL pipelines became plausible once optimization issues were handled more carefully.",
                ),
            ],
            "audit_notes": [
                "The bottleneck is strong, but the current future-paper set is not especially security-specific.",
            ],
        },
    },
    "RTLv3-0024": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in domain adaptation through retrieval augmentation. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already good. The revision just shortens it.",
            "edits": [
                "Trimmed duplicated wording.",
            ],
        },
        "gold_answer": (
            "A credible pre-cutoff bottleneck was the mismatch between embedding-level similarity and task-relevant domain alignment. "
            "Historical RAG papers show that retrieval can surface passages that look semantically close yet remain logically, factually, or professionally misaligned with what the downstream task actually needs, especially in specialized domains. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be retrieval-augmented domain adaptation for technical and scientific problem solving, where domain-aware indexing and reasoning-sensitive retrieval matter much more than generic semantic similarity."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was mostly strong. The refinement keeps the vector-similarity-gap intuition but makes the future opportunity less overfitted to a small set of sampled papers.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the alignment-gap bottleneck.",
                "Made the future opportunity scientific and technical domain adaptation more generally rather than naming a single narrow wave.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2403.01432",
                    "Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge",
                    "https://arxiv.org/abs/2403.01432",
                    "Supports the idea that retrieval quality and task fit remain central in less popular or specialized knowledge settings.",
                ),
                _paper(
                    "2409.13731",
                    "KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation",
                    "https://arxiv.org/abs/2409.13731",
                    "Provides direct evidence that professional-domain retrieval needs stronger alignment than vanilla semantic similarity.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2510.00919",
                    "Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving",
                    "https://arxiv.org/abs/2510.00919",
                    "Moderately supports the emergence of high-difficulty scientific domain adaptation through RAG.",
                ),
            ],
            "audit_notes": [
                "A healthy item after a relatively small refinement.",
            ],
        },
    },
    "RTLv3-0025": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in evaluating multi-turn retrieval-augmented dialogue systems. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already natural. The revision mainly standardizes the phrasing.",
            "edits": [
                "Aligned wording with other refined items.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was the lack of robust evaluation benchmarks and diagnostics for multi-turn retrieval behavior, especially in domain-specific settings. "
            "Historical work such as MTRAG and LexRAG shows that multi-turn RAG systems are not well served by single-turn benchmarks, and that legal or other professional conversations need domain-specific evaluation of turn-to-turn retrieval quality and response grounding. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be richer multi-turn evaluation frameworks for domain-specific dialogue, including benchmarks that separately score retrieval carryover, dialogue coherence, and evidence use across turns."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was close, but the multi-agent-evaluation add-on was not really supported. The refined answer keeps the benchmark gap and narrows the opportunity to stronger multi-turn evaluation frameworks.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Removed the unsupported multi-agent evaluation claim.",
                "Centered the answer on multi-turn and domain-specific evaluation design.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2501.03468",
                    "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems",
                    "https://arxiv.org/abs/2501.03468",
                    "Direct evidence that multi-turn RAG requires dedicated evaluation rather than single-turn proxies.",
                ),
                _paper(
                    "2502.20640",
                    "LexRAG: Benchmarking Retrieval-Augmented Generation in Multi-Turn Legal Consultation Conversation",
                    "https://arxiv.org/abs/2502.20640",
                    "Strong support that domain-specific multi-turn evaluation remains underdeveloped.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.21856",
                    "KnowMT-Bench: Benchmarking Knowledge-Intensive Long-Form Question Answering in Multi-Turn Dialogues",
                    "https://arxiv.org/abs/2509.21856",
                    "Moderately supports the emergence of richer multi-turn evaluation benchmarks after the cutoff.",
                ),
            ],
            "audit_notes": [
                "This item looks strong after the refinement.",
            ],
        },
    },
    "RTLv3-0026": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in black-box retrieval augmentation for large language models. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original prompt was understandable, but the attached evidence supports a more modest claim than the original gold answer made. The revision keeps the task while tightening its scope.",
            "edits": [
                "Made the bottleneck about configuration and deployment sensitivity rather than an unsupported leap to citation-aware RAG.",
            ],
        },
        "gold_answer": (
            "A cautious pre-cutoff bottleneck for black-box RAG is the lack of reliable deployment guidance for retrieval configuration and integration choices when the base model cannot be updated internally. "
            "Historical comparisons of retrieval and fine-tuning, plus domain-specific RAG benchmarks, show that performance is highly sensitive to retrieval setup, yet practitioners often lack principled guidance for chunking, query formulation, and evidence presentation in black-box systems. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be more reliable black-box RAG recipes for specialized domains, including traceable or citation-aware retrieval setups."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original answer overclaimed that optimal-configuration issues directly unlocked citation-aware RAG. The revised answer is more cautious and treats citation-aware retrieval as only one possible downstream opportunity rather than the uniquely evidenced next step.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Kept the configuration-sensitivity problem.",
                "Downgraded the certainty of the future opportunity.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2302.00083",
                    "In-Context Retrieval-Augmented Language Models",
                    "https://arxiv.org/abs/2302.00083",
                    "Supports the black-box in-context retrieval setting where the model itself is not updated internally.",
                ),
                _paper(
                    "2402.13178",
                    "Benchmarking Retrieval-Augmented Generation for Medicine",
                    "https://arxiv.org/abs/2402.13178",
                    "Shows that performance depends strongly on practical retrieval design choices in a high-stakes black-box-like setting.",
                ),
                _paper(
                    "2312.05934",
                    "Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs",
                    "https://arxiv.org/abs/2312.05934",
                    "Supports the broader point that retrieval configuration choices materially affect outcomes when the model is not retrained.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2601.06979",
                    "MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education",
                    "https://arxiv.org/abs/2601.06979",
                    "Only weak support: it shows continued deployment of domain-specific black-box RAG rather than directly validating the refined bottleneck.",
                ),
            ],
            "audit_notes": [
                "Another provisional item that would benefit from stronger future evidence during the full pass.",
            ],
        },
    },
    "RTLv3-0027": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in knowledge-graph-based retrieval augmentation for large language models. "
            "Support it with recurring limitations or failure cases from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already strong. The revision mainly trims it.",
            "edits": [
                "Minor wording cleanup.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was that knowledge-graph-based RAG often flattened structured graph information into text or otherwise failed to preserve relational structure during retrieval. "
            "Historical work on customer-service KG-RAG and later analyses of graph-versus-text retrieval both show that preserving relational structure is crucial but technically awkward. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be hybrid graph-text retrieval pipelines that keep graph structure explicit while still supporting flexible natural-language generation."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was already good. The refinement mostly sharpens the structural-preservation claim and keeps the opportunity focused on hybrid graph-text retrieval.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Clarified the structured-data bottleneck.",
                "Kept the future opportunity narrow and well supported.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2404.17723",
                    "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering",
                    "https://arxiv.org/abs/2404.17723",
                    "Shows practical benefits and limitations of knowledge-graph retrieval in real QA settings.",
                ),
                _paper(
                    "2410.20724",
                    "Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation",
                    "https://arxiv.org/abs/2410.20724",
                    "Direct evidence that preserving useful graph structure remains technically delicate.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.04716",
                    "KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering",
                    "https://arxiv.org/abs/2509.04716",
                    "Moderately supports the emergence of stronger hybrid graph-aware retrieval methods.",
                ),
            ],
            "audit_notes": [
                "A healthy item after light refinement.",
            ],
        },
    },
    "RTLv3-0028": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in iterative retrieval-generation pipelines for large language models. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was strong already. The revision only reduces repetition.",
            "edits": [
                "Minor wording cleanup.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was weak adaptivity in successive retrieval steps. "
            "Best-practices work on RAG and later multimodal retrieval-generation benchmarks both show that iterative systems often fail to refine queries effectively based on prior evidence, leading to redundant retrieval or low-precision evidence chains. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be feedback-driven iterative retrieval systems whose query strategy changes in response to intermediate reasoning state, especially in multi-step or multimodal tasks."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was good. The refinement mostly makes the bottleneck more concrete and avoids overselling a broad multimodal wave.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the adaptive-query bottleneck.",
                "Narrowed the future claim to feedback-driven retrieval adaptation.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2407.01219",
                    "Searching for Best Practices in Retrieval-Augmented Generation",
                    "https://arxiv.org/abs/2407.01219",
                    "Supports the claim that iterative retrieval behavior depends heavily on retrieval and query-management design choices.",
                ),
                _paper(
                    "2411.02937",
                    "Benchmarking Multimodal Retrieval Augmented Generation with Dynamic VQA Dataset and Self-adaptive Planning Agent",
                    "https://arxiv.org/abs/2411.02937",
                    "Directly supports the value and difficulty of self-adaptive planning in iterative multimodal retrieval.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.07666",
                    "MoLoRAG: Bootstrapping Document Understanding via Multi-modal Logic-aware Retrieval",
                    "https://arxiv.org/abs/2509.07666",
                    "Moderately supports the continued move toward adaptive logic-aware retrieval after the cutoff.",
                ),
            ],
            "audit_notes": [
                "A solid item after a small cleanup.",
            ],
        },
    },
    "RTLv3-0029": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in graph-based retrieval integration for language models. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already good. The revision mainly shortens it.",
            "edits": [
                "Minor wording cleanup.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was inefficient graph retrieval over rigid or poorly structured graph representations. "
            "Historical work such as GRAG, LightRAG, and later analyses of graph-based RAG shows that graph retrieval can become expensive, brittle, or semantically mismatched when traversal and representation are not adapted to the language model's needs. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be more efficient linear or hierarchical graph retrieval methods that improve scalability and contextual alignment."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was already fairly strong. The refinement tightens the bottleneck around retrieval efficiency and representation quality and keeps the future opportunity aligned with the sampled future papers.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Kept the rigid-representation theme.",
                "Aligned the opportunity more directly with linear and low-cost graph traversal methods.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2405.16506",
                    "GRAG: Graph Retrieval-Augmented Generation",
                    "https://arxiv.org/abs/2405.16506",
                    "Direct evidence that graph retrieval quality and structure design materially affect downstream generation.",
                ),
                _paper(
                    "2410.05779",
                    "LightRAG: Simple and Fast Retrieval-Augmented Generation",
                    "https://arxiv.org/abs/2410.05779",
                    "Supports the importance of efficient retrieval structure and token economy.",
                ),
                _paper(
                    "2410.20724",
                    "Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation",
                    "https://arxiv.org/abs/2410.20724",
                    "Further supports the mismatch problem between graph structure and language-model retrieval needs.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2510.10114",
                    "LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora",
                    "https://arxiv.org/abs/2510.10114",
                    "Strong post-cutoff support for the refined future opportunity.",
                ),
                _paper(
                    "2510.13193",
                    "ReMindRAG: Low-Cost LLM-Guided Knowledge Graph Traversal for Efficient RAG",
                    "https://arxiv.org/abs/2510.13193",
                    "Further strong support for efficient graph traversal as the realized next step.",
                ),
            ],
            "audit_notes": [
                "One of the strongest items in this batch.",
            ],
        },
    },
    "RTLv3-0030": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in retrieval-augmented conversational question answering. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was clear and natural. The revision only standardizes its shape.",
            "edits": [
                "Minor wording cleanup.",
            ],
        },
        "gold_answer": (
            "A defensible pre-cutoff bottleneck was the lack of end-to-end optimization over retrieval and generation across dialogue turns. "
            "Historical work such as IM-RAG and MTRAG shows that conversational QA systems struggle to maintain coherent retrieval state over multiple turns, which leads to context drift and weaker grounding. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be turn-aware conversational QA systems with better cross-turn retrieval control, personalization, and state tracking."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was broadly reasonable. The revision keeps the core bottleneck but makes the future opportunity more modest because the sampled post-cutoff evidence is not especially direct.",
            "historical_support": "strong",
            "future_support": "weak",
            "changes": [
                "Kept the end-to-end and cross-turn integration bottleneck.",
                "Made the future opportunity more cautious.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2405.13021",
                    "IM-RAG: Multi-Round Retrieval-Augmented Generation Through Learning Inner Monologues",
                    "https://arxiv.org/abs/2405.13021",
                    "Direct evidence that multi-round retrieval and generation need tighter joint optimization.",
                ),
                _paper(
                    "2501.03468",
                    "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems",
                    "https://arxiv.org/abs/2501.03468",
                    "Further shows that multi-turn conversational grounding remains a distinct challenge.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2602.23184",
                    "MTRAG-UN: A Benchmark for Open Challenges in Multi-Turn RAG Conversations",
                    "https://arxiv.org/abs/2602.23184",
                    "Weak-to-moderate support that the area continued to move toward explicit multi-turn evaluation and control.",
                ),
            ],
            "audit_notes": [
                "Useful item, but the current future-paper set should be strengthened later.",
            ],
        },
    },
    "RTLv3-0031": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in retrieval-augmented educational dialogue generation. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already good. The revision only reduces repetition.",
            "edits": [
                "Minor wording cleanup.",
            ],
        },
        "gold_answer": (
            "A credible pre-cutoff bottleneck was reliance on static learner or student models in retrieval-augmented tutoring systems. "
            "Historical work on pedagogical teacher-student agents and grounded educational QA shows that retrieval helps, but systems still struggle to adapt retrieval and explanation strategy as the learner state evolves. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be personalized educational dialogue systems with dynamic learner modeling that conditions both retrieval and response generation on longitudinal student state."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was fairly good. The refinement keeps the learner-model bottleneck and makes the downstream opportunity slightly more conservative.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Kept the static-student-model bottleneck.",
                "Framed the opportunity around dynamic learner-state conditioning rather than claiming it was already fully realized.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2505.19173",
                    "Investigating Pedagogical Teacher and Student LLM Agents: Genetic Adaptation Meets Retrieval Augmented Generation Across Learning Style",
                    "https://arxiv.org/abs/2505.19173",
                    "Direct evidence that learner modeling and adaptation matter in educational RAG systems.",
                ),
                _paper(
                    "2310.03184",
                    "Retrieval-augmented Generation to Improve Math Question-Answering: Trade-offs Between Groundedness and Human Preference",
                    "https://arxiv.org/abs/2310.03184",
                    "Provides background that retrieval alone does not settle personalization and pedagogical quality.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2601.06979",
                    "MedTutor: A Retrieval-Augmented LLM System for Case-Based Medical Education",
                    "https://arxiv.org/abs/2601.06979",
                    "Moderately supports retrieval-augmented tutoring systems becoming more specialized and longitudinal.",
                ),
                _paper(
                    "2510.01800",
                    "REBot: From RAG to CatRAG with Semantic Enrichment and Graph Routing",
                    "https://arxiv.org/abs/2510.01800",
                    "Additional moderate support for richer adaptive retrieval structures in educational dialogue settings.",
                ),
            ],
            "audit_notes": [
                "A decent item, though it would benefit from more explicit tutoring-evaluation historical papers.",
            ],
        },
    },
    "RTLv3-0032": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in ablation or concept-manipulation frameworks for visual generative modeling and diffusion. "
            "Support it with recurring limitations or failure evidence from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original prompt was understandable, but the original gold answer jumped too quickly to reinforcement-learning-based ablation. The revision keeps the task and narrows it to a better-supported editing-consistency problem.",
            "edits": [
                "Focused the prompt on consistent concept manipulation rather than a vague ablation umbrella.",
            ],
        },
        "gold_answer": (
            "A defensible pre-cutoff bottleneck was the trade-off between low-overhead editing and consistency across views, frames, or repeated edits. "
            "MasaCtrl shows how valuable tuning-free control can be for consistent editing, while video-generation work also highlights how maintaining consistency across time remains difficult. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be multi-round scene-editing systems that allow more complex and persistent concept manipulation without heavy per-task tuning."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original answer overclaimed that the natural next step was reinforcement-learning-based ablation. The refined answer stays with the much better supported low-overhead-versus-consistency trade-off and points to multi-round scene editing as the more defensible downstream opportunity.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Dropped the weakly supported RL-ablation claim.",
                "Recentered the item on consistent multi-round concept manipulation.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2304.08465",
                    "MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing",
                    "https://arxiv.org/abs/2304.08465",
                    "Direct evidence for the editing-overhead versus consistency trade-off.",
                ),
                _paper(
                    "2401.09047",
                    "VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models",
                    "https://arxiv.org/abs/2401.09047",
                    "Provides supporting context that consistency over time remains a core challenge in diffusion generation.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2511.13713",
                    "Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine",
                    "https://arxiv.org/abs/2511.13713",
                    "Moderately supports multi-round scene editing as a realized next-step opportunity.",
                ),
            ],
            "audit_notes": [
                "A much cleaner item after removing the unsupported RL leap.",
            ],
        },
    },
}


def load_slice(path: Path, start: int, end: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if i < start:
                continue
            if i > end:
                break
            rows.append(json.loads(line))
    return rows


def load_existing() -> list[dict]:
    with REFINED_TASKS.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_new_records() -> list[dict]:
    public_rows = load_slice(PUBLIC_TASKS, 17, 32)
    internal_rows = load_slice(INTERNAL_TASKS, 17, 32)
    internal_by_id = {row["task_id"]: row for row in internal_rows}

    records = []
    for public_row in public_rows:
        task_id = public_row["task_id"]
        internal_row = internal_by_id[task_id]
        override = BATCH2[task_id]
        record = {
            "task_id": task_id,
            "internal_task_id": internal_row["internal_task_id"],
            "family": public_row["family"],
            "domain": public_row["domain"],
            "horizon": public_row["horizon"],
            "title": public_row["title"],
            "source_release_path": str(RELEASE_DIR),
            "original_subtype": public_row["subtype"],
            "subtype": OLD_TO_NEW_SUBTYPE[public_row["subtype"]],
            "subtype_taxonomy_version": "2026-04-23-pilot-v1",
            "original_question": public_row["question"],
            "question": override["question"],
            "gold_answer_original": internal_row["gold_answer"],
            "gold_answer": override["gold_answer"],
            "expected_answer_points": COMMON_EXPECTED_POINTS,
            "question_quality": override["question_quality"],
            "answer_audit": override["answer_audit"],
            "ground_truth": override["ground_truth"],
            "review_metadata": {
                "pilot_batch": "RTLv3-0017-0032",
                "refined_at": "2026-04-23",
                "review_scope": [
                    "question_quality",
                    "answer_quality",
                    "subtype_remap",
                ],
                "used_web_verification": True,
            },
        }
        records.append(record)
    return records


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_log(path: Path, records: list[dict]) -> None:
    status_counts = Counter(record["answer_audit"]["status"] for record in records)
    known_caveats = [
        "RTLv3-0005",
        "RTLv3-0007",
        "RTLv3-0008",
        "RTLv3-0011",
        "RTLv3-0016",
        "RTLv3-0021",
        "RTLv3-0022",
        "RTLv3-0026",
        "RTLv3-0030",
    ]

    lines = [
        "# Benchmark Refine Log 20260423",
        "",
        "## Scope",
        "",
        "- Source release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/benchmark_release`",
        "- Output file: `benchmark_release/task_refined.jsonl`",
        "- This pilot pass currently refines the first 32 public tasks (`RTLv3-0001` to `RTLv3-0032`).",
        "- Review dimensions: question quality, answer correctness with web verification, and subtype redesign.",
        "- Future novelty cleanup was not rerun in this pass; this is a manual refinement layer on top of the existing release.",
        "- Supplementation was not performed in this pass; weak-support items were flagged for later evidence expansion.",
        "",
        "## Subtype Remap",
        "",
    ]
    for old, new in OLD_TO_NEW_SUBTYPE.items():
        lines.append(f"- `{old}` -> `{new}`")

    lines.extend(
        [
            "",
            "## Batch Summary",
            "",
            f"- Tasks reviewed: {len(records)}",
            f"- `validated_with_refinement`: {status_counts.get('validated_with_refinement', 0)}",
            f"- `substantially_corrected`: {status_counts.get('substantially_corrected', 0)}",
            "",
            "## Per-Task Audit",
            "",
            "| task_id | subtype | audit_status | historical_support | future_support | note |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for record in records:
        lines.append(
            f"| {record['task_id']} | {record['subtype']} | {record['answer_audit']['status']} | "
            f"{record['answer_audit']['historical_support']} | {record['answer_audit']['future_support']} | "
            f"{record['answer_audit']['summary']} |"
        )

    lines.extend(
        [
            "",
            "## Known Caveats",
            "",
            f"- Still needs stronger evidence supplementation in the full pass: {', '.join(f'`{task_id}`' for task_id in known_caveats)}.",
            "- `RTLv3-0008` and `RTLv3-0011` still overlap semantically and should be reconsidered for deduplication later.",
            "- Several original gold answers were directionally plausible but too aggressive about future realization; this pilot consistently made those claims more conservative.",
            "",
            "## Files Written",
            "",
            "- `benchmark_release/task_refined.jsonl`",
            "- `docs/benchmark_refine_log_20260423.md`",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    existing = load_existing()
    existing_ids = {row["task_id"] for row in existing}
    new_records = [row for row in build_new_records() if row["task_id"] not in existing_ids]
    merged = existing + new_records
    merged.sort(key=lambda row: int(row["task_id"].split("-")[1]))
    write_jsonl(REFINED_TASKS, merged)
    write_log(LOG_PATH, merged)
    print(f"Existing rows: {len(existing)}")
    print(f"Added rows: {len(new_records)}")
    print(f"Total rows written: {len(merged)}")


if __name__ == "__main__":
    main()
