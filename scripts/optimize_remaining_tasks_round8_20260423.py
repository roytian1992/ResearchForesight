import json
from collections import Counter
from copy import deepcopy
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
TASK_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"


def paper(paper_id, title, note):
    return {
        "paper_id": paper_id,
        "title": title,
        "url": f"https://arxiv.org/abs/{paper_id}",
        "note": note,
    }


UPDATES = {
    "RTLv3-0388": {
        "gold_answer": (
            "A more defensible bottleneck in pre-September 2025 mask-based conditional control is that masks "
            "mainly constrain where content should appear, but they do not reliably encode what semantic content "
            "should populate the region. ControlNet, MultiDiffusion, and Uni-ControlNet improved spatial control, "
            "yet the mask often remained a geometric scaffold rather than a semantically grounded representation, "
            "which led to inconsistent object identity, weak region coherence, or boundary-level artifacts in "
            "complex edits. If that bottleneck were reduced, the clearest near-term opportunity would be "
            "segmentation-aware or joint image-and-mask diffusion systems, where generation and pixel-level "
            "annotation are produced coherently instead of treating masks as an external binary constraint."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by grounding the bottleneck in semantic insufficiency of masks rather than generic controllability language.",
            "historical_support": "moderate",
            "future_support": "strong",
            "changes": [
                "Replaced the vague mask-control answer with a semantics-aware bottleneck tied to ControlNet-style methods.",
                "Linked the opportunity to joint image-and-mask generation using Seg4Diff and JoDiffusion-style follow-on work.",
            ],
        },
    },
    "RTLv3-0396": {
        "gold_answer": (
            "A key unresolved bottleneck in one-step diffusion distillation before September 2025 was that methods "
            "such as Consistency Models, InstaFlow, and Shortcut Models worked best for single-image synthesis, but "
            "their extreme step reduction made it much harder to preserve temporal or multi-view consistency when "
            "extending to video and 3D settings. The core issue was not just raw quality loss, but instability once "
            "structured outputs required causality, long-range coherence, or geometry consistency across views. If "
            "that bottleneck were reduced, the most plausible near-term opportunity would be one-step causal video "
            "generation and very fast 3D scene generation, which is exactly the direction later reflected by "
            "FlashWorld and one-step causal video generation work."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by anchoring the bottleneck in image-to-video or image-to-3D transfer failure rather than generic efficiency rhetoric.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Narrowed the bottleneck to failure of one-step distillation under temporal or multi-view consistency constraints.",
                "Used post-cutoff one-step video and 3D papers as direct support for the opportunity claim.",
            ],
        },
    },
    "RTLv3-0397": {
        "gold_answer": (
            "The most defensible bottleneck was that pre-September 2025 temporal text-video alignment metrics were "
            "still too clip-centric: they could score short motions or local composition, but they did not reliably "
            "measure event ordering, persistence of objects, or cross-shot coherence in longer videos. "
            "T2V-CompBench strengthened compositional evaluation, while papers such as SEINE exposed how much of the "
            "literature still focused on short transitions rather than multi-scene temporal narratives. If this "
            "evaluation bottleneck were reduced, a plausible near-term opportunity would be long-form temporal "
            "alignment benchmarks and metrics that make story-level video generation and unified spacetime modeling "
            "easier to compare and optimize."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 keeps this item but recasts it around long-form evaluation gaps, which is better supported than the earlier overconfident claim.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Rewrote the answer around clip-level versus long-form temporal evaluation rather than pretending the post-cutoff realization was direct.",
                "Softened the opportunity claim to a plausible metric-development direction instead of a near-certain realized wave.",
            ],
        },
    },
    "RTLv3-0403": {
        "question": (
            "Using literature available before August 31, 2025, identify the most consequential unresolved "
            "technical bottleneck in CLIP-based prompt-alignment metrics for visual generative modeling. Then infer "
            "which concrete research opportunity would become meaningfully more viable if that bottleneck were "
            "addressed over the subsequent six months. Ground your answer in recurring failure cases or metric "
            "limitations documented in the historical literature."
        ),
        "gold_answer": (
            "A stronger bottleneck than raw metric latency is weak construct validity: CLIP-based alignment scores "
            "are often too coarse to detect compositional mistakes, attribute-binding failures, or subtle visual "
            "artifacts that humans readily notice. Pre-cutoff work on T2I metrics repeatedly found that CLIP-style "
            "scores can correlate poorly with fine-grained prompt faithfulness once images are visually plausible at "
            "a global level. If that bottleneck were reduced, the clearest opportunity would be reliable metric-guided "
            "post-training, reranking, and inference-time optimization for prompt-faithful image generation and "
            "inpainting, rather than depending on brittle human evaluation or coarse proxy scores."
        ),
        "question_quality": {
            "status": "light_rewrite_round8",
            "summary": "Round 8 makes the prompt more natural by focusing on CLIP-based prompt-alignment metrics instead of the broader and more ambiguous alignment-metrics phrasing.",
            "edits": [
                "Clarified that the task is about CLIP-based prompt-alignment metrics for generative modeling.",
            ],
        },
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 replaces an implausible compute-cost answer with a well-supported metric-validity bottleneck.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Dropped the earlier low-latency claim, which was not the central weakness of CLIP-based evaluation.",
                "Reframed the opportunity around metric-guided optimization and prompt-faithfulness control.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2404.04251", "TS2: Type-Sensitive Text-to-Image Generation Evaluation", "arXiv.org"),
                paper("2411.02437", "TypeScore: Beyond CLIP-Based Metrics for Fine-Grained Text-to-Image Evaluation", "arXiv.org"),
                paper("2412.13989", "What Makes a Good Metric for Text-to-Image Generation?", "arXiv.org"),
            ],
            "future_supporting_papers": [
                paper("2512.21104", "FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting", "Proceedings of the AAAI Conference on Artificial Intelligence"),
                paper("2510.15857", "BLIP3o-NEXT: Next Frontier of Native Image Generation", "arXiv.org"),
            ],
            "audit_note": "Round 8 replaced weak generation-model citations with metric-specific historical support.",
        },
    },
    "RTLv3-0404": {
        "gold_answer": (
            "A historically grounded bottleneck in hybrid autoregressive-diffusion generation is error propagation in "
            "the autoregressive stage: once early discrete tokens or conditions are predicted incorrectly, the later "
            "diffusion refinement stage cannot always recover the right global structure. Show-o and HART make the "
            "hybrid recipe attractive because it is faster than pure diffusion, but the speed gain comes with "
            "fragility to early token mistakes. If this bottleneck were reduced, the clearest near-term opportunity "
            "would be explicit condition-error refinement in hybrid generators, together with extension to more "
            "structured settings such as multi-view or video generation where autoregressive consistency matters even more."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by tying the answer to a concrete and later-realized condition-error bottleneck in hybrid autoregressive-diffusion generation.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Made the bottleneck specifically about unrecoverable autoregressive condition errors.",
                "Aligned the opportunity with later condition-error refinement and multi-view autoregressive diffusion work.",
            ],
        },
    },
    "RTLv3-0405": {
        "gold_answer": (
            "A more defensible bottleneck in cross-modal diffusion before September 2025 was the absence of a shared "
            "and efficient latent representation that preserves structure when moving across modalities such as images, "
            "video, 3D shapes, and sensor data. OmniTokenizer, SlotDiffusion, and SALAD each show pieces of the "
            "problem: tokenizers or latent spaces that work for one modality often lose object structure, part "
            "structure, or cross-modal consistency when transferred elsewhere. If that bottleneck were reduced, a "
            "plausible near-term opportunity would be few-step generation in harder non-RGB domains, especially fast "
            "3D generation and generative modeling for radar or other structured sensor modalities."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 keeps this item but narrows the answer to shared latent representation quality, which is better matched to the cited cross-modal papers.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced the overly broad 'diffusion is slow' claim with a representation-level bottleneck.",
                "Tied the opportunity specifically to few-step 3D and radar generation rather than generic cross-modal acceleration.",
            ],
        },
    },
    "RTLv3-0406": {
        "gold_answer": (
            "The key bottleneck was that early token-merging methods could cut computation, but they were not "
            "sufficiently attention-aware or hardware-aware, so aggressive merging often damaged visual quality and "
            "did not always translate into the best real-world speedups. Token Merging for Fast Stable Diffusion "
            "already exposed the quality-compression trade-off, and later work such as ToMA showed that the merge "
            "strategy itself needed to respect attention structure and GPU efficiency. If this bottleneck were "
            "addressed, the most concrete opportunity would be much higher-ratio token reduction in large diffusion "
            "transformers without large fidelity loss, making fast image and video generation more practical on "
            "resource-constrained hardware."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item with a precise compression-quality bottleneck that is directly supported by ToMA-style follow-on work.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Refined the bottleneck from generic similarity merging to attention-aware and hardware-aware merging.",
                "Linked the opportunity to high-ratio token reduction rather than a vague mobile-deployment claim.",
            ],
        },
    },
    "RTLv3-0407": {
        "gold_answer": (
            "The strongest recurring bottleneck in accelerated autoregressive image generation was the speed-quality "
            "trade-off at high resolution and long sequence length. HART, ImageFolder, and Infinity-style "
            "autoregressive models all improve throughput, but they still expose the same problem: token folding, "
            "coarse compression, or shortcut decoding can remove exactly the detail and consistency that make large "
            "images convincing. If this bottleneck were reduced, the clearest near-term opportunity would be unified "
            "spatiotemporal autoregressive generation, where the same fast sequential backbone extends more naturally "
            "from still images to videos or other longer visual sequences."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by tightening it around the well-supported speed-quality trade-off and its link to later unified autoregressive visual generation.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Condensed the answer around the central acceleration-versus-fidelity trade-off.",
                "Linked the opportunity to later unified spatiotemporal autoregressive models rather than speculative generic expansion.",
            ],
        },
    },
    "RTLv3-0452": {
        "expected_answer_points": [
            "Identifies a specific long-video evidence-selection bottleneck grounded in pre-cutoff literature, rather than a vague claim about video-model weakness.",
            "Explains the failure mode using concrete historical limitations such as key-frame compression, fixed decomposition, or missed temporal evidence.",
            "Infers a near-term opportunity in agentic temporal focusing or active perception for long-video reasoning.",
        ],
        "gold_answer": (
            "A more defensible bottleneck in video understanding and reasoning before November 30, 2025 was static "
            "or weak evidence selection over long videos. Systems such as VideoAgent and OmAgent avoid processing "
            "every frame, but they still depend on compressed summaries, key-frame selection, or fixed decomposition "
            "schemes that can miss the precise temporal evidence needed for downstream reasoning. If that bottleneck "
            "were reduced, the most plausible near-term opportunity would be agentic temporal focusing and active "
            "perception for long-video reasoning, where the system learns to zoom into the right spans or modalities "
            "on demand instead of committing to one early summary."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by grounding it in dynamic evidence selection rather than the earlier specialized-domain knowledge claim.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Replaced the weak internal-knowledge claim with a much better supported temporal-focusing bottleneck.",
                "Used VideoZoomer and active-perception follow-on work as direct support for the opportunity.",
            ],
        },
    },
    "RTLv3-0478": {
        "gold_answer": (
            "A defensible bottleneck in information-extraction instruction tuning before late 2025 was task-definition "
            "and domain fragility: models tuned for one extraction schema or domain often transferred poorly, which "
            "forced repeated domain-specific adaptation and limited the appeal of a single instruction-tuned IE model. "
            "Work on cross-domain NER and task-definition bias both pointed to this mismatch. If that bottleneck were "
            "reduced, the most plausible near-term opportunity would be high-quality synthetic supervision for "
            "low-cost single-shot or lightly tuned IE pipelines, rather than retraining bespoke extraction systems for "
            "every new schema."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 keeps this task but tightens it around schema-transfer and domain-fragility, which is much closer to the cited IE literature.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Replaced the too-narrow dependence-on-domain-models claim with a broader and better grounded transfer bottleneck.",
                "Softened the opportunity to synthetic supervision for low-cost IE adaptation.",
            ],
        },
    },
    "RTLv3-0480": {
        "gold_answer": (
            "A more defensible bottleneck in medical-domain fine-tuning is heuristic data selection that ignores what "
            "the base model already knows, causing redundant examples to dominate scarce adaptation budgets. The "
            "medical data-selection literature explicitly argues that domain adaptation benefits from modeling example "
            "difficulty and the base model's knowledge state rather than just sampling by heuristic relevance. If "
            "that bottleneck were reduced, a plausible near-term opportunity would be far more data-efficient medical "
            "adaptation, including stronger local or specialty-specific medical assistants that need less labeled data "
            "to reach useful domain performance."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 retains the knowledge-aware data-selection idea but rewrites the opportunity more conservatively so it matches the evidence.",
            "historical_support": "strong",
            "future_support": "weak",
            "changes": [
                "Kept the medically relevant data-selection bottleneck but removed overclaiming about immediate realized methods.",
                "Reframed the opportunity as data-efficient specialty adaptation rather than a specific post-cutoff wave.",
            ],
        },
    },
    "RTLv3-0483": {
        "gold_answer": (
            "A more coherent bottleneck in reinforcement-learning-based fine-tuning is that the useful RL skills it "
            "produces are expensive to acquire and hard to transfer across domains or lightweight adaptation setups. "
            "Historical work on parameter-efficient preference alignment repeatedly documents trade-offs between "
            "quality, memory, and compute, which makes it difficult to reuse RL-style improvements broadly. If this "
            "bottleneck were reduced, a plausible near-term opportunity would be modular RL-skill transfer for "
            "continual adaptation, where models inherit reusable RL-acquired behaviors without paying the full "
            "alignment cost each time."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 replaces a loose preference-data narrative with a cleaner transferable-RL-skill bottleneck that better matches both the historical and future evidence.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Reframed the bottleneck around costly and non-transferable RL-acquired skills.",
                "Linked the opportunity to continual-adaptation style RL-skill injection rather than commodity-hardware RL tuning claims.",
            ],
        },
        "ground_truth": {
            "future_supporting_papers": [
                paper("2601.11258", "Knowledge is Not Enough: Injecting RL Skills for Continual Adaptation", "arXiv.org"),
            ],
            "audit_note": "Round 8 narrowed the future support set to the paper that most directly matches transferable RL skill injection.",
        },
    },
    "RTLv3-0504": {
        "gold_answer": (
            "The most defensible bottleneck in biomedical retrieval-augmented adaptation is the difficulty of "
            "combining heterogeneous biomedical evidence sources, such as ontologies, structured records, and free "
            "text, without fragmenting the reasoning chain. OpenTCM and CLI-RAG both point to the need for richer "
            "structure-aware retrieval than generic document RAG provides. If this bottleneck were reduced, the "
            "clearest near-term opportunity would be ontology-aware biomedical RAG systems that coordinate graph, "
            "terminology, and document retrieval inside one reasoning loop. The post-cutoff evidence is still "
            "narrower than ideal, so this should be treated as a plausible opportunity rather than a certainty."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 keeps this task but makes the opportunity claim explicitly cautious while preserving the strong ontology-aware bottleneck.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Retained the heterogeneous-knowledge integration bottleneck from round 2.",
                "Further softened the opportunity claim so it no longer overstates the narrow post-cutoff evidence.",
            ],
        },
    },
    "RTLv3-0505": {
        "question": (
            "Using literature available before December 1, 2025, identify one concrete unresolved bottleneck in "
            "dialogue systems that must combine persona information with retrieved external facts. Then infer which "
            "research opportunity would become more viable if that bottleneck were addressed within the following "
            "three months. Ground your answer in recurring limitations, hallucination patterns, or future-work "
            "signals from the historical literature."
        ),
        "gold_answer": (
            "A better grounded bottleneck is not generic retrieval error alone, but the failure to jointly maintain "
            "persona consistency and factual grounding across dialogue turns. Historical grounding papers show that "
            "retrieval can reduce hallucination, yet persona-aware systems still struggle with deciding which user "
            "context should influence the response and when that context should be overridden by external evidence. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be persona-aware retrieval and "
            "reasoning policies that selectively fuse user-specific memory with cited external facts instead of "
            "treating the two sources as interchangeable context."
        ),
        "question_quality": {
            "status": "light_rewrite_round8",
            "summary": "Round 8 rewrites the question into more natural language by explicitly naming persona information and external facts as the two grounding sources.",
            "edits": [
                "Replaced the abstract 'knowledge grounding' phrasing with a more concrete description of the dialogue setting.",
            ],
        },
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 makes this task usable by narrowing the bottleneck to persona-versus-fact fusion, which fits the dialogue setting better than the earlier generic RAG answer.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Refined the bottleneck from generic wrong retrieval to selective persona-and-fact fusion.",
                "Softened the opportunity claim to persona-aware retrieval policies instead of unrelated multimodal expansion claims.",
            ],
        },
    },
    "RTLv3-0508": {
        "gold_answer": (
            "A well-supported bottleneck in video retrieval augmentation is limited long-video context together with "
            "weak alignment between retrieved evidence and the exact entities or events needed for reasoning. "
            "VideoRAG explicitly targets the context-window problem, while visually aligned long-video retrieval work "
            "shows that plain text summaries can lose the multimodal evidence needed for correct answers. If this "
            "bottleneck were reduced, the clearest near-term opportunity would be fine-grained spatiotemporal video "
            "RAG, including faster and more reliable video QA systems that retrieve the right moments and entities "
            "instead of only coarse video-level context."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by replacing the placeholder gold answer with a concrete long-context and evidence-alignment bottleneck.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Replaced the placeholder answer with a specific long-video context and fine-grained evidence-alignment bottleneck.",
                "Updated the future support set to more relevant post-cutoff video-RAG papers.",
            ],
        },
        "ground_truth": {
            "future_supporting_papers": [
                paper("2512.15940", "R4: Retrieval-Augmented Reasoning for Vision-Language Models in 4D Spatio-Temporal Space", "arXiv.org"),
                paper("2601.01513", "FastV-RAG: Towards Fast and Fine-Grained Video QA with Retrieval-Augmented Generation", "arXiv.org"),
                paper("2601.06037", "TeleMem: Building Long-Term and Multimodal Memory for Agentic AI", "arXiv.org"),
            ],
            "audit_note": "Round 8 replaced a weak future paper with FastV-RAG and R4, which are much closer to the task topic.",
        },
    },
    "RTLv3-0511": {
        "gold_answer": (
            "A central bottleneck in hierarchical graph retrieval is efficient global-to-local search over large "
            "graphs without losing the higher-level community structure that tells the model where to look next. "
            "Pre-cutoff GraphRAG work repeatedly shows that flat retrieval is too shallow, while more structured "
            "approaches still struggle to identify the right graph region accurately and cheaply. If that bottleneck "
            "were reduced, the most plausible near-term opportunity would be deeper multi-stage GraphRAG systems that "
            "retrieve communities first and then adaptively integrate local evidence, rather than treating graph "
            "retrieval as one flat expansion step."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by grounding it in global-to-local graph navigation and adaptive integration, which is directly supported by ArchRAG-style work.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Replaced the earlier vague flat-versus-hierarchical claim with a clearer global-to-local retrieval bottleneck.",
                "Linked the opportunity to deeper staged GraphRAG systems rather than generic adaptive graph retrieval.",
            ],
        },
    },
    "RTLv3-0514": {
        "gold_answer": (
            "A key unresolved bottleneck in multi-turn retrieval-augmented conversational QA before December 2025 was "
            "that systems and benchmarks still handled later-turn questions poorly when the question was "
            "underspecified, non-standalone, or temporarily unanswerable. MTRAG explicitly documents these failure "
            "modes and shows that strong single-turn RAG systems do not automatically transfer to the multi-turn "
            "setting. If this bottleneck were reduced, the clearest near-term opportunity would be benchmark-driven "
            "development of conversation-state-aware retrieval and evaluation methods that decide when to retrieve, "
            "what to retrieve, and when a turn should be deferred or clarified."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this task by moving from a generic benchmark complaint to the specific later-turn failure modes documented by MTRAG.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Narrowed the bottleneck to underspecified, later-turn, and unanswerable conversational cases.",
                "Kept the opportunity benchmark-driven, but tied it to conversation-state-aware retrieval and evaluation.",
            ],
        },
    },
    "RTLv3-0515": {
        "gold_answer": (
            "A more defensible bottleneck in personalized retrieval-augmented conversational QA is shallow personal "
            "context retrieval: systems can store user profiles, but they still struggle to reason over which "
            "personal facts matter and how they should be combined across multiple retrieval steps. PersonaRAG moved "
            "the field toward user-centric retrieval, but later work on personalized QA makes clear that direct "
            "query-to-profile retrieval is not enough for multi-step personal context reasoning. If this bottleneck "
            "were reduced, the most plausible near-term opportunity would be personalized retrieval policies that "
            "explicitly reason about when and how to retrieve personal evidence across steps rather than treating "
            "personalization as one static memory lookup."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by tying the bottleneck to multi-step personal context reasoning, which is better matched to the follow-on literature.",
            "historical_support": "moderate",
            "future_support": "strong",
            "changes": [
                "Refined the bottleneck from a broad personalization gap to shallow personal-context retrieval.",
                "Updated the future support set to a paper directly about multi-step retrieval of personal context.",
            ],
        },
        "ground_truth": {
            "future_supporting_papers": [
                paper("2602.19317", "Learning to Reason for Multi-Step Retrieval of Personal Context in Personalized Question Answering", "unknown"),
                paper("2512.02502", "AskNearby: An LLM-Based Application for Neighborhood Information Retrieval and Personalized Cognitive-Map Recommendations", "GeoAI@SIGSPATIAL"),
            ],
            "audit_note": "Round 8 replaced a weakly matched recommender paper with direct support for multi-step personal-context retrieval.",
        },
    },
    "RTLv3-0516": {
        "gold_answer": (
            "A better grounded bottleneck in retrieval-augmented educational dialogue generation is the failure to "
            "jointly optimize factual grounding, learner modeling, and pedagogical response quality. Historical work "
            "already shows that retrieval can improve groundedness while still conflicting with human preference or "
            "teaching quality, and tutoring-assessment work suggests that retrieval accuracy alone does not solve "
            "pedagogical adaptation. If this bottleneck were reduced, a plausible opportunity would be educational "
            "assistants that explicitly track learner state while remaining retrieval-grounded, but the post-cutoff "
            "evidence is still more deployment-oriented than decisive as a research-wave confirmation."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 keeps this item but makes the opportunity claim explicitly cautious and better aligned with the educational-dialogue literature.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Retained the groundedness-versus-pedagogy bottleneck and added learner modeling more explicitly.",
                "Softened the opportunity claim because the post-cutoff evidence remains deployment-heavy.",
            ],
        },
    },
    "RTLv3-0541": {
        "gold_answer": (
            "A defensible bottleneck in cross-modal temporal alignment assessment is the lack of reliable automated "
            "metrics for fine-grained synchronization across modalities such as audio and video. Pre-cutoff "
            "audio-reactive video and avatar-synthesis papers emphasized synchronization quality, but evaluation still "
            "leaned heavily on human studies or coarse proxy scores that do not robustly capture sub-second "
            "misalignment. If this bottleneck were reduced, the clearest opportunity would be faster iteration on "
            "joint audio-video generation models with synchronization-aware objectives and evaluation pipelines, a "
            "direction that became more plausible with later unified audio-visual generation work."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 keeps the core synchronization-metric idea but removes overclaiming about the certainty of the post-cutoff realization.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Preserved the fine-grained synchronization bottleneck.",
                "Softened the opportunity claim to synchronization-aware audio-video generation and evaluation.",
            ],
        },
    },
    "RTLv3-EXP-VENUE-1113": {
        "question": (
            "Using literature published on or before August 31, 2025, identify the most plausible next-step "
            "research direction in zero-shot task generalization benchmarks for the following six months and name the "
            "likeliest top-tier venue bucket fit. Justify both the direction forecast and the venue fit using only "
            "pre-cutoff evidence, and treat both as probabilistic forecasts rather than certainties."
        ),
        "gold_answer": (
            "The most defensible next-step direction is planning-centric zero-shot task generalization benchmarks, "
            "with AAAI-like venues the likeliest top-tier fit. By August 2025, the strongest pre-cutoff momentum in "
            "this area was shifting from shallow task suites toward benchmarks that stress long-horizon planning, "
            "tool or environment interaction, and richer agent evaluation. Embodied planning is an especially strong "
            "variant of that direction, but the broader planning-benchmark umbrella is better supported by the "
            "historical trajectory and aligns more naturally with AAAI's benchmark-and-agent orientation."
        ),
        "question_quality": {
            "status": "light_rewrite_round8",
            "summary": "Round 8 removes the vague phrase 'this area' and explicitly names zero-shot task generalization benchmarks in the question.",
            "edits": [
                "Made the target topic explicit in the question rather than relying on the title for context.",
            ],
        },
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this venue-forecast item by inheriting the much stronger zero-shot planning evidence chain already established elsewhere in the dataset.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Aligned the answer with the stronger planning-benchmarks evidence used in the adjacent zero-shot planning tasks.",
                "Upgraded the venue forecast from a weak embodied-only claim to a planning-centric AAAI forecast with clearer support.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2308.03688", "AgentBench: Evaluating LLMs as Agents", "International Conference on Learning Representations"),
                paper("2310.04406", "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models", "International Conference on Machine Learning"),
            ],
            "future_supporting_papers": [
                paper("2509.01396", "DeepResearch Arena: The First Exam of LLMs' Research Abilities via Seminar-Grounded Tasks", "AAAI Conference on Artificial Intelligence"),
                paper("2509.03956", "World Model Implanting for Test-time Adaptation of Embodied Agents", "International Conference on Machine Learning"),
                paper("2510.27287", "Can LLMs Help You at Work? A Sandbox for Evaluating LLM Agents in Enterprise Environments", "Conference on Empirical Methods in Natural Language Processing"),
            ],
            "audit_note": "Round 8 supplemented the previously empty paper lists with the same planning-benchmark evidence chain used by the stronger adjacent zero-shot tasks.",
        },
    },
    "RTLv3-EXP-VENUE-1158": {
        "question": (
            "Using literature available before September 1, 2025, which one or two research directions in long video "
            "understanding fine-tuning appear most worth prioritizing for ACL-like venues in the upcoming submission "
            "cycle? Provide a short ranked plan grounded in pre-cutoff evidence and explain why those directions fit "
            "ACL's methodological profile better than nearby alternatives."
        ),
        "gold_answer": (
            "A defensible ACL-oriented ranked plan is: (1) fine-grained temporal grounding and evidence selection for "
            "long videos; (2) streaming or memory-efficient long-video understanding. The first direction is the "
            "stronger ACL fit because it couples language-conditioned reasoning with clear temporal localization and "
            "evaluation methodology. The second is also important, but it tilts slightly more toward systems "
            "efficiency. The immediate post-cutoff window supports both directions through papers on robust temporal "
            "grounding and streaming long-video understanding, so this task is usable once stated as a short ranked plan rather than a ranking over an unstated option set."
        ),
        "expected_answer_points": [
            "Produces a short ranked plan rather than an unstructured list.",
            "Explains why the selected direction(s) fit ACL-like multimodal and sequence-modeling preferences.",
            "Uses pre-cutoff methodological signals and near-term follow-on evidence without pretending the ranking is exact.",
        ],
        "question_quality": {
            "status": "light_rewrite_round8",
            "summary": "Round 8 fixes a structural flaw in the prompt: it previously demanded ranking only listed options even though no options were actually provided.",
            "edits": [
                "Removed the impossible 'rank only the listed options' instruction.",
                "Recast the task as a short ACL-oriented ranked plan.",
            ],
        },
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Round 8 makes this venue-planning item usable by fixing the broken prompt structure and giving a concrete ACL-facing ranked plan.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Rewrote the ill-posed question into an answerable ranked-plan prompt.",
                "Replaced the empty gold answer with two concrete long-video directions and an ACL-specific ordering.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2405.16009", "Streaming Long Video Understanding with Large Language Models", "arXiv.org"),
                paper("2403.10517", "VideoAgent: Long-form Video Understanding with Large Language Model as Agent", "European Conference on Computer Vision"),
                paper("2201.08071", "Temporal Sentence Grounding in Videos: A Survey and Future Directions", "IEEE Transactions on Pattern Analysis and Machine Intelligence"),
            ],
            "audit_note": "Round 8 supplemented the empty historical paper list with long-video understanding and temporal-grounding references that better justify the ACL-facing ranking.",
        },
    },
    "RTLv3-EXP-VENUE-1159": {
        "question": (
            "Using literature available on or before August 31, 2025, which one or two research directions in "
            "multimodal mathematical reasoning fine-tuning appear most worth prioritizing for AAAI-like venues? "
            "Provide a short ranked plan grounded in pre-cutoff evidence and explain why those directions fit "
            "AAAI's methodological profile better than nearby alternatives."
        ),
        "gold_answer": (
            "A defensible AAAI-oriented ranked plan is: (1) self-evolving iterative reflection or reward-guided "
            "fine-tuning for multimodal math reasoning; (2) difficulty-aware data sampling for multimodal "
            "post-training. The first direction more directly addresses the core reasoning failures that pre-cutoff "
            "VLM math papers exposed, while the second improves the training substrate and also surfaced in the "
            "immediate post-cutoff AAAI window. This makes the pair a much more concrete and venue-aligned gold "
            "answer than the earlier placeholder prose."
        ),
        "expected_answer_points": [
            "Produces a short ranked plan rather than an unstructured list.",
            "Explains why the selected direction(s) fit AAAI-like reasoning and benchmark preferences.",
            "Uses pre-cutoff multimodal-math evidence and near-term follow-on signals without overstating certainty.",
        ],
        "question_quality": {
            "status": "light_rewrite_round8",
            "summary": "Round 8 fixes the same structural flaw as the paired long-video task by removing a missing-option ranking instruction.",
            "edits": [
                "Removed the impossible 'rank only the listed options' instruction.",
                "Recast the task as a short AAAI-oriented ranked plan.",
            ],
        },
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Round 8 upgrades this item by inheriting the stronger multimodal-math evidence chain and converting the gold answer into a concrete ranked plan.",
            "historical_support": "moderate",
            "future_support": "strong",
            "changes": [
                "Rewrote the broken ranking prompt into an answerable venue-planning task.",
                "Replaced the placeholder answer with a concrete top-two plan directly supported by adjacent multimodal-math tasks and AAAI-window papers.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2502.11492", "Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding", "Annual Meeting of the Association for Computational Linguistics"),
                paper("2508.08688", "STELAR-VISION: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision", "AAAI Conference on Artificial Intelligence"),
                paper("2504.07491", "Kimi-VL Technical Report", "arXiv.org"),
            ],
            "audit_note": "Round 8 supplemented the empty historical paper list with the multimodal-math evidence chain used by the paired bottleneck task.",
        },
    },
}


def apply_updates(record):
    task_id = record["task_id"]
    spec = UPDATES[task_id]

    for key in ["question", "gold_answer", "expected_answer_points", "question_quality", "answer_audit"]:
        if key in spec:
            record[key] = deepcopy(spec[key])

    if "ground_truth" in spec:
        gt = record.setdefault("ground_truth", {})
        gt_spec = spec["ground_truth"]
        if "historical_supporting_papers" in gt_spec:
            gt["historical_supporting_papers"] = deepcopy(gt_spec["historical_supporting_papers"])
        if "future_supporting_papers" in gt_spec:
            gt["future_supporting_papers"] = deepcopy(gt_spec["future_supporting_papers"])
        notes = list(gt.get("audit_notes", []))
        audit_note = gt_spec.get("audit_note")
        if audit_note and audit_note not in notes:
            notes.append(audit_note)
        gt["audit_notes"] = notes

    review = record.setdefault("review_metadata", {})
    review["refined_at"] = "2026-04-23"
    review["review_depth"] = "manual_deep_round8"
    review["used_web_verification"] = True
    review["round8_targeted_fix"] = True


def main():
    before = Counter()
    rows = []
    for line in TASK_PATH.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        before[row.get("answer_audit", {}).get("status")] += 1
        if row["task_id"] in UPDATES:
            apply_updates(row)
        rows.append(row)

    after = Counter(row.get("answer_audit", {}).get("status") for row in rows)
    with TASK_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("before", dict(before))
    print("after", dict(after))
    print("updated", len(UPDATES))


if __name__ == "__main__":
    main()
