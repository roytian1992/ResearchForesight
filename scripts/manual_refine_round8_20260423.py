import json
from copy import deepcopy
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
TASK_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
LOG_PATH = ROOT / "docs" / "benchmark_refine_log_20260423_round8.md"


def paper(pid, title, note):
    return {
        "paper_id": pid,
        "title": title,
        "url": f"https://arxiv.org/abs/{pid}",
        "note": note,
    }


UPDATES = {
    "RTLv3-EXP-VENUE-1113": {
        "question": "Using literature published on or before August 31, 2025, identify the most plausible next concrete benchmark direction within zero-shot task generalization for the following six months and name the likeliest top-tier venue bucket. Justify both the direction forecast and the venue fit using only pre-cutoff evidence, and treat both as probabilistic forecasts rather than certainties.",
        "gold_answer": "The most plausible next-step direction is embodied planning benchmarks. By August 2025, the strongest zero-shot generalization work had already moved beyond static task suites toward planning-intensive agent evaluation, and papers such as AgentBench and Language Agent Tree Search made long-horizon planning failures highly visible. Among the candidate venue buckets, AAAI-like venues are the best fit: the forecasted contribution is an integrated benchmark for interactive agents in embodied or simulated environments, which is closer to AAAI's systems-and-agents profile than to a primarily language-centric venue. This is still a probabilistic forecast, but it is better supported than alternatives such as self-evolution or cybersecurity benchmark branches.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This venue forecast is now manually grounded in the already validated zero-shot generalization planning trajectory rather than a weak standalone guess.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Manually aligned the answer with the stronger neighboring zero-shot generalization agenda and direction-forecast tasks.",
                "Kept the venue claim probabilistic and tied it to AAAI's benchmark-and-agents profile."
            ],
        },
    },
    "RTLv3-EXP-VENUE-1158": {
        "question": "Using literature available before September 1, 2025, identify the one or two research directions in long video understanding fine-tuning that would have been the strongest ACL-oriented bets for the next submission cycle. Provide a ranked plan and justify it with pre-cutoff evidence about methodological momentum, recurring failure modes, and ACL-style contribution fit.",
        "gold_answer": "A defensible ACL-oriented ranking is: (1) temporal focusing or query-adaptive clip selection for long-video reasoning, and (2) reasoning-oriented fine-tuning that couples temporal grounding with explicit multi-step inference. Pre-cutoff long-video work repeatedly showed that simply feeding more frames overwhelmed context budgets and diluted the evidence actually needed for a question, so temporal focusing is the first priority. The second priority is reasoning-oriented fine-tuning, because even after relevant moments are found, models still struggle to chain evidence into grounded answers. This ranking fits ACL better than generic efficiency work because it emphasizes language-grounded temporal localization, reasoning traces, and benchmarkable inference behavior; the post-cutoff wave around RAVEN, StreamingVLM, Video-Thinker, and VideoZoomer moved in exactly this direction.",
        "expected_answer_points": [
            "Identifies one or two concrete directions instead of generic 'better fine-tuning' advice.",
            "Ranks the directions and explains why the first should come before the second.",
            "Links the ranking to ACL-style priorities such as language-grounded temporal reasoning, inference behavior, and benchmarkable methodology."
        ],
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The original prompt was structurally flawed because it referred to listed options that were not actually present. It was rewritten into a self-contained planning question.",
            "edits": [
                "Removed the invalid instruction to rank unspecified listed options.",
                "Made the target deliverable a ranked two-direction plan."
            ],
        },
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "This item is now usable after a manual rewrite of both the prompt and the gold answer. The previous version was structurally underspecified.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced a placeholder-style ranking answer with a concrete two-direction ACL plan.",
                "Grounded the ranking in long-context selection and reasoning bottlenecks rather than generic trend language."
            ],
        },
    },
    "RTLv3-EXP-VENUE-1159": {
        "question": "Using literature available before August 31, 2025, identify the one or two research directions in multimodal mathematical reasoning fine-tuning that would have been the strongest AAAI-oriented bets for the next submission cycle. Provide a ranked plan and justify it with pre-cutoff evidence about model failure modes, methodological momentum, and AAAI-style contribution fit.",
        "gold_answer": "A defensible AAAI-oriented ranking is: (1) iterative self-reflection or reward-guided fine-tuning for multimodal mathematical reasoning, and (2) difficulty-aware data selection for multimodal post-training. Pre-cutoff work on visual arithmetic and geometry repeatedly showed that vision-language models often produced superficially plausible reasoning but failed on exact symbolic or numerical execution, which makes explicit verification or reflection loops the most direct next step. Difficulty-aware sampling is the second priority because multimodal math performance is highly sensitive to curriculum quality and example difficulty. This ranking fits AAAI well because both directions emphasize algorithmic reasoning, structured evaluation, and general AI methodology rather than narrow task engineering; the post-cutoff literature, including MathSE and the difficulty-distinguish data-sampling paper, moved along these same lines.",
        "expected_answer_points": [
            "Identifies one or two concrete directions instead of generic post-training advice.",
            "Ranks the directions and explains why the first is the sharper near-term bet.",
            "Connects the plan to AAAI-style interests such as reasoning methodology and structured evaluation."
        ],
        "question_quality": {
            "status": "rewritten_for_clarity",
            "summary": "The original prompt referred to a missing candidate list. It was rewritten into a self-contained venue-planning task.",
            "edits": [
                "Removed the invalid 'rank only the listed options' instruction.",
                "Converted the task into a concrete one-or-two-direction planning question."
            ],
        },
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "This item is now substantially better specified after a manual rewrite of the prompt and answer around two concrete multimodal-math directions.",
            "historical_support": "moderate",
            "future_support": "strong",
            "changes": [
                "Replaced placeholder venue rhetoric with a concrete ranking of reflection-based and difficulty-aware post-training.",
                "Grounded the answer in the already audited multimodal mathematical reasoning bottleneck trajectory."
            ],
        },
    },
    "RTLv3-0388": {
        "gold_answer": "The key unresolved bottleneck was that mask-conditioned diffusion models mainly treated the mask as a geometric region, not as a semantically bound object or part specification. ControlNet, MultiDiffusion, and Uni-ControlNet delivered strong spatial controllability, but their failure cases repeatedly exposed boundary leakage, ambiguous object identity inside masked regions, and incoherent content when multiple semantics competed within the same mask. If that bottleneck were reduced, the clearest near-term opportunity was segmentation-aware or open-vocabulary mask conditioning, including joint image-and-segmentation generation. That opportunity is materially supported by post-cutoff work such as Seg4Diff and JoDiffusion, which move from pure mask geometry toward semantic region supervision and joint pixel-level annotation.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Manual review shows that this bottleneck-opportunity pair is defensible once the bottleneck is stated as semantic binding inside masked regions rather than generic mask control.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Sharpened the bottleneck from vague semantic incoherence to object/part binding within masked regions.",
                "Tied the opportunity directly to segmentation-aware mask conditioning and joint image-label generation."
            ],
        },
    },
    "RTLv3-0396": {
        "gold_answer": "The key unresolved bottleneck was not one-step image quality by itself, but the difficulty of transferring one-step distillation from single-image generation to structured outputs such as video and 3D scenes without losing temporal or multi-view consistency. Consistency Models, InstaFlow, and later one-step image work showed that one-step generation could be credible for images, yet they left causal consistency and cross-view coherence largely unresolved outside that setting. If this bottleneck were reduced, the most plausible opportunity was one-step video and 3D generation rather than just incrementally faster image synthesis. The post-cutoff literature supports exactly that reading, with work on high-quality 3D scene generation within seconds and one-step causal video generation.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "After manual review, this item is better framed as an extension bottleneck from images to temporally or spatially structured generation.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Recentered the bottleneck on transfer to video and 3D rather than generic efficiency rhetoric.",
                "Linked the near-term opportunity to the post-cutoff one-step video and 3D wave."
            ],
        },
    },
    "RTLv3-0397": {
        "gold_answer": "The most defensible bottleneck was the lack of evaluation methods that could score temporal text-video alignment beyond short, local clips. Pre-cutoff benchmarks were increasingly able to test whether a brief segment matched a prompt, but they were much weaker at checking whether a full generated video followed a temporally ordered instruction sequence or sustained semantic coherence across multiple events. If that bottleneck were reduced, the near-term opportunity would be long-form, event-level evaluation suites rather than another generic generation model. That opportunity is materially supported by later benchmark work such as LoCoT2V-Bench, which targets long-form and complex text-to-video generation more directly than earlier short-clip metrics.",
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "This item is now usable after a manual pivot away from weakly matched generation papers toward the clearer long-form evaluation bottleneck.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Reframed the bottleneck as short-clip-local evaluation bias rather than a generic lack of video quality metrics.",
                "Anchored the opportunity in long-form evaluation benchmarks rather than broad video-generation claims."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2407.14505", "T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation", "Direct historical support for fine-grained temporal and compositional evaluation in short or local settings."),
                paper("2504.03970", "VideoComp: Advancing Fine-Grained Compositional and Temporal Alignment in Video-Text Models", "Supports the need for fine-grained temporal-alignment evaluation rather than coarse global matching."),
                paper("2505.04946", "T2VTextBench: A Human Evaluation Benchmark for Textual Control in Video Generation Models", "Supports the pre-cutoff emphasis on narrow temporal control dimensions rather than long-form event ordering.")
            ],
            "future_supporting_papers": [
                paper("2510.26412", "LoCoT2V-Bench: A Benchmark for Long-Form and Complex Text-to-Video Generation", "Direct post-cutoff support for the long-form benchmark opportunity.")
            ],
        },
    },
    "RTLv3-0403": {
        "gold_answer": "The main bottleneck was that CLIP-based alignment scores were too global and too insensitive to fine-grained prompt faithfulness. Across compositional evaluation work, models could achieve decent CLIP-style similarity while still missing attributes, swapping relations, or omitting required entities, which made CLIP-based alignment unreliable as a sole metric for generative progress. If this bottleneck were reduced, the clearest opportunity would be fine-grained, concept-aware or region-aware evaluation and optimization for text-to-image generation and editing. That opportunity is supported by the post-cutoff move toward concept-level alignment analysis and prompt-alignment methods such as FreeInpaint, which explicitly target visual rationality and prompt-faithfulness beyond a single global similarity score.",
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original answer focused on runtime cost, which was not the core literature bottleneck. The item is now manually corrected around metric sensitivity and compositional faithfulness.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced the incorrect latency bottleneck with the more defensible problem of global CLIP scores missing compositional failures.",
                "Linked the opportunity to fine-grained evaluation and prompt-alignment methods."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2307.06350", "T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation", "Directly documents the limitations of coarse CLIP-style evaluation on compositional prompts."),
                paper("2503.11481", "T2I-FineEval: Fine-Grained Compositional Metric for Text-to-Image Evaluation", "Supports the move from global alignment to fine-grained concept-aware evaluation."),
                paper("2509.21227", "Evaluating Metrics for Compositional T2I Generation", "Shows that widely used image-generation metrics diverge in their correlation with human judgments on compositional cases.")
            ],
            "future_supporting_papers": [
                paper("2509.23457", "No Concept Left Behind: Test-Time Optimization for Compositional Text-to-Image Generation", "Supports concept-level alignment feedback as a practical follow-on opportunity."),
                paper("2512.21104", "FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting", "Directly supports the post-cutoff push toward prompt alignment and visual rationality beyond raw CLIP score."),
            ],
        },
    },
    "RTLv3-0404": {
        "gold_answer": "A defensible bottleneck in hybrid autoregressive diffusion generation was error propagation in the autoregressive stage: once a wrong token or condition was emitted early, later steps had limited ability to repair that mistake, so local prediction errors could corrupt the whole sample. This concern is visible in the hybrid-autoregressive literature around models such as HART and Show-o, where efficiency comes from sequential prediction but correction remains weak. If that bottleneck were reduced, the clearest near-term opportunity would be hybrid systems with explicit refinement or correction stages that can repair erroneous autoregressive conditions after the fact. The post-cutoff appearance of work on condition-error refinement in autoregressive image generation makes that opportunity materially more credible.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Manual review supports the original core intuition once the answer is stated as autoregressive error propagation and post-hoc correction.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Tightened the bottleneck around autoregressive error propagation instead of generic hybrid-generation weakness.",
                "Linked the opportunity to explicit refinement mechanisms that repair condition errors."
            ],
        },
    },
    "RTLv3-0405": {
        "gold_answer": "The most defensible unresolved bottleneck in cross-modal diffusion frameworks was the lack of a sufficiently shared latent or token interface across modalities. Historical work on image-video tokenization, object-centric diffusion, and 3D latent diffusion showed that each modality still depended on its own representation choices, making transfer brittle and slowing progress on unified cross-modal generation. If that bottleneck were reduced, the clearest opportunity would be more transferable few-step or latent-transport generation methods for modalities beyond images, especially 3D content and unconventional sensors such as radar. That opportunity is plausible and partially echoed by post-cutoff work on few-step 3D generation and radar latent diffusion, but the evidence is still not sharp enough for a full upgrade.",
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "Manual review improved the answer, but the bottleneck-to-opportunity chain is still weaker than most retained items because the future support remains broad and heterogeneous.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Replaced generic sampling-cost language with a more defensible representation-sharing bottleneck.",
                "Kept the item provisional because the post-cutoff evidence still spans too many loosely related modalities."
            ],
        },
    },
    "RTLv3-0406": {
        "gold_answer": "The core bottleneck was that early token-merging methods mostly relied on feature similarity and did not preserve the attention salience structure that diffusion transformers use to decide which tokens matter. As a result, aggressive merging often damaged fine details or complex scene structure even when the nominal speedup looked good. If that bottleneck were reduced, the clearest opportunity would be higher-ratio token compression that still preserves generation quality, making larger diffusion transformers and higher resolutions more practical. The post-cutoff appearance of ToMA is direct support for this reading because it explicitly moves toward attention-aware token merging for diffusion models.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This item upgrades cleanly after manual review because the post-cutoff attention-aware follow-on is a close match to the historical bottleneck.",
            "historical_support": "moderate",
            "future_support": "strong",
            "changes": [
                "Made the bottleneck explicitly about attention salience, not just generic merging quality loss.",
                "Linked the opportunity directly to attention-aware high-ratio token compression."
            ],
        },
    },
    "RTLv3-0407": {
        "gold_answer": "A central bottleneck in accelerated autoregressive image generation was the recurring trade-off between speed and fidelity: token folding, speculative decoding, or other acceleration tricks sped up decoding, but they also tended to blur detail, amplify artifacts, or destabilize high-resolution generation. Historical work such as Infinity, HART, and ImageFolder makes that tension clear. If that bottleneck were reduced, the most plausible opportunity would be extending fast autoregressive generation to spatiotemporal settings such as video or multi-view synthesis, where sequential modeling is natural but quality loss compounds quickly. The post-cutoff move toward unified spacetime autoregressive generation supports that opportunity more directly than the original answer did.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Manual review supports this item once the answer is framed as the speed-versus-fidelity trade-off that blocks spatiotemporal extension.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Condensed the bottleneck to the main historical speed-quality trade-off.",
                "Linked the opportunity to spacetime autoregressive generation rather than generic faster decoding."
            ],
        },
    },
    "RTLv3-0452": {
        "gold_answer": "The most consequential bottleneck was not generic lack of domain knowledge, but the inability of video-understanding systems to decide where and when to look in long videos while preserving a coherent reasoning state. Historical agent-style systems such as VideoAgent, DoraemonGPT, and OmAgent repeatedly had to decompose tasks, browse clips, or summarize long videos because naive full-video processing was too expensive and too noisy. If that bottleneck were reduced, the clearest near-term opportunity would be temporal-focusing and active-perception agents for long-video reasoning. The post-cutoff appearance of VideoZoomer and active-perception work makes that opportunity materially better supported than the original broad claim.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This item upgrades after manual review because the historical and future papers line up around temporal focus and active video exploration.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced the vague 'limited internal knowledge' bottleneck with the sharper problem of temporal focus and reasoning-state management.",
                "Linked the opportunity to reinforcement-learned temporal focusing and active perception."
            ],
        },
    },
    "RTLv3-0478": {
        "gold_answer": "A defensible bottleneck in information-extraction instruction tuning was schema and task-definition brittleness across domains. Pre-cutoff work showed that instruction-tuned IE systems often depended on domain-specific adaptation choices, label conventions, or retrieval scaffolding, which limited cross-domain transfer and made 'one-model-fits-all' behavior unreliable. If that bottleneck were reduced, the clearest near-term opportunity would be synthetic-data or inversion-based pipelines that cheaply produce task-aligned IE supervision for new schemas, including low-cost knowledge-graph extraction. That opportunity is better matched to the post-cutoff evidence, including InvertiTune, than the earlier answer about generic retraining cost.",
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original answer overreached on a weakly matched opportunity. This version is manually corrected around schema brittleness and synthetic supervision.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced generic domain-specific pretraining rhetoric with the sharper cross-domain schema brittleness problem.",
                "Aligned the opportunity with synthetic IE supervision and KG extraction."
            ],
        },
    },
    "RTLv3-0480": {
        "gold_answer": "A defensible bottleneck in medical domain fine-tuning was that data selection remained largely heuristic rather than knowledge-aware or difficulty-aware, which is especially costly when domain data are scarce and heterogeneous. Historical work such as 3DS already pointed toward decomposed difficulty-based selection, but the broader literature still lacked robust mechanisms for selecting examples based on what a medical model already knows and where it actually fails. If that bottleneck were reduced, the clearest near-term opportunity would be more sample-efficient medical adaptation pipelines that combine difficulty-aware selection with synthetic augmentation or lightweight model merging. The future evidence is still not perfect, but it is better aligned with this narrower claim than with the previous broad formulation.",
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "Manual review suggests the bottleneck is real, but the safest version is a narrower claim about knowledge-aware or difficulty-aware medical data selection.",
            "historical_support": "strong",
            "future_support": "weak",
            "changes": [
                "Narrowed the bottleneck to data-selection quality under medical data scarcity.",
                "Softened the opportunity to sample-efficient adaptation rather than a stronger realized-wave claim."
            ],
        },
    },
    "RTLv3-0483": {
        "gold_answer": "The clearest bottleneck in reinforcement-learning-based fine-tuning was the cost and instability of obtaining high-quality preference or reward signals at a scale that makes repeated adaptation practical. Historical papers on preference alignment repeatedly show that data curation choices, update efficiency, and scarce supervision interact in ways that make RL fine-tuning brittle and expensive. If that bottleneck were reduced, the most plausible near-term opportunity would be low-overhead continual RL adaptation, using sparse updates, merged teachers, or other mechanisms that reduce the burden of repeated preference collection. This is a better fit to the post-cutoff follow-on work than the earlier answer, though the evidence is still not strong enough for a full 'validated' label.",
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "This item is improved after manual review, but it remains a softer near-term opportunity call than the strongest validated tasks.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Refocused the bottleneck on reward-signal cost and update instability rather than only on preference-dataset curation.",
                "Tied the opportunity to continual low-overhead RL adaptation."
            ],
        },
    },
    "RTLv3-0504": {
        "gold_answer": "A defensible bottleneck in biomedical domain adaptation via retrieval augmentation is the difficulty of integrating heterogeneous biomedical evidence sources without losing ontology structure or fragmenting the evidence trail. Historical papers such as OpenTCM and CLI-RAG already point to the challenge of combining knowledge graphs, medical terminology, and free text within a single retrieval-and-generation loop. If that bottleneck were reduced, the clearest near-term opportunity would be ontology-aware biomedical GraphRAG systems that coordinate graph retrieval, terminology grounding, and document retrieval more reliably. The task is cleaner in this narrower form, but the post-cutoff evidence is still too thin and too narrow to justify a full upgrade.",
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "Manual review improved the answer, but the post-cutoff evidence remains too narrow to move this item out of the provisional bucket.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Kept the answer narrowly centered on heterogeneous biomedical evidence integration.",
                "Retained provisional status because the future support is still mostly one narrow follow-on line."
            ],
        },
    },
    "RTLv3-0505": {
        "gold_answer": "A defensible bottleneck in persona-plus-fact dialogue grounding is that systems still struggle to jointly retrieve, attribute, and reconcile user-specific context with external evidence. Historical papers on citation-enhanced chatbots, in-car QA, and hallucination detection make clear that wrong or weak retrieval easily contaminates the response, but persona settings add another layer because relevant user context and external facts can conflict or drift across turns. If that bottleneck were reduced, the clearest opportunity would be citation-aware, persona-conditioned retrieval and generation with explicit evidence tracking rather than generic broader-context RAG. That is a much better target than the earlier answer, but the post-cutoff evidence is still too indirect for a full upgrade.",
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "This item is cleaner after manual review, but the future support is still too indirect to justify more than a provisional label.",
            "historical_support": "strong",
            "future_support": "weak",
            "changes": [
                "Replaced the over-broad future opportunity with citation-aware persona-conditioned grounding.",
                "Kept the item provisional because the follow-on papers are not tightly dialogue-grounding specific."
            ],
        },
    },
    "RTLv3-0508": {
        "gold_answer": "The main bottleneck in video retrieval augmentation was coarse retrieval granularity: many systems could retrieve at the video or clip level, but they struggled to identify the exact segments, objects, or temporal slices needed for a query in long videos. Historical work such as VideoRAG and visually aligned long-video RAG makes that limitation explicit. If that bottleneck were reduced, the clearest near-term opportunity would be fast, fine-grained video QA and spatiotemporal reasoning with query-adaptive segment retrieval. The post-cutoff emergence of FastV-RAG and R4 is a close match to that opportunity, so this item is stronger after manual review than the previous placeholder-style answer.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This task upgrades cleanly after manual review because the future follow-ons match the historical temporal-localization bottleneck closely.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Replaced the placeholder gold answer with a concrete temporal-localization bottleneck.",
                "Anchored the opportunity in fine-grained video QA and 4D retrieval-augmented reasoning."
            ],
        },
    },
    "RTLv3-0511": {
        "gold_answer": "The most defensible bottleneck in hierarchical graph retrieval was the difficulty of balancing coarse community-level retrieval with fine-grained node or path evidence. Flat retrieval loses structure, but purely local graph expansion easily explodes context and misses the higher-level organization needed to choose which subgraph matters. Historical work such as ArchRAG and Fast Think-on-Graph already points to that tension. If that bottleneck were reduced, the clearest near-term opportunity would be adaptive hierarchical GraphRAG systems that retrieve coarsely, then drill down only where the graph actually supports the query. The post-cutoff appearance of Deep GraphRAG and related context-bubble methods is direct enough support to upgrade this item.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Manual review shows a clear match between the historical multi-level retrieval tension and the post-cutoff adaptive hierarchical GraphRAG wave.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Replaced the vague 'flat representations' claim with the sharper coarse-to-fine retrieval trade-off.",
                "Linked the opportunity to adaptive hierarchical GraphRAG."
            ],
        },
    },
    "RTLv3-0514": {
        "gold_answer": "A prominent unresolved bottleneck in multi-turn retrieval-augmented conversational QA was the lack of standardized evaluation that jointly measures retrieval timing, cross-turn state tracking, and response faithfulness. Historical papers such as IM-RAG, fine-grained conversational retrieval work, and MTRAG make clear that multi-turn RAG systems fail in ways that single-turn benchmarks do not capture. If that bottleneck were reduced, the clearest near-term opportunity would be more diagnostic benchmark suites and stronger cross-domain comparison protocols for multi-turn RAG. The immediate post-cutoff appearance of MTRAG-UN and multi-domain comparison work is a close enough match to validate this item after refinement.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This item upgrades after manual review because the historical evaluation bottleneck and the post-cutoff benchmark follow-ons align closely.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the core benchmark-gap answer but tightened the wording around retrieval timing and state tracking.",
                "Manually confirmed that the post-cutoff follow-ons are evaluation-centric rather than generic RAG papers."
            ],
        },
    },
    "RTLv3-0515": {
        "gold_answer": "The key bottleneck in personalized retrieval-augmented conversational QA was not personalization in the abstract, but reliably retrieving the right personal context over multiple memory sources without flooding the model with irrelevant user data. Historical work such as PersonaRAG already suggests that naive personalization either over-injects context or misses the user-specific facts that actually matter. If that bottleneck were reduced, the clearest near-term opportunity would be reasoning-based, multi-step retrieval of personal context rather than one-shot persona injection. The post-cutoff paper on learning to reason for multi-step retrieval of personal context is a close match, which makes this item much stronger than before.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This item upgrades after manual review because the future support now matches the bottleneck almost exactly.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Refined the bottleneck to multi-source personal-context retrieval rather than a generic personalization gap.",
                "Aligned the opportunity with reasoning-based multi-step personal retrieval."
            ],
        },
    },
    "RTLv3-0516": {
        "gold_answer": "A defensible bottleneck in retrieval-augmented educational dialogue generation is the three-way tension among factual grounding, pedagogical adaptation, and learner-state awareness. Historical educational and tutoring papers suggest that retrieval can improve factuality, but it does not by itself solve how a system should adapt explanations to a learner's current knowledge, misconceptions, or desired level of guidance. If that bottleneck were reduced, the clearest opportunity would be educational assistants that reason explicitly over learner state while grounding their responses in retrieved evidence. The item is stronger in this narrowed form, but the post-cutoff evidence is still too sparse and application-specific to justify a full upgrade.",
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "Manual review improved the bottleneck statement, but the future evidence remains too sparse for more than a provisional label.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Clarified that the bottleneck is a joint groundedness-pedagogy-learner-state problem.",
                "Retained provisional status because the post-cutoff support is still application-heavy rather than a clear research wave."
            ],
        },
    },
    "RTLv3-0541": {
        "gold_answer": "The main bottleneck in cross-modal temporal alignment assessment was the lack of automated metrics that reliably track fine-grained audio-video synchronization the way humans perceive it. Historical audio-reactive and audio-synchronized generation work repeatedly depended on human evaluation or coarse proxy metrics because subtle sub-second misalignment was hard to measure automatically. If that bottleneck were reduced, the clearest near-term opportunity would be unified audio-video evaluation suites with explicit cross-modal synchronization scoring rather than separate audio-only and video-only checks. That opportunity is materially supported by post-cutoff work such as T2AV-Compass, which explicitly argues that text-to-audio-video evaluation remained fragmented and introduces a unified benchmark for it.",
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This item upgrades after manual review because the post-cutoff benchmark evidence directly addresses the historical sync-evaluation gap.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Kept the core sync-metric bottleneck but replaced vague multimodal-foundation-model language with unified evaluation support.",
                "Anchored the opportunity in joint audio-video benchmarking rather than broader model scaling."
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                paper("2309.04509", "The Power of Sound (TPoS): Audio Reactive Video Generation with Stable Diffusion", "Historical support for the difficulty of evaluating temporal synchronization in audio-reactive generation."),
                paper("2403.05659", "Audio-Synchronized Visual Animation", "Supports the need for fine-grained alignment assessment rather than coarse quality metrics."),
                paper("2403.08764", "VLOGGER: Multimodal Diffusion for Embodied Avatar Synthesis", "Historical support for multimodal generation settings where cross-modal synchronization matters.")
            ],
            "future_supporting_papers": [
                paper("2512.21094", "T2AV-Compass: Towards Unified Evaluation for Text-to-Audio-Video Generation", "Direct post-cutoff support for the unified cross-modal evaluation opportunity."),
                paper("2601.03233", "LTX-2: Efficient Joint Audio-Visual Foundation Model", "Supports the rise of joint audio-video generation that increases pressure for better alignment assessment.")
            ],
        },
    },
}


ROUND8_TASK_IDS = list(UPDATES.keys())


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
        tid = row["task_id"]
        status_before = row.get("answer_audit", {}).get("status")
        counts_before[status_before] = counts_before.get(status_before, 0) + 1
        if tid in UPDATES:
            patch = UPDATES[tid]
            for key in ["question", "gold_answer", "expected_answer_points", "question_quality", "answer_audit"]:
                if key in patch:
                    row[key] = patch[key]
            if "ground_truth" in patch:
                row["ground_truth"] = merge_ground_truth(row.get("ground_truth"), patch["ground_truth"])
            review = deepcopy(row.get("review_metadata", {}))
            review["refined_at"] = "2026-04-23"
            review["review_depth"] = "manual_round8_web_checked"
            review["used_web_verification"] = True
            review["round8_manual_refine"] = True
            row["review_metadata"] = review
            touched.append((tid, row["answer_audit"]["status"], row["answer_audit"]["historical_support"], row["answer_audit"]["future_support"]))
        status_after = row.get("answer_audit", {}).get("status")
        counts_after[status_after] = counts_after.get(status_after, 0) + 1
        rows.append(row)

    TASK_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))

    log_lines = [
        "# Benchmark Refine Log 20260423 Round 8",
        "",
        "## Scope",
        "",
        "- This round manually re-audits the remaining low-evidence residue rather than applying another family template pass.",
        "- Focus: the last 3 venue-aware tasks and 20 remaining bottleneck-opportunity tasks.",
        "- Method: paper-level manual correction of question wording, gold answers, and audit labels; web verification used for the round.",
        "",
        "## Files Updated",
        "",
        "- `benchmark_release/task_refined.jsonl`",
        "- `docs/benchmark_refine_log_20260423_round8.md`",
        "- `scripts/manual_refine_round8_20260423.py`",
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
        log_lines.append(f"- `{tid}`: {status} (historical={historical}, future={future})")

    log_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `RTLv3-EXP-VENUE-1158` and `RTLv3-EXP-VENUE-1159` were structurally rewritten because the original prompts referred to missing candidate lists.",
            "- A small set of tasks remains provisional on purpose where the future-support chain is still too narrow or indirect (`RTLv3-0405`, `RTLv3-0504`, `RTLv3-0505`, `RTLv3-0516`).",
            "- Several previously placeholder-like answers were replaced with concrete bottlenecks that better match the cited historical and future papers.",
        ]
    )
    LOG_PATH.write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
