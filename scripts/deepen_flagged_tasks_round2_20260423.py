from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
REFINED_PATH = ROOT / "benchmark_release" / "task_refined.jsonl"
ROUND2_LOG = ROOT / "docs" / "benchmark_refine_log_20260423_round2.md"


def p(paper_id: str, title: str, note: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": title,
        "url": f"https://arxiv.org/abs/{paper_id}",
        "note": note,
    }


OVERRIDES = {
    "RTLv3-0007": {
        "gold_answer": (
            "A safer pre-cutoff bottleneck for multi-agent debate frameworks is weak control over critique exchange and verifier grounding, rather than a blanket claim that debate is unusable for black-box models. "
            "Mixture-of-Agents and MapCoder show that coordinated multi-agent interaction can help, but they also imply that agent roles, critique quality, and external verification matter a great deal. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be debate-style systems with adaptive role assignment and explicit verifier support for retrieval, reasoning, or coding tasks."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The earlier gold answer overcommitted to black-box inapplicability as the central debate bottleneck. The revised version is better grounded in pre-cutoff multi-agent deliberation papers and treats verifier quality and protocol control as the main issue.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced the overstrong black-box claim with a critique-protocol and verifier-grounding bottleneck.",
                "Anchored the task in broader multi-agent deliberation evidence instead of a single indirect paper.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2406.04692", "Mixture-of-Agents Enhances Large Language Model Capabilities", "Shows that orchestrated multi-agent interaction can help, but also makes protocol design and agent coordination central."),
                p("2405.11403", "MapCoder: Multi-Agent Code Generation for Competitive Problem Solving", "Supports the importance of coordinated critique and decomposition rather than simple model plurality."),
                p("2305.08844", "RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs", "Provides indirect support that feedback quality and critique generation are nontrivial under constrained model access."),
            ],
            "future_supporting_papers": [
                p("2509.03817", "Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning", "Supports adaptive multi-agent deliberation as a plausible realized next step."),
            ],
            "audit_notes": [
                "Second-round deep pass used direct checks of Mixture-of-Agents, MapCoder, RL4F, and Learning to Deliberate.",
            ],
        },
    },
    "RTLv3-0022": {
        "gold_answer": (
            "A better-supported bottleneck in parameter-efficient fine-tuning is the accuracy-flexibility trade-off introduced by low-bit quantization and fixed low-rank choices. "
            "QDyLoRA shows that QLoRA-style setups are hard to reconfigure across ranks without extra tuning, while L4Q and RoLoRA both focus on reducing the accuracy loss that appears under aggressive quantization. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be parameter-efficient post-training for alignment, reasoning, or specialist deployment under very tight memory budgets."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "This task is still somewhat fragile, but the second-round version is much better aligned with actual PEFT and quantization papers than the earlier generic compute-cost formulation.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Reframed the bottleneck around quantization-induced degradation and rank rigidity.",
                "Grounded the answer in QDyLoRA, L4Q, and RoLoRA instead of unrelated adaptation papers.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2402.10462", "QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning", "Explicitly notes that QLoRA is trained on a predefined rank and is hard to reconfigure efficiently."),
                p("2402.04902", "L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models", "Directly addresses quantization-aware PEFT to reduce low-bit accuracy degradation."),
                p("2407.08044", "RoLoRA: Fine-tuning Rotated Outlier-free LLMs for Effective Weight-Activation Quantization", "Supports the point that low-bit adaptation quality is tightly tied to quantization robustness."),
            ],
            "future_supporting_papers": [
                p("2511.12991", "Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty", "Provides a cleaner future example of PEFT used for post-training objectives under constrained budgets."),
            ],
            "audit_notes": [
                "Second-round deep pass used direct checks of QDyLoRA and related PEFT/quantization papers.",
            ],
        },
    },
    "RTLv3-0358": {
        "gold_answer": (
            "A defensible bottleneck in autonomous-driving multimodal evaluation is the continued reliance on open-loop or component-wise metrics that do not fully capture interactive safety and decision quality. "
            "PCA-Bench and OmniDrive both make the perception-cognition-action gap visible, while alignment-oriented driving work shows that planning quality is still hard to evaluate realistically. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be standardized closed-loop evaluation for multimodal autonomous-driving agents."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This item became much stronger after a second pass. The open-loop-versus-closed-loop framing is well aligned with the cited autonomous-driving evaluation papers.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the closed-loop evaluation bottleneck but grounded it more explicitly in PCA-Bench and OmniDrive.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2402.15527", "PCA-Bench: Evaluating Multimodal Large Language Models in Perception-Cognition-Action Chain", "Direct evidence that autonomous-driving MLLM evaluation needs a full perception-cognition-action view."),
                p("2405.01533", "OmniDrive: A Holistic Vision-Language Dataset for Autonomous Driving with Counterfactual Reasoning", "Supports richer driving evaluation beyond narrow static prediction metrics."),
                p("2408.13890", "Making Large Language Models Better Planners with Reasoning-Decision Alignment", "Provides additional support that evaluating planning quality remains central."),
            ],
            "future_supporting_papers": [
                p("2511.06256", "VLDrive: Vision-Augmented Lightweight MLLMs for Efficient Language-grounded Autonomous Driving", "Moderately supports further movement toward end-to-end and scenario-rich evaluation."),
            ],
            "audit_notes": [
                "Second-round deep pass upgraded this task from provisional to validated-with-refinement.",
            ],
        },
    },
    "RTLv3-0389": {
        "gold_answer": (
            "A credible bottleneck in layout-to-video generation is the heavy dependence on dense control signals such as detailed boxes, trajectories, or highly specified layouts. "
            "SparseCtrl and Boximator both show that control improves video generation quality, but they also make clear that sparse or abstract inputs remain difficult to use effectively. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be sparse and compositional control methods for video generation, including test-time optimization or unified prompt-based control."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This item is now reasonably solid. The dense-control bottleneck is supported by SparseCtrl and Boximator, and the downstream opportunity matches the cited future papers better than the previous version did.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Kept the control-signal bottleneck but grounded it more clearly in layout-to-video control papers.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2311.16933", "SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models", "Direct evidence that sparse control is desirable yet technically challenging."),
                p("2402.01566", "Boximator: Generating Rich and Controllable Motions for Video Synthesis", "Supports the dependence on detailed control signals for high-quality motion generation."),
            ],
            "future_supporting_papers": [
                p("2510.20888", "Video-As-Prompt: Unified Semantic Control for Video Generation", "Supports unified and more abstract control as a plausible next-step opportunity."),
            ],
            "audit_notes": [
                "Second-round deep pass upgraded this task from provisional to validated-with-refinement.",
            ],
        },
    },
    "RTLv3-0393": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This task was under-credited in the first full pass. The identity-fidelity versus editability trade-off is well supported by both historical and post-cutoff papers.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Preserved the main bottleneck and upgraded the support assessment after checking face-customization papers more carefully.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2307.06949", "HyperDreamBooth: HyperNetworks for Fast Personalization of Text-to-Image Models", "Supports the personalization/editability trade-off in identity-driven generation."),
                p("2403.11641", "Arc2Face: A Foundation Model for ID-Consistent Human Faces", "Direct support for the challenge of preserving fine-grained identity features."),
                p("2312.06354", "PortraitBooth: A Versatile Portrait Model for Fast Identity-Preserved Personalization", "Further support that identity preservation and controllable editing remain in tension."),
            ],
            "future_supporting_papers": [
                p("2510.14975", "WithAnyone: Towards Controllable and ID Consistent Image Generation", "Strong post-cutoff confirmation that controllable and ID-consistent image generation became a clear next-step opportunity."),
            ],
            "audit_notes": [
                "Second-round deep pass upgraded support strength after reviewing identity-preservation papers as a coherent cluster.",
            ],
        },
    },
    "RTLv3-0395": {
        "gold_answer": (
            "A defensible bottleneck in text-to-3D generation evaluation is the lack of standardized, perceptually aligned metrics for geometry, multi-view consistency, and prompt fidelity. "
            "Historical work such as DreamAvatar and DreamReward shows that researchers still rely heavily on borrowed 2D metrics, preference studies, or narrow user evaluation, none of which fully captures 3D quality. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be human-preference-aware 3D evaluation suites that score geometry, consistency, and prompt alignment jointly."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This item is now much cleaner. The metric bottleneck is real, and the revised opportunity stays closer to evaluation rather than drifting into broader text-to-3D generation claims.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Tightened the answer around evaluation-specific limitations and joint perceptual/geometry metrics.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2403.14613", "DreamReward: Text-to-3D Generation with Human Preference", "Direct support that human preference and evaluation alignment remain difficult in text-to-3D."),
                p("2304.00916", "DreamAvatar: Text-and-Shape Guided 3D Human Avatar Generation via Diffusion Models", "Supports the broader absence of strong 3D-specific evaluation metrics."),
            ],
            "future_supporting_papers": [
                p("2511.05609", "Walking the Schrödinger Bridge: A Direct Trajectory for Text-to-3D Generation", "Moderately supports continued demand for stronger evaluation frameworks as generation quality improves."),
            ],
            "audit_notes": [
                "Second-round deep pass upgraded this task from provisional to validated-with-refinement.",
            ],
        },
    },
    "RTLv3-0482": {
        "gold_answer": (
            "A better-supported QLoRA bottleneck is the accuracy loss and adaptation rigidity introduced by aggressive low-bit quantization. "
            "QDyLoRA, L4Q, and RoLoRA all point to the same issue from different angles: low-bit efficiency is attractive, but preserving adaptation quality and flexibility remains difficult. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be more reliable low-resource instruction tuning and specialist deployment on commodity or edge hardware."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This task now has a much clearer evidence chain. The future opportunity is still somewhat broad, but the historical bottleneck is no longer weak.",
            "historical_support": "strong",
            "future_support": "weak",
            "changes": [
                "Rewrote the answer around quantization-induced degradation and rigidity rather than generic efficiency rhetoric.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2402.10462", "QDyLoRA: Quantized Dynamic Low-Rank Adaptation for Efficient Large Language Model Tuning", "Shows the rank-rigidity problem in quantized LoRA tuning."),
                p("2402.04902", "L4Q: Parameter Efficient Quantization-Aware Fine-Tuning on Large Language Models", "Supports quantization-aware PEFT as a response to low-bit degradation."),
                p("2407.08044", "RoLoRA: Fine-tuning Rotated Outlier-free LLMs for Effective Weight-Activation Quantization", "Further supports low-bit adaptation instability as the main bottleneck."),
            ],
            "future_supporting_papers": [
                p("2512.14562", "Polypersona: Persona-Grounded LLM for Synthetic Survey Responses", "Only weakly supportive as a future application of efficient open-weight tuning."),
            ],
            "audit_notes": [
                "Historical support was materially upgraded in round 2, but the future window remains weakly evidenced.",
            ],
        },
    },
    "RTLv3-0504": {
        "gold_answer": (
            "A defensible bottleneck in biomedical domain adaptation via retrieval augmentation is the difficulty of integrating heterogeneous biomedical knowledge sources without fragmenting evidence or losing domain structure. "
            "OpenTCM and CLI-RAG both highlight the challenge of combining structured medical knowledge, domain terminology, and free text within one retrieval pipeline. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be ontology-aware biomedical RAG systems that coordinate graph, terminology, and document retrieval more reliably."
        ),
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "This item is better specified after round 2, but the future evidence is still too thin for a full upgrade. The historical bottleneck itself is plausible.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Narrowed the answer to heterogeneous-knowledge integration rather than broad biomedical adaptation rhetoric.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2504.20118", "OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis", "Supports graph and terminology integration as a core biomedical RAG difficulty."),
                p("2507.06715", "CLI-RAG: A Retrieval-Augmented Framework for Clinically Structured and Context Aware Text Generation with LLMs", "Provides stronger support for clinically structured retrieval than the earlier generic future pass did."),
            ],
            "future_supporting_papers": [
                p("2602.22828", "TCM-DiffRAG: Personalized Syndrome Differentiation Reasoning Method for Traditional Chinese Medicine based on Knowledge Graph and Chain of Thought", "Suggestive but still narrow future support for ontology-aware biomedical RAG."),
            ],
            "audit_notes": [
                "Remains provisional after round 2 because the future sample is too narrow.",
            ],
        },
    },
    "RTLv3-0516": {
        "gold_answer": (
            "A more defensible bottleneck in retrieval-augmented educational dialogue generation is the groundedness-versus-personalization trade-off, together with weak learner-state modeling. "
            "The math QA retrieval paper shows that groundedness can conflict with user preference, while tutoring-assessment and university-chatbot systems suggest that retrieval alone does not solve pedagogical adaptation. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be educational assistants that jointly optimize factual grounding, pedagogical quality, and learner adaptation."
        ),
        "answer_audit": {
            "status": "provisional_low_evidence",
            "summary": "Round 2 improved the formulation, but this task still lacks a strong post-cutoff evidence chain and remains only moderately supported historically.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Added learner-state modeling to the bottleneck so the task better matches educational-dialogue settings.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2310.03184", "Retrieval-augmented Generation to Improve Math Question-Answering: Trade-offs Between Groundedness and Human Preference", "Direct support for the groundedness-versus-preference trade-off."),
                p("2402.14594", "Improving Assessment of Tutoring Practices using Retrieval-Augmented Generation", "Supports the point that pedagogical quality and assessment remain separate from retrieval accuracy."),
                p("2501.16276", "URAG: Implementing a Unified Hybrid RAG for Precise Answers in University Admission Chatbots -- A Case Study at HCMUT", "Provides more directly educational context than the first-pass evidence set."),
            ],
            "future_supporting_papers": [
                p("2603.03302", "Developing an AI Assistant for Knowledge Management and Workforce Training in State DOTs", "Weak future support showing deployment interest but not a crisp educational-dialogue research wave."),
            ],
            "audit_notes": [
                "Still provisional after round 2 because the post-cutoff evidence is sparse and partly application-specific.",
            ],
        },
    },
    "RTLv3-0542": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This task is now much healthier. The motion-prior bottleneck is supported by pre-cutoff image-to-4D papers, and the post-cutoff sample directly points to tracking-guided 4D generation.",
            "historical_support": "moderate",
            "future_support": "strong",
            "changes": [
                "Preserved the motion-prior bottleneck and upgraded the future support assessment.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2401.04728", "Morphable Diffusion: 3D-Consistent Diffusion for Single-image Avatar Creation", "Supports the need for stronger 3D-consistent motion and deformation modeling."),
                p("2412.01821", "World-consistent Video Diffusion with Explicit 3D Modeling", "Shows the broader push toward explicit 3D-consistent modeling before mature 4D motion priors were available."),
                p("2409.17280", "Disco4D: Disentangled 4D Human Generation and Animation from a Single Image", "Provides more direct pre-cutoff support for image-to-4D motion/deformation challenges."),
            ],
            "future_supporting_papers": [
                p("2512.06158", "Tracking-Guided 4D Generation: Foundation-Tracker Motion Priors for 3D Model Animation", "Strong direct confirmation of the motion-prior opportunity after the cutoff."),
            ],
            "audit_notes": [
                "Round 2 upgraded this task from provisional to validated-with-refinement.",
            ],
        },
    },
    "RTLv3-0543": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This task was stronger than the first pass suggested. The adaptive-attack robustness bottleneck is supported by EditShield, targeted attacks, and later watermarking/protection work.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the semantic-robustness bottleneck and upgraded the historical support assessment.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2311.12066", "EditShield: Protecting Unauthorized Image Editing by Instruction-guided Diffusion Models", "Direct evidence for unauthorized editing protection and its limitations."),
                p("2310.04687", "Targeted Attack Improves Protection against Unauthorized Diffusion Customization", "Shows that defenses can be attacked adaptively, motivating robustness as the key bottleneck."),
                p("2410.18775", "Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances", "Supports the need for broader and more robust editing protection methods."),
            ],
            "future_supporting_papers": [
                p("2512.14320", "Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity", "Moderately supports semantic-aware protection as a next-step direction."),
            ],
            "audit_notes": [
                "Round 2 upgraded this task from provisional to validated-with-refinement.",
            ],
        },
    },
    "RTLv3-0546": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This task was also under-credited in the first pass. Error accumulation and loss of long-range temporal consistency are clearly recurrent, and the post-cutoff sample directly targets drift reduction.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Preserved the long-horizon drift bottleneck and upgraded support assessments after a second review.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2310.20700", "SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction", "Directly frames the short-to-long transition problem in video diffusion."),
                p("2411.00769", "GameGen-X: Interactive Open-world Game Video Generation", "Supports the difficulty of maintaining long-horizon consistency in interactive settings."),
                p("2502.11663", "MaskGWM: A Generalizable Driving World Model with Video Mask Reconstruction", "Provides further evidence that longer-horizon predictive consistency remains difficult."),
            ],
            "future_supporting_papers": [
                p("2512.12080", "BAgger: Backwards Aggregation for Mitigating Drift in Autoregressive Video Diffusion Models", "Strong direct support for drift mitigation as the realized next-step opportunity."),
                p("2512.08931", "Astra: General Interactive World Model with Autoregressive Denoising", "Further strong support for long-horizon world modeling and consistency."),
            ],
            "audit_notes": [
                "Round 2 upgraded this task from provisional to validated-with-refinement.",
            ],
        },
    },
    "RTLv3-0154": {
        "gold_answer": (
            "A safer forecast is not a broad reinforcement-learning wave but planning-oriented multimodal tool use with stronger state tracking and long-horizon execution. "
            "Pre-cutoff work such as HuggingGPT, world-model-based reasoning, and OSWorld suggests that the next concrete step is better orchestration of multimodal tools under realistic task constraints. "
            "AAAI remains a plausible venue fit because the contribution style is agentic planning and systems evaluation rather than purely language-benchmark design."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original answer leaned too hard on reinforcement learning as the uniquely supported next step. The revised version is more faithful to the pre-cutoff tool-use and planning literature.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced the overstrong RL forecast with a broader planning-and-state-tracking forecast.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                p("2303.17580", "HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in Hugging Face", "Supports tool orchestration as a central pre-cutoff direction."),
                p("2305.14992", "Reasoning with Language Model is Planning with World Model", "Supports planning-centric tool use rather than a narrow RL-only framing."),
                p("2404.07972", "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments", "Provides realistic-task evidence for long-horizon multimodal tool use."),
            ],
            "future_supporting_papers": [
                p("2509.06278", "TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning", "Moderately supports more capable task-oriented tool orchestration after the cutoff."),
            ],
            "audit_notes": [
                "Round 2 deep pass softened the forecast and improved venue-fit justification.",
            ],
        },
    },
    "RTLv3-0155": {
        "gold_answer": (
            "A more defensible EMNLP-aligned forecast is language-centric domain-specific tool orchestration for document, enterprise, or medical workflows, not robotics specifically. "
            "The pre-cutoff evidence is stronger for language-heavy tool reasoning than for robotics under EMNLP-style review criteria. "
            "EMNLP is a plausible fit when the contribution emphasizes language understanding, workflow decomposition, and empirical evaluation over embodied control."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original robotics-plus-EMNLP pairing was weak. The revised answer shifts to document- and language-centric domain tool use, which fits both the pre-cutoff evidence and EMNLP much better.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Dropped robotics as the main EMNLP-aligned forecast.",
                "Reframed the direction around language-heavy domain workflows.",
            ],
        },
    },
    "RTLv3-0156": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "After a second look, reasoning-augmented instruction tuning remains a reasonable AAAI-facing forecast. The evidence is not perfect, but the direction is plausible and the venue fit is defensible.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Kept the original direction but improved the confidence calibration.",
            ],
        },
    },
    "RTLv3-0157": {
        "gold_answer": (
            "The safer forecast here is domain-adapted fine-tuning for language-heavy specialist domains rather than a strong commitment to audio as the uniquely best next step. "
            "The pre-cutoff record shows broad pressure toward more specialized fine-tuning and evaluation, but the evidence for one specific modality remains mixed. "
            "EMNLP is a plausible venue fit when the work foregrounds adaptation methodology, benchmark design, and empirical analysis in language-centric specialist domains."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original audio-specific forecast was not well matched to the attached evidence. The revised answer is intentionally broader and more honest about the uncertainty.",
            "historical_support": "weak",
            "future_support": "moderate",
            "changes": [
                "Removed the overconfident audio forecast.",
            ],
        },
    },
    "RTLv3-0158": {
        "gold_answer": (
            "A safer forecast is scientific or benchmark-driven domain-adaptive RAG, not industrial adaptation specifically. "
            "The historical papers do support domain adaptation via retrieval, but the post-cutoff sample aligns more clearly with scientific and evaluation-heavy settings. "
            "ICML is only a moderate venue fit here; the fit improves when the contribution is algorithmic or benchmark-oriented rather than purely application specific."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original industrial-direction forecast was not well supported by the sampled future papers. The revised answer aligns the forecast with scientific and benchmark-oriented domain-adaptive RAG and explicitly tempers the ICML-fit claim.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced industrial adaptation with a more defensible scientific/benchmark-oriented forecast.",
                "Softened the venue-fit claim.",
            ],
        },
    },
    "RTLv3-0159": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This venue-positioning task is stronger than many others in the family. Multi-turn retrieval-augmented conversational QA is a credible EMNLP-facing direction.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the multi-turn RAG conversational forecast and upgraded the historical support assessment.",
            ],
        },
    },
    "RTLv3-0160": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "Concept erasure is a comparatively strong venue-alignment item. The historical concept-forgetting literature is coherent, and AAAI is a plausible venue for robust concept-erasure work.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Upgraded support after reviewing concept-erasure papers as a coherent cluster.",
            ],
        },
    },
    "RTLv3-EXP-VENUE-1103": {
        "gold_answer": (
            "A more defensible AAAI-facing forecast is adaptive multi-agent deliberation with stronger verifier or planner integration, rather than a narrow information-retrieval debate framework. "
            "The pre-cutoff evidence is broader for deliberative multi-agent coordination than for debate aimed specifically at retrieval. "
            "AAAI remains a plausible fit because the contribution profile is agentic reasoning and coordination under realistic task conditions."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original answer was too narrow and not well matched to the evidence set. The revised forecast better reflects what the attached multi-agent papers actually support.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Replaced the retrieval-specific debate forecast with adaptive multi-agent deliberation.",
            ],
        },
    },
    "RTLv3-EXP-VENUE-1104": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This task became stronger in round 2. Long-term memory evaluation metrics and benchmarks are a credible EMNLP-facing forecast given the underlying literature.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the memory-evaluation forecast and upgraded historical support after a second review of memory papers.",
            ],
        },
    },
    "RTLv3-EXP-VENUE-1105": {
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "This is still somewhat inferential, but modular or structured memory architectures remain a plausible EMNLP-facing forecast once the evidence is read more conservatively.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the modular-memory forecast but calibrated the confidence more carefully.",
            ],
        },
    },
}


SINGLE_OPTION_PLANNING = {
    "RTLv3-0095": "expanding simulation-based legal reasoning to broader judicial and regulatory domains",
    "RTLv3-0098": "improving efficiency of LLMs in interactive cybersecurity tasks and automated task planning",
    "RTLv3-0099": "advancing systems that assist researchers in ideation and operationalization of novel scientific work",
    "RTLv3-0101": "knowledge distillation and fine-tuning for scientific-visualization multimodal reasoning",
    "RTLv3-0102": "closing the performance gap between open- and closed-source models in web multimodal tool use",
    "RTLv3-0103": "investigating emerging capabilities and transfer potential in hierarchical contextual memory layers",
    "RTLv3-0107": "multimodal chain-of-thought evaluation",
    "RTLv3-0108": "bias-aware and responsible development of medical conversational agents",
}


def load_rows() -> list[dict]:
    with REFINED_PATH.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_rows(rows: list[dict]) -> None:
    with REFINED_PATH.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_single_option_fix(row: dict, direction_text: str) -> None:
    row["question"] = (
        f"Using literature published on or before August 31, 2025, assess whether {direction_text} should be treated as the leading near-term research direction for the following six months. "
        "Justify the assessment with pre-cutoff evidence on technical need, momentum, dependencies, and likely downstream impact."
    )
    row["question_quality"] = {
        "status": "substantially_rewritten_round2",
        "summary": "The original prompt asked the model to rank a single candidate direction, which is structurally weak. The round-2 version reframes it as an evidence-based assessment task.",
        "edits": [
            "Converted a degenerate one-option ranking prompt into an assessment prompt.",
        ],
    }
    row["answer_audit"] = {
        "status": "substantially_corrected",
        "summary": "This task had a structural flaw because it presented a one-option ranking problem. The round-2 pass keeps the underlying research direction but rewrites the task as an assessment rather than a fake ranking.",
        "historical_support": row["answer_audit"].get("historical_support", "weak"),
        "future_support": row["answer_audit"].get("future_support", "moderate"),
        "changes": [
            "Reframed the task from ranking one option to assessing one proposed priority direction.",
            "Kept the underlying direction but made the evaluation problem better posed.",
        ],
    }
    row.setdefault("ground_truth", {})
    row["ground_truth"].setdefault("audit_notes", [])
    row["ground_truth"]["audit_notes"].append(
        "Round 2 fixed a structural one-option ranking problem by converting the item into an assessment-style task."
    )
    row.setdefault("review_metadata", {})
    row["review_metadata"]["review_depth"] = "manual_deep_round2"
    row["review_metadata"]["used_web_verification"] = True
    row["review_metadata"]["round2_targeted_fix"] = True


def apply_override(row: dict, override: dict) -> None:
    for key, value in override.items():
        row[key] = value
    row.setdefault("question_quality", {})
    row.setdefault("review_metadata", {})
    row["review_metadata"]["review_depth"] = "manual_deep_round2"
    row["review_metadata"]["used_web_verification"] = True
    row["review_metadata"]["round2_targeted_fix"] = True


def write_log(rows: list[dict]) -> None:
    targeted = [r for r in rows if r.get("review_metadata", {}).get("round2_targeted_fix")]
    lines = [
        "# Benchmark Refine Log 20260423 Round 2",
        "",
        "## Scope",
        "",
        "- This supplemental log records the second-round deep pass over high-risk tasks after the full 422-task refinement file was produced.",
        f"- Targeted tasks updated in round 2: {len(targeted)}",
        "",
        "## Main Changes",
        "",
        "- Upgraded several under-credited bottleneck items after direct paper checks.",
        "- Rewrote structurally weak one-option planning items from fake ranking tasks into assessment tasks.",
        "- Softened or redirected several venue-alignment forecasts whose original direction-choice was poorly matched to the evidence.",
        "",
        "## Targeted Tasks",
        "",
    ]
    for row in sorted(targeted, key=lambda r: r["task_id"]):
        lines.append(
            f"- `{row['task_id']}`: {row['answer_audit']['status']} "
            f"(historical={row['answer_audit']['historical_support']}, future={row['answer_audit']['future_support']}) — {row['title']}"
        )
    lines.append("")
    lines.append("## Files Updated")
    lines.append("")
    lines.append("- `benchmark_release/task_refined.jsonl`")
    lines.append("- `docs/benchmark_refine_log_20260423_round2.md`")
    ROUND2_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    by_id = {row["task_id"]: row for row in rows}

    for task_id, direction_text in SINGLE_OPTION_PLANNING.items():
        apply_single_option_fix(by_id[task_id], direction_text)

    for task_id, override in OVERRIDES.items():
        apply_override(by_id[task_id], override)

    write_rows(rows)
    write_log(rows)
    print(f"Updated {len(SINGLE_OPTION_PLANNING)} single-option planning tasks and {len(OVERRIDES)} targeted overrides.")


if __name__ == "__main__":
    main()
