from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path("/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight")
RELEASE_DIR = ROOT / "benchmark_release"
DOCS_DIR = ROOT / "docs"

PUBLIC_TASKS = RELEASE_DIR / "tasks.jsonl"
INTERNAL_TASKS = RELEASE_DIR / "tasks_internal_full.jsonl"
OUTPUT_PATH = RELEASE_DIR / "task_refined.jsonl"
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


OVERRIDES = {
    "RTLv3-0001": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in how long-term memory is evaluated for LLM agents. "
            "Support the bottleneck with recurring limitations, failure cases, or explicit gaps in the pre-cutoff record, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved. "
            "Base both claims only on pre-cutoff evidence."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original wording was natural enough but repetitive. The revision keeps the same task while tightening the temporal constraint and reducing template-like phrasing.",
            "edits": [
                "Shortened duplicated instructions about cutoff discipline.",
                "Made the dependency between the bottleneck and the downstream opportunity more explicit.",
            ],
        },
        "gold_answer": (
            "A defensible bottleneck in the pre-cutoff literature was the lack of evaluation setups that genuinely stress very long-horizon memory use. "
            "Work such as LoCoMo showed that earlier studies usually covered only a few sessions, leaving very long-term dialogue memory, temporal consistency, and retrieval effectiveness underexplored. "
            "That means researchers often could not tell whether an agent had robust long-term memory or was only handling short-context recall. "
            "If this bottleneck were removed, a concrete near-term opportunity would be benchmark and metric design for very-long-horizon memory, including temporally grounded retrieval, event-consistency scoring, and memory-conditioned planning evaluation."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original gold answer overemphasized standardized metrics and parametric-update constraints, while the strongest historical evidence in the cited papers points more directly to insufficient long-horizon evaluation settings.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Reframed the bottleneck from a generic metrics gap to inadequate very-long-horizon evaluation coverage.",
                "Removed the unsupported jump from evaluation problems to closed-model parametric update limitations.",
                "Kept the future opportunity at the level of benchmark and metric design, which is better supported than a broader claimed research wave.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2402.17753",
                    "Evaluating Very Long-Term Conversational Memory of LLM Agents",
                    "https://arxiv.org/abs/2402.17753",
                    "Explicitly argues that prior work covered only a small number of chat sessions and that very long-term memory evaluation remained underexplored.",
                ),
                _paper(
                    "2308.10144",
                    "ExpeL: LLM Agents Are Experiential Learners",
                    "https://arxiv.org/abs/2308.10144",
                    "Useful as supporting context on memory and experience accumulation, but not the main evidence for the revised bottleneck.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.08222",
                    "Exploratory Retrieval-Augmented Planning For Continual Embodied Instruction Following",
                    "https://arxiv.org/abs/2509.08222",
                    "Provides moderate post-cutoff support that memory-sensitive planning and retrieval evaluation remained an active near-term opportunity.",
                ),
            ],
            "audit_notes": [
                "The original future grounding was directionally reasonable but weaker than the original prose suggested.",
            ],
        },
    },
    "RTLv3-0002": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in multi-turn reinforcement-learning frameworks for LLM agents. "
            "Support the bottleneck with documented limitations, failure cases, or acknowledged gaps, then explain one concrete six-month research opportunity that would become plausible if that bottleneck were addressed. "
            "Use only pre-cutoff evidence."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original question was clear but overlong. The revised version is shorter and keeps the same inferential burden.",
            "edits": [
                "Condensed repeated instructions.",
                "Made the six-month dependency the center of the prompt.",
            ],
        },
        "gold_answer": (
            "A concrete bottleneck in the pre-cutoff literature was reliance on fixed, prompted LLM evaluators or value functions together with shallow search. "
            "TS-LLM explicitly notes that a pre-trained LLM used as a value function can lack the task knowledge needed for effective guidance, and it also highlights that prior approaches mainly handled low-depth search problems rather than genuinely long-horizon planning. "
            "If that bottleneck were addressed, a plausible near-term opportunity would be adaptive multi-turn reinforcement-learning frameworks that learn when to deliberate, revise, or re-rank candidate trajectories instead of depending on a fixed prompting heuristic."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original gold answer had the right general direction, but it leaned too hard on uncertainty-aware RL as the unique next step. The revised answer stays closer to what the historical evidence and sampled future papers actually support.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the fixed-value-function bottleneck but tied it directly to shallow-search limitations from the historical paper.",
                "Replaced the overly specific uncertainty-aware claim with a broader adaptive deliberation opportunity.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2309.17179",
                    "Alphazero-like Tree-Search can Guide Large Language Model Decoding and Training",
                    "https://arxiv.org/abs/2309.17179",
                    "Directly states that prompted pre-trained LLMs used as value functions can be inadequate and that earlier work focused on low search depth.",
                ),
                _paper(
                    "2302.06692",
                    "Guiding Pretraining in Reinforcement Learning with Large Language Models",
                    "https://arxiv.org/abs/2302.06692",
                    "Supports the broader point that exploration and long-horizon guidance remain difficult in large environments.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.03817",
                    "Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning",
                    "https://arxiv.org/abs/2509.03817",
                    "Provides moderate post-cutoff evidence for adaptive deliberation and coordination as a realized next step.",
                ),
            ],
            "audit_notes": [
                "The original answer was directionally plausible, but the sampled future evidence does not specifically prove uncertainty-aware value estimation.",
            ],
        },
    },
    "RTLv3-0003": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in reasoning-aware multi-turn reinforcement learning. "
            "Support it with recurring limitations, failure cases, or implicit constraints from multiple pre-cutoff studies, then explain one concrete research opportunity that would become plausible within the next six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already reasonably natural. The revision trims repeated guardrails and keeps the task specific.",
            "edits": [
                "Reduced duplicated wording about ex ante reasoning.",
            ],
        },
        "gold_answer": (
            "A defensible bottleneck in reasoning-aware multi-turn reinforcement learning was unreliable state abstraction during sequential decision making. "
            "Language-guided world-model papers showed that agents often needed to verify or correct hypothesized world states, while strategic-play work also exposed unstable, biased action selection over long interactions. "
            "Without trustworthy intermediate state representations, reasoning-aware RL cannot reliably support long-horizon planning. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be process-supervised planning-and-execution frameworks that explicitly score state verification, not just final outcomes."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was close in spirit, but it overstated the evidence for synthetic-data generation as the main realized opportunity. The revised answer stays closer to state verification and process supervision.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Kept the world-model reliability theme but avoided overclaiming that the cited history conclusively established a single bottleneck.",
                "Replaced the weaker synthetic-data opportunity with planning-and-execution supervision, which better matches the sampled future paper.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2301.12050",
                    "Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling",
                    "https://arxiv.org/abs/2301.12050",
                    "States that agents benefit from explicitly hypothesizing and then verifying world states before acting.",
                ),
                _paper(
                    "2310.18940",
                    "Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game",
                    "https://arxiv.org/abs/2310.18940",
                    "Supports the broader point that reasoning quality and action selection can become unstable across multi-turn strategic settings.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2510.05691",
                    "DecEx-RAG: Boosting Agentic Retrieval-Augmented Generation with Decision and Execution Optimization via Process Supervision",
                    "https://arxiv.org/abs/2510.05691",
                    "Provides moderate post-cutoff evidence for explicit supervision over planning and execution rather than only endpoint reward.",
                ),
            ],
            "audit_notes": [
                "Historical evidence is decent but not as cleanly concentrated as in some other tasks, so the refined answer stays conservative.",
            ],
        },
    },
    "RTLv3-0004": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in tool-augmented reasoning for LLM agents. "
            "Ground it in recurring limitations or failure cases from the historical record, then explain one concrete research opportunity that would become plausible within the following six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was clear but verbose. The revision makes the core task more direct.",
            "edits": [
                "Shortened the wording without changing scope.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was brittle tool use caused by missing explicit state tracking and planning. "
            "Reasoning-as-planning work argued that capable reasoning needs a world model, while web and desktop agent evaluations showed that prompt-only agents often fail to keep track of environment state, recover from mistakes, or coordinate multi-step tool actions. "
            "If this bottleneck were addressed, a concrete near-term opportunity would be training-based tool-using agents with explicit planning-action-reflection loops and persistent state representations instead of relying only on one-shot prompting."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was broadly correct. The revision makes the bottleneck more concrete by centering state tracking and recovery rather than using 'internal world model' as a vague umbrella term.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Grounded the claim in observed environment-state and recovery failures from the cited benchmark papers.",
                "Narrowed the opportunity to explicit planning-action-reflection training loops.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2305.14992",
                    "Reasoning with Language Model is Planning with World Model",
                    "https://arxiv.org/abs/2305.14992",
                    "Provides the conceptual case that strong reasoning requires an explicit world-model-like planning mechanism.",
                ),
                _paper(
                    "2404.07972",
                    "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments",
                    "https://arxiv.org/abs/2404.07972",
                    "Shows that multimodal agents still struggle with long-horizon tool execution in real computer environments.",
                ),
                _paper(
                    "2401.01614",
                    "GPT-4V(ision) is a Generalist Web Agent, if Grounded",
                    "https://arxiv.org/abs/2401.01614",
                    "Supports the need for grounding and careful state handling in web-agent tool use.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.06278",
                    "TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning",
                    "https://arxiv.org/abs/2509.06278",
                    "Moderately supports the move toward more autonomous, explicitly structured tool use.",
                ),
            ],
            "audit_notes": [
                "Future evidence is supportive but not a perfect one-to-one confirmation of the revised bottleneck.",
            ],
        },
    },
    "RTLv3-0005": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in embodied agent guidance strategies. "
            "Support it with concrete limitations or failure evidence from pre-cutoff work, then explain one research opportunity that would become plausible within the following six months if that bottleneck were resolved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original prompt was broad enough to invite diffuse answers. The revision keeps the task open but sharpens the requirement that the bottleneck come from embodied guidance evidence rather than generic agent trends.",
            "edits": [
                "Tightened the scope around embodied guidance evidence.",
            ],
        },
        "gold_answer": (
            "The strongest defensible bottleneck in the cited pre-cutoff work is unreliable exploration guidance. "
            "The in-context exploration paper shows that LLM-guided exploration does not reliably produce strong search behavior on its own, while LLMLight also points to generalization and robustness limits when guidance policies are deployed in sequential control settings. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be exploration-aware guidance systems that maintain compact state summaries or learned exploration objectives, instead of depending on ad hoc prompting or external interventions."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original gold answer made an unsupported jump from embodied-guidance evidence to multi-agent RL for traffic control. The revised answer stays much closer to the historical bottleneck actually visible in the cited papers.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Removed the unsupported leap to multi-agent reinforcement learning for traffic signal control as the main opportunity.",
                "Reframed the opportunity around exploration-aware guidance, which is directly tied to the historical evidence.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2403.15371",
                    "Can large language models explore in-context?",
                    "https://arxiv.org/abs/2403.15371",
                    "Directly studies whether prompt-only LLM systems can explore effectively and shows clear limitations.",
                ),
                _paper(
                    "2312.16044",
                    "LLMLight: Large Language Models as Traffic Signal Control Agents",
                    "https://arxiv.org/abs/2312.16044",
                    "Useful as evidence that language-guided sequential control still faces robustness and transfer issues.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.15233",
                    "Video2Roleplay: A Multimodal Dataset and Framework for Video-Guided Role-playing Agents",
                    "https://arxiv.org/abs/2509.15233",
                    "Only weakly supportive. It shows continued work on guidance-rich embodied or situated agents but does not cleanly validate the original gold answer.",
                ),
            ],
            "audit_notes": [
                "This task still needs stronger post-cutoff supplementation if it is to become a canonical benchmark item.",
            ],
        },
    },
    "RTLv3-0006": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in long-term memory architectures for LLM agents. "
            "Support it with recurring architectural limitations or empirical failure cases, then explain one concrete research opportunity that would become plausible within the next six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was serviceable but repetitive. The revised version makes the architecture focus more explicit.",
            "edits": [
                "Clarified that the evidence should be architectural rather than generic memory rhetoric.",
            ],
        },
        "gold_answer": (
            "A defensible bottleneck in pre-cutoff long-term memory architectures was the lack of reliable write, update, and retrieval policies for persistent memory. "
            "Agents often depended on growing context windows or ad hoc retrieval buffers, yet very-long-term dialogue work still found major gaps in temporal and causal recall. "
            "That suggests the problem was not simply 'more memory' but how to maintain and update memory in a way the agent can use consistently over time. "
            "If this bottleneck were addressed, a concrete near-term opportunity would be structured memory architectures with explicit read-write policies, temporal consistency checks, and memory-conditioned planning."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer had the right intuition but leaned too heavily on closed-model parametric updates, which were not the clearest architecture bottleneck in the sampled evidence.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Shifted the bottleneck from inaccessible parametric updates to poor write/update/retrieval policy design.",
                "Aligned the opportunity with structured persistent memory rather than generic modularity claims.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2402.17753",
                    "Evaluating Very Long-Term Conversational Memory of LLM Agents",
                    "https://arxiv.org/abs/2402.17753",
                    "Shows that memory-augmented systems still struggle on very-long-term temporal and causal understanding.",
                ),
                _paper(
                    "2308.10144",
                    "ExpeL: LLM Agents Are Experiential Learners",
                    "https://arxiv.org/abs/2308.10144",
                    "Provides supporting context on how agents accumulate and externalize experience when direct model adaptation is limited.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.17459",
                    "PRINCIPLES: Synthetic Strategy Memory for Proactive Dialogue Agents",
                    "https://arxiv.org/abs/2509.17459",
                    "Moderately supports the emergence of more explicit persistent-memory structures after the cutoff.",
                ),
            ],
            "audit_notes": [
                "The revised answer is still partly inferential because the historical sample mixes evaluation and architecture papers.",
            ],
        },
    },
    "RTLv3-0007": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in multi-agent debate frameworks. "
            "Support it with explicit limitations or failure cases from the pre-cutoff record, then explain one concrete research opportunity that would become plausible within the next six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original prompt read naturally, but the internal evidence attached to it is narrower than the title suggests. The revision keeps the task but makes the evidence burden stricter.",
            "edits": [
                "Tightened the requirement for explicit evidence because the historical support set is thin.",
            ],
        },
        "gold_answer": (
            "The strongest cautious formulation here is that early multi-agent deliberation systems lacked robust mechanisms for adapting critique or debate protocols when models were fixed black boxes. "
            "The cited historical evidence is not perfect debate-specific support, but it does show that multi-agent feedback schemes become hard to transfer when the underlying model cannot be fine-tuned. "
            "If that bottleneck were addressed, a plausible near-term opportunity would be black-box-compatible debate frameworks with learned role adaptation and dynamic control over when agents should challenge, revise, or defer."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original gold answer overstated what the cited historical paper proves. RL4F is relevant to multi-agent feedback and black-box constraints, but it is not a clean debate-framework paper. This task remains usable only with a more cautious answer and a note that historical evidence should be supplemented.",
            "historical_support": "weak",
            "future_support": "moderate",
            "changes": [
                "Downgraded the certainty of the bottleneck claim.",
                "Explicitly noted that the current historical support set needs supplementation with more debate-specific papers.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2305.08844",
                    "RL4F: Generating Natural Language Feedback with Reinforcement Learning for Repairing Model Outputs",
                    "https://arxiv.org/abs/2305.08844",
                    "Relevant to multi-agent feedback under black-box constraints, but only indirect support for debate frameworks.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.03817",
                    "Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning",
                    "https://arxiv.org/abs/2509.03817",
                    "Provides moderate support for adaptive multi-agent deliberation after the cutoff.",
                ),
            ],
            "audit_notes": [
                "This is one of the clearest cases where the current gold set should be treated as provisional until stronger historical papers are added.",
            ],
        },
    },
    "RTLv3-0008": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in multi-agent systems for software engineering. "
            "Support it with recurring failure cases or documented shortcomings, then explain one concrete research opportunity that would become plausible within the following six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already clear. The revision only reduces template phrasing.",
            "edits": [
                "Shortened repeated temporal-discipline language.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was weak repository-level understanding combined with cascading coordination errors across agents. "
            "MAGIS directly studies GitHub issue resolution and shows that agent systems still struggle to gather the right context, localize the real fault, and keep multi-agent workflows aligned over long issue-repair trajectories. "
            "If that bottleneck were addressed, a concrete near-term opportunity would be software-engineering agent systems with tighter repository representations, integrated testing and debugging loops, and better coordination over end-to-end issue resolution."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original answer was directionally correct, but its future-grounding claims were stronger than the sampled future papers justify. The revised version keeps the core bottleneck and narrows the claimed opportunity.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Kept the repository-level context and coordination bottleneck.",
                "Removed the claim that the sampled future set clearly demonstrates mature end-to-end issue resolution at scale.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2403.17927",
                    "MAGIS: LLM-Based Multi-Agent Framework for GitHub Issue Resolution",
                    "https://arxiv.org/abs/2403.17927",
                    "Direct evidence that multi-agent software engineering still struggled on realistic repository-level issue resolution.",
                ),
                _paper(
                    "2310.05292",
                    "How to Teach Programming in the AI Era? Using LLMs as a Teachable Agent for Debugging",
                    "https://arxiv.org/abs/2310.05292",
                    "Supportive context on debugging and interactive programming assistance, though less directly tied to multi-agent systems.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2511.18467",
                    "Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems",
                    "https://arxiv.org/abs/2511.18467",
                    "Shows continued post-cutoff activity around multi-agent software systems, but not a clean proof that the original bottleneck was solved.",
                ),
            ],
            "audit_notes": [
                "This task is worth keeping, but its future evidence set should be rebuilt with more issue-resolution-specific papers.",
            ],
        },
    },
    "RTLv3-0009": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in embodied agent navigation and interaction. "
            "Ground it in documented limitations or failure cases, especially in large environments or language-guided world modeling, then explain one concrete research opportunity that would become plausible within the next six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was strong already. The revision mainly trims repetition.",
            "edits": [
                "Reduced duplicated wording around the ex ante perspective.",
            ],
        },
        "gold_answer": (
            "A credible bottleneck in the pre-cutoff literature was the combination of inefficient exploration and unreliable world-state hypotheses in large embodied environments. "
            "ELLM shows that unguided novelty-seeking can be wasteful when most discovered novelty is irrelevant, while language-guided world-model work shows that agents often need explicit verification and correction of their hypothesized states. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be embodied systems that couple guided exploration with memory-backed state verification, enabling longer and more coordinated navigation-and-interaction tasks."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer identified the right two ingredients, but it jumped too quickly to multi-agent collaboration. The revised answer stays with guided exploration plus state verification.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Preserved the exploration-plus-world-model theme.",
                "Replaced the narrower multi-agent-collaboration claim with a more directly supported embodied-planning opportunity.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2302.06692",
                    "Guiding Pretraining in Reinforcement Learning with Large Language Models",
                    "https://arxiv.org/abs/2302.06692",
                    "Directly shows that intrinsic exploration is not enough in large environments.",
                ),
                _paper(
                    "2301.12050",
                    "Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling",
                    "https://arxiv.org/abs/2301.12050",
                    "Shows the need for explicit hypothesis verification in language-guided world modeling.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2511.19430",
                    "Cook and Clean Together: Teaching Embodied Agents for Parallel Task Execution",
                    "https://arxiv.org/abs/2511.19430",
                    "Moderately supports the move toward longer-horizon embodied coordination after the cutoff.",
                ),
            ],
            "audit_notes": [
                "Future support is decent but still broader than the exact revised bottleneck.",
            ],
        },
    },
    "RTLv3-0010": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in long-term memory management for LLM agents. "
            "Support it with recurring failure evidence or acknowledged gaps in the historical record, then explain one concrete six-month research opportunity that would become plausible if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original wording was good but repetitive. The revised version is more concise.",
            "edits": [
                "Reduced duplicated prompt scaffolding.",
            ],
        },
        "gold_answer": (
            "A clear bottleneck in the pre-cutoff literature was failure to preserve and selectively retrieve temporally grounded memory as interaction histories grew. "
            "Very-long-term conversational evaluation showed that performance drops sharply when agents must recall events, relations, and temporal details across extended dialogue. "
            "If this bottleneck were addressed, a concrete near-term opportunity would be persistent strategy and memory-management modules for long-horizon dialogue agents, including explicit state tracking and selective recall over sustained interactions."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was mostly sound. The revision replaces the vague phrase 'limited effective context length' with the more task-relevant failure mode of preserving and retrieving temporally grounded memory.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Tightened the bottleneck around selective long-range recall and temporal grounding.",
                "Anchored the opportunity to persistent strategy memory, which is directly visible in the future sample.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2402.17753",
                    "Evaluating Very Long-Term Conversational Memory of LLM Agents",
                    "https://arxiv.org/abs/2402.17753",
                    "Strong direct evidence that long-horizon memory management remained unsolved before the cutoff.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.17459",
                    "PRINCIPLES: Synthetic Strategy Memory for Proactive Dialogue Agents",
                    "https://arxiv.org/abs/2509.17459",
                    "Strong post-cutoff support for explicit persistent strategy memory in long-horizon dialogue settings.",
                ),
            ],
            "audit_notes": [
                "This is one of the cleaner items in the first 16 after refinement.",
            ],
        },
    },
    "RTLv3-0011": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in multi-agent software engineering. "
            "Support it with recurring empirical shortcomings from the historical record, then explain one concrete research opportunity that would become plausible within the next six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original prompt was understandable, but it largely duplicated task 0008 in meaning. The revision keeps it usable while making the multi-agent software-engineering focus more explicit.",
            "edits": [
                "Reduced redundancy with neighboring tasks by centering multi-agent coordination in software workflows.",
            ],
        },
        "gold_answer": (
            "A defensible pre-cutoff bottleneck was weak coordination over repository-scale software workflows: agents could assist on subtasks, but they still struggled to maintain shared context, localize failures correctly, and keep long repair trajectories on track. "
            "MAGIS provides direct evidence that realistic GitHub issue resolution remains difficult for multi-agent systems. "
            "If this bottleneck were reduced, a plausible near-term opportunity would be multi-agent software-engineering systems with stronger role coordination plus automated test, debug, and verification feedback loops."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original gold answer was directionally fine, but its future claims were too specific relative to the sampled evidence and it overlapped heavily with task 0008. The revision keeps the item while making it narrower and more cautious.",
            "historical_support": "moderate",
            "future_support": "weak",
            "changes": [
                "Narrowed the claim to coordination and shared-context failures.",
                "Avoided overclaiming that the future sample proves mature multi-agent debugging and requirements-engineering pipelines.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2403.17927",
                    "MAGIS: LLM-Based Multi-Agent Framework for GitHub Issue Resolution",
                    "https://arxiv.org/abs/2403.17927",
                    "The clearest historical source for repository-level and coordination-related failure modes.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2511.18467",
                    "Shadows in the Code: Exploring the Risks and Defenses of LLM-based Multi-Agent Software Development Systems",
                    "https://arxiv.org/abs/2511.18467",
                    "Shows that the area remained active after the cutoff, but does not by itself validate all of the original gold claims.",
                ),
            ],
            "audit_notes": [
                "This item still overlaps semantically with task 0008 and may need deduplication at the full-dataset stage.",
            ],
        },
    },
    "RTLv3-0012": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in ablation studies of the training stages used in RL fine-tuning for large language models. "
            "Support it with recurring limitations or failure evidence, then explain one concrete research opportunity that would become plausible within the next six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original wording was specific but overly long. The revision keeps the same technical focus with less scaffolding.",
            "edits": [
                "Shortened the prompt while preserving the training-stage emphasis.",
            ],
        },
        "gold_answer": (
            "A concrete bottleneck in the pre-cutoff literature was stage instability: model quality could degrade early and unpredictably across KL budgets or training phases, making it hard to isolate what each stage of RL fine-tuning was actually contributing. "
            "That is visible in work on reinforced fine-tuning and in analyses of overoptimization in direct alignment algorithms. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be stage-aware training curricula and cleaner ablation protocols that separate the effects of supervised preparation, reward shaping, and RL optimization."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was largely on the right track. The revision keeps the stage-instability claim but removes some overconfident language about a specific future wave.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the instability bottleneck.",
                "Made the opportunity about stage-aware curricula and measurement rather than a broad claim of already realized multi-stage consensus.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2401.08967",
                    "ReFT: Reasoning with Reinforced Fine-Tuning",
                    "https://arxiv.org/abs/2401.08967",
                    "Provides evidence that training-stage design materially affects reasoning performance and stability.",
                ),
                _paper(
                    "2406.02900",
                    "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms",
                    "https://arxiv.org/abs/2406.02900",
                    "Shows instability and degradation patterns across alignment regimes and KL settings.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2512.05107",
                    "STARE-VLA: Progressive Stage-Aware Reinforcement for Fine-Tuning Vision-Language-Action Models",
                    "https://arxiv.org/abs/2512.05107",
                    "Provides moderate post-cutoff evidence for explicitly stage-aware fine-tuning design.",
                ),
            ],
            "audit_notes": [
                "The historical evidence is strong, but the future sample spans multiple domains, so the refined answer stays conservative.",
            ],
        },
    },
    "RTLv3-0013": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in evaluating chain-of-thought reasoning with reinforcement-learning methods. "
            "Support the bottleneck with documented limitations or failure cases, then explain one concrete research opportunity that would become plausible within the following six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already clear. The revision mainly removes redundant phrasing.",
            "edits": [
                "Condensed the cutoff language.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was dependence on single or overly narrow reasoning traces when training or evaluating RL-based chain-of-thought systems. "
            "That setup makes it hard to distinguish robust reasoning from brittle path-specific behavior, because many valid reasoning trajectories are never considered. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be process-level evaluation and reward modeling that score multiple candidate traces, confidence, and structural validity rather than only a single rewarded path."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was already one of the better ones. The refinement mostly sharpens the opportunity around process-level and multi-trace evaluation.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Kept the single-trace brittleness bottleneck.",
                "Clarified that the opportunity is multi-trace and confidence-aware evaluation rather than just domain specialization.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2401.08967",
                    "ReFT: Reasoning with Reinforced Fine-Tuning",
                    "https://arxiv.org/abs/2401.08967",
                    "Direct evidence that RL-based reasoning methods depend heavily on the structure of the reasoning traces they optimize.",
                ),
                _paper(
                    "2406.14532",
                    "RL on Incorrect Synthetic Data Scales the Efficiency of LLM Math Reasoning by Eight-Fold",
                    "https://arxiv.org/abs/2406.14532",
                    "Supports the broader point that reasoning supervision and evaluation should not collapse to a single canonical path.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2510.10072",
                    "Unilaw-R1: A Large Language Model for Legal Reasoning with Reinforcement Learning and Iterative Inference",
                    "https://arxiv.org/abs/2510.10072",
                    "Moderately supports the move toward richer reasoning supervision and iterative inference.",
                ),
            ],
            "audit_notes": [
                "This item is solid after a relatively small refinement.",
            ],
        },
    },
    "RTLv3-0014": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in evaluating chain-of-thought reasoning through supervised fine-tuning. "
            "Support it with explicit historical limitations or failure cases, then explain one concrete research opportunity that would become plausible within the next six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original wording was clear but the downstream opportunity it invited was broader than the cited evidence justified. The revised version still asks the same kind of question but makes the evaluation focus sharper.",
            "edits": [
                "Tightened the evaluation focus so the answer does not drift too far into unrelated multimodal claims.",
            ],
        },
        "gold_answer": (
            "A defensible pre-cutoff bottleneck was that supervised fine-tuning often improved chain-of-thought availability without giving researchers a reliable way to evaluate reasoning-path fidelity, especially for smaller models or failure-heavy settings. "
            "The CoT Collection demonstrates that model size matters substantially for reasoning quality, and later work on learning from previous mistakes shows that reasoning evaluation needs better error analysis than final-answer accuracy alone. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be evaluation protocols that score reasoning-path quality, error type, and recovery behavior across text, tables, and other structured inputs, rather than only whether the final answer is correct."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original gold answer made too large a jump from small-model CoT generalization issues to multimodal CoT evaluation. The revised answer keeps the historical signal and narrows the downstream opportunity to richer reasoning-fidelity evaluation.",
            "historical_support": "strong",
            "future_support": "moderate",
            "changes": [
                "Removed the unsupported claim that multimodal chain-of-thought evaluation was the main realized opportunity.",
                "Reframed the opportunity around reasoning-fidelity and error-analysis evaluation.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2305.14045",
                    "The CoT Collection: Improving Zero-shot and Few-shot Learning of Language Models via Chain-of-Thought Fine-Tuning",
                    "https://arxiv.org/abs/2305.14045",
                    "Shows that the benefits of supervised CoT fine-tuning vary substantially with model capacity.",
                ),
                _paper(
                    "2403.20046",
                    "Can LLMs Learn from Previous Mistakes? Investigating LLMs' Errors to Boost for Reasoning",
                    "https://arxiv.org/abs/2403.20046",
                    "Provides strong support that error-aware evaluation is necessary beyond final-answer accuracy.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.06278",
                    "TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning",
                    "https://arxiv.org/abs/2509.06278",
                    "Moderately supports richer structured-input reasoning evaluation after the cutoff.",
                ),
                _paper(
                    "2510.10072",
                    "Unilaw-R1: A Large Language Model for Legal Reasoning with Reinforcement Learning and Iterative Inference",
                    "https://arxiv.org/abs/2510.10072",
                    "Provides additional moderate support for evaluating richer reasoning processes.",
                ),
            ],
            "audit_notes": [
                "The revised answer is meaningfully better aligned with the actual historical papers than the original gold set.",
            ],
        },
    },
    "RTLv3-0015": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in human-preference alignment benchmarks for large language models. "
            "Support it with recurring limitations or empirical failures from pre-cutoff work, then explain one concrete research opportunity that would become plausible within the following six months if the bottleneck were solved."
        ),
        "question_quality": {
            "status": "light_edit",
            "summary": "The original prompt was already good. The revision only removes repeated phrasing.",
            "edits": [
                "Minor wording cleanup.",
            ],
        },
        "gold_answer": (
            "A strong pre-cutoff bottleneck was that preference-alignment methods often improved the targeted preference objective while failing to reliably preserve broader pretrained capabilities. "
            "Historical work on RRHF, contrastive preference optimization, and weak-to-strong supervision all points to instability, supervision-quality sensitivity, or incomplete capability recovery under preference-oriented fine-tuning. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be purpose-specialized preference benchmarks that explicitly test capability retention while aligning to narrow cultural, stylistic, or domain-specific preferences."
        ),
        "answer_audit": {
            "status": "validated_with_refinement",
            "summary": "The original answer was broadly correct. The refinement tightens the bottleneck around capability preservation and reduces some overconfident language about the future wave.",
            "historical_support": "strong",
            "future_support": "strong",
            "changes": [
                "Made capability retention the central bottleneck.",
                "Kept the downstream opportunity at the benchmark-design level, which is better supported by the future sample.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2312.09390",
                    "Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision",
                    "https://arxiv.org/abs/2312.09390",
                    "Strong direct evidence that aligned systems may fail to recover the full capabilities of stronger base models under weak supervision.",
                ),
                _paper(
                    "2304.05302",
                    "RRHF: Rank Responses to Align Language Models with Human Feedback without tears",
                    "https://arxiv.org/abs/2304.05302",
                    "Supports the broader claim that preference alignment remains sensitive and can distort capability tradeoffs.",
                ),
                _paper(
                    "2401.08417",
                    "Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation",
                    "https://arxiv.org/abs/2401.08417",
                    "Supports supervision-quality dependence and domain-specific performance tradeoffs.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.01035",
                    "We Politely Insist: Your LLM Must Learn the Persian Art of Taarof",
                    "https://arxiv.org/abs/2509.01035",
                    "Strong post-cutoff example of culturally specific preference benchmarking and alignment.",
                ),
            ],
            "audit_notes": [
                "This item remains one of the healthier benchmark entries after refinement.",
            ],
        },
    },
    "RTLv3-0016": {
        "question": (
            "Using literature published on or before August 31, 2025, identify one unresolved bottleneck in evaluating multimodal fine-tuning methods. "
            "Support it with explicit gaps or failure cases from the pre-cutoff literature, then explain one concrete research opportunity that would become plausible within the next six months if that bottleneck were solved."
        ),
        "question_quality": {
            "status": "moderate_edit",
            "summary": "The original prompt was understandable but broad enough to invite unsupported answers. The revision keeps the multimodal-evaluation focus but makes the evidence requirement tighter.",
            "edits": [
                "Sharpened the evaluation scope so the answer does not drift into general multimodal fine-tuning.",
            ],
        },
        "gold_answer": (
            "A defensible pre-cutoff bottleneck was the lack of reliable evaluation procedures for multimodal hallucination, factual consistency, and reward quality without dense human annotation. "
            "The strongest historical support in this sample comes from work on preference optimization for video multimodal models, which highlights how hard it is to judge outputs with language-model rewards alone. "
            "If this bottleneck were reduced, a concrete near-term opportunity would be AI-assisted and reasoning-aware multimodal evaluators that can score factual grounding, hallucination, and sycophancy across text-image-video settings at much larger scale."
        ),
        "answer_audit": {
            "status": "substantially_corrected",
            "summary": "The original gold answer was directionally plausible but too broad for the historical support set, and one cited historical paper was only weakly relevant. The revised answer keeps the item usable by centering multimodal evaluation reliability rather than claiming a broad label-free-evaluation wave.",
            "historical_support": "moderate",
            "future_support": "moderate",
            "changes": [
                "Dropped the overbroad claim that task-specific human annotation was the single key bottleneck across the area.",
                "Narrowed the opportunity to scalable multimodal factuality and sycophancy evaluation.",
            ],
        },
        "ground_truth": {
            "historical_supporting_papers": [
                _paper(
                    "2404.01258",
                    "Direct Preference Optimization of Video Large Multimodal Models from Language Model Reward",
                    "https://arxiv.org/abs/2404.01258",
                    "Provides the clearest historical evidence here: evaluating multimodal outputs with language-model rewards is difficult and noisy.",
                ),
                _paper(
                    "2310.13023",
                    "GraphGPT: Graph Instruction Tuning for Large Language Models",
                    "https://arxiv.org/abs/2310.13023",
                    "Only weakly relevant historical support and should likely be replaced during full-dataset cleanup.",
                ),
            ],
            "future_supporting_papers": [
                _paper(
                    "2509.16149",
                    "Pointing to a Llama and Call it a Camel: On the Sycophancy of Multimodal Large Language Models",
                    "https://arxiv.org/abs/2509.16149",
                    "Moderately supports the emergence of richer multimodal evaluation targets beyond plain task accuracy.",
                ),
                _paper(
                    "2510.16455",
                    "RAVEN: Robust Advertisement Video Violation Temporal Grounding via Reinforcement Reasoning",
                    "https://arxiv.org/abs/2510.16455",
                    "Additional moderate support that multimodal evaluation is moving toward richer reasoning-based assessment.",
                ),
            ],
            "audit_notes": [
                "This item should receive stronger historical-paper supplementation in the full pass.",
            ],
        },
    },
}


def load_head(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle, start=1):
            if i > n:
                break
            rows.append(json.loads(line))
    return rows


def build_records() -> list[dict]:
    public_rows = load_head(PUBLIC_TASKS, 16)
    internal_rows = load_head(INTERNAL_TASKS, 16)
    internal_by_id = {row["task_id"]: row for row in internal_rows}

    records = []
    for public_row in public_rows:
        task_id = public_row["task_id"]
        internal_row = internal_by_id[task_id]
        override = OVERRIDES[task_id]
        answer_audit = override["answer_audit"]
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
            "answer_audit": answer_audit,
            "ground_truth": override["ground_truth"],
            "review_metadata": {
                "pilot_batch": "RTLv3-0001-0016",
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

    lines = [
        "# Benchmark Refine Log 20260423",
        "",
        "## Scope",
        "",
        "- Source release: `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/benchmark_release`",
        "- Output file: `benchmark_release/task_refined.jsonl`",
        "- This pilot pass refines the first 16 public tasks (`RTLv3-0001` to `RTLv3-0016`).",
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
            "- `RTLv3-0005`, `RTLv3-0007`, `RTLv3-0008`, `RTLv3-0011`, and `RTLv3-0016` still need stronger evidence supplementation in the full-dataset pass.",
            "- `RTLv3-0008` and `RTLv3-0011` overlap semantically and should be reconsidered for deduplication later.",
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
    records = build_records()
    write_jsonl(OUTPUT_PATH, records)
    write_log(LOG_PATH, records)
    print(f"Wrote {len(records)} refined tasks to {OUTPUT_PATH}")
    print(f"Wrote log to {LOG_PATH}")


if __name__ == "__main__":
    main()
