# Venue Prior Knowledge for Evaluation

## Purpose

- make `venue_aware_research_positioning` less like a pure bucket-matching task
- let the evaluator use soft venue priors about:
  - typical scope
  - typical contribution package
  - reviewer expectation / evidence package
  - nearby compatible venue families
- explicitly allow multiple compatible venues when the answer distinguishes primary fit from secondary fit

## Implemented In

- `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight/src/researchworld/experiment_eval_aux.py`

## Current Prior Profiles

- `ACL / EMNLP / NAACL`
  - NLP / computational linguistics / multilingual / resource-and-evaluation oriented
  - strong fit for empirical studies, resources, benchmarks, analyses
  - reviewers typically expect strong baselines, ablations, error analysis, human evaluation or dataset quality discussion

- `ICLR / ICML / NeurIPS`
  - broad ML / representation learning / rigorous ML methodology
  - strong fit for learning methods, theory, benchmark methodology, foundation-model evaluation, infrastructure
  - reviewers typically expect strong experiments, ablations, robustness / scaling discussion, and sometimes theory or broader ML significance

- `AAAI / IJCAI`
  - broad AI with stronger tolerance for reasoning / planning / agents / integrated AI framing
  - fit is better when the answer makes AI-task relevance explicit rather than only claiming deep-learning novelty

- `KDD / SIGIR`
  - data science / knowledge discovery / retrieval / ranking / information access
  - fit is better when the answer specifies retrieval, ranking, search, data-driven impact, evaluation, reproducibility, or deployment-facing package

- `CVPR / ICCV / ECCV`
  - computer vision / vision-language / visual benchmark / qualitative + quantitative evaluation

## Evaluation Policy Change

- nearby compatible venue families remain acceptable
- high score now requires:
  - a clear primary fit
  - a venue-typical contribution package
  - venue-typical reviewer / evidence expectations
  - if multiple compatible venue families are mentioned, an explicit primary-vs-secondary distinction

## Official Sources Used For Bucket-Level Priors

- ACL 2024 main conference call:
  - https://2024.aclweb.org/calls/main_conference_papers/
- EMNLP 2024 main conference call:
  - https://2024.emnlp.org/calls/main_conference_papers/
- ICLR 2024 call for papers:
  - https://iclr.cc/Conferences/2024/CallForPapers
- NeurIPS 2024 call for papers:
  - https://nips.cc/Conferences/2024/CallForPapers
- ICML 2025 call for papers:
  - https://icml.cc/Conferences/2025/CallForPapers
- KDD 2025 research track call for papers:
  - https://kdd2025.kdd.org/research-track-call-for-papers/

## Notes

- this is currently applied in `family auxiliary` evaluation, not as a standalone benchmark field
- the priors are intentionally soft: they should guide scoring, not force one exact venue when several compatible venues are plausible
