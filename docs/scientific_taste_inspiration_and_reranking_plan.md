# 从 “AI Can Learn Scientific Taste” 到 Benchmark Agent 适配的启发

本文记录论文 **AI Can Learn Scientific Taste**（Tong et al., 2026, arXiv:2603.14473）对当前 `ResearchInsightBenchmark` 工作的直接启发，并将其转化为对 **ARIS** 与 **ResearchAgent** 的可执行增强方案。

## 1. 这篇论文最值得吸收的点

### 1.1 重点不在执行，而在 judgment

该论文强调：相比“会不会写代码、跑实验、读论文”，AI scientist 更缺的是 **scientific taste**，即：

- 判断什么问题更重要
- 判断什么方向更可能形成高影响力后续
- 提出更有潜力的 idea

这和我们 benchmark 的目标高度一致。我们当前的任务家族，本质上都在考察一种 **forward-looking scientific judgment**。

### 1.2 candidate selection 可能比 generation 更关键

该论文的方法虽然包含训练，但它最重要的工程启发不是 RL 本身，而是：

- 先得到多个 candidate
- 再用一个更偏“judgment/taste”的模块去排序和选择

这和我们当前对 ARIS / ResearchAgent 的观察一致：

- 它们通常能生成“像样的答案”
- 但 final selection 经常选错
- 特别是在 `direction_forecasting` 上，问题往往不是不会生成，而是没有选中 benchmark 真正对应的 future cluster

### 1.3 pairwise preference 是很好的辅助思路，但不应直接替代主 benchmark

该论文大量依赖 pairwise preference。这个思路值得吸收，但不能直接替代我们的主 benchmark，因为我们的 benchmark 还要求：

- open-ended answer generation
- evidence traceability
- strict temporal cutoff
- family-specific structural grounding

所以更合理的做法是：

- 保留当前 benchmark 主体
- 在 method 内部引入 taste-aware reranking
- 在未来的 benchmark 扩展中考虑 pairwise auxiliary track

## 2. 对当前 benchmark 叙事的启发

可以把我们的 benchmark 更明确地表述为：

> a benchmark for evidence-grounded scientific judgment under strict temporal cutoff.

或者更口语化地说：

> 我们测的不是 AI 会不会复述文献，而是它能不能在历史证据约束下，做出有价值的研究判断。

这和 scientific taste 的概念相邻，但我们的约束更严格：

- 不是只看 citation preference
- 不是只看 pairwise choice
- 而是要求 agent 在 bounded historical evidence 下给出完整、可追溯、可事后验证的答案

## 3. Taste-aware reranking: 对 ARIS 的落地方案

### 3.1 不改变 ARIS 主干

仍然保留：

1. survey
2. ideation
3. review
4. render

Taste-aware 改造只放在：

- survey 后的 evidence packet construction
- review 阶段的 candidate reranking
- render 前的 final selection sanity check

### 3.2 需要新增的 taste 特征

#### Forecasting

优先加分：

- **future-cluster formation potential**：是否更像未来会形成一个研究簇，而不是单篇 paper trick
- **expected-deliverable alignment**：是否贴近 task frame 中的 expected deliverable
- **generality / transferability**：是否有跨设置、跨任务、跨子场景的扩展潜力
- **community uptake plausibility**：是否符合历史趋势、venue 偏好、代表论文后续延展逻辑
- **artifact-title reuse penalty**：是否只是把旧 paper 标题做轻微改写
- **over-specialization penalty**：是否过于局部，导致和 hidden future cluster 错位

#### Bottleneck

优先加分：

- **recurring-friction centrality**：是不是 recurring limitation，而不是偶发问题
- **unlock leverage**：如果解决，是否能打开 downstream opportunity
- **mechanism explicitness**：bottleneck 与 opportunity 之间是否有清晰技术机制
- **generic bottleneck penalty**：是否只是“数据不够/评测不足”这类弱诊断

#### Planning

优先加分：

- **dependency realism**：技术依赖是否真实存在
- **ordering plausibility**：优先级排序是否合理
- **field leverage**：该计划是否具有较高研究杠杆
- **crowded-direction penalty**：是否把已经拥挤、边际收益低的方向排得过高

#### Venue-aware

优先加分：

- **venue-fit realism**：是否符合 venue 常见的 empirical framing / contribution pattern
- **positioning clarity**：方向、主张、评测风格之间是否对齐
- **reviewer-appeal plausibility**：是否符合该 venue 常见接受偏好

### 3.3 推荐打分公式

对 ARIS 的 candidate reranking，建议在原有 score 上增加：

```text
final_score =
  evidence_support
+ family_grounding_prior
+ historical_plausibility
+ expected_deliverable_alignment
+ scientific_taste_bonus
- generic_penalty
- artifact_reuse_penalty
- over_specialization_penalty
```

其中：

- `scientific_taste_bonus` 不应压过 evidence support
- 但在 forecasting 上，应足够压过 paper-title overlap 这种弱信号

## 4. Taste-aware reranking: 对 ResearchAgent 的落地方案

ResearchAgent 的问题不是不会读文献，而是太容易停在“综述模式”。

### 4.1 不改变主干

仍然保留：

1. literature retrieval
2. reading / aggregation
3. synthesis

新增 taste-aware 层放在 synthesis 前后：

- synthesis 前：evidence compression into decision packet
- synthesis 后：candidate judgement and selection

### 4.2 建议新增模块

#### Module A: decision packet

把 retrieval/read 的结果压成：

- historical state
- central unresolved issue
- strongest successor signals
- likely inflection
- venue / dependency cues

#### Module B: multi-candidate judgement

不要只生成一个答案。先生成 `k` 个 candidate，再让 judgment head 排序。

建议至少输出：

- 1 个偏保守 candidate
- 1 个偏高杠杆 candidate
- 1 个偏结构化/grounded candidate

然后再排序。

#### Module C: anti-survey penalty

ResearchAgent 需要显式惩罚：

- 综述式但无 forward commitment
- 列举很多方向但不做判断
- 只总结 pre-cutoff literature，却不输出 next-step judgement

## 5. 对后续 benchmark 扩展的启发

这篇论文还提示我们，未来可以扩展一个 **pairwise scientific judgment track**。例如：

- 给两个 candidate future directions，让模型选哪个更可能在未来 materialize
- 给两个 bottleneck，让模型选哪个更 high-leverage
- 给两个 plan，让模型选哪个更 technically grounded

这个 track 的优点是：

- 自动评测更稳
- 更容易隔离 judgement 能力
- 可与主 benchmark 的开放式任务形成互补

但这应该作为扩展，不应替代当前主 benchmark。

## 6. 实施优先级建议

### 第一优先级

- ARIS forecasting reranker 接入 `expected_deliverable_alignment + scientific_taste_bonus`
- 把 title-derived historical topic overlap 降权

### 第二优先级

- ResearchAgent 增加 `decision packet + multi-candidate judgement`
- 增加 anti-survey penalty

### 第三优先级

- 增加 pairwise auxiliary evaluation track
- 用于分析“系统是否具备 taste，但 generation 表达不稳定”这类情况

## 7. 一句话总结

这篇论文对我们的最大启发不是“去做 citation-RL”，而是：

> 对 research agent 而言，生成候选并不难，难的是用一个更像 scientific judge 的模块，把真正高价值、可实现、与未来发展更对齐的 candidate 选出来。
