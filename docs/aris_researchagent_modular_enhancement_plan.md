# ARIS 与 ResearchAgent 模块化增强方案（不改变核心方法）

本文给出一个针对 **ARIS** 与 **ResearchAgent** 的增强方案。目标不是把它们重写成新的 agent，而是在 **不改变核心方法主干** 的前提下，增加一层对 `ResearchInsightBenchmark` 更友好的模块，使其在 benchmark 上的表现更稳定。

---

## 1. 设计原则

### 1.1 什么可以改

允许增加以下“外插模块”：

1. **task-family router**
2. **offline retrieval adapter**
3. **evidence filtering / reranking**
4. **family-specific reasoning scaffold**
5. **candidate answer reranker / self-critique**
6. **benchmark-facing final composer**
7. **JSON repair + retry / failure fallback**

这些模块本质上属于：
- 输入适配
- 检索增强
- 输出约束
- 稳定性工程

不改变原方法的核心 reasoning backbone。

### 1.2 什么不应该改

原则上不做以下事情：

1. 把 **ARIS** 改写成新的 planner / 自定义 agent graph
2. 把 **ResearchAgent** 的核心 literature assistant 主链替换成 ResearchArc 的 signal-abstraction 流程
3. 把 benchmark 的 hidden GT 或 future evidence 暴露给推理阶段
4. 用一套统一的“强行模板”替代原方法原生的候选生成与审查流程

---

## 2. 两个方法的原始定位

### 2.1 ARIS

ARIS 原生更像：

- `survey -> ideation -> review` 的轻量研究工作流
- research copilot / skill workflow
- 用调研、构思、批判式审查来形成研究建议

它擅长：
- 组织文献景观
- 生成候选研究方向
- 做轻量 review / selection

它天然较弱的点：
- 对严格的 temporal-cutoff 任务不一定足够敏感
- 对 bottleneck → opportunity 这种机制链任务不一定足够锋利
- 对 future alignment 的判断未必稳定

### 2.2 ResearchAgent

ResearchAgent 原生更像：

- literature-grounded research assistant
- 检索论文、阅读论文、整理论文、形成分析的 agent

它擅长：
- 文献检索与综合
- 研究辅助分析
- 论文级别的信息组织

它天然较弱的点：
- 更偏 understanding / synthesis，而不是 ex-ante forecasting
- 对 family-specific 的任务结构未必天然对齐
- 容易输出“很像综述”的答案，而不是“明确 forward judgment”的答案

---

## 3. ARIS 的增强方案

### 3.1 核心保留

保留 ARIS 的主流程：

1. `literature survey`
2. `idea generation`
3. `critical review / selection`
4. `final answer rendering`

也就是说，ARIS 仍然是“先调研，再想法生成，再审查”的 agent，而不是被改写成别的框架。

### 3.2 建议新增模块

#### Module A: Task-family router

作用：
- 在 ARIS survey 之前，先识别任务属于：
  - bottleneck opportunity discovery
  - direction forecasting
  - strategic research planning
  - venue-aware research positioning

影响：
- 不改 ARIS 主流程，只决定后续 query hints、review rubric 和最终渲染方式。

预期提升：
- `Future Alignment`
- family-specific grounding

---

#### Module B: Offline retrieval adapter

作用：
- 统一接 benchmark 的离线 KB
- 所有输入证据严格受 `time_cutoff` 约束
- 同时支持：
  - paper-level retrieval
  - structure retrieval
  - section retrieval
  - pageindex retrieval（尤其 bottleneck）

影响：
- 不改变 ARIS 的 survey 动作，只替换输入证据源。

预期提升：
- `Evidence-Grounded Factuality`
- `Evidence Traceability`

---

#### Module C: Evidence landscape enhancer

作用：
- 在 survey 阶段额外提供：
  - representative papers
  - recurring limitations
  - future-work signals
  - venue distribution
  - successor topic candidates
  - historical likelihood cues

这层不负责直接回答问题，而是让 ARIS survey 到的“研究景观”更像 benchmark 真正想测的结构。

预期提升：
- `Future Alignment`
- `Opportunity / Forecast / Dependency Grounding`

---

#### Module D: Family-specific ideation scaffold

作用：
- 在 `idea generation` 阶段加任务骨架，但不替代 ARIS 自己生成候选。

建议骨架：

- **bottleneck**：`historical limitation -> central unresolved mechanism -> unlocked opportunity`
- **forecasting**：`current signals -> inflection -> 1-3 plausible next directions`
- **planning**：`candidate directions -> dependencies / risks -> ranked plan`
- **venue-aware**：`technical direction -> contribution framing -> venue fit`

预期提升：
- family-specific grounding
- `Future Alignment`

---

#### Module E: Review-stage reranker

作用：
- 保留 ARIS 的 review 机制，但增加：
  - historical plausibility
  - specificity bonus
  - generic-answer penalty
  - evidence-anchor bonus

换句话说，不替 ARIS 做决定，而是给它更适合 benchmark 的审查标准。

预期提升：
- `Future Alignment`
- `Evidence-Grounded Factuality`
- family-specific grounding

---

#### Module F: Benchmark-facing final composer

作用：
- 保留 review 选出的核心内容
- 最后一层把答案写成：
  - 先给明确结论
  - 再给最关键的 evidence anchors
  - 少空话，少泛化句

注意：
- 这层应该是“answer shaping”，不是重写 reasoning。

预期提升：
- `Evidence Traceability`
- `Evidence-Grounded Factuality`

---

### 3.3 ARIS 最小增强包（推荐先做）

如果只做最小版本，建议先实现：

1. task-family router
2. offline retrieval adapter
3. family-specific ideation scaffold
4. review-stage reranker
5. benchmark-facing final composer

这五个模块已经足够形成一个“明显更强但仍然是 ARIS”的版本。

---

## 4. ResearchAgent 的增强方案

### 4.1 核心保留

保留 ResearchAgent 的主链：

1. literature retrieval
2. literature reading / aggregation
3. synthesis / answer drafting

ResearchAgent 仍然应被看作一个 literature-grounded research assistant，而不是被改造成新的 research-trajectory agent。

### 4.2 建议新增模块

#### Module A: Task decomposition front-end

作用：
- 在原始问题进入 ResearchAgent 之前，先拆出统一的问题骨架：
  - `historical state`
  - `central unresolved issue`
  - `forward implication`
  - （如需要）`ranking / venue / dependency`

这样做不会替它回答问题，只是减少“读了很多论文却不知道问题究竟要什么”的情况。

预期提升：
- family-specific grounding
- `Future Alignment`

---

#### Module B: Family-aware retrieval expansion

作用：
- 保留原始 retrieval
- 但根据 family 增加不同的 query expansion：
  - bottleneck: limitation / failure / unresolved challenge
  - forecasting: emerging direction / trend / successor line
  - planning: roadmap / prerequisite / dependency / open problem
  - venue-aware: benchmark / evaluation / empirical framing / venue fit

预期提升：
- `Evidence-Grounded Factuality`
- `Future Alignment`

---

#### Module C: Evidence filtering and citation pruning

作用：
- 检索回来后增加一层精筛：
  - 去掉只有关键词命中、但不支持具体推理的 paper
  - 控制证据数量
  - 保留更高价值的 limitation / method / result / future-work evidence

这样能减少“citation stuffing”。

预期提升：
- `Evidence Traceability`
- `Evidence-Grounded Factuality`

---

#### Module D: Family-specific reasoning scaffold

作用：
- 在 synthesis 阶段加 reasoning scaffold：

- **bottleneck**：要求显式输出 `bottleneck -> why unresolved -> downstream opportunity`
- **forecasting**：要求显式输出 `signals -> inflection -> predicted direction`
- **planning**：要求显式输出 `ranked priorities -> dependencies -> trade-offs`
- **venue-aware**：要求显式输出 `technical positioning -> contribution framing -> likely venue fit`

这一步依然不替换 ResearchAgent 的“读文献并综合”的核心，而是让综合更符合 benchmark contract。

预期提升：
- family-specific grounding
- `Future Alignment`

---

#### Module E: Candidate answer reranker / self-critique

作用：
- 让 ResearchAgent 生成 2--3 个候选答案
- 再用轻量 judge 从以下维度打分：
  - specificity
  - evidence support
  - task-family fit
  - forward commitment

选最强的一版输出。

预期提升：
- `Future Alignment`
- `Evidence Traceability`
- family-specific grounding

---

#### Module F: Final answer normalizer

作用：
- 把最终答案收束成 benchmark 友好的风格：
  - 先结论
  - 再 evidence anchors
  - 再必要的 caveat

ResearchAgent 原生可能更像综述式回答，这一层的目标就是避免“写得很多，但判断不够 sharp”。

预期提升：
- `Evidence Traceability`
- `Evidence-Grounded Factuality`

---

### 4.3 ResearchAgent 最小增强包（推荐先做）

如果只做最小版本，建议先实现：

1. task decomposition front-end
2. family-aware retrieval expansion
3. evidence filtering and citation pruning
4. family-specific reasoning scaffold
5. candidate answer reranker

这套组合更适合 ResearchAgent，因为它原生已经有较强的 literature synthesis 主链。

---

## 5. 两者对指标的主要提升方向

### 5.1 ARIS

优先希望提升：

1. `Future Alignment`
2. `Evidence-Grounded Factuality`
3. family-specific grounding

原因：
- ARIS 通常能整理和构思，但 forward judgment 不够稳
- 容易出现“方向不够尖锐 / 结论不够具体”的问题

### 5.2 ResearchAgent

优先希望提升：

1. family-specific grounding
2. `Future Alignment`
3. `Evidence Traceability`

原因：
- ResearchAgent 往往能做文献综合
- 但容易偏综述，不一定能把 reasoning 对齐到 benchmark 的任务结构

---

## 6. 推荐实验顺序

### Phase 1: Pilot

先不要全量直接重跑，建议先做 12--20 题 pilot：

- 4 families 都覆盖
- 4 domains 尽量覆盖
- 看以下指标是否有实质提升：
  - `Evidence-Grounded Factuality`
  - `Future Alignment`
  - `Evidence Traceability`
  - family-specific grounding

### Phase 2: Small-batch refinement

如果 pilot 有收益，再针对最弱 family 做定向微调：

- ARIS 优先救 bottleneck / forecasting
- ResearchAgent 优先救 planning / bottleneck

### Phase 3: Full run

只有当 pilot 里至少有两个核心指标显著提升时，才值得进入全量评测或更大规模重跑。

---

## 7. 一句话总结

- **ARIS**：更适合做“外层强化”，重点增强 survey 输入、candidate review 与 final answer shaping。
- **ResearchAgent**：更适合做“中层结构增强”，重点增强 task decomposition、evidence filtering 与 family-specific synthesis scaffold。
- 两者都应坚持同一个原则：

> **不替换原方法的核心 backbone，只增加更适合 benchmark 的模块层。**
