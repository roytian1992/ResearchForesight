# 候选任务质量筛选指标 与 实验测评指标 的区分

## 1. 为什么需要区分

当前 benchmark 流程里存在两套不同的打分体系：

1. **构建阶段的候选任务质量筛选**
2. **实验阶段的模型答案测评**

它们服务于完全不同的问题。如果都统称为 `judge`，会在代码、文档和结果解释上产生混淆。因此，构建阶段的打分更准确地应理解为 **candidate-quality screening**，而不是 benchmark evaluation。

---

## 2. 构建阶段：候选任务质量筛选

### 2.1 目标

这一步回答的问题是：

> 这条候选题目本身，是否足够好，能否进入 benchmark？

因此它评估的是 **任务质量**，而不是某个模型的答题能力。

### 2.2 当前字段命名

为避免歧义，当前代码已引入更清晰的别名：

- `candidate_quality_judge`: 推荐使用的名称
- `judge`: 兼容旧脚本保留的历史字段

两者当前保存的是同一份构建阶段筛选结果，但后续文档和新代码应优先使用 `candidate_quality_judge`。

### 2.3 这一步评什么

这一步采用 **family-specific LLM-as-judge rubric**。不同任务家族使用不同维度，因为不同 family 的“好题”标准不同。

例如：

#### Direction Forecasting
- family_fit
- temporal_discipline
- historical_grounding
- next_direction_specificity
- trajectory_alignment
- future_ground_truth_strength
- surface_naturalness
- public_benchmark_suitability

#### Bottleneck-Opportunity Discovery
- family_fit
- temporal_discipline
- deep_reading_need
- bottleneck_specificity
- historical_grounding
- opportunity_linkage
- future_ground_truth_strength
- surface_naturalness
- public_benchmark_suitability

#### Strategic Research Planning
- family_fit
- temporal_discipline
- planning_scope_clarity
- prioritization_quality
- evidence_grounding
- strategic_value
- future_alignment
- surface_naturalness
- public_benchmark_suitability

### 2.4 这一步的意义

它本质上在过滤以下问题：

- 题目是否像这个 family
- 时间边界是否清晰
- 问题是否具体、可答、可判
- gold answer 是否足够明确
- future window 是否能提供足够强的 ground truth 支撑
- 表达是否适合公开 benchmark

因此，这一步是 **task construction quality control**，不是模型实验测评。

---

## 3. 实验阶段：模型答案测评

### 3.1 目标

这一步回答的问题是：

> 给定同一道 benchmark 题，某个模型或 agent 的回答质量如何？

因此它评估的是 **方法表现**。

### 3.2 当前使用的主指标

目前实验阶段的指标体系包括：

#### Universal Primary
- Evidence-Grounded Factuality
- Future Alignment
- Evidence Traceability

#### 其他指标
- Task Fulfillment
- family-specific metric

这些指标直接针对 **模型输出**，与候选题构建阶段的筛选分不是一回事。

---

## 4. 推荐的术语约定

后续文档、代码和表格中建议统一采用以下术语：

### 构建阶段
- Candidate Quality Judge
- Candidate Screening Score
- Candidate Quality Filtering

### 实验阶段
- Benchmark Evaluation
- Answer Evaluation
- Model Performance Metrics

不建议继续用一个笼统的 `judge score` 同时指代两者。

---

## 5. 当前实现策略

为了不破坏已有脚本，当前采用 **非破坏式兼容**：

- 旧字段 `judge` 保留
- 新字段 `candidate_quality_judge` 同步写入
- 新脚本优先读取 `candidate_quality_judge`
- 如果不存在，再回退到 `judge`

这保证了：

1. 老产物仍然能用
2. 新代码的语义更清晰
3. 后续逐步清理命名时不会影响已有流程

---

## 6. 一个简洁结论

一句话概括：

> `candidate_quality_judge` 是在评“这道题能不能进 benchmark”；  
> 实验指标是在评“模型把这道题答得好不好”。

