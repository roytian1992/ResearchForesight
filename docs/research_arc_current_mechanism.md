# ResearchArc 当前机制说明（accepted v6 optimized）

本文记录当前被接受的 ResearchArc 版本，即 `src/researchworld/research_arc_v6.py` 对应的 **v6 optimized / full168** 机制。

## 1. 总体定位
ResearchArc 不是单纯的 RAG，也不是直接从检索结果生成答案。它的核心思路是：

1. 先围绕任务做 **高召回证据恢复**；
2. 再做 **任务焦点对齐** 与 **family-specific 先验推断**；
3. 把原始证据压缩成一个 **historical signal map**；
4. 最后基于这个 signal map 做 **mechanism-level reasoning** 和 **final refinement**。

也就是说，它强调“先组织研究信号，再回答问题”，而不是“检索到什么就直接写什么”。

---

## 2. 当前 pipeline
在 `ResearchArcV6.run_task(...)` 中，当前主流程如下：

### Step 1. Task parsing
- 解析任务 family、domain、horizon、time cutoff、task contract。
- 根据 family 映射到更具体的任务子类型：
  - `bottleneck_opportunity_discovery` → `mechanism_bottleneck_opportunity`
  - `direction_forecasting` → `mechanism_inflection_forecast`
  - `strategic_research_planning` → `ranked_agenda_with_dependencies`

### Step 2. Policy routing
- 由 `PolicyRouter` 决定当前任务走哪个 family head：
  - `BottleneckHeadV4`
  - `DirectionHeadV3`
  - `PlanningHeadV3`

### Step 3. Evidence backbone construction
- `EvidenceBackbone(domain_id).build(task=task)` 负责构建离线证据包。
- 当前主要证据源包括：
  - paper-level evidence
  - structure-level evidence
  - section-level evidence
- 这些证据都来自 cutoff 之前的离线 benchmark/support packet 资产，不依赖联网搜索。

### Step 4. Focus resolution
- `FocusResolver` 对候选 node / topic 做任务对齐，找到真正和题目最相关的焦点。
- 这个阶段不是只看关键词，而是综合 task、support packet、ranked nodes 等信息来定位任务焦点。

### Step 5. Evidence formatting
- `EvidenceFormatter` 将证据整理成更适合 reasoning 的中间结构，例如：
  - focus summary
  - paper evidence
  - structure evidence
  - section evidence
  - subdirection candidates
  - bottleneck candidates
  - opportunity candidates

### Step 6. Family-specific prior synthesis
- 根据 family，调用对应 head 生成一版 family-specific 先验：
  - bottleneck 类：找历史瓶颈、机会及候选链路
  - direction 类：找正在分化/抬升的后继方向
  - planning 类：找可排序的 agenda / dependency 轴
- 这一层提供的是“任务框架上的先验草图”，不是最终答案。

### Step 7. Historical signal abstraction
- `SignalAbstractorV6` 会把检索到的原始证据进一步压缩为结构化信号图。
- 当前抽象结果包含：
  - `observations`
  - `tradeoffs`
  - `recurring_bottlenecks`
  - `inflection_points`
  - `emerging_directions`
  - `agenda_axes`
  - `successor_topic_candidates`
- 这一层的目标是把“论文证据”转成“研究轨迹信号”。

### Step 8. Mechanism-level reasoning
- `MechanismReasonerV6` 基于前面的 signal abstraction + family head prior，生成家族化推理结果。
- 当前版本的关键点：
  - **bottleneck family** 更强调历史瓶颈 → 后续机会的因果链；
  - **direction / planning family** 更强调显式 evidence anchor 与可追溯表达；
  - 会优先使用 support snapshot 中真实出现过的 paper title / evidence title；
  - 避免把中间过程里的内部打分、split pressure 等“机器痕迹”直接暴露到最终答案里；
  - 对机器式 topic label 做自然语言化处理，减少下划线风格短语。

### Step 9. Final refinement
- `FinalRefinerV6` 负责最后一轮答案整理。
- 当前目标包括：
  - 保留 family-specific 推理骨架；
  - 把答案写成 benchmark 期望的自然语言风格；
  - 强化 evidence traceability；
  - 去掉内部变量感、评分感、节点感的表达。

---

## 3. 当前 accepted 版本相对旧版的关键改动
当前接受的是 **pilot12c → full168** 这条线，核心不是重写，而是做小步优化：

1. **保留 ResearchArc 原本“signal abstraction → mechanism reasoning → refinement”的主干**；
2. **对 bottleneck / direction / planning 做 family-specific 的差异化处理**，而不是统一模板；
3. **加强显式 evidence anchoring**，尤其是 direction/planning；
4. **允许 bottleneck 保留更强的机制化表达**，避免因为过度“规整化”而损失洞察；
5. **自然化 topic label 和中间术语**，减少 benchmark 输出中的机器痕迹。

简化地说：

> 当前版本的优化重点，不是“多检索”，而是“让 family-specific reasoning 和 evidence traceability 更平衡”。

---

## 4. 当前版本的效果取向
accepted v6 optimized 的经验画像是：

- 比旧版 **Evidence-Grounded Factuality** 更稳；
- 比旧版 **Future Alignment** 略强；
- 比旧版 **Evidence Traceability** 略强；
- 在三类 family-specific grounding 上仍保持 **第 1 名**；
- 代价是某些高层规划感/覆盖感指标数值略有下降，但名次未变。

因此我们把它作为当前默认 ResearchArc 版本。

---

## 5. 当前输出与诊断字段
`run_task(...)` 最终会返回：
- `task_parse`
- `policy`
- `retrieval_plan`
- `focus`
- `formatted_evidence`
- `evidence_bundle`
- `head_result`
- `signal_abstraction`
- `mechanism_reasoning`
- `refinement`
- `retrieval_mode`
- `evidence`
- `diagnostics`
- `answer`

其中当前 `retrieval_mode` 为：

`support_packet+paper+structure+section+signal_abstraction+mechanism_reasoning+final_refinement`

这也是当前 ResearchArc 和普通 hybrid RAG 最重要的区别之一。
