# ARIS / CoI 适配详细记录

更新时间：`2026-04-18`  
仓库根目录：`/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight`

本文记录我们把 `ARIS` 与 `CoI-Agent` 适配到 `ResearchForesight` 离线 benchmark 的实际改造点。重点不是泛泛介绍方法，而是说明：

1. 原方法骨架保留了什么
2. 离线 benchmark 适配具体改了什么
3. 哪些改动是 benchmark-facing 的工程增强
4. 当前版本的已知局限是什么

---

## 1. 适配目标

统一目标有四个：

1. 保留原方法的核心推理骨架，而不是把所有方法都改写成同一个 RAG agent。
2. 把在线检索替换成 benchmark 的离线 KB，严格受 `time_cutoff` 约束。
3. 让方法能直接回答 benchmark 的四类任务，而不是只输出论文原生 demo 风格的自由文本。
4. 让最终答案尽量直接 benchmark-facing，减少评测前的二次重写。

统一约束有两个：

1. 不把 future GT 或 hidden eval label 暴露到方法推理阶段。
2. family-specific 的增强可以做，但要明确区分“保留原骨架”和“为 benchmark 补的任务头/约束层”。

---

## 2. 相关代码与脚本位置

### 2.1 ARIS

- 主实现：`src/researchworld/aris_offline.py`
- 运行脚本：`scripts/run_aris_offline.py`

### 2.2 CoI

- 主实现：`src/researchworld/coi_agent_offline.py`
- 离线检索适配：`src/researchworld/coi_offline_retrieval.py`
- 单进程运行脚本：`scripts/run_coi_agent_offline.py`
- 分片 supervisor：`scripts/run_coi_agent_offline_sharded.py`

---

## 3. ARIS 的适配

## 3.1 保留的原始骨架

`ARIS` 当前仍然保留原始的高层流程：

1. `literature survey`
2. `idea generation`
3. `critical review / selection`
4. `final answer rendering`

在本地实现里，对应主干仍然是：

- `run_research_lit(...)`
- `run_idea_creator(...)`
- `run_research_review(...)`
- `render_answer(...)`

也就是说，我们没有把 `ARIS` 改成“先检索、再直接回答”的普通 baseline；核心仍然是先 survey，再多候选 ideation，再 review 选解。

## 3.2 离线化替换

离线化主要发生在证据入口层：

1. 接入 `OfflineKnowledgeBase`
2. 所有 retrieval 都受任务 `time_cutoff` 约束
3. 检索来源统一换成 benchmark 内部资产：
   - `paper retriever`
   - `structure retriever`
   - `section retriever`
   - `pageindex retriever`

当前 `ARIS` 的 `retrieval_mode` 是 `offline_kb_hybrid_aris`，本质上是“保留 ARIS workflow 的离线 hybrid retrieval”。

## 3.3 任务路由与 task decomposition

为了让 `ARIS` 在 benchmark 上不至于一上来就跑偏，我们在正式 survey 之前加了两层适配：

1. `route_task(...)`
2. `decompose_task(...)`

其中 `decompose_task(...)` 会产出：

- `historical_state`
- `central_friction`
- `expected_deliverable`
- `extrapolation_boundary`
- `must_preserve`

这层的作用不是替代 ARIS 思考，而是把 benchmark 的答题 contract 提前显式化。尤其是：

- forecasting：强制收敛到 `trajectory + concrete next direction`
- bottleneck：强调 `one-step downstream opportunity`
- strategic / venue：如果题面给了显式候选方向，必须后续全程保留

## 3.4 family-aware query 构造

`ARIS` 原本非常依赖前面的 literature landscape 质量，所以我们对 query 构造做了较深的 family-aware 增强。

基础输入来自：

- `focus`
- `title`
- `question`
- `task_frame`
- `domain-specific expansions`

再叠加 family-specific query：

- bottleneck：`limitation / failure / unresolved challenge / future work opportunity`
- forecasting：`trend / emerging direction / trajectory / future work`
- venue：`venue / evaluation / benchmark / empirical trend`
- strategic：`open problem / dependency / bottleneck / future work`

这一步的目的很直接：让 `ARIS` survey 到的文献景观更像 benchmark 真正要问的问题，而不是停留在泛主题综述层。

## 3.5 证据组织增强

`ARIS` 现在 gather 的不是单一 paper list，而是一整套 evidence bundle：

- `paper_evidence`
- `structure_evidence`
- `section_evidence`
- `pageindex_evidence`
- `successor_topic_candidates`

此外还会计算：

- `historical_likelihood_signals`
- `signal_digest`
- `evidence_digest`
- `family_packet`

这使得后面的 ideation / review 不只是“看几篇摘要随便猜”，而是能看到：

- recurring bottlenecks
- momentum topics
- dependency axes
- top venue hits
- target venue cue
- candidate successor topics

## 3.6 family packet 作为 benchmark 适配层

`ARIS` 当前最重要的 benchmark 适配层是 `family_packet`。

这层是把 evidence 压成 family-specific scaffold，供 ideation / review / render 复用。当前已经明确支持：

### Bottleneck

会注入：

- `preferred_bottlenecks`
- `preferred_unlocks`
- `canonical_bottleneck_hints`
- `canonical_opportunity_hints`
- `unlock_chains`

核心目标是防止模型把：

- 症状写成 bottleneck
- 多跳远期研究计划写成 immediate opportunity
- artifact 名称直接当作机会本身

### Forecasting

会注入：

- `forecast_guardrails`
- `expected_direction_aliases`
- `trend_transitions`
- `trend_direction_candidates`
- `trajectory_estimate`

核心目标是约束答案靠近“历史信号支持的下一步方向”，避免直接跳到过宽泛或过远期的 umbrella topic。

### Strategic

会注入：

- `trend_transitions`
- `trend_direction_candidates`
- `explicit_direction_candidates`

核心目标是让 strategic comparative 题不再凭空发明方向标签，而是在题面已有候选方向中做排序与依赖分析。

### Venue

会注入：

- `primary_venue_bucket`
- `compatible_venue_buckets`
- `package_expectations`
- `contrastive_not_best_for`

核心目标是让 venue 定位不只是“像某某会议”，而是显式说明：

- primary fit
- secondary compatible families
- reviewer/package expectation
- weaker-fit venues

## 3.7 ideation 阶段的 family-specific 硬约束

`run_idea_creator(...)` 这层的 prompt 现在已经不是通用候选生成，而是按 family 分支强化。

关键增强：

1. 三个候选必须真的不同，不能只是同义改写。
2. 候选必须尽量贴近 historical future-work / topic label，而不是 ad hoc umbrella phrase。
3. Candidate 1 明确要求是“最 evidence-conservative 的近一步”。

额外的 family-specific contract：

### Strategic / Venue

如果题面给了 `candidate_directions`，则：

- 只能在这组方向里排序
- 每个候选都必须覆盖全部候选方向且恰好一次
- 禁止自创第三个 label

这是最近一轮修复的关键。之前 strategic comparative 题的主要坏点之一，就是 render/review 阶段会把题面 label 改坏。

### Forecasting

增加了 `forecast_guardrail`：

- primary next direction 尽量贴近历史支持的 label
- 至少一个候选显式映射历史 transition

### Bottleneck

增加了 `unlock-chain guidance`：

- bottleneck 必须更 upstream
- opportunity 必须是一跳 unlock
- 避免 artifact-like answer

## 3.8 review 阶段的 reranking 增强

`ARIS` 的 review 本来就是关键步骤，我们在这层没有拿掉 review，而是把 reranker 改得更 benchmark-aware。

当前 `_candidate_quality_score(...)` 会综合：

- `evidence_bonus`
- `specificity_bonus`
- `family_fit_bonus`
- `signal_anchor_bonus`
- `packet_bonus`
- `scientific_taste_bonus`
- `expected_alignment_bonus`

同时加入惩罚项：

- `generic_penalty`
- `artifact_reuse_penalty`
- `expected_divergence_penalty`
- `over_specialization_penalty`

这一步是 `ARIS` 当前能在 strategic 上明显改善的核心原因之一。它不再只看“语言上像不像一个好答案”，而是显式看：

- 与 family packet 是否对齐
- 与趋势候选是否贴近
- 是否过泛
- 是否用了 artifact / benchmark 这种偷懒标签
- 是否满足 explicit ranking contract

## 3.9 render 阶段的 benchmark-facing 输出

`ARIS` 没有走一个独立的通用 answer adapter，而是把最终 `render_answer(...)` 直接做成 benchmark-facing prose。

当前 family-specific 的 render 行为：

- bottleneck：必须显式给出 `bottleneck + downstream opportunity + linkage`
- forecasting：必须显式给出 `trajectory + next direction + why-next`
- strategic：输出排序项，并带 `first milestone / why now / dependency / risk`
- venue：输出排序项，并带 `technical rationale / venue-fit rationale / reviewer package / secondary families`

另外，对 strategic / venue 还额外加了 `_enforce_explicit_direction_contract(...)`，如果 render 把题面候选方向改坏，会强制拉回。

## 3.10 结论：ARIS 当前属于哪种适配

`ARIS` 目前是：

- 方法骨架保留：强
- 离线化替换：强
- family-specific benchmark 适配：强
- benchmark-facing 输出投影：强

换句话说，它仍然是 `survey -> ideation -> review` 的 ARIS，但 family packet、reranker、render contract 这些层已经明显是 benchmark 专门增强。

---

## 4. CoI 的适配

## 4.1 保留的原始骨架

`CoI` 当前保留的核心主线仍然是：

1. 找 anchor papers
2. 沿 anchor 建 paper chain / idea chain
3. 从 chain 中抽 idea / experiment / entity / reference
4. 基于 chain 提炼 trend 与 future
5. 再投影成 benchmark 答案

也就是说，方法核心仍然是“`chain-of-ideas backbone`”，不是普通的 top-k RAG。

## 4.2 离线化替换

离线化发生在两层：

1. `OfflineKnowledgeBase`
2. `CoIOfflineRetrievalAdaptor`

其中 `CoIOfflineRetrievalAdaptor` 做了三件事：

1. 从 `support_packets` 读取 domain packet
2. 构建 task-specific candidate pool
3. 负责 family-aware query 和 packet / paper 侧的离线检索融合

`CoI` 现在所有 anchor 与 transition 检索都走本地 KB，并受 `time_cutoff` 控制。

## 4.3 candidate pool 注入

这是 `CoI` 适配里非常关键的一层。

`build_candidate_pool(...)` 会把 benchmark 自带的 packet 信息注入给 CoI backbone，包括：

- packet 本身
- representative papers
- emergent descendants
- top limitations
- historical future work

然后再在 paper 侧打出：

- `packet_match`
- `focus_score`
- `family_score`
- `query_hit_count`

作用是让 `CoI` 不只是盲目沿 paper citation / title 相似性扩展，而是更贴着 benchmark 当前这道题的 topic packet。

## 4.4 family-aware query 与 anchor 选择

`CoI` 的 query 不是只从题面直接抽，而是做了 family-aware 扩展：

- bottleneck：偏 limitation / failure / challenge / open problem
- forecasting：偏 trend / trajectory / emerging direction
- strategic：偏 open problem / trade-off / priority

随后 anchor 选择会综合：

- query 命中
- packet 命中
- pool bonus
- paper 检索分数

最终只保留 top anchors。

当前 `max_anchor_papers = 4`，并且还可以通过环境变量控制内部 anchor 并行。

## 4.5 chain building 的 benchmark 改造

`build_faithful_chain(...)` 是 CoI 当前最核心的适配点。

它的流程是：

1. 对 anchor 先抽 `paper profile`
2. 向前做 `forward` 扩展
3. 再向后做 `backward` 扩展
4. 组装出 `idea_chain_text`
5. 补 paper/fulltext evidence

每个 profile 当前抽取的是：

- `idea`
- `experiment`
- `entities`
- `references`

并缓存到磁盘与内存。

## 4.6 transition candidate 的预筛选与批判定

这是 `CoI` 近期最关键的加速与稳定性改造之一。

之前慢的一个重要原因是 transition 选择太“硬 LLM 化”了。现在我们先做轻量预筛选，再做小批量判定。

当前 `_prefilter_transition_candidates(...)` 会组合：

- embedding similarity
- keyword overlap
- retrieval combined score
- packet score
- query score
- temporal score

做一个加权 `prefilter_score`，然后只保留短名单。

之后 `_judge_transition_batch(...)` 再对 shortlist 做一次 JSON 判定：

- 只选 single best adjacent paper
- 如果没有真相邻的，也允许给 `backup_paper_id`

这一步的实际意义：

1. 大幅减少 LLM 直接判断的大候选池规模
2. 把 CoI 的“链条连续性”判断保留给 LLM，而不是彻底规则化
3. 避免每一步 transition 都在几十篇候选上做重判

## 4.7 paper profile / content / entity 的缓存

`CoI` 比 `ARIS` 慢很多，一个核心原因就是它会反复看论文内容并抽 profile。

当前已经加了多层缓存：

- `paper_content` cache
- `paper_profile` cache
- `entity_summary` cache
- `anchor_branch` cache
- embedding text cache

这意味着：

1. 同一 paper 不会重复抽 profile
2. 同一 anchor branch 失败重启后可以直接复用
3. 重跑 / resume 时不会完全从零开始

## 4.8 anchor 级并行与分片运行

`CoI` 现在的并行有两层：

### 进程外并行

`run_coi_agent_offline_sharded.py` 会：

- 把任务切成多个 shard
- 每个 shard 单独启动 `run_coi_agent_offline.py`
- 自动监控、重启、合并 `results_merged.jsonl`

### 进程内并行

`run_task_faithful(...)` 内部会对多个 anchor branch 做 `ThreadPoolExecutor` 并行。

也就是说，当前的 CoI 不是单纯“每题单线程死跑”，而是：

- shard 级并发
- anchor 级并发
- 缓存复用

不过即便如此，它依然明显比 ARIS 慢，因为每个 anchor branch 里本身还包含：

- profile extraction
- forward/backward transition judgment
- trend generation
- future generation
- benchmark projection

## 4.9 family-specific answer head 与 schema projection

`CoI` 当前并不是所有 family 都在 backbone 末端直接输出最终答案，而是增加了 benchmark-facing 的 family head / adapter。

主流程里会先跑多个 candidate chains，再按 family 进入：

- `direction_forecasting` -> `_run_direction_head(...)`
- `strategic_research_planning` -> `_run_planning_head(...)`
- 其它 family -> `_run_bottleneck_head(...)`

这点要特别注意：

1. `CoI` 当前对 `direction` 和 `strategic` 有显式 family head
2. `venue` 目前并没有一套像 `ARIS` 那样独立、对称的 venue-specific head
3. `venue` 当前更接近落在“其它 family”的通用投影路径上，不算完整适配

这也是 `CoI` 当前在 venue 指标上长期偏弱的重要原因之一。

## 4.10 统一结构化投影层

`CoI` 在 chain/trend/future 之后，还会过一层 `_project_structured_answer(...)`。

这层会根据 family 生成不同 schema：

- strategic：`ranked_directions`
- forecasting：`trajectory_label + next_directions`
- 其它：`bottleneck + opportunity + linkage`

然后再经过一层 critique / repair，最后由 `_verbalize_structured_answer(...)` 变成自然语言答案。

这层是典型的 benchmark 适配层，不是 CoI 论文原生 backbone 的一部分。

## 4.11 CoI 当前慢在哪里

当前 `CoI` 的主要耗时大头仍然是：

1. `extract_paper_profile(...)`
2. transition batch judgment
3. trend generation
4. future generation
5. family projection / critique

如果展开看，单个 anchor branch 实际上是：

1. `build_chain`
2. `judge_forward / judge_backward`
3. `extract_profile`
4. `chain_built`
5. `trend`
6. `future`
7. `structured projection`

所以它慢并不只是“脚本没并发”，而是方法本身就比 ARIS 更重，而且我们又把它变成了 benchmark-aware 的多阶段版本。

## 4.12 结论：CoI 当前属于哪种适配

`CoI` 当前是：

- 方法骨架保留：强
- 离线化替换：强
- retrieval / candidate pool benchmark 适配：强
- family-specific head：中
- 四个 family 的对称性：弱于 ARIS

换句话说，`CoI` 的 offline backbone 很完整，但它目前还不是四类任务都完全 benchmark-native 的版本。

---

## 5. ARIS 与 CoI 的适配差异

可以把两者理解成两种不同的适配策略。

### ARIS 更像

- 保留 survey -> ideation -> review
- 把 family contract 早早打进 task decomposition / family packet / reranker / renderer
- 强调最终答案的 benchmark-facing 可控性

因此它更适合：

- strategic comparative
- venue ranking / positioning
- 需要明确排序和理由结构的任务

### CoI 更像

- 保留 paper-chain / idea-chain backbone
- 先保证 chain 连贯性，再从 chain 提炼 trend / future
- benchmark 适配主要发生在 retrieval adaptor 和末端 schema projection

因此它更适合：

- traceability 强的任务
- 需要沿技术演化链做解释的任务

但它现在的问题也更明显：

- 成本高
- 慢
- family head 不够对称
- venue 适配明显不如 ARIS 完整

---

## 6. 哪些属于“方法保真”，哪些属于“benchmark 特化”

### 方法保真的部分

#### ARIS

- survey -> ideation -> review 主干
- 候选生成后再 review 的工作方式

#### CoI

- anchor paper -> chain expansion -> trend/future 提炼 主干
- 通过 chain 而不是纯 top-k evidence 汇总来形成判断

### benchmark 特化的部分

#### ARIS

- task decomposition
- family packet
- explicit ranking contract enforcement
- historical likelihood reranker
- benchmark-facing renderer

#### CoI

- candidate pool / packet 注入
- family-aware query expansion
- transition prefilter + shortlist judge
- structured schema projection
- family-specific benchmark answer head

---

## 7. 当前已知边界与风险

## 7.1 ARIS

1. 适配已经比较深，虽然主骨架还在，但 family packet / reranker / render contract 明显是 benchmark 特化层。
2. 这类适配在 benchmark 上通常有效，但也意味着和原论文的“原教旨实现”已经不是一回事。
3. 它对 evidence 组织质量依赖很强，如果某个 domain 的 structure / section / pageindex 不够全，review 阶段还是可能过泛。

## 7.2 CoI

1. 现在不是四个 family 都做了对称适配，尤其 venue 明显不完整。
2. 运行代价很高，full run 成本和耗时都重。
3. cache / shard / anchor parallel 已经加了，但方法本身依然偏重。
4. 当前虽然加入了 embedding + keyword prefilter，但链条质量仍然高度依赖 transition judgment 的稳定性。

---

## 8. 后续建议

### ARIS

1. 继续沿 family packet 路线做细化，而不是推倒重写。
2. 优先补 bottleneck / forecasting 上的 family-specific signal，让其像 current strategic 一样更稳。
3. 在维持方法骨架不变的前提下，继续强化 renderer 前的 deterministic guardrail。

### CoI

1. 如果继续做 benchmark 主力方法，应该补齐 venue-specific head，而不是继续让 venue 走通用路径。
2. 如果继续优化速度，优先盯：
   - profile extraction 次数
   - transition shortlist 大小
   - anchor 数量
   - projection / critique 次数
3. 如果只想保留 CoI 的特色优势，则更适合把它定位成“高 traceability、强链式解释”的方法，而不是全面型最快 baseline。

---

## 9. 一句话总结

- `ARIS`：我们保留了 `survey -> ideation -> review` 的方法主干，并把 query、family packet、review reranker、render contract 全部做成 benchmark-facing。
- `CoI`：我们保留了 `paper-chain / idea-chain` 主干，补了离线 retrieval adaptor、candidate pool、transition prefilter、cache 与 family projection，但目前 family 适配仍不如 ARIS 对称，尤其 venue 还不完整。
