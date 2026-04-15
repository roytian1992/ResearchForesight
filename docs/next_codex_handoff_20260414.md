# Next Codex Handoff — 2026-04-14

本文给下一个 Codex session 使用，重点记录：
1. 当前 benchmark / release / KB 的稳定状态
2. 已完成的主要实验与结果
3. ARIS / ResearchAgent 当前问题
4. 下一步建议的改良方向
5. 进入工作前应先看哪些文件

---

## 0. 一句话结论

当前已经稳定的部分是：
- **benchmark 数据与 release** 已经整理并 push
- **官方离线 KB** 已经放到 GitHub
- **168 题主评测结果** 已经定稿，当前最强方法是 **ResearchArc**
- **CoI** 是强基线，尤其 `Evidence Traceability` 最强
- **ARIS-Offline v7** 和 **ResearchAgent-Offline** 仍明显偏弱，尤其在：
  - `Evidence-Grounded Factuality`
  - `Future Alignment`

下一阶段的主要工作，不是再折腾 benchmark 构建，而是：
- **继续增强 ARIS / ResearchAgent**
- 目标是至少不要差于 `Native LLM` / `Hybrid RAG`
- 但增强时**不能改掉方法本身的核心骨架**

---

## 1. 仓库与发布状态

仓库：
- `~/ResearchForesight`
- remote: `git@github.com:roytian1992/ResearchForesight.git`

本次已 push 的清理提交：
- commit: `56d35ab951ba5a2f2237c5902aa2345e0b557099`

### 1.1 当前命名后的 release

位于：`data/releases/`

- `benchmark_halfyear`：437 tasks
- `benchmark_quarter`：96 tasks
- `benchmark_full`：533 tasks

说明：
- 之前带数字的旧名字（如 `benchmark_v1_halfyear_440` / `benchmark_v1_mixed_571`）已经改掉
- `mixed` 已改为 `full`

### 1.2 官方 offline bundle

位于：
- `benchmark_release/benchmark_v3_20260407_venue/`

重要内容：
- `tasks.jsonl`
- `manifest.json`
- `kb/`

这个 `kb/` 就是当前 offline methods 默认使用的离线知识库。

### 1.3 KB 现状

已确认：
- `ARIS-Offline` 与 `ResearchAgent-Offline` 默认都读 `release_dir / kb`
- 代码位置：
  - `scripts/run_aris_offline.py`
  - `scripts/run_researchagent_offline.py`

已确认现在 **官方 offline KB 已经在 git / GitHub 上**。

注意：
- 这次只把 **官方 bundle 的 KB** 放上去了
- 没有把旧 `data/releases/benchmark_halfyear/kb` 与 `future_kb` 一并公开提交，避免仓库继续膨胀



## 1.4 重要路径与工作目录

### 当前实际工作根目录
- 用户工作根目录：`/vepfs-mlp2/c20250513/241404044/users/roytian`
- 当前 repo：`/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchForesight`
- 论文目录：`/vepfs-mlp2/c20250513/241404044/users/roytian/papers/ResearchBenchmark`

说明：
- 下一个 session 默认应在：`/vepfs-mlp2/c20250513/241404044/users/roytian`
- 不要把东西乱装到 `/` 或系统级目录
- 用户之前明确要求：尽量都在真实工作目录下操作

### 关键代码路径
- ResearchArc：`src/researchworld/research_arc_v6.py`
- ARIS-Offline：`src/researchworld/aris_offline.py`
- ResearchAgent-Offline：`src/researchworld/researchagent_offline.py`
- Offline eval 主逻辑：`src/researchworld/eval_v3_1.py`

### 关键 runner 路径
- ARIS runner：`scripts/run_aris_offline.py`
- ResearchAgent runner：`scripts/run_researchagent_offline.py`
- release 组装脚本：`scripts/assemble_incremental_releases.py`

### Benchmark / release 路径
- 根 README：`README.md`
- 官方 offline bundle：`benchmark_release/benchmark_v3_20260407_venue/`
- 官方 offline KB：`benchmark_release/benchmark_v3_20260407_venue/kb/`
- 命名后的 releases：
  - `data/releases/benchmark_halfyear/`
  - `data/releases/benchmark_quarter/`
  - `data/releases/benchmark_full/`

### 结果与评测路径
- 总结果表：`results/final_metrics/final_metric_results_summary.md`
- ResearchArc 168 结果：`results/research_arc_v6_opt_full168/`
- ARIS 168 结果：`results/aris_offline_168_v7_qwen/`
- ResearchAgent 168 结果：`results/researchagent_offline_168_qwen/`
- 标准聚合结果：`tmp/final_metric_bundle_v1/`

### 关键文档路径
- ResearchArc 机制：`docs/research_arc_current_mechanism.md`
- CoI / ARIS 适配：`docs/coi_aris_adaptation_notes.md`
- ARIS / ResearchAgent 增强计划：`docs/aris_researchagent_modular_enhancement_plan.md`
- ARIS 验证重规划：`docs/aris_validation_replan.md`
- scientific taste 启发：`docs/scientific_taste_inspiration_and_reranking_plan.md`
- 本交接文档：`docs/next_codex_handoff_20260414.md`

### 常用 pilot 划分文件
- `tmp/pilot24_balanced_core_ids.txt`
- `tmp/pilot36_balanced_core_ids.txt`

### Git / 发布相关
- GitHub repo：`https://github.com/roytian1992/ResearchForesight`
- 当前已 push 的 cleanup commit：`56d35ab951ba5a2f2237c5902aa2345e0b557099`

### 操作注意事项
- 用户明确要求：**用 `tmux`，不要用 `nohup`**
- 用户明确要求：**后续每一个 method 的问答生成阶段，至少开 `4 workers` 做题目级并发；不要再单进程顺序跑完整套 100 题**
- 尽量不要做 destructive git 操作
- 这个 repo 目前本地还有很多未提交实验结果与临时文件，开始前先 `git status` 看清楚
- `CoI-Agent-Offline` 当前代码支持外部依赖 fallback：
  - 优先找 `ResearchForesight/external/CoI-Agent`
  - 若不存在，会自动回退到 `/vepfs-mlp2/c20250513/241404044/users/roytian/ResearchTrajectoryLab/external/CoI-Agent`
  - 也可用环境变量 `RESEARCHFORESIGHT_COI_PATH` 显式覆盖

---

## 2. Benchmark 当前稳定设定

### 2.1 当前主评测集

主结果表使用的是：
- **168 题标准评测集**

结果汇总文件：
- `results/final_metrics/final_metric_results_summary.md`

补充更新（2026-04-16）：
- `data/releases/benchmark_core100/` 已不再是旧 100 题
- 现已直接覆盖为从 `data/releases/benchmark_full_curated_polished/` 严格筛出的 **251 题** 子集
- 当前筛选规则：
  - `bottleneck_opportunity_discovery` 要求 `ground_truth.future_descendants` 非空
  - `direction_forecasting` 要求 `ground_truth.emergent_descendants` 非空
  - `strategic_research_planning` 要求 `ground_truth.direction_records` 非空
  - `venue_aware_research_positioning` 要求 `ground_truth.direction_records` 非空
- 当前 `benchmark_core100` family 分布：
  - `bottleneck_opportunity_discovery`: 56
  - `direction_forecasting`: 67
  - `strategic_research_planning`: 76
  - `venue_aware_research_positioning`: 52
- 251 个 task_id 已写入：
  - `data/releases/benchmark_core100/task_ids.txt`
  - `tmp/benchmark_core100_strict251_task_ids.txt`
- `tasks_hidden_eval_v3.jsonl` 和 `tasks_hidden_eval_v3_1.jsonl` 也已同步重建

补充更新（2026-04-16，数据回捞）：
- 已新建扩容版 curated release：
  - `data/releases/benchmark_full_curated_recovered21/`
- 它是在 `benchmark_full_curated_polished` 基础上，回捞了 `21` 个原先被一刀切删除、但实际上仍保留 `candidate_directions` / `direction_records` 的 `q1_agenda_priority_selection`
- 扩容后：
  - curated task 总数：`443`
  - 剩余 dropped：`90`
- 对应 strict 规则下，可保留任务数从 `251` 增长到 `272`
  - task id 文件：`tmp/benchmark_full_curated_recovered21_strict272_task_ids.txt`
  - strict family 分布：
    - `bottleneck_opportunity_discovery`: 56
    - `direction_forecasting`: 67
    - `strategic_research_planning`: 97
    - `venue_aware_research_positioning`: 52
  - strict horizon 分布：
    - `half_year`: 239
    - `quarter`: 33
- 新 release 也已经重建：
  - `tasks_hidden_eval_v3.jsonl`
  - `tasks_hidden_eval_v3_1.jsonl`
- 仍未回捞的 `90` 个 dropped tasks 里，还有一小批 gold answer 具备明显“已排序方向”结构，可作为下一波 heuristic recovery 候选：
  - `tmp/dropped_agenda_gold_rankable35_task_ids.txt`

### 2.2 当前使用的指标

当前只保留：

#### Primary
- `Evidence-Grounded Factuality`
- `Future Alignment`
- `Evidence Traceability`

#### Family-specific
- `Opportunity Grounding`
- `Forecast Grounding`
- `Technical Dependency Grounding`
- `Venue Positioning Grounding`

说明：
- 官方评测口径固定为 `3 primary + 4 family-specific`
- `Task Fulfillment`、`Strategic Intelligence`、`Research Value`、`Uncertainty Calibration`、`Temporal Leakage` 都不再保留在当前正式评测代码里
- 2026-04-15 起，相关旧测评代码已从当前主代码路径移除，不要再恢复或继续汇报这些诊断类指标
- 运行策略也已更新：后续 core100 实验中，method answer generation 默认按 `>=4 workers` 题目级并发执行；评测并发仍可独立配置
- 用户最在意的是：
  - `Evidence-Grounded Factuality`
  - `Future Alignment`

### 2.3 Full release venue 扩充进展（2026-04-16）

当前用于 full release venue 扩充分析的基线 release：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75/`

对应 audit 脚本与报告：
- 脚本：`scripts/audit_full_release_venue_coverage.py`
- 报告目录：`tmp/venue_coverage_audit/benchmark_full_curated_recovered21_bottleneck18_expanded75/`

基线 release 的 venue 现状：
- 总题数：`518`
- strict：`365`
- venue public：`100`
- venue strict：`52`
- venue strict quarterly：`0`

关键结论：
- 之前 venue 缺口不只是“题数少”，而是 **strict 可用来源的 bucket 很偏**
- 直接看 top-1 bucket 会误判“没法补”；实际不少 `strategic_research_planning` strict 题在 `target_window_stats.top_venue_buckets` 里已经带了次强 bucket 支撑
- 可直接从现有 strict planning 题派生补洞的 bucket 主要包括：
  - `llm_agent`: `ACL`, `ICML`, `ICLR`, `NeurIPS`
  - `llm_finetuning_post_training`: `ACL`
  - `rag_and_retrieval_structuring`: `ICML`, `KDD`
  - `visual_generative_modeling_and_diffusion`: `ICML`
- 当前数据里**确实没有可直接 strict 回捞**的 bucket 主要包括：
  - `rag`: `ACL`, `SIGIR`, `WSDM`, `WWW`
  - `visual`: `CVPR`, `ECCV`, `ICCV`, `NeurIPS`
  - `llm_finetuning_post_training`: `ICLR`, `ICML`, `NeurIPS`
  - `llm_agent`: `IJCAI`

已新增 targeted venue augmentation 脚本：
- `scripts/augment_release_with_targeted_venue_coverage.py`

该脚本策略：
- 先补 `venue strict = 0`、但在 strict planning 源里已有 bucket 支撑的 domain × bucket 组合
- 再尽量给每个 domain 补 `quarter` venue（如果存在可用 strict 源）
- bucket 支撑定义为：`ground_truth.target_window_stats.top_venue_buckets` 中该 bucket 计数 `> 0`
- 不再局限于只取 dominant/top-1 bucket

按默认参数生成的新候选 release：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover/`

当前结果：
- 总题数：`528`
- strict：`375`
- venue public：`110`
- venue strict：`62`
- venue strict quarterly：`3`

新增补齐的 bucket：
- `llm_agent`: `ACL`, `ICML`, `ICLR`, `NeurIPS`
- `llm_finetuning_post_training`: `ACL`
- `rag_and_retrieval_structuring`: `ICML`, `KDD`
- `visual_generative_modeling_and_diffusion`: `ICML`
- 另外补了 `quarter` venue：
  - `llm_agent / AAAI`
  - `llm_finetuning_post_training / AAAI`
  - `rag_and_retrieval_structuring / ICML`

新 release 的 audit 报告：
- `tmp/venue_coverage_audit/benchmark_full_curated_recovered21_bottleneck18_expanded75_venuecover/`

仍然补不起来的 bucket（当前数据确实无 strict 支撑）：
- `llm_agent`: `IJCAI`
- `llm_finetuning_post_training`: `ICLR`, `ICML`, `NeurIPS`
- `rag_and_retrieval_structuring`: `ACL`, `SIGIR`, `WSDM`, `WWW`
- `visual_generative_modeling_and_diffusion`: `CVPR`, `ECCV`, `ICCV`, `NeurIPS`

所以下一步如果还要继续把 full release 的 venue 做全，重点不是再从现有 release 里“抠”，而是：
- 新扩一批真正带这些 bucket 支撑的 planning/venue 候选
- 特别是 `rag -> SIGIR/WWW/WSDM`、`visual -> CVPR/ICCV/ECCV`

---

## 3. 当前 168 题主结果（最重要）

来源：
- `results/final_metrics/final_metric_results_summary.md`

### 3.1 Primary 指标

| Method | Evidence-Grounded Factuality | Future Alignment | Evidence Traceability |
|---|---:|---:|---:|
| ARIS-Offline v7 | 0.4388 | 0.3269 | 0.5635 |
| CoI | 0.5277 | 0.4567 | **0.7180** |
| Hybrid RAG | 0.5209 | 0.3960 | 0.5191 |
| Native LLM | 0.4160 | 0.4411 | 0.0555 |
| ResearchAgent-Offline | 0.4169 | 0.3134 | 0.4461 |
| ResearchArc | **0.5423** | **0.4592** | 0.6275 |

### 3.2 结论

- **ResearchArc** 是当前总体最好方法
- **CoI** 是强 baseline，尤其 `Evidence Traceability` 最强
- **Hybrid RAG** 比 Native 更稳，但 `Future Alignment` 不强
- **ARIS-Offline v7**：traceability 尚可，但事实性与 future alignment 偏弱
- **ResearchAgent-Offline**：目前整体最弱之一，尤其 primary 两项不够用

### 3.3 Family-specific 排名结论

- `Opportunity Grounding`：ResearchArc 第 1
- `Forecast Grounding`：ResearchArc 第 1
- `Technical Dependency Grounding`：ResearchArc 第 1

这说明：
- ResearchArc 的优势主要来自 **family-specific reasoning scaffold + signal abstraction + mechanism reasoning**
- 但 ARIS / ResearchAgent 现在还没有把“forward judgment”真正做出来

---

## 4. 当前已经相对稳定的方法机制

### 4.1 ResearchArc

机制说明文件：
- `docs/research_arc_current_mechanism.md`

当前 accepted 版本：
- `src/researchworld/research_arc_v6.py`
- 对应结果目录：`results/research_arc_v6_opt_full168/`

关键点：
- 不是普通 RAG
- 主干是：
  1. task parsing
  2. policy routing
  3. evidence backbone
  4. focus resolution
  5. evidence formatting
  6. family head prior
  7. signal abstraction
  8. mechanism reasoning
  9. final refinement

核心经验：
- 先把历史证据整理成 **historical signal map**，再做判断
- 这比“检索完直接写答案”更适合 benchmark

### 4.2 CoI

适配说明：
- `docs/coi_aris_adaptation_notes.md`

状态：
- 已经是比较深的 benchmark-oriented adaptation
- backbone 仍是 `Chain-of-Ideas`
- 强项在：
  - 链式证据组织
  - 趋势/方向/瓶颈链路抽取
  - traceability

### 4.3 ARIS

适配说明：
- `docs/coi_aris_adaptation_notes.md`
- `src/researchworld/aris_offline.py`
- runner: `scripts/run_aris_offline.py`

当前保留的核心骨架：
1. survey
2. ideation
3. review
4. render

当前问题：
- survey / ideation / review 流程保住了
- 但 **final selected answer 经常不够 sharp**
- 特别是 `direction_forecasting`，容易给出：
  - 太泛
  - 太保守
  - 太像 survey 总结，而不是 forward call

### 4.4 ResearchAgent

当前实现：
- `src/researchworld/researchagent_offline.py`
- runner: `scripts/run_researchagent_offline.py`

原始定位：
- 文献检索 / 阅读 / 综合型 research assistant

当前问题：
- 太容易进入 **survey mode**
- 输出看起来像 literature summary，而不是明确的 future judgment
- family-specific task contract 对齐不够
- primary 两项表现不够

---

## 5. ARIS / ResearchAgent 已经总结出的主要问题

这是下一个 session 最应该接着做的部分。

### 5.1 ARIS 当前问题

1. **candidate generation 不是主要瓶颈，candidate selection 才是**
   - 它往往能生成像样候选
   - 但 review 阶段常常选错

2. **太保守 / 太泛化**
   - 容易选“看起来安全”的方向
   - 但 benchmark 更看重是否对 future cluster 做了对的判断

3. **forecasting 最弱**
   - 没有把 “current signals -> inflection -> successor topic” 这条链做硬

4. **Evidence Traceability 有时上升，但会以牺牲 factuality / future alignment 为代价**
   - 之前已经出现过这种 regression
   - 所以不能只追 citation 或 evidence density

### 5.2 ResearchAgent 当前问题

1. **读得很多，但不 commit**
   - 输出像综述
   - 缺 forward call

2. **缺 family-specific synthesis scaffold**
   - bottleneck / forecasting / planning 的目标结构不同
   - 现在经常被统一成 generic literature synthesis

3. **缺 candidate-level judgment / reranking**
   - 直接 single-pass synthesis 不够
   - 应该至少生成多版 candidate 再选

4. **Evidence filtering 仍不够强**
   - 容易 citation stuffing
   - 也容易把与推理无关但主题相关的论文塞进答案

---

## 6. 已形成的改良方向（不要推翻方法主干）

关键原则：
- **不能把 ARIS / ResearchAgent 改写成新的 ResearchArc**
- 只能增加模块
- 不改变方法骨架

### 6.1 ARIS 推荐增强方向

参考文件：
- `docs/aris_researchagent_modular_enhancement_plan.md`
- `docs/scientific_taste_inspiration_and_reranking_plan.md`
- `docs/aris_validation_replan.md`

推荐的最小增强包：
1. `task-family router`
2. `offline retrieval adapter`
3. `family-specific ideation scaffold`
4. `review-stage reranker`
5. `benchmark-facing final composer`

更具体地说：
- **保留** ARIS 的 `survey -> ideation -> review -> render`
- 重点加强的是：
  - review 阶段的 reranking
  - 对 generic / conservative candidate 的惩罚
  - 对具体 successor cluster、expected deliverable、community uptake plausibility 的偏好

尤其 forecasting：
- 应强化：
  - `future-cluster formation potential`
  - `expected-deliverable alignment`
  - `community uptake plausibility`
- 应惩罚：
  - `artifact-title reuse`
  - `over-specialization`
  - `generic safe answer`

### 6.2 ResearchAgent 推荐增强方向

参考文件：
- `docs/aris_researchagent_modular_enhancement_plan.md`
- `docs/scientific_taste_inspiration_and_reranking_plan.md`

推荐的最小增强包：
1. `task decomposition front-end`
2. `family-aware retrieval expansion`
3. `evidence filtering / citation pruning`
4. `family-specific reasoning scaffold`
5. `candidate answer reranker / self-critique`

最关键的是：
- 不要 single-pass 直接写答案
- 应先压成 **decision packet**，再生成多个 candidate，再做 judgement

建议 ResearchAgent 至少输出：
- 1 个保守 candidate
- 1 个高杠杆 candidate
- 1 个最 grounded candidate

然后再通过 judge 选择。

### 6.3 来自 “AI Can Learn Scientific Taste” 的启发

文件：
- `docs/scientific_taste_inspiration_and_reranking_plan.md`

最重要 insight：
- 当前 benchmark 测的，本质上是 **scientific judgment / taste under cutoff constraints**
- 很多时候问题不是 generate 不出来，而是 **选错了 final answer**
- 所以下一阶段应强调：
  - taste-aware reranking
  - multi-candidate judgement
  - anti-survey penalty

---

## 7. 实验流程上的经验与坑

### 7.1 不要再用超小样本直接决定是否全量

之前已经发现：
- 极小样本（例如很窄的 forecasting-only smoke）容易误导
- 小样本升了，全量可能掉
- 小样本掉了，也不一定代表全量必掉

因此后续验证建议：
- 先用 **balanced pilot**
- 不要只测某一个 family 的几个题

推荐验证集：
- `tmp/pilot24_balanced_core_ids.txt`
- `tmp/pilot36_balanced_core_ids.txt`

说明文件：
- `docs/aris_validation_replan.md`

### 7.2 Full run 之前的 gate

至少满足：
- target metric 在 balanced split 上提升
- `Evidence-Grounded Factuality` 和 `Future Alignment` 不能明显回退
- 不能出现某个 family grounding 明显崩掉

### 7.3 用户明确不想要的事情

- 不要用 `nohup`
- 用 `tmux`
- 不要乱改稳定 CUDA 环境
- 一切尽量在真实工作目录下进行
- 尽量不要做 destructive git 操作

---

## 8. 下一个 session 建议的实际工作顺序

### Option A：优先救 ARIS

建议顺序：
1. 先读：
   - `docs/coi_aris_adaptation_notes.md`
   - `docs/aris_researchagent_modular_enhancement_plan.md`
   - `docs/scientific_taste_inspiration_and_reranking_plan.md`
   - `docs/aris_validation_replan.md`
2. 检查 `src/researchworld/aris_offline.py`
3. 在 ARIS 中加入更强的：
   - family-aware reranker
   - anti-generic penalty
   - multi-candidate selection
4. 先跑 `pilot24` 或 `pilot36`
5. 如果 primary 两项确实改善，再上 168

### Option B：优先救 ResearchAgent

建议顺序：
1. 先读：
   - `docs/aris_researchagent_modular_enhancement_plan.md`
   - `docs/scientific_taste_inspiration_and_reranking_plan.md`
2. 检查 `src/researchworld/researchagent_offline.py`
3. 加入：
   - decision packet
   - 2~3 个 candidate generation
   - judgement / reranking
   - anti-survey penalty
4. 先跑 balanced pilot
5. 通过后再上 168

如果只能二选一，我建议：
- **先做 ARIS**，因为 ARIS 当前已经有一定 traceability 基础，离提升到“至少不输 RAG”更近

---

## 9. 当前最该看的文件

### 结果与总表
- `results/final_metrics/final_metric_results_summary.md`

### 方法说明
- `docs/research_arc_current_mechanism.md`
- `docs/coi_aris_adaptation_notes.md`
- `docs/aris_researchagent_modular_enhancement_plan.md`
- `docs/scientific_taste_inspiration_and_reranking_plan.md`
- `docs/aris_validation_replan.md`

### 关键实现
- `src/researchworld/research_arc_v6.py`
- `src/researchworld/aris_offline.py`
- `src/researchworld/researchagent_offline.py`

### runners
- `scripts/run_aris_offline.py`
- `scripts/run_researchagent_offline.py`

### benchmark / release
- `README.md`
- `benchmark_release/benchmark_v3_20260407_venue/README.md`
- `benchmark_release/benchmark_v3_20260407_venue/kb/`

---

## 10. 最后一句提醒

后续工作的评价标准很简单：
- 不看花哨 workflow
- 不看是否更像“科研 agent”
- 主要看三件事：
  1. `Evidence-Grounded Factuality` 有没有升
  2. `Future Alignment` 有没有升
  3. 是否至少不比 `Native LLM` / `Hybrid RAG` 差

如果一个改动只提升了 `Evidence Traceability`，但 primary 前两项掉了，原则上不接受。

---

## 11. 2026-04-16 数据集扩充进展

### 当前新的 full release 扩充版

已新增一个可直接用于后续实验的 release：
- `data/releases/benchmark_full_curated_recovered21_bottleneck18_expanded75`

来源基线：
- base release 是 `data/releases/benchmark_full_curated_recovered21_bottleneck18`

当前结果：
- total tasks: `518`
- strict tasks: `365`
- strict quarterly tasks: `105`

strict family 分布：
- bottleneck: `89`
- forecasting: `94`
- strategic: `130`
- venue: `52`

新增任务共 `75` 条：
- q1 pool: `63`
- cluster pool: `12`

新增 family 分布：
- bottleneck: `15`
- forecasting: `27`
- strategic: `33`

### 扩充脚本

新增脚本：
- `scripts/expand_full_release_with_candidate_pools.py`

默认行为：
- 从 `benchmark_full_curated_recovered21_bottleneck18` 出发
- 合并：
  - `tmp/q1_short_candidates/all_candidates.judged.jsonl`
  - `tmp/cluster_expansion_v1/all_candidates.judged.jsonl`
- 对现有 release 以及新加入条目按 `normalized title` 去重
- 只保留 strict-ready 条目

选择规则：
- 正常通过：`judge.overall_score >= 0.55`
- 额外恢复一批“judge 明显坏掉但任务本身其实高质量”的条目：
  - 条件是 `overall_score == 0`
  - 但 numeric subscores 的 mean `>= 0.85`
  - 当前通过这条规则恢复了 `39` 条

本次 `75` 条新增里：
- `36` 条来自正常 score threshold
- `39` 条来自 suspicious zero-judge recovery

### 产物状态

这个 expanded release 已经补齐：
- `tasks.jsonl`
- `tasks_hidden_eval.jsonl`
- `tasks_build_trace.jsonl`
- `tasks_internal_full.jsonl`
- `task_ids.txt`
- `strict_task_ids.txt`
- `expanded_candidates_report.json`
- `tasks_hidden_eval_v3.jsonl`
- `tasks_hidden_eval_v3_manifest.json`
- `tasks_hidden_eval_v3_1.jsonl`
- `tasks_hidden_eval_v3_1_manifest.json`

### 已知 residual issue

base release 里本来就有 `1` 个重复标题，不是这次扩充新引入的：
- `Bottleneck and Opportunity Discovery in Dynamic Retrieval with Pre-Encoded Knowledge Fusion`
- 一个是 half-year：`RTLv3-EXP-1029`
- 一个是 quarter：`RTLv3-0510`

这次先保留，避免对旧 release 语义做额外改写；如果后面需要完全清理标题冲突，可以只对 quarter 那条补一个 `(Quarterly)` 后缀。
