# Benchmark 评测结果汇总

## 评测范围
- 任务总数：**168**
- 对比方法：**Native LLM**、**Hybrid RAG**、**ResearchArc（accepted v6 optimized）**、**CoI**、**ARIS-Offline v7**、**ResearchAgent-Offline**
- 结果按照两层指标汇报：
  1. **Primary**
  2. **Family-specific**
- **Task Fulfillment** 不再纳入主结果表。
- ResearchArc 当前接受版本机制说明：`docs/research_arc_current_mechanism.md`

## 指标层级

### Primary
- **Evidence-Grounded Factuality**
- **Future Alignment**
- **Evidence Traceability**

### Family-specific
- **Bottleneck / Opportunity Discovery** → **Opportunity Grounding**
- **Direction Forecasting** → **Forecast Grounding**
- **Strategic Research Planning** → **Technical Dependency Grounding**

## 1. Primary 指标

| Method | Evidence-Grounded Factuality | Future Alignment | Evidence Traceability |
|---|---:|---:|---:|
| ARIS-Offline v7 | 0.4388 | 0.3269 | 0.5635 |
| CoI | 0.5277 | 0.4567 | **0.7180** |
| Hybrid RAG | 0.5209 | 0.3960 | 0.5191 |
| Native LLM | 0.4160 | 0.4411 | 0.0555 |
| ResearchAgent-Offline | 0.4169 | 0.3134 | 0.4461 |
| ResearchArc | **0.5423** | **0.4592** | 0.6275 |

## 2. Family-specific 指标

### 2.1 Bottleneck / Opportunity Discovery

| Method | Opportunity Grounding |
|---|---:|
| ARIS-Offline v7 | 0.5875 |
| CoI | 0.7627 |
| Hybrid RAG | 0.6919 |
| Native LLM | 0.6124 |
| ResearchAgent-Offline | 0.5558 |
| ResearchArc | **0.8301** |

### 2.2 Direction Forecasting

| Method | Forecast Grounding |
|---|---:|
| ARIS-Offline v7 | 0.3112 |
| CoI | 0.4647 |
| Hybrid RAG | 0.5002 |
| Native LLM | 0.4706 |
| ResearchAgent-Offline | 0.3784 |
| ResearchArc | **0.8084** |

### 2.3 Strategic Research Planning

| Method | Technical Dependency Grounding |
|---|---:|
| ARIS-Offline v7 | 0.7382 |
| CoI | 0.8121 |
| Hybrid RAG | 0.8143 |
| Native LLM | 0.7436 |
| ResearchAgent-Offline | 0.7317 |
| ResearchArc | **0.8599** |

## 3. 分任务类别结果

### 3.1 Bottleneck / Opportunity Discovery（46 题）

| Method | Evidence-Grounded Factuality | Future Alignment | Evidence Traceability | Opportunity Grounding |
|---|---:|---:|---:|---:|
| ARIS-Offline v7 | 0.4879 | 0.4479 | 0.4098 | 0.5875 |
| CoI | 0.5002 | 0.4341 | **0.7092** | 0.7627 |
| Hybrid RAG | **0.5478** | 0.4432 | 0.3986 | 0.6919 |
| Native LLM | 0.4399 | **0.5154** | 0.0723 | 0.6124 |
| ResearchAgent-Offline | 0.4342 | 0.3786 | 0.4207 | 0.5558 |
| ResearchArc | 0.5337 | 0.3335 | 0.5942 | **0.8301** |

### 3.2 Direction Forecasting（49 题）

| Method | Evidence-Grounded Factuality | Future Alignment | Evidence Traceability | Forecast Grounding |
|---|---:|---:|---:|---:|
| ARIS-Offline v7 | 0.4356 | 0.2550 | 0.5803 | 0.3112 |
| CoI | 0.5663 | **0.5672** | 0.6004 | 0.4647 |
| Hybrid RAG | 0.5271 | 0.3867 | 0.5399 | 0.5002 |
| Native LLM | 0.5326 | 0.4254 | 0.0055 | 0.4706 |
| ResearchAgent-Offline | 0.4208 | 0.2246 | 0.2791 | 0.3784 |
| ResearchArc | **0.6051** | 0.4406 | **0.6361** | **0.8084** |

### 3.3 Strategic Research Planning（73 题）

| Method | Evidence-Grounded Factuality | Future Alignment | Evidence Traceability | Technical Dependency Grounding |
|---|---:|---:|---:|---:|
| ARIS-Offline v7 | 0.4101 | 0.2989 | 0.6490 | 0.7382 |
| CoI | **0.5192** | 0.3968 | **0.8024** | 0.8121 |
| Hybrid RAG | 0.4998 | 0.3725 | 0.5811 | 0.8143 |
| Native LLM | 0.3227 | 0.4048 | 0.0785 | 0.7436 |
| ResearchAgent-Offline | 0.4034 | 0.3319 | 0.5742 | 0.7317 |
| ResearchArc | 0.5056 | **0.5508** | 0.6426 | **0.8599** |

## 原始结果文件
- `tmp/final_metric_bundle_v1/final_overall.csv`
- `tmp/final_metric_bundle_v1/final_by_family.csv`
- `results/aris_offline_168_v7_qwen/results.jsonl`
- `results/aris_offline_168_v7_qwen/eval_v31/summary.json`
- `results/aris_offline_168_v7_qwen/eval_v4/summary.json`
- `results/aris_offline_168_v7_qwen/eval_aux/summary.json`
- `results/researchagent_offline_168_qwen/results.jsonl`
- `results/researchagent_offline_168_qwen/eval_v31/summary.json`
- `results/researchagent_offline_168_qwen/eval_v4/summary.json`
- `results/researchagent_offline_168_qwen/eval_aux/summary.json`
- `results/research_arc_v6_opt_full168/results.jsonl`
- `results/research_arc_v6_opt_full168/eval_v31/summary.json`
- `results/research_arc_v6_opt_full168/eval_v4/summary.json`
- `results/research_arc_v6_opt_full168/eval_aux/summary.json`
