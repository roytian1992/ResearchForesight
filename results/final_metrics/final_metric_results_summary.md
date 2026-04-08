# Benchmark 评测结果汇总

## 评测范围
- 任务总数：**168**
- 对比方法：**Native LLM**、**Hybrid RAG**、**ResearchArc**、**CoI**
- 结果按照三层指标汇报：
  1. **Primary**
  2. **Diagnostic**
  3. **Family-specific**

## 指标层级

### Primary
- **Fact**
- **Future Alignment**
- **Evidence Traceability**

### Diagnostic
- **Temporal Leakage**
- **Task Fulfillment**

### Family-specific
- **Bottleneck / Opportunity Discovery** → **Opportunity Grounding**
- **Direction Forecasting** → **Forecast Grounding**
- **Strategic Research Planning** → **Technical Dependency Grounding**

## 1. Primary 指标

| Method | Fact | Future Alignment | Evidence Traceability |
|---|---:|---:|---:|
| CoI | 0.5277 | 0.4567 | 0.7180 |
| Hybrid RAG | 0.5209 | 0.3960 | 0.5191 |
| Native LLM | 0.4160 | 0.4411 | 0.0555 |
| ResearchArc | 0.5185 | 0.4563 | 0.6243 |

## 2. Diagnostic 指标

| Method | Temporal Leakage | Task Fulfillment |
|---|---:|---:|
| CoI | 0.6687 | 0.7452 |
| Hybrid RAG | 0.8588 | 0.7746 |
| Native LLM | 0.9301 | 0.7699 |
| ResearchArc | 0.4729 | 0.8047 |

## 3. Family-specific 指标

### 3.1 Bottleneck / Opportunity Discovery

| Method | Opportunity Grounding |
|---|---:|
| CoI | 0.7627 |
| Hybrid RAG | 0.6919 |
| Native LLM | 0.6124 |
| ResearchArc | 0.9285 |

### 3.2 Direction Forecasting

| Method | Forecast Grounding |
|---|---:|
| CoI | 0.4647 |
| Hybrid RAG | 0.5002 |
| Native LLM | 0.4706 |
| ResearchArc | 0.8820 |

### 3.3 Strategic Research Planning

| Method | Technical Dependency Grounding |
|---|---:|
| CoI | 0.8121 |
| Hybrid RAG | 0.8143 |
| Native LLM | 0.7436 |
| ResearchArc | 0.8908 |

## 4. 分任务类别结果

### 4.1 Bottleneck / Opportunity Discovery（46 题）

| Method | Fact | Future Alignment | Evidence Traceability | Temporal Leakage | Task Fulfillment | Opportunity Grounding |
|---|---:|---:|---:|---:|---:|---:|
| CoI | 0.5002 | 0.4341 | 0.7092 | 0.7970 | 0.6966 | 0.7627 |
| Hybrid RAG | 0.5478 | 0.4432 | 0.3986 | 0.9857 | 0.7444 | 0.6919 |
| Native LLM | 0.4399 | 0.5154 | 0.0723 | 0.9857 | 0.7912 | 0.6124 |
| ResearchArc | 0.5283 | 0.3710 | 0.6207 | 0.7074 | 0.7476 | 0.9285 |

### 4.2 Direction Forecasting（49 题）

| Method | Fact | Future Alignment | Evidence Traceability | Temporal Leakage | Task Fulfillment | Forecast Grounding |
|---|---:|---:|---:|---:|---:|---:|
| CoI | 0.5663 | 0.5672 | 0.6004 | 0.5339 | 0.7018 | 0.4647 |
| Hybrid RAG | 0.5271 | 0.3867 | 0.5399 | 0.6804 | 0.7344 | 0.5002 |
| Native LLM | 0.5326 | 0.4254 | 0.0055 | 0.9061 | 0.7229 | 0.4706 |
| ResearchArc | 0.5610 | 0.4287 | 0.6289 | 0.1922 | 0.8352 | 0.8820 |

### 4.3 Strategic Research Planning（73 题）

| Method | Fact | Future Alignment | Evidence Traceability | Temporal Leakage | Task Fulfillment | Technical Dependency Grounding |
|---|---:|---:|---:|---:|---:|---:|
| CoI | 0.5192 | 0.3968 | 0.8024 | 0.6784 | 0.8050 | 0.8121 |
| Hybrid RAG | 0.4998 | 0.3725 | 0.5811 | 0.8986 | 0.8205 | 0.8143 |
| Native LLM | 0.3227 | 0.4048 | 0.0785 | 0.9112 | 0.7879 | 0.7436 |
| ResearchArc | 0.4837 | 0.5286 | 0.6235 | 0.5134 | 0.8201 | 0.8908 |

## 原始结果文件
- `tmp/final_metric_bundle_v1/final_overall.csv`
- `tmp/final_metric_bundle_v1/final_by_family.csv`
