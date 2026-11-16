---
标题: AnalysisRouter
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/analysis.py]
---

## 是什么

`backend/api/routers/analysis.py` 中的因子分析路由，把计算、IC/IR、稳定性、暴露、归因、监测、相关性与报表等分析能力暴露为 HTTP 端点。

## 关键事实

- 全部为 POST 端点，共 19 个：`/calculate`、`/ic`、`/stability`、`/multi-period`、`/decay`、`/exposure`、`/effectiveness`、`/attribution`、`/monitoring`、`/correlation/enhanced`、`/correlation/interpret`、`/correlation/mixed-type`、`/quantile-returns`、`/cumulative-returns`、`/turnover`、`/tear-sheet`、`/weighted-ic`、`/factor-importance`、`/lookahead-bias`。
- 内部函数 `_resolve_factor_code` 先查 factors 表（含停用因子），再遍历 generated_factors 表匹配名称，均未命中返回 404。
- 依赖服务包括 [[AnalysisService]]、[[FactorStabilityService]]、[[EnhancedAnalysisService]]、[[FactorExposureService]]、[[FactorEffectivenessService]]、[[FactorAttributionService]]、[[FactorMonitoringService]]。
- 滚动相关计算调用 `calculate_rolling_ic`，避免路由层自行实现 Pearson 口径。
- 响应统一经 `sanitize_dict` 与 `safe_numeric_value` 清洗后再返回。
- 请求模型按端点拆分，如 `CalculateRequest`（因子名 + 股票列表 + 区间 + 频率）、`ICAnalysisRequest`、`StabilityRequest`、`QuantileReturnsRequest`、`WeightedICRequest`、`CorrelationAnalysisRequest`、`LookaheadBiasRequest`。
- 端点挂载于 [[FactorFlowAPI应用]] 的路由聚合中。
