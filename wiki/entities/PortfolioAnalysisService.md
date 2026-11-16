---
标题: PortfolioAnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/portfolio_analysis_service.py]
---

## 是什么

`backend/services/portfolio_analysis_service.py` 中的组合分析服务，覆盖暴露、集中度、风险指标与权重优化的组合层分析。

## 关键事实

- 提供行业暴露、因子暴露、集中度与综合分析入口。
- 集中度以 Gini 系数衡量。
- 风险指标统一委托 [[RiskMetrics]]，传入基准时补充跟踪误差与 beta，符合 [[单一真相源]]。
- 权重求解与方法对比底层由 [[WeightOptimizer]] 承担。
- 另提供把因子值合成为组合打分的入口。
