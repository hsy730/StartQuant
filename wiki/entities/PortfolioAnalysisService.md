---
标题: PortfolioAnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/portfolio_analysis_service.py]
---

## 是什么

`backend/services/portfolio_analysis_service.py` 中的组合分析服务，负责暴露、集中度、风险指标与权重优化的组合层分析。

## 关键事实

- 类定义在该文件第 19 行，提供 `calculate_industry_exposure`、`calculate_factor_exposure`、`calculate_concentration`、`analyze_portfolio_comprehensive`。
- 集中度用 `_calculate_gini` 计算 Gini 系数。
- `calculate_risk_metrics` 内部委托 `risk_metrics.calculate_risk_metrics`，传入基准时再补充 tracking_error 与 beta，符合 [[单一真相源]]。
- `optimize_weights` 与 `compare_weight_methods` 负责权重求解与方法对比，底层为 [[WeightOptimizer]]。
- `calculate_combined_factor_score` 把因子值合成为组合打分。
- 分析维度见 [[组合集中度与暴露]]，风险指标口径见 [[RiskMetrics]]。
