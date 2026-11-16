---
标题: FactorReturnAnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_return_analysis_service.py]
---

## 是什么

`backend/services/factor_return_analysis_service.py` 第 65 行的因子收益分析服务，提供分组收益、累计收益、多空利差与换手率分析。

## 关键事实

- 配置类 `FactorReturnAnalysisConfig` 默认 `n_quantiles=5`、`quantile_method="equal_weight"`、`forward_period=1`、`min_samples_per_group=5`、`enable_bootstrap=True`、`bootstrap_n=1000`。
- `calculate_quantile_returns(factor_data, factor_name, price_column="close")` 按日期逐截面 `qcut`，输出各组平均收益、t 检验、多空利差与单调性检验。
- `calculate_cumulative_returns(..., long_short=True)` 构建多空组合并输出累计收益曲线与汇总统计。
- `calculate_turnover_analysis` 输出换手率、自相关与稳定性评分，并对换手水平生成文字解读。
- `_calculate_spread` 使用 `ttest_ind_from_stats(equal_var=False)` 做 Welch's t 检验；`_test_monotonicity` 以相邻组同向比例 0.8 为单调门槛。
- `_bootstrap_quantile_returns` 默认重抽样 1000 次，多股票按日期聚类、单股票按行重抽样。
- 夏普与最大回撤分别委托 `calculate_sharpe`、`calculate_risk_metrics`，与 [[RiskMetrics]] 保持同一口径。
- 分层口径与单调性判定见 [[横截面分层与单调性]]。
