---
标题: StatisticsService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/statistics_service.py]
---

## 是什么

`backend/services/statistics_service.py` 第 40 行的高级统计分析服务，为因子有效性、稳定性、交互与拥挤度提供统计检验方法。

## 关键事实

- `t_test_ic(ic_series, confidence_level=0.95)` 对 IC 序列做单样本 t 检验，输出 t 值、p 值、是否显著、均值、标准差与置信区间。
- 常数 IC 序列（标准差小于 1e-10）单独处理：均值为正时 t 值记为无穷、p 值为 0，均值为 0 时 p 值为 1，避免退化结果，与 [[零波动率边界处理]] 同一思路。
- `test_monotonicity(quantile_returns, alternative="increasing")` 对各层均值做 Spearman 秩相关单向检验，p 值小于 0.05 视为单调。
- `calculate_factor_decay(df, factor_name, max_periods=10)` 逐期计算 IC 得到衰减曲线；`calculate_ic_predictability(ic_series, lag=5)` 输出 IC 自相关序列与其绝对值均值。
- `calculate_periodic_ic(..., period="M")` 按周期重采样求 IC；`calculate_rolling_ic_stability` 默认窗口为 20/60/120/252 日。
- `calculate_market_regime_ic` 按给定市场环境分组求 IC；`analyze_factor_interactions` 用二阶多项式特征描述因子交互项。
- `calculate_factor_crowding` 以 `1 / (1 + 滚动标准差)` 表示拥挤度，窗口默认 20；`calculate_turnover` 以信号差分绝对值均值定义换手率。
- 分层收益统计委托 `calculate_risk_metrics` 计算年化收益与夏普，IC 计算委托 `calculate_ic`，除法统一走 [[安全除法]]，体现 [[单一真相源]]。
- `calculate_factor_correlation_matrix` 对因子列做 Spearman 相关矩阵，与 [[FactorCorrelationService]] 的完整诊断相比只提供矩阵本身。
- scipy 调用由 `_suppress_scipy_warnings` 上下文管理器局部抑制告警，不影响其他模块。
