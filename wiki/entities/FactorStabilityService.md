---
标题: FactorStabilityService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_stability_service.py, backend/constants.py]
---

## 是什么

`backend/services/factor_stability_service.py` 中的因子稳定性服务，从分布、时序、离散度与滚动四个角度评估因子是否稳定。

## 关键事实

- 类定义在该文件第 29 行，常量 `MIN_STABILITY_DAYS`、`ANNUAL_TRADING_DAYS`、`SEMI_ANNUAL_WINDOW` 来自 `backend/constants.py`。
- `calculate_distribution_stability(factor_series, window=ANNUAL_TRADING_DAYS)` 按年度窗口切段比较，长度不足 `window × 2` 时抛 `ValueError`。
- `calculate_time_series_stability` 评估序列自身的时序稳定性。
- `calculate_coefficient_of_variation` 用变异系数衡量离散程度。
- `calculate_rolling_stability(windows=[20, 60, SEMI_ANNUAL_WINDOW, ANNUAL_TRADING_DAYS])` 多窗口滚动评估。
- 窗口规则见 [[稳定性判定窗口]]，稳定性得分会进入 [[ComprehensiveScoringService]] 的评分。
