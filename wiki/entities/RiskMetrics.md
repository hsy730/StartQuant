---
标题: RiskMetrics
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/risk_metrics.py, backend/constants.py]
---

## 是什么

`backend/services/risk_metrics.py` 是风险与收益指标的统一计算入口，全部基于 empyrical 实现，不自行推导公式。

## 关键事实

- `calculate_risk_metrics(returns, risk_free_rate=RISK_FREE_RATE, annual_trading_days=ANNUAL_TRADING_DAYS)` 返回 10 项指标：total_return、annual_return、volatility、sharpe_ratio、sortino_ratio、max_drawdown、calmar_ratio、win_rate、var_95、cvar_95。
- 常量来自 `backend/constants.py`：`ANNUAL_TRADING_DAYS=252`、`RISK_FREE_RATE=0.03`、`VAR_CONFIDENCE_CUTOFF=0.05`。
- `calculate_relative_metrics(strategy_returns, benchmark_returns, ...)` 是 alpha / beta / information_ratio / tracking_error 的统一入口。
- 另有 `calculate_sharpe`、`calculate_volatility` 两个单项快捷函数。
- 零波动率、空序列与非有限值的处理规则见 [[零波动率边界处理]]；除法保护见 [[安全除法]]。
- 该模块是 [[单一真相源]] 原则在风险指标上的落地。
