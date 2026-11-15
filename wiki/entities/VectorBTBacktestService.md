---
标题: VectorBTBacktestService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/vectorbt_backtest_service.py]
---

## 是什么

`backend/services/vectorbt_backtest_service.py` 中基于 vectorbt 的回测服务，覆盖单因子、多因子、横截面与分块回测。

## 关键事实

- 类定义在该文件第 81 行，构造参数含 `commission_rate=0.0003`、`slippage=0.0`、`slippage_mode="custom"`。
- 主要入口：`single_factor_backtest`、`multi_factor_backtest`、`cross_sectional_backtest`、`chunked_single_factor_backtest`、`run_vectorbt_backtest_from_weights`。
- 长区间回测先由 `_auto_chunk_config` 按频率与总 bar 数自动分块，再由 `_stitch_equity_curves` 拼接净值曲线。
- 智能滑点由 `set_smart_slippage` 调用滑点检测器完成，返回结果含 recommended / conservative / aggressive 三档与 confidence、reasoning。
- 指标计算包含 `calculate_metrics`、VaR/CVaR（`_calculate_var_cvar`）与交易明细格式化。
- 成本参数的量纲约束见 [[量纲一致性]]，回测结论通常再由 [[ComprehensiveScoringService]] 汇总评分。
