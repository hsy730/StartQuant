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

- 构造参数集中管理手续费、滑点与滑点模式，成本参数量纲必须与收益口径一致，见 [[量纲一致性]]。
- 对外按因子个数与截面方式区分多个回测入口，也支持直接由权重序列驱动回测。
- 长区间回测先自动分块再拼接净值曲线，避免一次性加载全量数据。
- 智能滑点能力委托滑点检测器，返回结果含分档建议、置信度与理由。
- 指标计算含常规绩效指标与 VaR/CVaR；回测结论通常再由 [[ComprehensiveScoringService]] 汇总评分。
