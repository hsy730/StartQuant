---
标题: StrategyComparisonService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/strategy_comparison_service.py]
---

## 是什么

`backend/services/strategy_comparison_service.py` 第 17 行的策略对比服务，在同一份行情数据上批量回测多个策略并比较表现。

## 关键事实

- 主入口 `compare_strategies(df, strategy_names, strategy_params=None)` 先对入参 `df` 做 `.copy()`，保证调用方数据不被修改。
- 策略实例统一从 [[StrategyRegistry]] 的 `get_strategy(name, **params)` 取用，回测与指标计算沿用策略基类，体现 [[策略信号与权重分离]]。
- 返回四块内容：`strategies`（各策略回测与指标）、`metrics_comparison`（指标表）、`equity_curves`（净值曲线）、`statistical_tests` 与 `ranking`。
- 指标表以策略名为索引，缺失值保留为 `None` 而非填 0，避免把缺数当成零值。
- 两两检验要求对齐后样本不少于 30 条，否则跳过并告警；同时给出 Welch's t 检验、配对 t 检验与相关系数。
- 排名对年化收益、夏普、最大回撤分别取秩（回撤反向），再按可用指标求秩均值，NaN 秩跳过，最终得分低者名次靠前。
- `generate_comparison_report` 依据对比结果拼接纯文本报告，指标对比、显著性检验与排名依次成节。
- 显著性阈值来自常量 `STATISTICAL_SIGNIFICANCE_ALPHA = 0.05`。
