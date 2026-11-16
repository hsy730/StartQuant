---
标题: StrategyComparisonService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/strategy_comparison_service.py]
---

## 是什么

`backend/services/strategy_comparison_service.py` 中的策略对比服务，在同一份行情数据上批量回测多个策略并给出横向对比结论。

## 关键事实

- 入参行情数据先做副本，保证调用方数据不被修改。
- 策略实例统一从 [[StrategyRegistry]] 获取，回测与指标沿用策略基类，遵循 [[策略信号与权重分离]]。
- 输出包含四块：各策略回测明细、指标对比表、净值曲线，以及两两统计检验与排名。
- 指标缺失保留为 null 而非补零；两两检验在样本对齐后不足下限时跳过并告警。
- 排名按多个核心指标取秩后求均值，显著性阈值取自 [[全局常量模块]]。
