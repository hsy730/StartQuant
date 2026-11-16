---
标题: StrategyRegistry
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/strategy_registry.py, backend/strategies/]
---

## 是什么

`backend/services/strategy_registry.py` 中的策略注册表，按名称登记内置策略模板并统一对外提供实例。

## 关键事实

- 以「名称到实现类」的映射维护注册关系，内置等权、市值、动量与均值回归四个模板。
- 对外提供注册自定义策略、按名取实例与列举全部可用策略三项能力。
- 请求未注册的名称时抛错并列出可用项，不做静默回退。
- 所有策略继承同一基类，信号与权重分成两个方法，规则见 [[策略信号与权重分离]]；回测本身不在本模块执行，见 [[VectorBTBacktestService]]。
