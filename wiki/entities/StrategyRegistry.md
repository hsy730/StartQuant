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

- 类定义在该文件第 15 行，内部用 `_strategies` 字典保存策略名到实现类的映射。
- 内置四个模板：`equal_weight`（等权）、`market_cap`（市值）、`momentum`（动量）、`mean_reversion`（均值回归）。
- 对外方法：`register` 注册自定义策略、`get_strategy` 按名取实例、`list_strategies` 列出全部可用策略。
- 请求未注册的策略名时抛 `ValueError`，错误信息中列出可用策略，不做静默回退。
- 所有策略继承 `BaseStrategy`，信号与权重分成两个方法，规则见 [[策略信号与权重分离]]。
- 回测本身不在此处执行，见 [[VectorBTBacktestService]]。
