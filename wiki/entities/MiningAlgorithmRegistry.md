---
标题: MiningAlgorithmRegistry
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/mining_algorithm_registry.py]
---

## 是什么

`backend/services/mining_algorithm_registry.py` 是因子挖掘算法的工厂注册表，按算法名创建对应的挖掘服务实例。

## 关键事实

- 核心接口为 `register_algorithm(name, factory)`、`create_algorithm(name, **kwargs)`、`get_registered_algorithms()`、`is_algorithm_registered(name)`。
- 内置注册 5 个算法名：genetic、pysr、tree_prescreen、gflownet、deep_implicit，模块加载时由 `_register_builtin_algorithms()` 自动完成。
- genetic 对应遗传算法服务，pysr 对应符号回归服务；tree_prescreen、gflownet、deep_implicit 共用一个工厂，最终实例化为 `DualMiningService`。
- 三个统一调度算法通过 `downstream_algorithm` 指定下游挖掘算法，默认 genetic。
- 请求未知算法名时记录 warning 并返回 `None`，不抛异常。
- 新增算法只需实现工厂函数并在此文件注册，调用方无需改动。
