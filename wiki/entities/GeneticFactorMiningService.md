---
标题: GeneticFactorMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/genetic_factor_mining_service.py]
---

## 是什么

`backend/services/genetic_factor_mining_service.py` 中基于 DEAP 遗传规划的因子挖掘服务。

## 关键事实

- 继承 [[BaseMiningService]]，服务名为「遗传规划」。
- 默认 `population_size=50`、`n_generations=20`、`cx_prob=0.7`、`mut_prob=0.3`、`elite_size=5`。
- 默认 `parsimony_coeff=0.001`、`diversity_penalty_coeff=0.1`、`max_tree_depth=17`、`max_base_factors=30`。
- 工具集使用单点交叉 `cxOnePoint`、均匀变异 `mutUniform` 并挂深度限制装饰器，防止表达式膨胀。
- 选择默认使用 `selTournament`（`tournsize=3`），可切换为 NSGA-II。
- 每代保留精英个体，支持取消标志与进度回调。
- 与 [[PySRFactorMiningService]]、[[GFlowNetMiningService]] 共用基类的适应度路由与验证口径。
