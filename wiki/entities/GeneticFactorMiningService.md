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
- 用单点交叉与均匀变异配合深度限制装饰器，抑制表达式膨胀。
- 选择默认采用锦标赛，可切换为 NSGA-II；每代保留精英个体。
- 支持简约度惩罚、多样性惩罚、树深与基础因子数量上限等搜索约束。
- 与 [[PySRFactorMiningService]]、[[GFlowNetMiningService]] 共用基类的适应度路由与验证口径。
