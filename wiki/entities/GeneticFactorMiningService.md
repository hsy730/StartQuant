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
- 内置终端使用统计：按代累计各基础因子在种群中的出现次数，并统计其进入 HOF 的次数，`hof_hits=0` 判定为死终端，死终端占比过半时输出因子池冗余告警。
- 截面 IC 取逐日 Spearman 的**有符号**均值；`|IC|` 仅作为方向无关的适应度，IR 的分子分母必须来自同一有符号序列，截面样本数下限为 10 只股票。
- 重复子树惩罚在 NSGA-II 下只叠加到复杂度目标（`n_dups ≤ 2` 每个 0.15，之后 0.4/个），不进入 IC 目标；改按 `duplicate_ratio` 相对量纲缩放的 A/B（50×10、13 因子、10 只股票、3 个种子）显示 HOF 终端数与死终端差异落在噪声内，单目标模式下终端数上升（2.07→2.63）但最优 IC 下降（0.266→0.183），因此保留绝对惩罚。
- 适应度评估含线程池与缓存，同一进程内连续多次运行结果不可复现；A/B 必须每次独立进程、用多个随机种子取均值。
- 与 [[PySRFactorMiningService]]、[[GFlowNetMiningService]] 共用基类的适应度路由与验证口径。
