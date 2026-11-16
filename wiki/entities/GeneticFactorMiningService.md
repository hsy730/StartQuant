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
- 评估股票样本在整轮进化中固定（覆盖基类 `_refresh_stock_sample`，仅在重建股票池时重新抽样）：基类每代随机重抽会让精英沿用的旧 fitness 与新样本上的子代不可比，NSGA-II / 锦标赛退化为噪声比较。
- NSGA-II 的 IC 目标不做 `max(…, 0)` 截断：负 IC 个体之间需保留相对排序，否则劣质个体并列、支配关系与拥挤度失去区分度。
- 代际 300s 评估超时 `break` 后，未评估个体统一填惩罚 fitness（`(0.0, 999.0)` / `(0.0,)`），否则 `stats.compile` 取 `fitness.values[0]` 会抛 `IndexError`。
- SymPy 代数等价去重（规范形）默认关闭，由 `use_sympy_dedup` 开关控制：`sp.simplify()` 对大 GP 树可能卡住 16+ 分钟，且命中率低；关闭后去重只依赖 zobrist（结构 + 交换律同构）。`_sympy_canonical_key_with_timeout` 统一吞掉异常返回 `None`，sympy 缺失时自动降级。
- 因子值缓存跨代保留：评估股票样本整轮固定后，同一棵树的因子值在所有代相同，无需按代清空（原 `_cache_clear` 已删除）。
- 逐股 / 逐个体耗时日志为 DEBUG 级：热路径每次评估约 40 条 INFO，20 代运行可产生 1.7 万+ 行，INFO 只保留每代汇总与超时告警。
- 与 [[PySRFactorMiningService]]、[[GFlowNetMiningService]] 共用基类的适应度路由与验证口径。
