---
标题: 适应度目标路由与Z-Score归一化
类型: 概念
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/base_mining_service.py]
---

## 一句话定义

按 `fitness_objective` 在 IC、IR 与组合分数之间切换适应度，并用代际 Z-Score 消除量纲差异。

## 核心要点

- 支持四种目标：`ic_mean`（默认）、`ir_ratio`、`sharpe`、`combined`；`sharpe` 直接复用 IR，不做额外计算。
- 适应度取 `spearman_ic` 与 `pearson_ic` 两类结果中的最优绝对 IC，IR 由均值除以标准差得到，除零保护走 [[安全除法]]。
- `combined` 公式为 `0.6 × Norm(IC) + 0.4 × Norm(IR)`：z 值先截断到 [-3, 3]，再线性映射到 [0, 1]。
- 第一代使用先验冷启动常量（IC 均值 0.03、标准差 0.02；IR 均值 0.5、标准差 0.3），之后每代按收集值更新。
- 统计量至少需要 5 个有效值；标准差取 `max(σ, max(0.01 × μ, 0.005))` 作为下界，防止种群收敛时 Z-Score 爆炸。
- 批量模式在全部评估结束后重算 `combined` 分数、同步 validation 分数并重新排名。

## 适用场景 / 边界

- 适用：遗传规划、GFlowNet 等需要把 IC 与 IR 合成为一个标量适应度的迭代搜索。
- 边界：该分数只用于搜索排序，不代表因子通过验证，最终判定仍以 [[FactorValidationService]] 的检查结果为准；不同代的绝对分值不可直接横向比较。
