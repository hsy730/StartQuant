---
标题: PySRFactorMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/pysr_factor_mining_service.py]
---

## 是什么

`backend/services/pysr_factor_mining_service.py` 中基于 PySR 的符号回归因子挖掘服务。

## 关键事实

- 依赖 `pysr`，未安装时构造即抛出 `ImportError`。
- 默认 `niterations=40`、`populations=30`、`procs=8`、`population_size=33`。
- 默认 `maxsize=30`、`maxdepth=5`、`parsimony=0.0032`，约束默认 `{"/": (-1, 9)}`。
- 默认二元算子 `+ - * /`，一元算子 `neg abs log sqrt`。
- 单标的建模至少 50 个有效样本，截面模式每只股票至少 20 个。
- 截面模式优先用 [[AlphalensAnalysisService]] 计算 IC/IR，失败回退到 [[FactorValidationService]]。
- 结果按验证分排序，`combined` 目标下再做批量 Z-Score 归一化。
- 复杂度与样本约束见 [[符号回归复杂度与样本门槛]]。
