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
- 通过迭代次数、种群数量、算子集合与复杂度上限控制搜索空间。
- 对单标的与截面模式分别设定最小有效样本门槛。
- 截面模式优先用 [[AlphalensAnalysisService]] 计算 IC/IR，失败回退到 [[FactorValidationService]]。
- 结果按验证分排序，合成目标下再做批量归一化；复杂度约束见 [[符号回归复杂度与样本门槛]]。
