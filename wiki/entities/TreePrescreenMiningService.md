---
标题: TreePrescreenMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/tree_prescreen_mining_service.py]
---

## 是什么

`backend/services/tree_prescreen_mining_service.py` 中的挖掘服务，先用树模型筛选因子，再把筛选结果交给下游算法搜索表达式。

## 关键事实

- 继承 [[BaseMiningService]]，作为命名明确的挖掘算法接入统一挖掘流程。
- 树模型可在自动、LightGBM 与 XGBoost 之间选择，自动模式按依赖可用性决定。
- 树模型超参、重要性类型与阈值、Top-K 数量以及参与评估的股票数上限均可通过参数覆盖，Top-K 为零表示自动决定。
- 下游算法可选遗传规划或符号回归，相关参数原样透传。
- 两阶段流程见 [[树模型预筛选两阶段管道]]，结果走 [[挖掘结果统一验证口径]]。
