---
标题: TreePrescreenMiningService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/tree_prescreen_mining_service.py]
---

## 是什么

`backend/services/tree_prescreen_mining_service.py` 中先用树模型筛因子、再交给下游算法搜索表达式的挖掘服务。

## 关键事实

- 继承 [[BaseMiningService]]，服务名为「树模型预筛选」。
- 树模型可选 `auto`、`lightgbm`、`xgboost`，`auto` 按依赖可用性决定。
- 默认 `n_estimators=100`、`max_depth=5`、`learning_rate=0.1`，重要性类型默认 `gain`。
- 重要性阈值默认 0.01，`top_k=0` 表示自动取 min(10, 特征数)。
- 默认至少 50 只股票参与评估，`max_eval_stocks=50`。
- 下游算法默认 `genetic`，也可切 `pysr`，相应参数逐一透传。
- 两阶段流程见 [[树模型预筛选两阶段管道]]，结果最终走 [[挖掘结果统一验证口径]]。
