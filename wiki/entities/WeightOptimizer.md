---
标题: WeightOptimizer
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/weight_optimizer_service.py]
---

## 是什么

`backend/services/weight_optimizer_service.py` 中的组合权重优化器，是多因子权重计算的统一入口。

## 关键事实

- 类定义在 `backend/services/weight_optimizer_service.py` 第 18 行，`METHODS` 支持 6 种方法：equal_weight、ic_weight、ir_weight、max_sharpe、min_variance、risk_parity。
- 入口 `calculate_weights(factor_values, factor_names, method="equal_weight", returns=None, factor_data_dict=None)` 返回 `{"weights": {因子: 权重}, "method": 方法名}`。
- 传入未知 method 时记录 warning 并回退到等权，不抛异常。
- 入口对 `factor_values` 做防御性复制，不修改调用方传入的数据。
- 传入 alphalens 格式的 `factor_data_dict` 时，IC/IR 加权委托 alphalens 计算横截面 IC，口径与 [[横截面SpearmanIC]] 一致。
- 组合权重是 [[因子研究链路]] 的最后一环，前序环节见 [[FactorPreprocessingPipeline]]。
