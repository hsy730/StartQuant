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

- 支持等权、IC 加权、IR 加权、最大夏普、最小方差与风险平价六种方法。
- 单一入口返回权重字典与方法名，入参数据做防御性复制，不修改调用方数据。
- 传入未知方法时告警并回退等权，不抛异常。
- 传入 alphalens 格式数据时委托其计算横截面 IC，口径与 [[横截面SpearmanIC]] 一致。
- 位于 [[因子研究链路]] 末端，前序预处理环节见 [[FactorPreprocessingPipeline]]。
