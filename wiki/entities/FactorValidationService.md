---
标题: FactorValidationService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_validation_service.py, backend/constants.py]
---

## 是什么

`backend/services/factor_validation_service.py` 中的因子验证服务，用多项统计检查叠加未来函数检测给因子打分，并判定是否通过。

## 关键事实

- IC 与 IR 阈值取自全局常量，换手率、相关性上限与 IC 类型可另行配置。
- 验证流程依次覆盖 IC、RankIC、IR、换手率、稳定性与相关性，稳定性检查委托 [[FactorStabilityService]]。
- 通过条件为全部基础检查通过，且未来函数风险不高于低风险档。
- 综合得分上限 100，按各维度加权，未来函数风险按等级乘性扣分。
- 批量验证中已通过的因子会作为后续相关性检查的参照；模块导出全局单例。
