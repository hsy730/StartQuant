---
标题: FactorEffectivenessService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_effectiveness_service.py]
---

## 是什么

`backend/services/factor_effectiveness_service.py` 中的因子有效性分析服务，从散点关系、IC 时序、事件响应与收益衰减四个角度衡量因子对未来收益的解释力。

## 关键事实

- IC 与衰减曲线优先由 alphalens 计算，不可用时回退到自实现版本。
- 横截面 IC 走统一计算入口，服务内不重复实现。
- 事件响应按分位阈值划分事件组，阈值与分组口径见 [[事件响应阈值分组]]。
- 输出面向绘图，散点、时序与衰减数据都整理成前端可直接消费的结构。
