---
标题: FactorEffectivenessService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_effectiveness_service.py]
---

## 是什么

`backend/services/factor_effectiveness_service.py` 中的因子有效性分析服务，覆盖散点、IC 时序、事件响应与衰减。

## 关键事实

- 入口为 `analyze_effectiveness(factor_data, factor_name, future_periods=[1, 5, 10, 20])`。
- IC 计算优先使用 alphalens，失败时回退到时序 IC 或横截面 IC。
- 横截面 IC 委托统一计算入口，不在此服务内另写实现。
- 事件响应默认分位阈值 0.8、持有期 `[1, 3, 5, 10]` 日，规则见 [[事件响应阈值分组]]。
- 衰减曲线同样优先由 alphalens 计算，失败时用自实现版本。
- 散点数据把因子值与未来收益配对，供前端绘图。
