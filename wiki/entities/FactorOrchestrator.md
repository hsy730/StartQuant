---
标题: FactorOrchestrator
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_orchestrator_service.py]
---

## 是什么

`backend/services/factor_orchestrator_service.py` 中的因子编排器，把表达式因子送进一条固定顺序的验证流水线并输出结论与报告。

## 关键事实

- 以配置对象描述流水线，各阶段可开关，默认启用偏差检测、IC、分组回测与报告类阶段。
- 偏差检测默认失败即终止，IC 设有通过阈值。
- 各阶段实现为独立步骤，结果汇总为状态、Markdown 报告与结构化数据三部分。
- 状态区分通过、拒绝与部分通过三档。
- 流水线设计见 [[一键验证流水线]]。
