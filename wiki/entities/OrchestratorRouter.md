---
标题: OrchestratorRouter
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/orchestrator.py]
---

## 是什么

`backend/api/routers/orchestrator.py` 中的编排与模型路由，暴露因子验证、排序模型训练预测与模型注册中心管理三类能力。

## 关键事实

- 端点分三组：因子编排验证、排序模型训练预测、模型注册中心管理。
- 批量验证返回多因子对比，单次验证返回单因子完整结论。
- 排序模型支持按股票列表拉数训练与直接传入已有数据训练两种入口。
- 模型阶段晋升、删除与查询均转交注册中心服务，失败返回 404。
- 依赖 [[FactorOrchestrator]]、[[StockRankerService]] 与 [[ModelRegistry]]，挂载于 [[FactorFlowAPI应用]]。
