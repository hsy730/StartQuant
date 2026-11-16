---
标题: OrchestratorRouter
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/orchestrator.py]
---

## 是什么

`backend/api/routers/orchestrator.py` 中的编排与模型路由，暴露因子一键验证、批量验证、排序模型训练预测，以及模型注册中心的阶段晋升与查询。

## 关键事实

- 共 14 个端点，分三组：`/orchestrator/validate` 与 `/orchestrator/batch-validate`；`/stock-ranker/train`、`/train-with-data`、`/predict`、`GET /models`、`DELETE /models/{model_id}`、`GET /explain/{model_id}`；模型注册中心的列表、元数据、版本、晋升、删除与统计。
- 批量验证返回多因子对比表，单次验证返回单因子完整结论。
- 排序模型支持两种训练入口：按股票列表拉数训练，或直接传入已有数据训练。
- 模型晋升从请求体读取目标阶段，缺省为 `staging`，委托 `model_registry.promote`。
- 模型删除、元数据与版本查询均转为 `model_registry` 对应方法，失败返回 404。
- 模型统计端点独立暴露，返回注册中心的汇总计数。
- 依赖 [[FactorOrchestrator]]、[[StockRankerService]] 与 [[ModelRegistry]]。
- 端点挂载于 [[FactorFlowAPI应用]]。
- 涉及模型注册中心的端点统一用服务层单例，不直接操作数据库。
