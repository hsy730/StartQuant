---
标题: FactorRouter
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/factors.py]
---

## 是什么

`backend/api/routers/factors.py` 中的因子管理路由，负责因子增删改查、批量生成、预筛选、公式验证、复制、简化，以及暂存池的查看与提升。

## 关键事实

- 共 15 个端点：`GET /`、`GET /stats`、`GET /{factor_id}`、`POST /`、`PUT /{factor_id}`、`DELETE /{factor_id}`、`POST /batch-generate`、`POST /preselect`、`POST /validate`、`POST /{factor_id}/copy`、`POST /{factor_id}/simplify`、`GET /generated/`、`GET /generated/{generated_id}`、`POST /generated/{generated_id}/promote`、`DELETE /generated/{generated_id}`。
- 列表接口支持按 `category` 与 `source` 过滤；统计接口单列，避免被路径参数吞掉。
- 复制因子时按「原名_序号」生成新名称，序号取现有同前缀用户因子的最大值加一。
- `/validate` 先拒绝含控制字符的代码，再调用 `factor_service.validate_factor_code` 实际执行校验。
- `/batch-generate` 按表达式去重，只回传前 20 个示例与总数；`/preselect` 当前为占位实现，原样返回入参。
- 提升接口依赖 [[GeneratedFactorRepository]] 读取暂存记录，并在写入前做表达式安全校验。
- 直接依赖 [[FactorService]] 与 [[FactorGeneratorService]]。
- 端点挂载于 [[FactorFlowAPI应用]]。
- 清理暂存记录只删除暂存池条目，不影响已提升的正式因子。
