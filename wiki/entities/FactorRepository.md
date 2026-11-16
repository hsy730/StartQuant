---
标题: FactorRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/factor_repository.py]
---

## 是什么

`backend/repositories/factor_repository.py` 中的因子表数据访问类，封装查询、写入、删除与来源计数。

## 关键事实

- 构造时接收一个 SQLAlchemy `Session`，不自行创建连接。
- `get_all` 支持按 `source` 过滤与 `active_only` 只取启用记录，结果按分类、名称排序。
- `get_by_name` 默认只查启用记录，传 `include_inactive=True` 才包含禁用记录。
- `delete` 为硬删除；记录 `source` 为 `preset` 时抛出 `ValueError("预置因子不能删除")`。
- `get_preset_count` 与 `get_user_count` 分别统计 `source` 为 `preset`/`user` 且 `is_active=1` 的数量。
- 向上被 [[FactorService]] 与 [[AnalysisService]] 调用，会话来源见 [[数据库会话生命周期管理]]。
