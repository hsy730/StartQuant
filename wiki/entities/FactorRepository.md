---
标题: FactorRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/factor_repository.py]
---

## 是什么

`backend/repositories/factor_repository.py` 中的因子表仓储类，封装因子记录的查询、写入、删除与来源计数。

## 关键事实

- 构造时注入 SQLAlchemy 会话，不自建连接。
- 查询支持按来源过滤与只看启用记录，结果按分类与名称排序。
- 默认视角只含启用记录，需要时才显式包含已禁用记录。
- 删除为硬删除，预置因子受保护不可删。
- 向上被 [[FactorService]] 调用，会话来源见 [[数据库会话生命周期管理]]。
