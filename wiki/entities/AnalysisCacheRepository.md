---
标题: AnalysisCacheRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/factor_repository.py]
---

## 是什么

`backend/repositories/factor_repository.py` 中的分析结果缓存表数据访问类。

## 关键事实

- 以唯一键 `cache_key` 查询缓存记录。
- 提供创建、更新、按主键删除三个方法，均在方法内提交。
- `delete_old_cache(days=7)` 删除 `created_at` 早于当前时间减指定天数的记录，并返回删除行数。
- 缓存键的构造与失效规则见 [[缓存键与TTL失效]]。
- 与 [[FactorRepository]] 同属仓储层，但未实现异常回滚。
