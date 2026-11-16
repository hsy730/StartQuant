---
标题: CacheRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/cache_repository.py]
---

## 是什么

`backend/repositories/cache_repository.py` 中的缓存元数据表数据访问类。

## 关键事实

- `create` 写入 `cache_key`、`file_path`、`created_at`、`ttl`、`size`，初始化 `access_count=0`、`expired=False`。
- `get_by_key` 按唯一键查询单条元数据。
- `update_access` 调用模型的 `mark_as_accessed` 后提交，累加访问次数并刷新最后访问时间。
- `mark_as_expired` 直接把 `expired` 置为真。
- `get_all_expired` 在内存中逐条调用 `is_expired()` 过滤。
- `get_stats` 返回总数、总大小（字节与 MB）、总访问次数、过期数与活跃数。
- `clear_all` 用一条批量删除语句清空并返回行数。
- 缓存键与过期语义的规则见 [[缓存键与TTL失效]]。
