---
标题: CacheService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/cache_service.py]
---

## 是什么

`backend/services/cache_service.py` 中的智能缓存服务，管理缓存文件、元数据、TTL、命中统计与清理。

## 关键事实

- 类定义在该文件第 20 行，缓存目录默认 `settings.AKSHARE_CACHE_DIR`，数据以 `{md5}.pkl` 落盘。
- 元数据（键、TTL、访问统计、过期标记）存数据库，由 `CacheRepository` 读写，与磁盘数据分离。
- 对外方法：`get`、`set`、`delete`、`exists`、`cleanup_expired`、`get_stats`、`clear_all`。
- `set(cache_key, data, ttl=None)` 中 `ttl=None` 表示永不过期，单位秒；`get` 命中时更新访问统计，过期或文件损坏时返回 `default`。
- 缓存文件损坏会自动删除文件与元数据，不向调用方抛异常。
- 命中与未命中次数保存在实例内存字段 `_hits`、`_misses` 中，`get_stats` 汇总输出。
- 设计规则见 [[缓存键与TTL失效]]，主要调用方为 [[DataService]]。
