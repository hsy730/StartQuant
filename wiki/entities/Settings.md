---
标题: Settings
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/core/settings.py]
---

## 是什么

`backend/core/settings.py` 中的应用配置类与模块级全局实例，集中定义路径、数据库、缓存、预处理与日志参数。

## 关键事实

- 基于 pydantic-settings 的 `BaseSettings`，从 `.env` 读取配置，编码 UTF-8、大小写不敏感。
- 默认 `DATABASE_URL` 为 `sqlite:///{DB_DIR}/factorflow.db`。
- 缓存默认 TTL 为 7 天，自动清理开启且间隔 24 小时，滚动窗口 `ROLLING_WINDOW=252`。
- 默认分析区间为 `2020-01-01` 至 `2024-12-31`，日志级别 `INFO`，单文件 10MB、保留 5 份备份。
- 实例化时创建数据、缓存、数据库、报告、akshare 缓存与日志目录，并在数据库缺失时复制样本库。
- 全局 `settings` 被 [[数据库会话生命周期管理]] 读取以构造数据库引擎。
