---
标题: DataRouter
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/data.py]
---

## 是什么

`backend/api/routers/data.py` 中的行情与缓存路由，提供个股数据查询、预设股票池列举与缓存管理。

## 关键事实

- 共 6 个端点：`GET /stock/{code}`、`GET /stock-pools`、`GET /stock-pools/{pool_id}/stocks`、`GET /cache/stats`、`POST /cache/cleanup`、`POST /cache/clear`。
- 个股数据必须传起止日期，无数据返回 404，响应转成 `index`/`columns`/`data` 三字段矩阵结构。
- 内置 8 个股票池：`sz50`、`cyb50`、`hs300_sample`、`hs300`、`zz500_sample`、`zz500`、`zz1000_sample`、`zz1000`。
- 带 `_sample` 后缀的池只取前 50 只，用于快速体验横截面 IC；全量池分别对应 300、500、1000 只。
- 每个池记录名称、描述与指数代码，`sz50` 对应 000016、`cyb50` 对应 399673。
- 缓存三个端点分别返回统计、清理过期项与清空全部缓存。
- 取数委托 [[DataService]]，缓存管理委托 [[CacheService]]。
- 响应经 `sanitize_dict` 清洗后再返回，避免 NaN 与 NumPy 类型破坏 JSON。
