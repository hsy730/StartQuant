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

- 端点分为个股行情、股票池与缓存管理三组，个股查询必须显式给出起止日期。
- 行情响应被转换成 index/columns/data 三字段的矩阵结构，无数据返回 404。
- 内置多个宽基与板块股票池，部分池提供抽样版本以便快速体验横截面 IC。
- 取数委托 [[DataService]]，缓存管理委托 [[CacheService]]。
- 响应统一经清洗函数过滤 NaN 与 NumPy 类型后再返回。
