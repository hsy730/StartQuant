---
标题: DataService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/data_service.py]
---

## 是什么

`backend/services/data_service.py` 中的行情数据服务，负责股票数据获取、缓存与可交易性掩码构建，设计口径为 Mask-First。

## 关键事实

- 类定义在该文件第 42 行，数据来源为 akshare，日线与分钟线均使用 `adjust="qfq"` 前复权。
- 缓存键由 `股票代码_起始日期_结束日期` 取 MD5，默认 TTL 取 `settings.CACHE_DEFAULT_TTL`，通过全局 `cache_service` 读写。
- 板块涨跌幅配置：主板 ±10%、创业板 ±20%、科创板 ±20%、北交所 ±30%。
- `TradableMaskConfig` 控制掩码条件：`check_limit_up`、`check_limit_down`、`check_suspended`、`check_new_stock`、`new_stock_threshold=250`、`volume_threshold=0.0`。
- 主要方法：`get_stock_data`、`get_stock_minute_data`、`get_multiple_stocks_data`、`get_industry_classification`、`get_market_cap_data`、`incremental_update`、`get_cache_stats`、`cleanup_cache`、`clear_cache`。
- `_preprocess_data` 负责列名标准化与 `tradable_mask` 构建，规则见 [[可交易性掩码优先]]，缓存机制见 [[缓存键与TTL失效]]。
