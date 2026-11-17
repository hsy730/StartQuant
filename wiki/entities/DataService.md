---
标题: DataService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-17
来源: [backend/services/data_service.py]
---

## 是什么

`backend/services/data_service.py` 中的行情数据服务，负责股票数据获取、缓存与可交易性掩码构建，设计口径为 Mask-First。

## 关键事实

- 数据来源为 akshare，日线与分钟线均取前复权；缓存以代码与日期区间哈希为键，TTL 取自配置。
- 涨跌幅限制按主板、创业板、科创板与北交所分档，写死在服务内部。
- 通过一个掩码配置对象控制涨停、跌停、停牌、次新与流动性等过滤条件。
- 对外覆盖个股与批量取数、分钟数据、行业分类、市值数据、增量更新与缓存运维。
- 预处理负责列名标准化与掩码构建，规则见 [[可交易性掩码优先]]，缓存机制见 [[缓存键与TTL失效]]。
- 所有 akshare 网络调用经共享锁 `backend/utils/akshare_lock.AKSHARE_LOCK` 串行执行：akshare 的 JS 签名引擎 `py_mini_racer` 非线程安全，多线程并发取数会在 `mini_racer.dll` 内部硬崩溃（exit `0x80000003`，Python 无法捕获；实测 160 只股票冷缓存、15 线程 3 次崩 2 次）。锁只覆盖网络调用本身，缓存命中路径不经过锁，因此 `get_multiple_stocks_data` 的并行解码仍然有效，冷缓存则退化为逐只拉取。本服务内的入口：日线、分钟线、行业列表/成分股、个股信息；同一把锁也被股票池路由的 `_call_akshare_with_timeout`、`factor_service` 的 akshare 健康检查与 `FactorAttributionService` 的基准指数取数复用（RLock，允许同线程嵌套）。
