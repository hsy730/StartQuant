---
标题: JSON字段入库前转字典
类型: 概念
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/backtest_repository.py]
---

## 一句话定义

净值曲线与分层收益在落库前先递归转成普通字典，避免 ORM 的 JSON 列序列化失败。

## 核心要点

- `equity_curve` 带 `to_dict` 方法时先转换成字典。
- `quantile_returns` 逐键处理：对象转 `to_dict`，字典再逐值转换。
- 转换后的结果写入回测结果表的 JSON 列。
- 未转换的 DataFrame 等对象不能直接写入 JSON 列。
- 对外响应时再由 [[JSON安全序列化]] 统一清洗 NaN 与 NumPy 类型。

## 适用场景 / 边界

只覆盖一层或两层带 `to_dict` 的对象；更深的嵌套结构需要调用方自行展开。
