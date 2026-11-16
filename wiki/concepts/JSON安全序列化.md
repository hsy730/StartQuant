---
标题: JSON安全序列化
类型: 概念
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/utils/serialization.py]
---

## 一句话定义

NumPy 与 Pandas 的值在返回前统一转成 JSON 安全类型，NaN 与无穷一律转为 null。

## 核心要点

- `safe_numeric_value` 把 `None`、NaN、Inf 统一转为 `None`，非数值类型回落到默认值。
- `sanitize_dict` 递归处理 dict、list 与 tuple，覆盖 ndarray、Series、Timestamp 等容器与标量。
- `pd.Timedelta` 显式转为 `None`，避免时间差对象无法序列化。
- 未知类型先尝试 `float()`，失败则返回 `None`，不向上抛异常。
- 对外始终给 null 而不是 0.0，与 [[安全除法]] 的缺省语义一致。

## 适用场景 / 边界

适用于路由与服务返回值、缓存写入前的清洗；不适用于需要保留 NaN 语义的内部数值计算。
