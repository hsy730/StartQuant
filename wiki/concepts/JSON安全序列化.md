---
标题: JSON安全序列化
类型: 概念
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/utils/serialization.py, backend/repositories/backtest_repository.py]
---

## 一句话定义

NumPy、Pandas 与 ORM JSON 列的值在输出或落库前统一转为安全类型，NaN 与无穷一律转为 null。

## 核心要点

- `safe_numeric_value` 把 `None`、NaN、Inf 统一转为 `None`，非数值类型回落到默认值。
- `sanitize_dict` 递归处理 dict、list 与 tuple，覆盖 ndarray、Series、Timestamp 等容器与标量。
- `pd.Timedelta` 显式转为 `None`；未知类型先尝试 `float()`，失败则返回 `None`，不向上抛异常。
- 落库前带 `to_dict` 的对象（净值曲线、分层收益）先递归转普通字典再写入 JSON 列，DataFrame 不能直接入库。
- 对外始终给 null 而不是 0.0，与 [[安全除法]] 的缺省语义一致。

## 适用场景 / 边界

适用于路由与服务返回值、缓存写入前的清洗，以及回测结果的 JSON 列落库；不适用于需要保留 NaN 语义的内部数值计算。
