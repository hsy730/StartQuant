---
标题: SerializationUtils
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/utils/serialization.py]
---

## 是什么

`backend/utils/serialization.py` 提供的 JSON 安全序列化工具函数集。

## 关键事实

- `safe_numeric_value(value, default=None)` 返回 float 或 `None`，NaN 与 Inf 归为 `None`。
- `sanitize_dict` 递归处理 dict、list、tuple、ndarray 与 Series。
- `np.bool_` 转 bool，`np.integer` 转 int，`np.floating` 转 float。
- `pd.Timestamp` 转字符串，`pd.Timedelta` 转 `None`。
- 未知类型先尝试 float 转换，失败返回 `None`。
- 与 [[安全除法]] 共同维持不可计算即 null 的对外语义，规则见 [[JSON安全序列化]]。
