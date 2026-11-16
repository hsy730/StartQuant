---
标题: SerializationUtils
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/utils/serialization.py]
---

## 是什么

`backend/utils/serialization.py` 中的 JSON 安全序列化工具函数集，负责把数值与容器对象清洗成可被 JSON 序列化的形式。

## 关键事实

- 提供标量清洗与容器递归清洗两类入口，NaN、Inf 与转换失败的值统一归为 null。
- 递归范围覆盖 dict、list、tuple、ndarray 与 Series，并识别 numpy 标量类型与 pandas 时间类型。
- 时间戳输出字符串，时间差与未知类型无法转换时返回 null。
- 与 [[安全除法]] 共同维持「不可计算即 null」的对外语义。
