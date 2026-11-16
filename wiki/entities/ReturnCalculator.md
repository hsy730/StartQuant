---
标题: ReturnCalculator
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/utils/return_calculator.py]
---

## 是什么

`backend/utils/return_calculator.py` 中的未来收益率计算函数，作为统一收益口径入口。

## 关键事实

- 唯一对外函数为 `calculate_future_return(df, price_column="close", period=1)`。
- 计算式为 `pct_change(period).shift(-period)`，即 t 日决策、t+N 日收益。
- 索引为 MultiIndex 时按 `level=1`（资产）分组后 `transform`，保留原索引结构。
- 单层索引直接对价格列计算。
- 分组口径见 [[面板按资产分组计算]]，产出的收益序列供 [[横截面SpearmanIC]] 使用。
