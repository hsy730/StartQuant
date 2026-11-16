---
标题: FactorNeutralizationService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_neutralization_service.py]
---

## 是什么

`backend/services/factor_neutralization_service.py` 第 29 行的因子中性化服务，提供市值、行业与联合三种线性回归残差中性化。

## 关键事实

- 类常量 `MIN_SAMPLES=10`；模块级常量 `MIN_INDUSTRY_SIZE=5` 控制最小行业样本量。
- `neutralize_market_cap(df, factor_name, market_cap_column="market_cap")` 用 `sklearn` 的 `LinearRegression` 对 `log(市值)` 回归取残差。
- `neutralize_industry(df, factor_name, industry_column="industry")` 用行业哑变量回归；少于 2 个行业时原样返回并告警。
- `neutralize_both` 先完成市值过滤与小行业过滤，再统一构造"行业哑变量 + log(市值)"特征矩阵，避免过滤顺序造成对齐错误。
- 三种方法都通过 `_build_result_series` 保持原索引，被过滤位置填 `NaN`，不就地修改入参。
- `get_industry_classification` 与 `add_industry_classification` 委托 [[DataService]] 获取行业分类，失败时统一标记为 `unknown`。
- 具体方法论见 [[回归残差中性化]]；因子被中性化后仍走同一套 [[单一真相源]] 的评估入口。
