---
标题: FactorNeutralizationService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_neutralization_service.py]
---

## 是什么

`backend/services/factor_neutralization_service.py` 中的因子中性化服务，用线性回归残差剥离市值与行业的影响。

## 关键事实

- 支持市值、行业与两者联合三种中性化；联合路径先统一过滤再构造特征矩阵，避免过滤顺序造成对齐错位。
- 行业样本量有下限，行业数不足时原样返回并告警。
- 结果保持原索引，被过滤的位置置为缺失值，不就地修改入参。
- 行业分类统一委托 [[DataService]]，取不到时归为未知。
- 方法论见 [[回归残差中性化]]。
