---
标题: FactorPreprocessingPipeline
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_preprocessing_pipeline.py]
---

## 是什么

`backend/services/factor_preprocessing_pipeline.py` 中的因子预处理管道，按固定顺序完成缺失值处理、去极值、中性化与标准化。

## 关键事实

- 行为由配置对象驱动，默认采用 MAD 去极值、Z-score 标准化与联合中性化。
- 内部顺序固定：缺失值处理、无穷值清理、去极值、中性化、标准化。
- 同时提供单因子、数据框与多股票三种粒度的入口，均返回处理结果与统计信息。
- 样本不足最小要求时原样返回并标记跳过，同时记录告警。
- 顺序与规则见 [[因子预处理四步顺序]]，下游消费见 [[因子研究链路]]。
