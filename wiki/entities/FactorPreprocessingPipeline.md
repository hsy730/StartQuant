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

- 类定义在 `backend/services/factor_preprocessing_pipeline.py` 第 81 行，参数由 `PreprocessingConfig` dataclass 提供。
- 默认配置：MAD 去极值、`winsorize_n_sigma=3.0`、Z-score 标准化、联合中性化开启、缺失值填 0、`min_samples=10`、横截面模式。
- 对外方法：`process_single_factor`、`process_factor_dataframe`、`process_multi_stock_factors`，返回（处理后数据, 处理统计字典）。
- 内部执行顺序：缺失值处理 → 无限值清理 → 去极值 → 中性化（默认 `_neutralize_joint` 联合回归）→ 标准化。
- 样本数低于 `min_samples` 时原样返回并标记 `skipped`，同时记录 warning 日志。
- 处理规则来源见 [[因子预处理四步顺序]]，下游消费者见 [[因子研究链路]]。
