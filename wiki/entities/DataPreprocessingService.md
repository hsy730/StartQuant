---
标题: DataPreprocessingService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/data_preprocessing_service.py]
---

## 是什么

`backend/services/data_preprocessing_service.py` 中的数据集级清洗服务，负责异常值、增量合并与数据质量校验。

## 关键事实

- `detect_outliers` 支持 `std` 与 `iqr` 两种口径，规则见 [[异常值检测双口径与处理方法]]。
- `handle_outliers` 支持 `clip`、`remove`、`replace`、`replace_median` 四种处理方式。
- `incremental_update` 以日期为键做并集合并，重复日期保留新数据，返回按索引排序的结果。
- `validate_data_quality` 依次检查必需列、空值、无穷值与空表，返回布尔值与消息。
- `standardize_columns` 默认把中文列名映射为英文，设置日期索引并把数值列转为数值类型。
- `fill_missing_values` 支持 `ffill`、`bfill`、`interpolate` 与 `mean` 四种填充方式。
