---
标题: FactorImportService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_import_service.py]
---

## 是什么

`backend/services/factor_import_service.py` 第 17 行的外部因子导入服务，把 CSV 或 DataFrame 中的日期与因子值登记为平台因子。

## 关键事实

- 两个导入入口：`import_from_csv(csv_file_path, ...)` 与 `import_from_dataframe(df, ...)`，参数结构一致。
- 默认列名：日期列 `date`、因子值列 `factor_value`；缺列、空表或名称重复都会抛错并返回失败结果。
- 默认分类为 `导入`，因子来源固定写为 `user`，描述为空时自动填文件名或固定说明。
- 成功返回 `success`、`factor_id`、`factor_name`、`row_count`，CSV 入口额外返回 `date_range`。
- `validate_csv_format` 返回 `valid`、`errors`、`warnings` 与 `info` 四部分，用于上传前检查。
- `get_import_template` 提供含日期与因子值两列的模板。
- 因子代码由 `_generate_import_code` 生成，按日期与因子列拼装为可计算表达式。
- 导入后的因子与挖掘产物一样，可继续走 [[FactorValidationService]] 与版本管理流程。
