---
标题: FactorImportService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_import_service.py]
---

## 是什么

`backend/services/factor_import_service.py` 中的外部因子导入服务，把 CSV 或 DataFrame 中的日期与因子值登记成平台因子。

## 关键事实

- 提供文件与内存数据两条导入通道，内部共用同一套列解析逻辑。
- 依赖约定的日期列与因子值列；缺列、空表或名称重复都会失败并返回失败结果。
- 导入因子统一标记为用户来源的“导入”分类，描述缺失时自动补默认文案。
- 另提供上传前格式预检与导入模板获取两项辅助能力。
- 导入后的因子继续走常规评估与版本管理流程，见 [[FactorValidationService]]。
