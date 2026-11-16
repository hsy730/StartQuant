---
标题: FactorVersionService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_version_service.py, backend/models/factor_version.py]
---

## 是什么

`backend/services/factor_version_service.py` 中的因子版本服务，编排版本创建、历史查询、对比与回滚，模块底部导出全局实例 `factor_version_service`。

## 关键事实

- 通过 `FactorVersionRepository` 读写版本表，通过 [[FactorRepository]] 读写因子主表。
- 版本号由版本计数递增生成，以主版本加次版本的形式编码。
- 回滚先自动备份当前版本，再覆盖因子代码与描述，最后切换当前标记。
- 版本模型记录因子、版本号、代码、描述、变更原因、创建时间与当前标记。
- 规则说明见 [[版本快照与回滚]]，因子定义来源见 [[FormulaCompilerService]]。
