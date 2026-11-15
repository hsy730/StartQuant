---
标题: FactorVersionService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_version_service.py, backend/models/factor_version.py]
---

## 是什么

`backend/services/factor_version_service.py` 中的因子版本服务，负责版本创建、历史查询、对比与回滚，模块底部导出全局实例 `factor_version_service`。

## 关键事实

- 类定义在该文件第 12 行，通过 `FactorVersionRepository` 读写 `factor_versions` 表，通过 `FactorRepository` 读写 `factors` 表。
- 对外方法：`create_version`、`rollback_to_version`、`get_version_history`、`get_current_version_info`、`delete_version`、`compare_versions`。
- 版本号由 `_generate_version_code` 生成，规则为 `v{count // 10 + 1}.{count % 10}`。
- 回滚流程先备份当前版本（变更原因标注"回滚前自动备份"），再覆盖 `factors` 表的 code 与 description，最后把目标版本设为 `is_current`。
- `FactorVersionModel` 字段为 id、factor_id、version_code、code、description、change_reason、created_at、is_current。
- 规则说明见 [[版本快照与回滚]]，因子定义来源见 [[FormulaCompilerService]]。
