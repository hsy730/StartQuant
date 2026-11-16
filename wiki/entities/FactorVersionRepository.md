---
标题: FactorVersionRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/factor_version_repository.py]
---

## 是什么

`backend/repositories/factor_version_repository.py` 中的因子版本仓储类。

## 关键事实

- `create` 新建版本并置 `is_current=True`，创建前先把该因子其他版本全部置为非当前。
- `get_by_factor_id` 按创建时间倒序返回该因子的全部版本。
- `get_current_version` 以 `factor_id` 与 `is_current` 两个条件取当前版本。
- `get_by_version_code` 按因子与版本号定位。
- `set_current` 先调用内部方法清空旧标记，再置目标版本为当前。
- `delete_by_factor_id` 批量删除该因子全部版本并返回行数，`get_version_count` 返回版本数量。
- 单当前约束见 [[当前版本唯一性]]，回滚流程由 [[FactorVersionService]] 编排。
