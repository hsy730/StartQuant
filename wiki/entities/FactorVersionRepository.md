---
标题: FactorVersionRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/factor_version_repository.py]
---

## 是什么

`backend/repositories/factor_version_repository.py` 中的因子版本仓储类，负责 `factor_versions` 表的读写与单当前版本约束的维护。

## 关键事实

- 以因子为主线，支持按因子、版本号与当前标记三种方式检索版本。
- 新建版本前先清除该因子其他版本的当前标记。
- 版本列表按创建时间倒序返回，并支持按因子批量删除与数量统计。
- 单当前约束见 [[当前版本唯一性]]，回滚流程由 [[FactorVersionService]] 编排。
