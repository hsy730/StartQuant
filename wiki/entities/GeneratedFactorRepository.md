---
标题: GeneratedFactorRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/generated_factor_repository.py]
---

## 是什么

`backend/repositories/generated_factor_repository.py` 中的生成因子表数据访问类。

## 关键事实

- `get_by_expression` 按表达式查询记录，用于写入前去重。
- `get_all_valid` 取 `is_valid` 为真的记录，按创建时间倒序。
- `get_all_unsaved` 取 `is_saved` 为假的记录，按创建时间倒序。
- `mark_saved(id, factor_name)` 置 `is_saved=True` 并写入 `factor_name`，记录不存在时返回 `False`。
- `delete` 按主键删除并提交，记录不存在时返回 `False`。
- `to_dict` 输出表达式、生成方式、IC/IR、换手、稳定性、验证分、两个状态标记、复杂度与附加信息。
- 状态语义见 [[生成因子去重与保存标记]]，入库门控由 [[FactorService]] 承担。
