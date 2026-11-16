---
标题: GeneratedFactorRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/generated_factor_repository.py]
---

## 是什么

`backend/repositories/generated_factor_repository.py` 中的生成因子表数据访问类，负责候选因子的查询、状态标记与序列化。

## 关键事实

- 支持按表达式查询，用于写入前去重。
- 可按有效性标记与保存标记过滤，结果按创建时间倒序。
- 写保存标记时同步回写对应因子名，记录不存在时返回失败。
- 序列化输出表达式、生成方式、绩效指标、状态标记与复杂度等概览信息。
- 状态语义见 [[生成因子去重与保存标记]]，入库门控由 [[FactorService]] 承担。
