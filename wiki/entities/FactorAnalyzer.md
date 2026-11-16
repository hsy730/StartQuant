---
标题: FactorAnalyzer
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_analyzer.py]
---

## 是什么

`backend/services/factor_analyzer.py` 第 36 行的因子分析器，把挖掘结果逐条验证后写入 `generated_factors` 表，并组装前端需要的汇总结果。

## 关键事实

- 构造函数接收因子计算器、原始数据与收益率序列三个依赖。
- `analyze(mining_result, source="unknown")` 遍历候选，单候选验证超时 30 秒；超时后线程池以 `wait=False` 关闭，避免阻塞整批。
- 输出包含 `discovered_factors`、`total_discovered`、`valid_factors`、`best_fitness` 与 `avg_fitness`，并透传算法特有元数据。
- `_unified_validate` 优先用预计算值，有效点少于 10 个返回空结果；IC/IR 取绝对值，IR 截断到 [-5, 5]。
- 当验证分数、IC、IR 全部接近 0 时，回退到算法原始适应度，并把 `overall_passed` 置为 False。
- `_store_factor` 以表达式查重：已存在则更新指标与来源，不存在则新建且 `is_saved=False`。
- 数据库写入失败时调用 `_fallback_analyze`，仍返回内存中的候选结果，不让整次挖掘失败。
- 验证标准依赖 [[FactorValidationService]]，入库后的版本留痕由 [[FactorVersionService]] 负责。
