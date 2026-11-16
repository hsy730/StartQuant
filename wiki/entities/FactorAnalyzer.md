---
标题: FactorAnalyzer
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_analyzer.py]
---

## 是什么

`backend/services/factor_analyzer.py` 中的因子分析器，把挖掘结果逐条验证后写入 `generated_factors` 表，并组装前端需要的汇总结果。

## 关键事实

- 构造时注入因子计算器、原始数据与收益率序列三个依赖。
- 逐候选验证带单条超时保护，超时后以非阻塞方式关闭线程池，避免拖住整批。
- 统一验证优先使用预计算值，有效点不足时返回空结果，IC/IR 取绝对值并做区间截断。
- 验证分数与 IC/IR 同时接近零时回退到算法原始适应度，并判定为未通过。
- 入库按表达式查重，写入失败时降级返回内存结果；验证标准依赖 [[FactorValidationService]]，版本留痕由 [[FactorVersionService]] 负责。
