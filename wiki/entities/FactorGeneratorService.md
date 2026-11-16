---
标题: FactorGeneratorService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_generator_service.py]
---

## 是什么

`backend/services/factor_generator_service.py` 中的组合因子生成器，用运算符、统计函数与技术指标批量拼装候选表达式并做首轮预筛。

## 关键事实

- 按二元组合、滚动统计、技术指标与混合模式四类策略产出候选，组合深度与数量均有上限。
- 除法一律走安全除法，避免候选表达式出现除零。
- 预筛以滚动 IC 计算 IR，并按 IC 阈值、IR 阈值与有效值比例过滤。
- 有效样本过少的候选直接判为无效，不进入后续评估。
- 生成的候选交由 [[FactorAnalyzer]] 评估、[[FactorValidationService]] 验证。
