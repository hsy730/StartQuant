---
标题: ComprehensiveScoringService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/comprehensive_scoring_service.py, backend/constants.py]
---

## 是什么

`backend/services/comprehensive_scoring_service.py` 中的综合评分服务，把因子、策略、组合的多维指标压成百分制总分并给出评级。

## 关键事实

- 三个评分入口分别对应因子、策略与组合，各自持有独立的默认权重表。
- 评分上限常量取自公共常量模块，总分统一截断到百分制区间并保留两位小数。
- 评级由总分映射得到，权重可被调用方覆盖。
- 附带滑点敏感度分析与成本建议生成能力。
- 评分映射与截断规则见 [[评分归一与截断]]。
