---
标题: ComprehensiveScoringService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/comprehensive_scoring_service.py, backend/constants.py]
---

## 是什么

`backend/services/comprehensive_scoring_service.py` 中的综合评分服务，把因子、策略、组合的多维指标压成 0–100 总分并给出评级。

## 关键事实

- 类定义在该文件第 16 行，`MAX_SCORE = 100.0` 来自 `backend/constants.py`。
- 三个入口：`score_factor`、`score_strategy`、`score_portfolio`，各自有独立默认权重。
- 因子评分默认权重为 ic 0.35、ir 0.30、stability 0.20、turnover 0.15；策略评分默认权重为 return 0.3、risk 0.25、efficiency 0.2、stability 0.15、cost 0.1。
- 总分统一以 `max(0.0, min(round(total, 2), MAX_SCORE))` 截断，`_get_grade` 依据总分给出评级。
- 附带 `analyze_slippage_sensitivity` 与滑点建议生成方法，用于评估成本敏感度。
- 评分映射与截断规则见 [[评分归一与截断]]，汇报依赖 [[因子研究链路]] 中的 IC/IR 结果。
