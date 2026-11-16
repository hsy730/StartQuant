---
标题: PositionAnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/position_analysis_service.py]
---

## 是什么

`backend/services/position_analysis_service.py` 中的持仓分析服务，从仓位序列计算统计、变化、时长与集中度。

## 关键事实

- 提供 `analyze_positions`、`analyze_position_history`、`calculate_position_concentration` 三个入口。
- `analyze_positions` 默认初始资金 1000000，先取仓位绝对值再算均值、最大值与最小值。
- 统计包含零仓位比例与满仓比例，满仓定义为绝对权重不低于 0.9。
- 持仓变化用 `diff().abs()` 的均值与最大值衡量。
- 连续持仓区间用于统计持仓段数与总持仓天数，平均持仓为两者之商。
- 换手为权重变化绝对值之和除以 2，口径见 [[持仓时长与换手口径]]。
- 持仓价值按仓位乘初始资金换算，集中度实现见 [[单票集中度占位实现]]。
