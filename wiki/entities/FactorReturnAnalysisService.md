---
标题: FactorReturnAnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_return_analysis_service.py]
---

## 是什么

`backend/services/factor_return_analysis_service.py` 中的因子收益分析服务，衡量因子分组后的收益结构、多空表现与换手特征。

## 关键事实

- 行为由配置对象控制，分组数、前向期、最小样本与自助法次数均有默认值。
- 按日逐截面分组，输出各组平均收益、组间差异检验与单调性检验。
- 单调性以相邻组方向一致的比例作为门槛。
- 夏普与回撤等风险指标复用 [[RiskMetrics]] 的口径，不另起一套算法。
- 分层与单调性判定见 [[横截面分层与单调性]]。
