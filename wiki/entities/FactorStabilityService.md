---
标题: FactorStabilityService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_stability_service.py, backend/constants.py]
---

## 是什么

`backend/services/factor_stability_service.py` 中的因子稳定性服务，从分布、时序、离散度与滚动四个角度判断因子表现是否可延续。

## 关键事实

- 判定窗口常量来自 [[全局常量模块]]，年度窗口是默认主口径。
- 分布稳定性按年度窗口切段比较，样本跨度不足下限时直接报错。
- 另有独立入口分别衡量时序稳定性、变异系数与多窗口滚动稳定性。
- 窗口取值规则见 [[稳定性判定窗口]]。
- 稳定性得分会进入综合评分，见 [[ComprehensiveScoringService]]。
