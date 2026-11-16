---
标题: FactorValidationService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_validation_service.py, backend/constants.py]
---

## 是什么

`backend/services/factor_validation_service.py` 第 38 行的因子验证服务，用六项检查加未来函数检测给因子质量打分，并给出是否通过。

## 关键事实

- 构造参数默认值：`ic_threshold=IC_PASS_THRESHOLD`（0.02）、`ir_threshold=IR_PASS_THRESHOLD`（0.5）、`turnover_threshold=0.5`、`max_correlation=0.8`、`ic_type="time_series"`。
- `validate_factor` 依次执行 IC、RankIC、IR、换手率、稳定性、相关性六项验证；稳定性检查委托 [[FactorStabilityService]]，相关性检查以已入库因子为参照。再由 [[LookaheadBiasDetector]] 做未来函数检测，判定口径见 [[前视偏差判定]]。
- `overall_passed` 要求六项基础检查全部通过，且未来函数风险等级属于 `safe`、`low` 或缺失；高风险或严重风险直接不通过。
- 综合得分上限 100：IC 与 RankIC 各最多 25 分（绝对值 × 300 封顶），IR 最多 20 分，稳定性 15 分，换手率最多 15 分（`15 - turnover × 30`）。
- 未来函数按风险等级乘性扣分：高风险/严重按 `1 - risk_score/100` 缩放，中等风险乘 0.5，低风险乘 0.9。
- `batch_validate` 按顺序验证多个因子，已通过的因子会加入 `existing_factors`，参与后续因子的相关性检查。
- 时序 IC 模式固定追加一条告警："时序IC仅评估择时能力，建议使用横截面IC"。
- 模块底部导出全局实例 `factor_validation_service`。
