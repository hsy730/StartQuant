---
标题: WeightedICService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/weighted_ic_service.py, backend/constants.py]
---

## 是什么

`backend/services/weighted_ic_service.py` 第 63 行的加权 IC 服务，把多个因子的 IC 序列按不同权重方案合成并给出因子重要性。

## 关键事实

- 加权方法枚举 `WeightingMethod` 含五种：`equal_weight`、`ir_weight`、`abs_ic_weight`、`decay_weight`、`optimal_weight`，默认 `ir_weight`。
- 配置类 `WeightedICConfig` 默认 `decay_half_life=60`、`min_observations=20`、`lookback_window=252`、`correlation_adjustment=True`、`risk_aversion=1.0`。
- `calculate_weighted_ic(factor_ic_dict, return_series=None, factor_correlation_matrix=None)` 输出加权 IC 的均值、标准差、IR、正比例与逐日序列。
- `_calculate_weights` 中 IR 加权取 `max(ir, 0)`，全负时记录告警并回退等权；衰减加权按 `0.5 ** (1 / decay_half_life)` 生成指数权重。
- `_adjust_for_correlation` 对绝对相关超过 `HIGH_CORRELATION_THRESHOLD=0.7` 的因子对折减权重较小的一方，折减系数为 `1 - (corr - 0.7) × 0.5`，折减后重新归一化。
- `calculate_factor_importance` 综合 IC、IR、稳定性与独特性给出排名与文字解读；`_calculate_stability_score` 衡量 IC 序列稳定性。
- 权重与折减规则见 [[加权IC与相关性折减]]，权重归一化统一走公共工具。
