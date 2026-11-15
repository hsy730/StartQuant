---
标题: LookaheadBiasDetector
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/lookahead_bias_detector.py]
---

## 是什么

`backend/services/lookahead_bias_detector.py` 中的未来函数统计检测器，通过多维度统计特征识别因子中的未来信息泄漏。

## 关键事实

- 类定义在该文件第 87 行；结果类型 `LookaheadBiasDetectionResult` 含 has_bias、risk_level、risk_score、checks、summary、recommendations。
- 风险等级枚举为 safe / low / medium / high / critical 五档。
- 默认阈值示例：`ic_abs_max=0.15`、`ir_max=3.0`、`ic_positive_ratio_max=0.85`、`autocorr_lag1_max=0.99`、`quantile_spread_daily_max=0.05`、`sharpe_max=10.0`、`win_rate_max=0.95`。
- 构造函数支持 `thresholds` 覆盖与 `strict_mode`；严格模式将 IC、IR、排名相关阈值乘以 0.6。
- 对外入口 `detect(factor_values, return_values=None, factor_name="factor", extra_context=None)` 与 `detect_cross_sectional(...)`；有效样本不足 20 条时跳过检测并走数据不足分支。
- 判定逻辑见 [[前视偏差判定]]。
