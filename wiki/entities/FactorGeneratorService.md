---
标题: FactorGeneratorService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_generator_service.py]
---

## 是什么

`backend/services/factor_generator_service.py` 第 17 行的组合因子生成器，按运算符、统计函数与技术指标批量产出候选表达式并做首轮预筛。

## 关键事实

- 内置三张能力表：运算符 6 种、统计函数 16 种、技术指标 10 种（SMA、EMA、RSI、MACD、BBANDS、STOCH、ADX、CCI、ATR、VOLATILITY）。
- `generate_binary_combinations(base_factors, max_depth=3, max_combinations=100)` 生成深度 1–3 的二元与嵌套组合，除法写成 `safe_divide(..., default=np.nan)`。
- `generate_statistical_combinations(base_factors, window_sizes=[5, 10, 20, 60], max_combinations=50)` 生成滚动统计表达式。
- 另有 `generate_indicator_combinations` 与 `generate_hybrid_factors` 分别覆盖技术指标与混合模式。
- `preselect_factors` 默认 `ic_threshold=IC_PASS_THRESHOLD`、`ir_threshold=IR_PASS_THRESHOLD`、`min_valid_ratio=0.7`。
- 预筛用 20 日滚动 Spearman IC 计算 IR；IC 标准差接近 0 且均值非 0 时把 IR 记为无穷大。
- `calculate_factor_metrics` 在有效点少于 10 个时返回 `valid=False`。
- 验证与入库环节见 [[FactorValidationService]] 与 [[FactorAnalyzer]]。
