---
标题: FactorAttributionService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_attribution_service.py]
---

## 是什么

`backend/services/factor_attribution_service.py` 中的因子归因服务，把收益拆成因子贡献与 alpha/beta 两部分。

## 关键事实

- 类定义在该文件第 25 行，主要入口 `analyze_attribution(...)`，内部由 `_calculate_contribution`、`_decompose_alpha_beta`、`_decompose_return` 完成拆解。
- `_calculate_contribution` 用 `close.shift(-1) / close − 1` 构造未来收益，再走 `calculate_cross_sectional_ic_from_panel`（Spearman、`min_stocks=5`）。
- 分组阈值为逐日 `quantile(0.7)` 与 `quantile(0.3)`，输出高暴露收益、低暴露收益、多空收益与贡献比例。
- 日 IC 少于 3 天时返回 `{"error": "数据不足以计算贡献度"}`；多于 1 天时用 `ttest_1samp` 给出 IC 显著性。
- 基准数据由 `_get_benchmark_data` 获取，alpha/beta 拆解依赖基准序列。
- 归因口径见 [[横截面分位归因]]，IC 口径见 [[横截面SpearmanIC]]。
