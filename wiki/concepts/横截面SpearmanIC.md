---
标题: 横截面SpearmanIC
类型: 概念
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/utils/ic_calculator.py, .trae/rules/code_smell_prevention.md]
---

## 一句话定义

IC 必须在每个交易日内做横截面 Spearman 相关，再对日序列取均值，而非池化全样本求一次相关。

## 核心要点

- 统一入口为 `backend/utils/ic_calculator.py` 的 `calculate_ic(factor, returns, method="spearman", min_samples=10)`，默认即 Spearman。
- `calculate_rank_ic` 是 Spearman 口径的快捷方式；`calculate_rolling_ic(window=20)` 默认 Spearman；面板数据走 `calculate_cross_sectional_ic_from_panel`。
- 计算前用 `factor.notna() & returns.notna()` 过滤，有效样本少于 `min_samples` 返回 `None`。
- IR 由日 IC 的均值与标准差得到，不可计算时返回 `None`，不得写 `0.0`（见 [[安全除法]]）。
- 禁止用 `.corr()` 的默认 Pearson 代替，也禁止先池化全量数据再求相关。

## 适用场景 / 边界

- 适用：单因子有效性检验、因子衰减分析、IC/IR 加权。
- 边界：需要 Pearson 口径时必须显式传 `method="pearson"`；传入含 NaN 的数据会污染结果，必须先清洗；样本不足时返回 `None` 而非 0。
