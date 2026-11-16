---
标题: 多重共线性VIF筛查
类型: 概念
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_correlation_service.py, backend/constants.py]
---

## 一句话定义

用方差膨胀因子衡量一个因子能被其余因子线性解释的程度，超过阈值即视为共线性风险。

## 核心要点

- VIF 计算依赖 `statsmodels` 的 `variance_inflation_factor`，有效样本少于 30 条时直接返回"样本不足"。
- 分档固定：大于 10 为 `severe`，5 到 10 为 `high`，2.5 到 5 为 `moderate`，小于等于 2.5 为 `low`；非有限值按 `severe` 处理。
- 只要最大 VIF 超过 5，就将 `has_issue` 置为真，并分别输出"严重共线性(VIF>10)"与"高度共线性(5<VIF≤10)"两类告警。
- 两因子重叠用相关系数判定：双链 [[横截面SpearmanIC]] 之外的绝对相关超过 `HIGH_CORRELATION_THRESHOLD=0.7` 时建议正交化或移除。
- 分散化另用 RMS 相关性度量，取相关矩阵上三角元素的均方根，`diversification_score = (1 - RMS) × 100`。

## 适用场景 / 边界

- 适用：多因子模型选因子、合成因子前的冗余检查、因子池扩容。
- 边界：VIF 只反映线性共线性，对非线性依赖不敏感；`statsmodels` 未安装时该环节降级为不可用，不能据此认定因子无共线性问题。
