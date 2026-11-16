---
标题: FactorExposureService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/factor_exposure_service.py]
---

## 是什么

`backend/services/factor_exposure_service.py` 中的因子暴露度分析服务，输出单票时间序列状态与整体分布。

## 关键事实

- 提供四个入口：`calculate_exposure_metrics`、`calculate_exposure_by_stock`、`calculate_percentile_distribution`、`calculate_rolling_exposure`。
- `calculate_exposure_metrics` 默认 `window=20`，返回当前值、时间序列分位、滚动均值与标准差、变异系数、分布统计与直方图。
- 代表样本由 `find_longest_stock` 选取，规则见 [[代表样本选取最长有效序列]]。
- 分位序列用 `rank(pct=True) * 100` 一次计算，避免逐点查询。
- 分布统计合并所有股票的所有因子值，分位点覆盖 p1 到 p99，直方图 50 桶。
- 变异系数与分位口径见 [[时间序列分位与变异系数]]。
