---
标题: AlphalensAnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/alphalens_analysis_service.py]
---

## 是什么

`backend/services/alphalens_analysis_service.py` 第 33 行的因子分析服务，把 alphalens-reloaded 的 IC、收益与换手率分析包装成统一字典输出。

## 关键事实

- `prepare_factor_data` 接收 `{股票代码: 因子值 Series}` 与收盘价矩阵，默认 `periods=(1, 5, 10)`、`quantiles=5`、`max_loss=0.50`，产出 MultiIndex 为 `date × asset` 的 alphalens 标准数据。
- 只有单只股票时给出"横截面IC分析需要多只股票数据"告警；所有因子值均为 NaN 或 Inf 时返回 `None`。
- `analyze_ic` 输出 Spearman Rank IC 与 Pearson IC 两套统计；Pearson 由服务自行按日计算，因为 alphalens 默认只提供 Spearman。
- `analyze_returns` 输出分位收益、因子收益、`factor_alpha_beta` 与多空价差（spread）。
- `analyze_turnover` 输出分位换手率与因子秩自相关，两者均来自 alphalens 的 `quantile_turnover` 与 `factor_rank_autocorrelation`。
- `full_analysis` 按"准备数据 → IC → 收益 → 换手率"的顺序串起全部分析，并在元数据中记录股票数、周期、分位与日期范围。
- 模块底部导出全局实例 `alphalens_analysis_service`；因子有效性分析服务通过其 `full_analysis` 获取 alphalens 结果，单股票场景再回退到自实现。
