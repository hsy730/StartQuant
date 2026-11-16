---
标题: EnhancedAnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/enhanced_analysis_service.py]
---

## 是什么

`backend/services/enhanced_analysis_service.py` 中的增强分析服务，提供 IC 显著性与多模块聚合分析。

## 关键事实

- 无内部状态的轻量服务，构造不持有数据。
- `calculate_ic_significance(factor_values, return_values, confidence_level=0.95)` 给出 IC 显著性结果。
- 有日期索引时先按日计算横截面 IC 再做检验，规则见 [[IC显著性单样本t检验]]。
- 横截面 IC 委托统一入口，参数为 `min_stocks=5`、`min_dates=2`、`method="spearman"`。
- `analyze_enhanced` 提供中性化、稳定性与摘要三个开关，摘要默认开启，其余默认关闭。
- 结果字段包括 IC 均值、t 统计量、p 值、显著性标记、样本数与截面数。
