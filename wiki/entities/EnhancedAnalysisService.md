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

- 无内部状态的轻量服务，构造时不持有任何数据。
- 显著性入口接收因子值与收益序列，可指定置信水平。
- 有日期索引时先按日计算横截面 IC 再做单样本检验，规则见 [[IC显著性单样本t检验]]。
- 横截面 IC 委托统一入口计算，并带有最小股票数与最小截面数约束。
- 聚合入口以中性化、稳定性与摘要三个开关控制子分析是否执行，结果包含 IC 统计量、显著性标记与样本规模。
