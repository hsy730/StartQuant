---
标题: AlphalensAnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/alphalens_analysis_service.py]
---

## 是什么

`backend/services/alphalens_analysis_service.py` 中的因子分析服务，把 alphalens-reloaded 的 IC、收益与换手率分析包装成统一的字典输出。

## 关键事实

- 输入为「股票代码 → 因子值序列」与收盘价矩阵，先转成 alphalens 标准的日期 × 标的双层索引面板。
- 在 alphalens 原生只提供 Spearman 口径之外，另行补算 Pearson IC。
- 收益与换手率分析分别产出分位收益、多空价差、分位换手率与因子秩自相关。
- 单股票与因子值全为无效值属于降级场景，前者给告警、后者直接返回空结果。
- 模块底部导出全局单例，因子有效性分析优先调用它，单股票场景再回退到自实现。
