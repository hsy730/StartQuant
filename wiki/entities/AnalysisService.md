---
标题: AnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/analysis_service.py]
---

## 是什么

`backend/services/analysis_service.py` 中的整体分析服务，把因子计算、预处理、IC/IR、alphalens、前视偏差检测与 SHAP 串成一次完整分析。

## 关键事实

- 以股票列表、因子列表与日期区间为入口，可选用缓存并指定滚动窗口长度。
- 预处理固定采用去极值、市值中性化、行业中性化与标准化组合，多股票时启用横截面模式与并行。
- alphalens 分析只在多股票场景运行，股票数过多时跳过行业分组。
- 前视偏差检测先构造下一期收益列再逐因子检测；SHAP 不可用时写入错误字段而不中断流程。
- 另提供加权 IC、报告生成与导出，预处理实现复用 [[FactorPreprocessingPipeline]]。
