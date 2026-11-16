---
标题: AnalysisService
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/services/analysis_service.py]
---

## 是什么

`backend/services/analysis_service.py` 中的整体分析服务，把因子计算、预处理、IC/IR、alphalens、未来函数检测与 SHAP 串成一次分析。

## 关键事实

- 入口 `analyze(stock_codes, factor_names, start_date, end_date, use_cache=True, rolling_window=252)`。
- 分析前固定使用一套预处理配置：MAD 去极值、市值中性化、行业中性化、zscore 标准化；股票数大于 1 时启用横截面模式。
- 多股票预处理时股票数大于 3 启用并行。
- 缓存键包含因子代码哈希，机制见 [[因子版本哈希缓存失效]]。
- 股票数不少于 2 时运行 alphalens 分析，股票数大于 30 时跳过行业分组。
- 未来函数检测会先补 `future_return_1`（收盘价一阶差分后前移一行）再逐因子检测。
- SHAP 不可用时结果中写入 `{"error": "SHAP not available"}`，不中断流程。
- IC/IR 计算复用统一横截面口径，见 [[横截面SpearmanIC]]。
- 另提供加权 IC、SHAP 计算、报告生成与报告导出方法。
- 预处理实现复用 [[FactorPreprocessingPipeline]]。
