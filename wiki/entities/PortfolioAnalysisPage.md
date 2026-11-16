---
标题: PortfolioAnalysisPage
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/pages/PortfolioAnalysis.tsx]
---

## 是什么

`frontend/react-antd/src/pages/PortfolioAnalysis.tsx` 中的组合分析页面，提供多因子权重优化与多种权重方法的横向对比。

## 关键事实

- 权重优化调用 `api.optimizeWeights`，方法对比调用 `api.compareWeightMethods`。
- 页面分「权重优化」与「方法对比」两个页签：前者至少选两个因子、默认最大夏普，后者同时选因子与方法。
- 对比结果同时渲染 IC 与 IR 两张图，并输出指标表格与排名。
- 优化结果可生成组合因子代码，并通过 `api.createFactor` 保存为自定义因子。
- 后端权重计算见 [[WeightOptimizer]]，接口封装来自 [[ApiClient]]。
