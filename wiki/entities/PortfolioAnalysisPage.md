---
标题: PortfolioAnalysisPage
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/pages/PortfolioAnalysis.tsx]
---

## 是什么

`frontend/react-antd/src/pages/PortfolioAnalysis.tsx` 中的组合分析页面，提供多因子权重优化与多种权重方法的横向对比两个页签。

## 关键事实

- 权重优化调用 `api.optimizeWeights`，方法对比调用 `api.compareWeightMethods`。
- 支持的方法与后端权重优化入口一致，默认最大夏普。
- 优化结果可转成组合因子代码，并通过 `api.createFactor` 保存。
- 对比结果同时渲染 IC 与 IR 两张图，并附指标表格。
- 交互组织见 [[权重优化与方法对比双页签]]，接口封装来自 [[ApiClient]]。
- 权重计算后端见 [[WeightOptimizer]]。
