---
标题: FactorDetailPage
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/pages/FactorDetail.tsx]
---

## 是什么

`frontend/react-antd/src/pages/FactorDetail.tsx` 中的因子详情页，围绕单个因子组织计算、IC、暴露、有效性、归因、监测、分层收益与全貌报告等分析面板。

## 关键事实

- 通过 URL 查询参数接收因子标识，再调用 `api.getFactorDetail` 拉取详情。
- 股票池选项来自 `api.getStockPools` 与 `api.getStockPoolStocks`，行情来自 `api.getStockData`。
- 因子值由 `api.calculateFactor` 计算，IC 由 `api.calculateIC` 计算。
- 分析面板分别调用 `analyzeExposure`、`analyzeEffectiveness`、`analyzeAttribution`、`analyzeMonitoring`、`detectLookaheadBias`。
- 收益与报告面板调用 `analyzeQuantileReturns`、`analyzeCumulativeReturns`、`generateTearSheet`。
- 支持在详情页直接校验公式、更新、复制与删除因子。
- 内置 `PreprocessingConfigPanel`，预处理参数在详情页即可调整。
- 图表统一用 echarts 渲染，样式由独立 CSS 文件承载。
- 接口封装来自 [[ApiClient]]，预处理面板见 [[PreprocessingConfigPanel]]。
