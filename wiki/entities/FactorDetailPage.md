---
标题: FactorDetailPage
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/pages/FactorDetail.tsx]
---

## 是什么

`frontend/react-antd/src/pages/FactorDetail.tsx` 中的因子详情页，是单个因子的分析工作台，把计算、IC、暴露、有效性、归因、监测、分层收益与全貌报告汇到同一页面。

## 关键事实

- 因子标识来自 URL 查询参数，页面据此拉取详情并驱动各分析面板。
- 各面板独立请求后端、彼此不共享中间结果，图表统一用 echarts 渲染。
- 页面内可直接完成公式校验、因子更新、复制与删除等管理动作。
- 内嵌预处理面板，预处理参数在详情页即可调整，见 [[PreprocessingConfigPanel]]。
- 所有后端调用经统一接口封装 [[ApiClient]]。
