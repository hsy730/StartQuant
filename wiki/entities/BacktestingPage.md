---
标题: BacktestingPage
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/pages/Backtesting.tsx]
---

## 是什么

`frontend/react-antd/src/pages/Backtesting.tsx` 中的回测页面，用表单配置单因子或多因子策略，提交后展示指标卡、图表与策略对比结果。

## 关键事实

- 表单费率与滑点以百分数输入，提交前除以 100 转成比例。
- 策略类型切换会重置不相关的因子字段，避免残留配置。
- 提交走 `api.runBacktest`，成功后把逐股票数据组装成图表入参。
- 指标区展示总收益、年化、夏普与最大回撤，涨用红色、跌用绿色。
- 回测结果图表由 [[BacktestCharts]] 渲染。
- 后端入口见 [[BacktestRouter]]，接口封装来自 [[ApiClient]]。
