---
标题: BacktestCharts
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [frontend/react-antd/src/components/BacktestCharts.tsx]
---

## 是什么

`frontend/react-antd/src/components/BacktestCharts.tsx` 中的回测图表组件，把单只股票的K线、因子、交易信号、净值与回撤画成一张联动组合图。

## 关键事实

- 入参是按股票代码分组的数据，组件内部用选择器切换标的。
- 信号可切换「实际成交」与「策略信号」两种展示口径。
- 因子曲线在绘制前做归一处理，见 [[因子与净值归一对比]]。
- 同时兼容单因子与多因子两种数据格式，旧格式自动补齐为空结构。
- 图表组织方式见 [[图表单实例与坐标联动]]。
- 由 [[BacktestingPage]] 调用。
