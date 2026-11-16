---
标题: BacktestRouter
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/backtest.py]
---

## 是什么

`backend/api/routers/backtest.py` 中的回测路由，提供单策略回测、多策略对比以及回测历史的查询与删除。

## 关键事实

- 端点分为单次回测、多策略对比与历史管理三类，写入类请求的默认值集中在请求模型中。
- 策略类型区分单因子与多因子，多因子时以列表首个因子作为主因子。
- 滑点支持智能与自定义两种模式，智能模式依据换手率估算成本。
- 回测执行放入线程池以免阻塞事件循环，未安装回测引擎时入口直接返回 503。
- 取数、因子与回测分别委托 [[DataService]]、[[FactorService]] 与 [[VectorBTBacktestService]]，端点挂载于 [[FactorFlowAPI应用]]。
