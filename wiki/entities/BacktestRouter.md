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

- 共 4 个端点：`POST /single`、`POST /comparison`、`GET /history`、`DELETE /history/{record_id}`。
- 单回测请求默认值：初始资金 100 万、佣金 0.0003、滑点 0、分位 50、方向多头、分组数 5、每笔 100 股、频率日线、分块模式 `auto`。
- 策略类型分 `single_factor` 与 `multi_factor`，多因子时取因子列表首个作为主因子。
- 滑点模式支持 `smart` 与 `custom`；智能模式调用回测服务的滑点检测，换手率按年化 12 倍传入。
- 回测执行放到线程池中，避免阻塞事件循环。
- 对比接口支持每策略单独配置因子、多空方向与 `top_pct`，年化按几何复利计算。
- 历史查询默认取 10 条，删除按记录 ID 操作。
- 依赖 [[VectorBTBacktestService]]、[[DataService]]、[[FactorService]] 与 [[BacktestRepository]]。
- 端点挂载于 [[FactorFlowAPI应用]]。
- 未安装回测引擎时在入口直接返回 503，并给出安装提示。
