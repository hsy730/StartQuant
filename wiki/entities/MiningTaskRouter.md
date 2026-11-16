---
标题: MiningTaskRouter
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/api/routers/mining.py, backend/api/main.py]
---

## 是什么

`backend/api/routers/mining.py` 中的挖掘任务路由模块，负责任务提交、状态轮询、结果获取、取消、恢复与历史查询。

## 关键事实

- 支持多种挖掘算法，请求模型统一给出种群、迭代、交叉变异、适应度目标与 IC 门限等参数。
- 任务状态与服务实例分存于模块级内存字典，便于取消与释放。
- 历史与检查点经 `MiningTaskRepository` 落库，与 [[挖掘任务与检查点模型]] 对应。
- 调度统一委托 [[DualMiningService]]，算法由 [[MiningAlgorithmRegistry]] 注册。
- 回收策略见 [[挖掘任务内存表与TTL回收]]，由 [[FactorFlowAPI应用]] 以 `/api/mining` 前缀挂载。
