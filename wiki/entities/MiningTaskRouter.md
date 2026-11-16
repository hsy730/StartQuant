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

- 模块声明支持五种算法：genetic（DEAP 遗传规划）、pysr（符号回归）、tree_prescreen（树模型预筛选管道）、gflownet、deep_implicit。
- 请求模型 `GeneticMiningRequest` 默认值：种群 50、迭代 10 代、交叉 0.7、变异 0.3、精英 5、适应度目标 `ic_mean`、IC 门限 0.03、算法 `genetic`、频率 `D`、基础因子抽样上限 30、NSGA-II 开启。
- 端点固定为：`POST /genetic`、`GET /status/{task_id}`、`GET /results/{task_id}`、`POST /cancel/{task_id}`、`GET /active`、`GET /history`、`GET /history/{task_id}`、`DELETE /history/{task_id}`、`POST /resume/{task_id}`、`GET /checkpoints/{task_id}`。
- 内存常量：`MAX_TASKS = 100`、`TASK_TTL_SECONDS = 86400`、`MAX_FITNESS_HISTORY_ENTRIES = 1000`。
- 模块级字典 `mining_tasks` 与 `mining_services` 分离保存任务状态与服务实例，服务实例用于取消与释放内存。
- 历史与检查点通过 `MiningTaskRepository` 落库，与 [[挖掘任务与检查点模型]] 对应。
- 调度统一委托给 [[DualMiningService]]，算法由 [[MiningAlgorithmRegistry]] 注册。
- 回收策略见 [[挖掘任务内存表与TTL回收]]，恢复能力见 [[挖掘检查点续跑]]。
- 由 [[FactorFlowAPI应用]] 以 `/api/mining` 前缀注册。
