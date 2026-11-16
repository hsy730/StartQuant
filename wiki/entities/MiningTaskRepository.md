---
标题: MiningTaskRepository
类型: 实体
状态: 草稿
创建: 2026-11-16
更新: 2026-11-16
来源: [backend/repositories/mining_task_repository.py]
---

## 是什么

`backend/repositories/mining_task_repository.py` 中的挖掘任务表数据访问类。

## 关键事实

- 同时支持按业务键 `task_id` 与主键 `id` 查询。
- `get_active_tasks` 取 `pending` 与 `running` 状态记录。
- `get_history(limit=50, offset=0)` 按创建时间倒序分页，`get_history_count` 返回总数。
- `update_status` 只更新模型上存在的属性（`hasattr` 判断），异常时回滚。
- `update_progress` 把状态置为 `running`，写入进度、当前代数、总代数、最优与平均适应度。
- `complete_task` 置 `completed`、进度 100 与完成时间，并从结果中提取最优与平均适应度。
- `abort_interrupted_tasks` 把遗留的 `pending`/`running` 批量置为 `aborted`，见 [[重启时中断任务回收]]。
- `to_summary_dict` 以结果中 `factors` 的长度给出 `factor_count`。
